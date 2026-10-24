"""Per-invocation MHC lowering; runtime consumes only resolved launchers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import partial
from types import MappingProxyType

import torch

from b12x._lib.compile_plan import attach_programs, compile_only_launches
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation._efficiency import capture_autotune_effort
from ._tuning import MhcConfig, MhcQuery, TUNING

_CONTROL_NAMES = (
    "B12X_AUTOTUNE_EFFORT",
    "B12X_AUTOTUNE_EXHAUSTIVE",
    "B12X_MHC_PREFILL_TF32_MMA",
    "B12X_MHC_PREFILL_BF16_MMA",
    "B12X_MHC_PREFILL_BF16_MIN_TOKENS",
    "B12X_MHC_PREFILL_TF32_MIN_TOKENS",
    "B12X_MHC_PREFILL_MIN_TOKENS",
    "B12X_MHC_PREFILL_BLOCK_M",
    "B12X_MHC_PREFILL_BLOCK_M_SIZE",
    "B12X_MHC_PREFILL_TILE_N",
    "B12X_MHC_PREFILL_COMPACT",
    "B12X_MHC_PREFILL_BF16_TMA",
    "B12X_MHC_DECODE_SPLITS",
    "B12X_MHC_DECODE_TILE_N",
    "B12X_MHC_DECODE_BF16X2",
    "B12X_MHC_PARTIALS_PER_CTA",
    "B12X_MHC_DECODE_FINALIZE_THREADS",
    "B12X_MHC_DECODE_FINALIZE_CTAS",
)
_CODEGEN_NAMES = (
    "_MHC_PDL",
    "_PREFILL_THREADS",
    "_PREFILL_GRAM_THREADS",
    "_PREFILL_TMA_COMPUTE_WARPS",
    "_PREFILL_TMA_THREADS",
    "_PREFILL_TMA_TILE_M",
    "_PREFILL_TMA_TILE_N",
    "_PREFILL_TMA_TILE_K",
    "_PREFILL_TMA_STAGES",
    "_PREFILL_FINALIZE_THREADS",
)


def _codegen_snapshot():
    # These compiler constants already have module-initialization lifetime.
    # Record the actual values used by the DSL, not a later environment reread.
    from . import _kernels

    return FrozenMapping({name: getattr(_kernels, name) for name in _CODEGEN_NAMES})


@dataclass(frozen=True)
class _NativeLaunch:
    route: str
    source_splits: int = 0
    decode_tile_n: int = 0
    decode_bf16x2: bool = False
    partials_per_cta: int = 4
    finalize_threads: int = 0
    finalize_groups: int = 1
    block_m: int = 2
    tile_n: int = 24


def _lower_native(query, config, device):
    controls = query.controls
    tokens, hidden = query.max_tokens, query.hidden_size
    capability = None if device.identity is None else device.identity.compute_capability
    route = query.operation
    if query.operation == "pre" and query.lagged_mix and config.backend == "tf32_tma":
        route = "lagged_tf32"
    elif route == "post_pre" and config.lagged_prepare:
        route = "post_pre"
    elif route == "post_pre":
        threshold = int(controls.get("B12X_MHC_PREFILL_MIN_TOKENS", "96"))
        bf16_threshold = int(controls.get("B12X_MHC_PREFILL_BF16_MIN_TOKENS", "384"))
        if config.backend == "tf32_tma":
            route = "tf32"
        elif (
            query.has_norm_weight
            and query.has_fn_bf16
            and tokens >= bf16_threshold
            and controls.get("B12X_MHC_PREFILL_BF16_MMA", "1") != "0"
        ):
            route = (
                "bf16_tma"
                if controls.get("B12X_MHC_PREFILL_BF16_TMA", "1") != "0"
                else "bf16_mma"
            )
        elif (
            query.has_norm_weight
            and tokens >= threshold
            and controls.get("B12X_MHC_PREFILL_BLOCK_M", "1") != "0"
        ):
            route = "block_m"
        elif (
            query.has_norm_weight
            and tokens >= threshold
            and controls.get("B12X_MHC_PREFILL_COMPACT", "1") != "0"
        ):
            route = "compact"
        else:
            route = "decode"
    source_splits = tile_n = 0
    partials = 4
    bf16x2 = False
    if route == "decode":
        raw = controls.get("B12X_MHC_DECODE_SPLITS")
        if raw is not None and raw != "":
            source_splits = int(raw)
            tile_n = (
                int(controls.get("B12X_MHC_DECODE_TILE_N", "3")) if source_splits else 0
            )
        elif capability == (12, 1) and hidden == 4096 and tokens >= 8:
            source_splits, tile_n = (8, 6) if tokens >= 10 else (4, 6)
        if source_splits and (
            source_splits <= 0 or source_splits > 32 or hidden % source_splits
        ):
            raise ValueError(
                "decode source splits must divide hidden size and be in [1, 32]"
            )
        if source_splits and (tile_n <= 0 or 24 % tile_n):
            raise ValueError("decode tile_n must divide 24")
        raw = controls.get("B12X_MHC_DECODE_BF16X2")
        bf16x2 = (
            raw != "0"
            if raw is not None
            else tokens == 16 and hidden == 4096 and source_splits > 0
        )
        bf16x2 = bf16x2 and query.bf16x2_eligible
    threads, groups = 0, 1
    if route in ("pre", "decode", "post_pre"):
        raw = controls.get("B12X_MHC_PARTIALS_PER_CTA")
        if config.lagged_prepare:
            partials = config.partials_per_cta
        elif raw is not None and raw != "":
            partials = int(raw)
        elif route != "post_pre" and capability == (12, 1) and hidden == 4096:
            partials = 25 if tokens >= 8 else 9 if tokens >= 4 else 4
        from ._kernels import _validate_post_pre_partials_per_cta

        partials = _validate_post_pre_partials_per_cta(partials)
    if route in ("pre", "decode"):
        raw = controls.get("B12X_MHC_DECODE_FINALIZE_THREADS")
        if raw is not None and raw != "":
            threads = int(raw)
        elif capability == (12, 1) and hidden == 4096:
            threads = (
                128
                if tokens >= 16
                else 512
                if tokens >= 13
                else 128
                if tokens >= 10
                else 512
                if tokens >= 8
                else 0
            )
        if threads and (
            threads <= 0 or threads > 1024 or threads % 32 or hidden % (2 * threads)
        ):
            raise ValueError("finalizer threads must evenly vectorize hidden size")
        if threads:
            raw = controls.get("B12X_MHC_DECODE_FINALIZE_CTAS")
            groups = (
                int(raw)
                if raw is not None
                else 8
                if threads == 128 and tokens >= 10
                else 1
            )
    return _NativeLaunch(
        route,
        source_splits,
        tile_n,
        bf16x2,
        partials,
        threads,
        groups,
        int(controls.get("B12X_MHC_PREFILL_BLOCK_M_SIZE", "2")),
        int(controls.get("B12X_MHC_PREFILL_TILE_N", "12" if hidden == 7168 else "24")),
    )


def _decode_query(payload):
    values = dict(payload)
    values["controls"] = FrozenMapping(values["controls"])
    values["codegen"] = FrozenMapping(values["codegen"])
    return MhcQuery(**values)


def compile_mhc(query_payload, config_payload, native_payload, ordinal):
    """Reuse the planned programs without reconstructing their fake operands."""
    query_payload = FrozenMapping(query_payload)
    if query_payload["codegen"] != _codegen_snapshot():
        raise ValueError(
            "MHC compiler code-generation snapshot differs from declaration"
        )
    compile_payload = query_payload.to_dict()
    compile_payload["controls"] = FrozenMapping(
        {
            name: value
            for name, value in query_payload["controls"].items()
            if name not in ("B12X_AUTOTUNE_EFFORT", "B12X_AUTOTUNE_EXHAUSTIVE")
        }
    )
    return _compile_mhc(
        FrozenMapping(compile_payload),
        FrozenMapping(config_payload),
        FrozenMapping(native_payload),
        ordinal,
    )


@dataclass(frozen=True)
class _MhcCompileOperands:
    mode: object
    tensors: dict
    # Fake operand metadata owns no executable programs.
    __b12x_programs__ = ()


@program_cache(scope="preparation")
def _fake_operands(query_payload, ordinal):
    """Reuse shape-faithful operands across a query's compile configurations."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    query = _decode_query(query_payload)
    device = torch.device("cuda", ordinal)
    m, h, s = query.max_tokens, query.hidden_size, query.split_k
    mode = FakeTensorMode()
    with mode:

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)

        values = dict(
            x=empty((m, h)),
            residual=empty((m, 4, h)),
            out=empty((m, 4, h)),
            mix=empty((m, 4), torch.float32),
        )
        if query.operation != "collapse":
            values.update(
                prev_post=empty((m, 4), torch.float32),
                prev_comb=empty((m, 4, 4), torch.float32),
                partials=empty((m, s, 25), torch.float32),
                fn=empty(
                    (
                        24,
                        h
                        if query.operation == "pre" and not query.expanded_residual
                        else 4 * h,
                    ),
                    torch.float32,
                ),
                fn_bf16=empty((24, 4 * h)),
                norm=empty((h,), getattr(torch, query.norm_weight_dtype))
                if query.has_norm_weight
                else None,
                scale=empty((3,), torch.float32),
                bias=empty((24,), torch.float32),
                y=empty((m, h)),
            )
    return _MhcCompileOperands(mode, values)


@program_cache(scope="preparation")
def _compile_mhc(query_payload, config_payload, native_payload, ordinal):
    """Compile exactly the chosen branch using shape-faithful CUDA FakeTensors."""
    from . import _kernels as kernels

    query = _decode_query(query_payload)
    config = MhcConfig.from_config(FrozenMapping(config_payload))
    native = _NativeLaunch(**native_payload)
    fake = _fake_operands(query_payload, ordinal)
    operands = fake.tensors
    programs = {}
    with fake.mode, compile_only_launches():
        x, residual, out = (operands[name] for name in ("x", "residual", "out"))
        if query.operation == "collapse":
            mix = operands["mix"] if query.collapse_weighted else None
            programs["collapse"] = kernels._run_mhc_collapse_launch(
                state=residual, pre_mix=mix, out=x
            )
            return programs
        prev_post, prev_comb, partials, fn, norm, scale, bias = (
            operands[name]
            for name in (
                "prev_post",
                "prev_comb",
                "partials",
                "fn",
                "norm",
                "scale",
                "bias",
            )
        )

        def post_pre_programs(pre_mix=None, y=None):
            args = dict(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                partials=partials,
                out=out,
            )
            if native.route in ("tf32", "bf16_tma", "bf16_mma"):
                programs["gram"] = kernels._run_mhc_post_pre_prefill_gram_launch(**args)
                if native.route == "tf32":
                    programs["project"] = kernels._run_mhc_prefill_tf32_project_launch(
                        out=out,
                        fn=fn,
                        partials=partials,
                        split_fp32_fn=query.lagged_mix,
                        **_projection_options(config),
                    )
                else:
                    programs["project"] = kernels._run_mhc_prefill_bf16_project_launch(
                        out=out,
                        fn_bf16=operands["fn_bf16"],
                        partials=partials,
                        use_tma=native.route == "bf16_tma",
                    )
            elif native.route == "block_m":
                programs["partial"] = (
                    kernels._run_mhc_post_pre_prefill_block_m_partial_launch(
                        **args,
                        fn=fn,
                        compute_gram=not query.lagged_mix,
                        block_m=native.block_m,
                        tile_n=native.tile_n,
                    )
                )
            elif native.route == "compact":
                programs["partial"] = kernels._run_mhc_post_pre_prefill_partial_launch(
                    **args,
                    fn=fn,
                    compute_gram=not query.lagged_mix,
                )
            else:
                programs["partial"] = kernels._run_mhc_post_pre_partial_launch(
                    **args,
                    fn=fn,
                    compute_gram=query.has_norm_weight and not query.lagged_mix,
                    pre_mix=pre_mix,
                    y=y,
                    native=native,
                )

        if query.operation == "post":
            programs["post"] = kernels._run_mhc_post_launch(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                out=out,
            )
            return programs
        if query.lagged_mix:
            mix, y = operands["mix"], operands["y"]
            if native.route == "lagged_tf32":
                from ._pre_prefill import prepare_lagged_prefill

                programs["prepare"] = prepare_lagged_prefill(residual, out, partials)
                programs["project"] = kernels._run_mhc_prefill_tf32_project_launch(
                    out=out,
                    fn=fn,
                    partials=partials,
                    split_fp32_fn=True,
                    **_projection_options(config),
                )
            elif query.operation == "pre":
                programs["partial"] = kernels._run_mhc_pre_partial_launch(
                    residual=residual if query.expanded_residual else x,
                    fn=fn,
                    partials=partials,
                    out=out,
                    compute_gram=False,
                    pre_mix=mix if config.lagged_prepare else None,
                    y=y if config.lagged_prepare else None,
                    partials_per_cta=native.partials_per_cta,
                )
            else:
                post_pre_programs(
                    mix if config.lagged_prepare else None,
                    y if config.lagged_prepare else None,
                )
            programs["finalize"] = kernels._run_mhc_finalize_gram_launch(
                residual=out,
                partials=partials,
                scale=scale,
                bias=bias,
                y=y,
                post=prev_post,
                comb=prev_comb,
                norm_weight=norm,
                pre_mix=mix,
                pre_out=mix,
                **_lagged_finalize_options(query, config, native),
            )
            if query.split_lagged:
                programs["input"] = kernels._run_mhc_lagged_norm_launch(
                    partials=partials, y=y, norm_weight=norm, norm_eps=query.norm_eps
                )
                options = _lagged_finalize_options(query, config, native)
                options["fuse_norm"] = False
                programs["coefficients"] = kernels._run_mhc_finalize_gram_launch(
                    residual=out,
                    partials=partials,
                    scale=scale,
                    bias=bias,
                    y=y,
                    post=prev_post,
                    comb=prev_comb,
                    norm_weight=norm,
                    pre_mix=mix,
                    pre_out=mix,
                    **options,
                )
            return programs
        if query.operation == "pre":
            programs["partial"] = kernels._run_mhc_pre_partial_launch(
                residual=residual if query.expanded_residual else x,
                fn=fn,
                partials=partials,
                out=out,
                compute_gram=query.has_norm_weight,
                partials_per_cta=native.partials_per_cta,
            )
        else:
            post_pre_programs()
        programs["finalize"] = kernels._run_mhc_finalize_gram_launch(
            residual=out,
            partials=partials,
            scale=scale,
            bias=bias,
            y=x,
            post=prev_post,
            comb=prev_comb,
            norm_weight=norm,
            **_finalize_options(query, config, native),
        )
    return programs


def _projection_options(config):
    return {
        "tile_m": config.projection_tile_m,
        "tile_n": config.projection_tile_n,
        "tile_k": config.projection_tile_k,
        "num_stages": config.projection_num_stages,
        "num_m_warps": config.projection_num_m_warps,
        "num_n_warps": config.projection_num_n_warps,
        "k_splits": config.projection_k_splits,
    }


def _finalize_options(query, config, native):
    compact = native.route in ("tf32", "bf16_tma", "bf16_mma", "block_m", "compact")
    return dict(
        rms_eps=query.rms_eps,
        hc_eps=query.hc_eps,
        sinkhorn_iters=query.sinkhorn_iters,
        norm_eps=query.norm_eps,
        fuse_norm=query.has_norm_weight,
        compact_partials=compact,
        compact_projection_splits=config.projection_k_splits
        if native.route == "tf32"
        else 1,
        active_source_splits=native.source_splits if native.route == "decode" else 0,
        single_cta_threads=native.finalize_threads,
        single_cta_groups=native.finalize_groups,
    )


def _lagged_finalize_options(query, config, native):
    return dict(
        rms_eps=query.rms_eps,
        hc_eps=query.hc_eps,
        sinkhorn_iters=query.sinkhorn_iters,
        norm_eps=query.norm_eps,
        fuse_norm=query.has_norm_weight,
        compact_partials=native.route
        in ("lagged_tf32", "tf32", "bf16_tma", "bf16_mma", "block_m", "compact"),
        compact_projection_splits=config.projection_k_splits
        if native.route in ("lagged_tf32", "tf32")
        else 1,
        active_source_splits=native.source_splits if native.route == "decode" else 0,
        lagged_prepared=config.lagged_prepare,
    )


def _post_pre_primary(query, config, native, programs):
    from . import _kernels as kernels

    if native.route in ("tf32", "bf16_tma", "bf16_mma"):
        gram = partial(
            kernels._run_mhc_post_pre_prefill_gram_launch, _prepared=programs["gram"]
        )
        if native.route == "tf32":
            project = partial(
                kernels._run_mhc_prefill_tf32_project_launch,
                _prepared=programs["project"],
                split_fp32_fn=query.lagged_mix,
                **_projection_options(config),
            )

            def primary(*, fn, fn_bf16=None, **args):
                gram(**args)
                project(out=args["out"], fn=fn, partials=args["partials"])
        else:
            project = partial(
                kernels._run_mhc_prefill_bf16_project_launch,
                _prepared=programs["project"],
                use_tma=native.route == "bf16_tma",
            )

            def primary(*, fn, fn_bf16, **args):
                gram(**args)
                project(out=args["out"], fn_bf16=fn_bf16, partials=args["partials"])
    else:
        if native.route == "block_m":
            entry = partial(
                kernels._run_mhc_post_pre_prefill_block_m_partial_launch,
                compute_gram=not query.lagged_mix,
                block_m=native.block_m,
                tile_n=native.tile_n,
            )
        elif native.route == "compact":
            entry = partial(
                kernels._run_mhc_post_pre_prefill_partial_launch,
                compute_gram=not query.lagged_mix,
            )
        else:
            entry = partial(
                kernels._run_mhc_post_pre_partial_launch,
                compute_gram=query.has_norm_weight and not query.lagged_mix,
                native=native,
            )

        def primary(*, fn_bf16=None, **args):
            return entry(**args, _prepared=programs["partial"])

    return primary


def _launchers(query, config, native, programs):
    from . import _kernels as kernels

    if query.operation == "post":
        return {
            "post": attach_programs(
                partial(kernels._run_mhc_post_launch, _prepared=programs["post"]),
                programs,
            )
        }
    if query.operation == "collapse":
        return {
            "collapse": attach_programs(
                partial(
                    kernels._run_mhc_collapse_launch, _prepared=programs["collapse"]
                ),
                programs,
            )
        }
    if query.lagged_mix:
        if native.route == "lagged_tf32":
            from ._pre_prefill import prepare_lagged_prefill

            prepare = partial(prepare_lagged_prefill, _prepared=programs["prepare"])
            project = partial(
                kernels._run_mhc_prefill_tf32_project_launch,
                _prepared=programs["project"],
                split_fp32_fn=True,
                **_projection_options(config),
            )

            def primary(*, residual, fn, partials, out, pre_mix, y):
                prepare(residual, out, partials)
                project(out=out, fn=fn, partials=partials)
        elif query.operation == "post_pre":
            entry = _post_pre_primary(query, config, native, programs)
            if config.lagged_prepare:
                primary = entry
            else:

                def primary(*, pre_mix, y, **args):
                    entry(**args)
        else:
            entry = partial(
                kernels._run_mhc_pre_partial_launch,
                partials_per_cta=native.partials_per_cta,
            )
            if config.lagged_prepare:
                primary = partial(
                    entry, _prepared=programs["partial"], compute_gram=False
                )
            else:

                def primary(*, pre_mix, y, **args):
                    entry(**args, _prepared=programs["partial"], compute_gram=False)

        finalize = partial(
            kernels._run_mhc_finalize_gram_launch,
            _prepared=programs["finalize"],
            **_lagged_finalize_options(query, config, native),
        )
        result = {
            "partial": attach_programs(primary, programs),
            "finalize": attach_programs(finalize, programs),
        }
        if query.split_lagged:
            result["input"] = attach_programs(
                partial(
                    kernels._run_mhc_lagged_norm_launch,
                    _prepared=programs["input"],
                    norm_eps=query.norm_eps,
                ),
                programs,
            )
            options = _lagged_finalize_options(query, config, native)
            options["fuse_norm"] = False
            result["coefficients"] = attach_programs(
                partial(
                    kernels._run_mhc_finalize_gram_launch,
                    _prepared=programs["coefficients"],
                    **options,
                ),
                programs,
            )
        return result
    finalize = partial(
        kernels._run_mhc_finalize_gram_launch,
        _prepared=programs["finalize"],
        **_finalize_options(query, config, native),
    )
    if query.operation == "pre":
        primary = partial(
            kernels._run_mhc_pre_partial_launch,
            _prepared=programs["partial"],
            compute_gram=query.has_norm_weight,
            partials_per_cta=native.partials_per_cta,
        )
    else:
        primary = _post_pre_primary(query, config, native, programs)
    return {
        "partial": attach_programs(primary, programs),
        "finalize": attach_programs(finalize, programs),
    }


def plan_mhc(caps, *, invocation=FrozenMapping(), override=None):
    from ._impl import B12XMHCScratchCaps, _MhcState, _layout_mhc_scratch

    if not isinstance(caps, B12XMHCScratchCaps):
        raise TypeError("caps must be B12XMHCScratchCaps")
    invocation = FrozenMapping(invocation)
    options = invocation.to_dict()
    operation = options.get("operation")
    if operation not in ("pre", "post", "post_pre", "collapse"):
        raise ValueError("MHC declarations require one explicit operation")
    if (
        operation in ("pre", "post_pre")
        and not {"rms_eps", "hc_eps", "sinkhorn_iters"} <= options.keys()
    ):
        raise ValueError("MHC declarations require their numerical recipe")
    supplied = set(options) - (
        set(MhcQuery.__dataclass_fields__)
        - {
            "dtype",
            "max_tokens",
            "hidden_size",
            "split_k",
            "controls",
            "codegen",
            "smem_limit",
        }
    )
    if supplied:
        raise ValueError(f"unknown MHC invocation fields: {sorted(supplied)}")
    controls = FrozenMapping(
        {
            **{name: os.environ[name] for name in _CONTROL_NAMES if name in os.environ},
            "B12X_AUTOTUNE_EFFORT": capture_autotune_effort().value,
        }
    )
    codegen = _codegen_snapshot()
    smem = (
        int(torch.cuda.get_device_properties(caps.device).shared_memory_per_block_optin)
        if caps.device.type == "cuda"
        else 0
    )
    query = MhcQuery(
        dtype=str(caps.dtype).removeprefix("torch."),
        max_tokens=caps.max_tokens,
        hidden_size=caps.hidden_size,
        split_k=caps.split_k,
        controls=controls,
        codegen=codegen,
        smem_limit=smem,
        **options,
    )
    if query.split_lagged and (
        not query.lagged_mix
        or not query.has_norm_weight
        or query.operation not in ("pre", "post_pre")
        or query.output_mode != "provided"
        or query.max_tokens >= 96
    ):
        raise ValueError(
            "split lagged execution requires provided normalized native decode buffers"
        )
    lowered = {}

    def native(config, device):
        key = (config, device.identity)
        if key not in lowered:
            lowered[key] = _lower_native(query, config, device)
        return lowered[key]

    def memory(config, device):
        if query.operation in ("post", "collapse") or query.output_mode == "functional":
            return MemoryRequirements()
        layout = _layout_mhc_scratch(caps)
        return MemoryRequirements(
            scratch=(
                scratch_buffer_spec(
                    "mhc.scratch", nbytes=layout.nbytes, device=caps.device
                ),
            )
        )

    def materialize(selection, device):
        selected_native = native(selection.config, device)
        compiled = compile_mhc(
            TUNING.encode_query(query),
            selection.config.to_dict(),
            asdict(selected_native),
            device.ordinal,
        )
        return _MhcState(
            caps=caps,
            query=query,
            config=selection.config,
            layout=_layout_mhc_scratch(caps),
            _scratch_specs=memory(selection.config, device).scratch,
            launchers=MappingProxyType(
                _launchers(query, selection.config, selected_native, compiled)
            ),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _device=caps.device,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.norm.mhc._preparation:compile_mhc",
                TUNING.encode_query(query),
                config.to_dict(),
                asdict(native(config, device)),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=memory,
        _materialize=materialize,
    )
