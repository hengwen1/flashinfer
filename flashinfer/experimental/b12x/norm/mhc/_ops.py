"""Compilation boundaries for mHC with fully caller-owned storage."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared

_BUFFER_FIELDS = ("partials", "y", "post_buffer", "comb_buffer", "out")


@torch.library.custom_op("b12x::mhc_bound", mutates_args=("buffers",))
def _bound(
    inputs: list[torch.Tensor],
    buffers: list[torch.Tensor],
    norm_weight: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    fn_bf16: torch.Tensor | None,
    operation: str,
    plan_handle: int,
    expected_m: int | None,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> None:
    from . import _impl

    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "norm.mhc")
    binding = _impl.B12XMHCBinding(
        state=state,
        plan=plan,
        expected_m=expected_m,
        pre_out=buffers[5] if len(buffers) == 6 else None,
        **dict(zip(_BUFFER_FIELDS, buffers[:5], strict=True)),
    )
    options = dict(
        binding=binding,
        _state=state,
        norm_weight=norm_weight,
        pre_mix=pre_mix,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
    )
    if operation == "pre":
        _impl._b12x_mhc_pre_impl(*inputs, **options)
    elif operation == "post_pre":
        _impl._b12x_mhc_post_pre_impl(*inputs, fn_bf16=fn_bf16, **options)
    else:
        raise ValueError("bound mHC requires pre or post_pre")


@_bound.register_fake
def _fake_bound(
    inputs,
    buffers,
    norm_weight,
    pre_mix,
    fn_bf16,
    operation,
    plan_handle,
    expected_m,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    norm_eps,
):
    pass


@torch.library.custom_op("b12x::mhc_lagged_input", mutates_args=("buffers",))
def _input(
    inputs: list[torch.Tensor],
    buffers: list[torch.Tensor],
    coefficients: list[torch.Tensor],
    norm_weight: torch.Tensor,
    pre_mix: torch.Tensor,
    operation: str,
    plan_handle: int,
) -> None:
    from . import _impl

    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "norm.mhc")
    if not state.query.split_lagged:
        raise ValueError("input phase requires a split lagged declaration")
    binding = _impl.B12XMHCBinding(
        state=state,
        plan=plan,
        partials=buffers[0],
        y=buffers[1],
        out=buffers[2],
        post_buffer=coefficients[0],
        comb_buffer=coefficients[1],
        pre_out=coefficients[2],
    )
    query = state.query
    options = dict(
        binding=binding,
        _state=state,
        _input_only=True,
        norm_weight=norm_weight,
        pre_mix=pre_mix,
        rms_eps=query.rms_eps,
        hc_eps=query.hc_eps,
        sinkhorn_iters=query.sinkhorn_iters,
        norm_eps=query.norm_eps,
    )
    entry = (
        _impl._b12x_mhc_pre_impl
        if operation == "pre"
        else _impl._b12x_mhc_post_pre_impl
    )
    entry(*inputs, **options)


@_input.register_fake
def _fake_input(
    inputs, buffers, coefficients, norm_weight, pre_mix, operation, plan_handle
):
    pass


@torch.library.custom_op(
    "b12x::mhc_lagged_coefficients", mutates_args=("coefficients",)
)
def _coefficients(
    partials: torch.Tensor,
    out: torch.Tensor,
    y: torch.Tensor,
    coefficients: list[torch.Tensor],
    scale: torch.Tensor,
    base: torch.Tensor,
    plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "norm.mhc")
    if not state.query.split_lagged:
        raise ValueError("coefficient phase requires a split lagged declaration")
    from . import _impl

    tokens, query = partials.shape[0], state.query
    if not 0 <= tokens <= query.max_tokens:
        raise ValueError("coefficient rows exceed the prepared capacity")
    _impl._validate_mhc_binding_views(
        partials=partials,
        y=y,
        out=out,
        post=coefficients[0],
        comb=coefficients[1],
        tokens=tokens,
        hidden_size=query.hidden_size,
        split_k=query.split_k,
        dtype=torch.bfloat16,
        device=state.caps.device,
    )
    _impl._validate_optional_view(
        coefficients[2],
        shape=(tokens, 4),
        dtype=torch.float32,
        device=state.caps.device,
        name="mHC pre_out",
    )
    _impl._validate_optional_view(
        scale, shape=(3,), dtype=torch.float32, device=partials.device, name="mHC scale"
    )
    _impl._validate_optional_view(
        base, shape=(24,), dtype=torch.float32, device=partials.device, name="mHC base"
    )
    if not partials.shape[0]:
        return
    state.launchers["coefficients"](
        residual=out,
        partials=partials,
        scale=scale,
        bias=base,
        y=y,
        post=coefficients[0],
        comb=coefficients[1],
        norm_weight=y,
        pre_mix=coefficients[2],
        pre_out=coefficients[2],
    )


@_coefficients.register_fake
def _fake_coefficients(partials, out, y, coefficients, scale, base, plan_handle):
    pass


def run_input(binding, inputs, *, norm_weight, pre_mix):
    """Write residual and normalized input; retain partials until coefficient completion."""
    if not complete(binding, pre_mix) or binding.plan is None:
        raise ValueError("split lagged input requires a complete caller-owned binding")
    if binding.state is not require_prepared(binding.plan, "norm.mhc"):
        raise ValueError("binding and plan have different prepared states")
    _input(
        list(inputs),
        [binding.partials, binding.y, binding.out],
        [binding.post_buffer, binding.comb_buffer, binding.pre_out],
        norm_weight,
        pre_mix,
        binding.state.query.operation,
        binding.plan.handle,
    )
    return binding.y


def run_coefficients(binding, scale, base):
    """Read retained partials and write the following boundary's FP32 coefficients."""
    if not complete(binding, binding.pre_out) or binding.plan is None:
        raise ValueError(
            "split lagged coefficients require a complete caller-owned binding"
        )
    if binding.state is not require_prepared(binding.plan, "norm.mhc"):
        raise ValueError("binding and plan have different prepared states")
    _coefficients(
        binding.partials,
        binding.out,
        binding.y,
        [binding.post_buffer, binding.comb_buffer, binding.pre_out],
        scale,
        base,
        binding.plan.handle,
    )


def complete(binding, pre_mix):
    return (
        binding is not None
        and all(getattr(binding, field) is not None for field in _BUFFER_FIELDS)
        and (pre_mix is None or binding.pre_out is not None)
    )


def run(
    binding,
    inputs,
    *,
    operation,
    norm_weight,
    pre_mix,
    fn_bf16=None,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    norm_eps,
    **outputs,
):
    if binding.state is not require_prepared(binding.plan, "norm.mhc"):
        raise ValueError("binding and plan have different prepared states")
    extras = [name for name, value in outputs.items() if value is not None]
    if extras:
        raise ValueError(
            "mHC binding owns scratch and output buffers; do not also pass "
            + ", ".join(extras)
        )
    buffers = [getattr(binding, field) for field in _BUFFER_FIELDS]
    if binding.pre_out is not None:
        buffers.append(binding.pre_out)
    _bound(
        list(inputs),
        buffers,
        norm_weight,
        pre_mix,
        fn_bf16,
        operation,
        binding.plan.handle,
        binding.expected_m,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        float(norm_eps),
    )
    tokens = inputs[0 if operation == "pre" else 1].shape[0]
    return tuple(
        value if value.shape[0] == tokens else value[:tokens]
        for value in (binding.out, binding.post_buffer, binding.comb_buffer, binding.y)
    )


@torch.library.custom_op("b12x::mhc_post_out", mutates_args=("out",))
def post(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    from . import _impl

    _impl._b12x_mhc_post_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        out=out,
        _state=_impl._prepared_state(
            plan_from_handle(plan_handle), output_mode="provided"
        ),
    )


@post.register_fake
def _fake_post(x, residual, prev_post, prev_comb, out, plan_handle):
    pass
