"""Capacity-compiled fused MXFP4 execution with caller-owned workspace."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from b12x._lib.compile_plan import (
    DeferredCuTeKernel,
    compile_only_launches,
    load_programs,
    program_keys,
)
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import align_up
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.preparation import FrozenMapping


@dataclass(frozen=True)
class FusedMXFP4Layout:
    caps: object
    views: tuple
    nbytes: int

    def scratch_specs(self):
        return (
            scratch_buffer_spec(
                "paged_indexer.mxfp4", nbytes=self.nbytes, device=self.caps.device
            ),
        )

    def shapes_and_dtypes(self):
        return tuple((spec.shape, spec.dtype) for spec in self.scratch_specs())

    def bind(self, *, scratch, score_width=None, **kwargs):
        storage = scratch_tensor(scratch, self.scratch_specs(), owner="fused MXFP4")
        adjustment = 0 if hasattr(storage, "fake_mode") else (-storage.data_ptr()) % 256
        views = {
            name: storage.narrow(0, offset + adjustment, size).view(dtype).view(shape)
            for name, shape, dtype, offset, size in self.views
        }
        width = (
            self.caps.max_candidates
            or self.caps.max_page_table_width * self.caps.page_size
        )
        return FusedMXFP4Runtime(
            scratch=MappingProxyType(views),
            score_width=width if score_width is None else score_width,
            **kwargs,
        )


@dataclass(frozen=True)
class FusedMXFP4Runtime:
    scratch: object
    score_width: int
    page_table: torch.Tensor
    cache_lengths: torch.Tensor
    active_width: torch.Tensor
    candidate_indices: torch.Tensor | None
    candidate_lengths: torch.Tensor | None
    candidate_output: torch.Tensor | None
    candidate_output_lengths: torch.Tensor | None


def resolve_fused_mxfp4_kwargs(query, config, device):
    """Map the typed configuration to concrete kernel constructor choices."""
    from .fused_indexer import KV_LAYOUT_PAGED

    fused = config.mxfp4
    width = query.max_candidates or query.max_page_table_width * query.page_size
    resident = (
        query.max_q_rows * fused.ctas_per_row
        <= device.identity.sm_count * fused.mxfp4_coop_residency
    )
    if config.fused_merge == "cooperative" and not resident:
        raise ValueError("cooperative MXFP4 CTAs must fit a resident GPU wave")
    serial = config.fused_merge == "serial" or not resident
    return dict(
        num_heads_static=query.num_q_heads,
        topk=512,
        kv_layout=KV_LAYOUT_PAGED,
        ctas_per_group=fused.ctas_per_row,
        num_sms=device.identity.sm_count,
        merge_threshold=width if serial else 0,
        max_seq_capacity=width,
        k_quant_page_stride=query.page_size * 68,
        mxfp4=True,
        candidates=bool(query.max_candidates),
        page_size=query.page_size,
        score_width=width,
        tile_schedule=fused.tile_schedule,
        grid_layout=fused.grid_layout,
        ordered_output=fused.ordered_output,
        output_position_capacity=query.max_page_table_width * query.page_size,
        state_reset_pdl=fused.state_reset_pdl,
        vector_output_pairs=fused.vector_output_pairs,
        mxfp4_warp_weights=fused.mxfp4_warp_weights,
        mxfp4_aligned_q=fused.mxfp4_aligned_q,
        mxfp4_query_offset_mode=fused.mxfp4_query_offset_mode,
        mxfp4_unsigned_candidate_pages=fused.mxfp4_unsigned_candidate_pages,
        mxfp4_merge_bits=fused.mxfp4_merge_bits,
        mxfp4_histogram_count_bits=fused.mxfp4_histogram_count_bits,
        mxfp4_histogram_bitmap=fused.mxfp4_histogram_bitmap,
        mxfp4_serial_state_words=fused.mxfp4_serial_state_words,
        mxfp4_small_output=fused.mxfp4_small_output,
        mxfp4_partition_count=fused.mxfp4_partition_count,
        mxfp4_score_batch_tiles=fused.mxfp4_score_batch_tiles,
        mxfp4_cutoff_strategy=fused.mxfp4_cutoff_strategy,
        mxfp4_threads=fused.mxfp4_threads,
        mxfp4_score_passes=fused.mxfp4_score_passes,
        mxfp4_coop_residency=fused.mxfp4_coop_residency,
        mxfp4_compact_shared=fused.mxfp4_compact_shared,
    )


def _kernel(query, config, device):
    from .fused_indexer import DSAFusedIndexerKernel

    return DSAFusedIndexerKernel(**resolve_fused_mxfp4_kwargs(query, config, device))


def workspace_layout(caps, query, config, device):
    """Lay out bounded carries and output temporaries in one aligned byte view."""
    rows = caps.max_q_rows
    if caps.candidate_topk_blocks:
        from .hierarchical_fused import hierarchical_scratch_shapes

        shapes = {
            name: (shape, dtype)
            for name, (dtype, shape) in hierarchical_scratch_shapes(
                rows, config.mxfp4.ctas_per_row
            ).items()
        }
    else:
        pack, state = _kernel(query, config, device).merge_scratch_elements(rows)
        shapes = {
            "selected_values": ((rows, 512), torch.float32),
            "pack_values": ((pack,), torch.float32),
            "pack_indices": ((pack,), torch.int32),
            "merge_state": ((state,), torch.int32),
        }
        if not config.mxfp4.ordered_output:
            shapes.update(
                raw_indices=((rows, 512), torch.int32),
                raw_values=((rows, 512), torch.float32),
            )
    offset, views = 0, []
    for name, (shape, dtype) in shapes.items():
        offset = align_up(offset, 256)
        size = dtype.itemsize
        for dim in shape:
            size *= dim
        views.append((name, tuple(shape), dtype, offset, size))
        offset += size
    return FusedMXFP4Layout(caps, tuple(views), offset + 255)


def _ordinary_arguments(values, views, score_width, ordered_output):
    from .kernel import _to_kernel_tensor

    lengths = values["cache_lengths"]
    output_ids = values["output_indices"] if ordered_output else views["raw_indices"]
    output_values = values["output_scores"] if ordered_output else views["raw_values"]
    ids = values["candidate_indices"]
    counts = values["candidate_lengths"]
    if ids is None:
        ids, counts = output_ids, lengths
    tensors = (
        values["q_mxfp4"],
        values["query_weights"],
        values["index_k_cache"],
        values["q_scales"],
        values["page_table"],
        lengths,
        lengths,
        lengths,
        output_ids,
        output_values,
        views["pack_values"],
        views["pack_indices"],
        views["merge_state"],
    )
    dtypes = (
        Uint8,
        BFloat16,
        Uint8,
        Uint8,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Float32,
        Float32,
        Int32,
        Int32,
    )
    alignments = (1, 2, 16, 1, 4, 4, 4, 4, 4, 4, 16, 16, 16)
    args = tuple(
        _to_kernel_tensor(t, d, assumed_align=a)
        for t, d, a in zip(tensors, dtypes, alignments, strict=True)
    )
    operands = tuple(
        _to_kernel_tensor(t, d, assumed_align=a)
        for t, d, a in (
            (values["q_scales"], Uint8, 1),
            (values["active_width"], Int32, 4),
            (ids, Int32, 4),
            (counts, Int32, 4),
        )
    )
    return (*args, current_cuda_stream(), (*operands, Int32(score_width)))


def _hierarchical_arguments(values, views, width, ctas):
    from .mxfp4 import _ptr

    rows = values["q_mxfp4"].shape[0]
    page_table = values["page_table"]
    stride = 0 if page_table.shape[0] == 1 else page_table.stride(0)
    tensors = (
        values["q_mxfp4"],
        values["q_scales"],
        values["query_weights"],
        values["index_k_cache"],
        page_table,
        values["cache_lengths"],
        values["active_width"],
        values["output_indices"],
        values["output_scores"],
        views["block_indices"],
        views["block_values"],
        views["partial_indices"],
        views["partial_values"],
        views["partial_counts"],
        views["state"],
    )
    dtypes = (
        Uint8,
        Uint8,
        BFloat16,
        Uint8,
        Int32,
        Int32,
        Int32,
        Int32,
        Float32,
        Int32,
        Float32,
        Int32,
        Float32,
        Int32,
        Int32,
    )
    local = tuple(_ptr(t, d) for t, d in zip(tensors, dtypes, strict=True))
    local = (
        *local,
        Int32(rows),
        Int32(width),
        Int32(page_table.shape[1]),
        Int64(stride),
        Int64(values["index_k_cache"].stride(0)),
        Int64(values["index_k_cache"].shape[0]),
    )
    merge = None
    if ctas > 1:
        merge = (*local[5:14], Int32(rows))
    return local, merge


@dataclass(frozen=True)
class FusedMXFP4Programs:
    fused: object
    reset: object
    local: object
    merge: object
    sort: object
    quantize: object
    write: object
    facts: FrozenMapping

    @property
    def __b12x_dependencies__(self):
        return tuple(
            p
            for p in (
                self.fused,
                self.reset,
                self.local,
                self.merge,
                self.sort,
                self.quantize,
                self.write,
            )
            if p is not None
        )

    @property
    def __b12x_programs__(self):
        return program_keys(self.__b12x_dependencies__)


def _compile_hierarchical(query, config, facts):
    from .hierarchical_fused import HierarchicalFusedMXFP4

    obj = HierarchicalFusedMXFP4(
        query.num_q_heads, query.page_size, config.mxfp4.ctas_per_row
    )
    dtypes = (
        Uint8,
        Uint8,
        BFloat16,
        Uint8,
        Int32,
        Int32,
        Int32,
        Int32,
        Float32,
        Int32,
        Float32,
        Int32,
        Float32,
        Int32,
        Int32,
    )
    pointers = tuple(
        make_ptr(d, 256, cute.AddressSpace.gmem, assumed_align=1) for d in dtypes
    )
    args = (
        *pointers,
        Int32(query.max_q_rows),
        Int32(1),
        Int32(1),
        Int64(1),
        Int64(query.page_size * 68),
        Int64(1),
        current_cuda_stream(),
    )
    local = b12x_compile(
        obj.local,
        *args,
        compile_spec=KernelCompileSpec.from_key(
            "attention.indexer.mxfp4_hierarchical_local", 2, (facts.to_dict(),)
        ),
    )
    merge = None
    if config.mxfp4.ctas_per_row > 1:
        merge = b12x_compile(
            obj.merge,
            *pointers[5:14],
            Int32(query.max_q_rows),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.indexer.mxfp4_hierarchical_merge", 2, (facts.to_dict(),)
            ),
        )
    return local, merge


@program_cache(scope="preparation")
def compile_fused_mxfp4(query_payload, config_payload, ordinal):
    """Compile selected programs from capacity metadata without live allocations."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x.preparation.device import detect_device
    from ._tuning import DsaIndexerConfig, DsaIndexerQuery
    from .mxfp4 import _compile, _FusedStateReset
    from .fused_indexer import _COOP_STATE_WORDS, _FUSED_COARSE1, _FUSED_STATE_ARRIVAL

    query = DsaIndexerQuery(**dict(query_payload))
    config = DsaIndexerConfig.from_config(FrozenMapping(config_payload))
    detected = detect_device(torch.device("cuda", ordinal))
    caps = SimpleNamespace(
        device=torch.device("cuda", ordinal),
        max_q_rows=query.max_q_rows,
        max_candidates=query.max_candidates,
        max_page_table_width=query.max_page_table_width,
        page_size=query.page_size,
        candidate_topk_blocks=query.candidate_topk_blocks,
    )
    kwargs = resolve_fused_mxfp4_kwargs(query, config, detected)
    facts = FrozenMapping(
        {
            "constructor": FrozenMapping(kwargs),
            "max_rows": query.max_q_rows,
            "hierarchical": bool(query.candidate_topk_blocks),
        }
    )
    fused = reset = local = merge = sort = None
    with torch.cuda.device(ordinal), FakeTensorMode(), compile_only_launches():
        if query.candidate_topk_blocks:
            local, merge = _compile_hierarchical(query, config, facts)
            sort = _compile("sort", (2048, True), ordinal)
        else:
            from .kernel import _to_kernel_tensor

            def empty(shape, dtype):
                return torch.empty(shape, dtype=dtype, device=caps.device)

            rows, heads = query.max_q_rows, query.num_q_heads
            width = query.max_candidates or query.max_page_table_width * query.page_size
            values = dict(
                q_mxfp4=empty((rows, heads, 64), torch.uint8),
                q_scales=empty((rows, heads, 4), torch.uint8),
                query_weights=empty((rows, heads), torch.bfloat16),
                index_k_cache=empty((1, query.page_size * 68), torch.uint8),
                page_table=empty((rows, query.max_page_table_width), torch.int32),
                cache_lengths=empty((rows,), torch.int32),
                active_width=empty((1,), torch.int32),
                output_indices=empty((rows, 512), torch.int32),
                output_scores=empty((rows, 512), torch.float32),
                candidate_indices=empty((rows, query.max_candidates), torch.int32)
                if query.max_candidates
                else None,
                candidate_lengths=empty((rows,), torch.int32)
                if query.max_candidates
                else None,
            )
            layout = workspace_layout(caps, query, config, detected)
            storage = empty((layout.nbytes,), torch.uint8)
            runtime = layout.bind(
                scratch=storage,
                page_table=values["page_table"],
                cache_lengths=values["cache_lengths"],
                active_width=values["active_width"],
                candidate_indices=values["candidate_indices"],
                candidate_lengths=values["candidate_lengths"],
                candidate_output=None,
                candidate_output_lengths=None,
            )
            kernel = _kernel(query, config, detected)
            fused = b12x_compile(
                kernel,
                *_ordinary_arguments(
                    values, runtime.scratch, width, config.mxfp4.ordered_output
                ),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.indexer.fused_indexer", 11, (facts.to_dict(),)
                ),
            )
            if kernel.merge_in_kernel:
                reset = b12x_compile(
                    _FusedStateReset(
                        rows,
                        kernel.state_words_per_group,
                        _FUSED_COARSE1,
                        _FUSED_STATE_ARRIVAL,
                        contiguous=kernel.state_words_per_group != _COOP_STATE_WORDS,
                    ),
                    _to_kernel_tensor(
                        runtime.scratch["merge_state"], Int32, assumed_align=16
                    ),
                    Int32(rows),
                    current_cuda_stream(),
                    compile_spec=KernelCompileSpec.from_key(
                        "attention.indexer.mxfp4_state_reset",
                        7,
                        (rows, kernel.state_words_per_group),
                    ),
                )
            if not config.mxfp4.ordered_output:
                sort = _compile("sort", (512, False), ordinal)
        quantize = _compile("quantize", (False, 64), ordinal)
        write = _compile("quantize", (True, query.page_size), ordinal)
    return FusedMXFP4Programs(fused, reset, local, merge, sort, quantize, write, facts)


@dataclass(frozen=True)
class FusedMXFP4Binding:
    state: object
    runtime: FusedMXFP4Runtime
    rows: int
    args: tuple
    merge_args: tuple | None
    sort_tensors: tuple | None
    output_indices: torch.Tensor
    owners: tuple


class MXFP4PreparedState:
    """Retain compiled programs and bind live buffers before graph capture."""

    def __init__(self, caps, query, config, device, programs):
        self.caps, self.query, self.config = caps, query, config
        self.layout = workspace_layout(caps, query, config, device)
        self.programs = programs
        with torch.cuda.device(caps.device):
            load_programs(programs)
            if not caps.candidate_topk_blocks:
                from .fused_indexer import _check_mxfp4_cooperative_residency

                launcher = programs.fused
                if isinstance(launcher, DeferredCuTeKernel):
                    launcher = launcher._load()
                _check_mxfp4_cooperative_residency(
                    _kernel(query, config, device),
                    launcher,
                    num_rows=caps.max_q_rows,
                    num_sms=device.identity.sm_count,
                )

    @property
    def inner(self):
        return self.layout

    @property
    def __b12x_dependencies__(self):
        return self.programs.__b12x_dependencies__

    @property
    def __b12x_programs__(self):
        return self.programs.__b12x_programs__

    def bind(
        self,
        *,
        scratch,
        q_mxfp4,
        q_scales,
        query_weights,
        index_k_cache,
        page_table,
        cache_lengths,
        active_width,
        output_indices,
        output_scores=None,
        candidate_indices=None,
        candidate_lengths=None,
        candidate_output=None,
        candidate_output_lengths=None,
        score_width=None,
        **_ignored,
    ):
        from .mxfp4 import bind_mxfp4

        rows = q_mxfp4.shape[0]
        if query_weights.ndim == 3 and query_weights.shape[-1] == 1:
            query_weights = query_weights.view(rows, self.caps.num_q_heads)
        values = dict(
            q_mxfp4=q_mxfp4,
            q_scales=q_scales,
            query_weights=query_weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            cache_lengths=cache_lengths,
            active_width=active_width,
            output_indices=output_indices,
            output_scores=output_scores,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
            candidate_output=candidate_output,
            candidate_output_lengths=candidate_output_lengths,
        )
        runtime = bind_mxfp4(self, scratch=scratch, score_width=score_width, **values)
        if output_scores is None:
            values["output_scores"] = runtime.scratch["selected_values"][:rows]
        fused = self.config.mxfp4
        if fused.mxfp4_aligned_q and (
            q_mxfp4.data_ptr() % 4 or q_scales.data_ptr() % 4
        ):
            raise ValueError(
                "aligned MXFP4 query loads require four-byte-aligned query and scales"
            )
        if fused.vector_output_pairs and (
            output_indices.data_ptr() % 8 or values["output_scores"].data_ptr() % 8
        ):
            raise ValueError(
                "paired MXFP4 output stores require eight-byte-aligned caller outputs"
            )
        merge_args = sort_tensors = None
        if self.caps.candidate_topk_blocks:
            args, merge_args = _hierarchical_arguments(
                values, runtime.scratch, runtime.score_width, fused.ctas_per_row
            )
            sort_tensors = (
                runtime.scratch["block_indices"],
                runtime.scratch["block_values"],
                candidate_output,
                runtime.scratch["block_values"],
                cache_lengths,
                active_width,
                candidate_output_lengths,
            )
        else:
            args = _ordinary_arguments(
                values, runtime.scratch, runtime.score_width, fused.ordered_output
            )
            if not fused.ordered_output:
                sort_tensors = (
                    runtime.scratch["raw_indices"],
                    runtime.scratch["raw_values"],
                    output_indices,
                    values["output_scores"],
                    cache_lengths,
                    active_width,
                    cache_lengths,
                )
        owners = (*values.values(), *runtime.scratch.values())
        return FusedMXFP4Binding(
            self, runtime, rows, args, merge_args, sort_tensors, output_indices, owners
        )

    def run(self, binding):
        from .mxfp4 import _launch

        if binding.state is not self:
            raise ValueError("MXFP4 binding belongs to a different prepared state")
        with torch.cuda.device(self.caps.device):
            rows = binding.rows
            stream = current_cuda_stream()
            if self.caps.candidate_topk_blocks:
                run_compiled(self.programs.local, (*binding.args, stream))
                if self.programs.merge is not None:
                    run_compiled(self.programs.merge, (*binding.merge_args, stream))
                _launch(
                    "sort",
                    (2048, True),
                    binding.sort_tensors,
                    (rows,),
                    launcher=self.programs.sort,
                )
            else:
                if self.programs.reset is not None:
                    run_compiled(
                        self.programs.reset, (binding.args[12], Int32(rows), stream)
                    )
                run_compiled(
                    self.programs.fused, (*binding.args[:13], stream, binding.args[14])
                )
                if self.programs.sort is not None:
                    _launch(
                        "sort",
                        (512, False),
                        binding.sort_tensors,
                        (rows,),
                        launcher=self.programs.sort,
                    )
        return binding.output_indices

    def quantize_query(self, query, **kwargs):
        from .mxfp4 import _quantize_q_mxfp4

        return _quantize_q_mxfp4(query, **kwargs, launcher=self.programs.quantize)

    def write_index_keys(self, keys, **kwargs):
        from .mxfp4 import _quantize_write_index_k_mxfp4

        return _quantize_write_index_k_mxfp4(
            keys, **kwargs, page_size=self.caps.page_size, launcher=self.programs.write
        )
