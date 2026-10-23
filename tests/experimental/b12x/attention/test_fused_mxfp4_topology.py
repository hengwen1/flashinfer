"""MXFP4 block topology under mutable capture and complete state cleanup."""

import cutlass
import pytest
import torch

from .test_fused_mxfp4 import _assert_selected, _case

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils.device import current_cuda_stream
from b12x.attention.dsa_indexer import fused_indexer as fused
from b12x.attention.dsa_indexer import mxfp4
from b12x.attention.dsa_indexer.kernel import _to_kernel_tensor
from b12x.preparation import CandidateResourceIneligibleError

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="MXFP4 topology requires SM120",
)


def _compile_topology(
    case, *, threads, passes, residency, compact, ctas, serial, vectors
):
    a = case.args
    rows = a["q_mxfp4"].shape[0]
    device = a["q_mxfp4"].device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    candidates = a.get("candidate_indices") is not None
    width = a["candidate_indices"].shape[1] if candidates else case.width
    pages = a["page_table"]
    if pages.shape[0] == 1:
        pages = pages.expand(rows, -1)
    kernel = fused.DSAFusedIndexerKernel(
        num_heads_static=a["q_mxfp4"].shape[1],
        topk=512,
        ctas_per_group=ctas,
        num_sms=sms,
        merge_threshold=width if serial else 0,
        max_seq_capacity=width,
        k_quant_page_stride=a["index_k_cache"].stride(0),
        mxfp4=True,
        candidates=candidates,
        page_size=case.page_size,
        score_width=width,
        ordered_output=True,
        output_position_capacity=case.width,
        state_reset_pdl=ctas > 1,
        vector_output_pairs=vectors,
        mxfp4_threads=threads,
        mxfp4_score_passes=passes,
        mxfp4_coop_residency=residency,
        mxfp4_compact_shared=compact,
    )
    pack_elements, state_elements = kernel.merge_scratch_elements(rows)
    pack_v = torch.empty(pack_elements, dtype=torch.float32, device=device)
    pack_i = torch.empty(pack_elements, dtype=torch.int32, device=device)
    state = torch.zeros(state_elements, dtype=torch.int32, device=device)
    tensors = (
        a["q_mxfp4"],
        a["query_weights"],
        a["index_k_cache"],
        a["q_scales"],
        pages,
        a["cache_lengths"],
        a["cache_lengths"],
        a["cache_lengths"],
        a["output_indices"],
        a["output_scores"],
        pack_v,
        pack_i,
        state,
    )
    dtypes = (
        cutlass.Uint8,
        cutlass.BFloat16,
        cutlass.Uint8,
        cutlass.Uint8,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
        cutlass.Int32,
    )
    args = []
    for tensor, dtype in zip(tensors, dtypes, strict=True):
        align = 1
        if tensor is a["output_indices"] or tensor is a["output_scores"]:
            align = 8 if vectors else 4
        elif tensor is pack_v or tensor is pack_i or tensor is state:
            align = 16
        args.append(_to_kernel_tensor(tensor, dtype, assumed_align=align))
    operands = tuple(
        _to_kernel_tensor(t, d, assumed_align=1)
        for t, d in (
            (a["q_scales"], cutlass.Uint8),
            (a["active_width"], cutlass.Int32),
            (a.get("candidate_indices", a["output_indices"]), cutlass.Int32),
            (a.get("candidate_lengths", a["cache_lengths"]), cutlass.Int32),
        )
    )
    operands = (*operands, cutlass.Int32(width))
    args = (*args, current_cuda_stream(), operands)
    config = (
        kernel.constructor_config,
        rows,
        case.page_size,
        case.width,
        candidates,
        threads,
        passes,
        residency,
        compact,
        ctas,
        serial,
        vectors,
    )
    launcher = b12x_compile(
        kernel,
        *args,
        compile_spec=KernelCompileSpec.from_key(
            "attention.indexer.mxfp4_topology_test", 1, config
        ),
    )
    reset = None
    reset_args = ()
    if ctas > 1:
        reset_args = (
            _to_kernel_tensor(state, cutlass.Int32, assumed_align=16),
            cutlass.Int32(rows),
        )
        reset = b12x_compile(
            mxfp4._FusedStateReset(
                rows,
                kernel.state_words_per_group,
                fused._FUSED_COARSE1,
                fused._FUSED_STATE_ARRIVAL,
                contiguous=kernel.state_words_per_group != fused._COOP_STATE_WORDS,
            ),
            *reset_args,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.indexer.mxfp4_topology_reset_test",
                2,
                (rows, kernel.state_words_per_group),
            ),
        )
    prepared = mxfp4._PreparedFusedMXFP4(
        launcher,
        args,
        (a["output_indices"], a["output_scores"], pack_v, pack_i, state),
        (*tensors, *a.values()),
        rows,
        reset,
        reset_args,
    )
    return kernel, prepared, sms


@pytest.mark.parametrize(
    "passes,ctas,residency,serial,gathered,page,width,vectors,rows",
    [
        (2, 1, 1, True, False, 64, 32768, False, 4),
        (4, 64, 1, True, False, 128, 32768, False, 4),
        (3, 64, 1, True, False, 64, 32768, True, 4),
        (2, 2, 1, False, True, 128, 32768, False, 4),
        (2, 64, 2, False, False, 128, 32768, False, 4),
        (2, 64, 2, False, True, 64, 32768, True, 4),
        (2, 2, 1, False, False, 128, 131072, False, 4),
        (2, 64, 1, True, False, 64, 131072, False, 4),
        (2, 256, 2, False, False, 128, 131072, False, 1),
    ],
)
def test_mxfp4_topology_mutable_capture(
    passes, ctas, residency, serial, gathered, page, width, vectors, rows
):
    case = _case(page_size=page, gathered=gathered, shared=gathered, width=width)
    if rows == 1:
        for name in ("q_mxfp4", "q_scales", "query_weights", "cache_lengths"):
            case.args[name] = case.args[name][:1].contiguous()
        case.args["page_table"] = case.args["page_table"][:1]
        case.args["output_indices"] = torch.empty(
            (1, 512), device="cuda", dtype=torch.int32
        )
        case.args["output_scores"] = torch.empty(
            (1, 512), device="cuda", dtype=torch.float32
        )
    if not serial and residency == 2:
        sms = torch.cuda.get_device_properties(
            case.args["q_mxfp4"].device
        ).multi_processor_count
        ctas = max(1, (2 * sms) // rows)
    case.mutate(0, width, "mixed")
    kernel, prepared, sms = _compile_topology(
        case,
        threads=512,
        passes=passes,
        residency=residency,
        compact=True,
        ctas=ctas,
        serial=serial,
        vectors=vectors,
    )
    capacity = fused._check_mxfp4_cooperative_residency(
        kernel, prepared.launcher, num_rows=rows, num_sms=sms
    )
    if not serial and rows * ctas > sms:
        assert capacity >= 2
    prepared()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            prepared()
        for step, visible, kind in (
            (0, width, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, width - 5, "zero"),
            (4, width, "negative"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in prepared.scratch:
                scratch.fill_(17)
            case.args["output_indices"].fill_(-17)
            case.args["output_scores"].fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[key] == after[key], key
            assert not bool(prepared.scratch[-1].count_nonzero())
            _assert_selected(case, expected, logical)
    graph.reset()


def test_mxfp4_topology_rejects_nonresident_cooperative_grid():
    case = _case(page_size=128, shared=True, width=32768)
    case.mutate(0, case.width, "mixed")
    sms = torch.cuda.get_device_properties(
        case.args["q_mxfp4"].device
    ).multi_processor_count
    kernel, prepared, sms = _compile_topology(
        case,
        threads=512,
        passes=2,
        residency=2,
        compact=False,
        ctas=max(1, (2 * sms) // 4),
        serial=False,
        vectors=False,
    )
    with pytest.raises(
        CandidateResourceIneligibleError, match="exceeds compiled kernel residency"
    ):
        fused._check_mxfp4_cooperative_residency(
            kernel, prepared.launcher, num_rows=4, num_sms=sms
        )


@pytest.mark.parametrize("count_bits,bitmap", [(32, False), (16, False), (16, True)])
def test_mxfp4_histogram_512_capture(monkeypatch, count_bits, bitmap):
    constructor = fused.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(
            mxfp4_merge_bits=16,
            mxfp4_histogram_count_bits=count_bits,
            mxfp4_histogram_bitmap=bitmap,
        )
        return constructor(**kwargs)

    monkeypatch.setattr(fused, "DSAFusedIndexerKernel", configured_kernel)
    test_mxfp4_topology_mutable_capture(2, 2, 1, False, True, 128, 32768, True, 4)


def test_mxfp4_compact_serial_512_capture(monkeypatch):
    constructor = fused.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(mxfp4_serial_state_words=2)
        return constructor(**kwargs)

    monkeypatch.setattr(fused, "DSAFusedIndexerKernel", configured_kernel)
    test_mxfp4_topology_mutable_capture(2, 64, 1, True, False, 64, 32768, False, 4)


def test_mxfp4_cyclic_512_capture(monkeypatch):
    constructor = fused.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(tile_schedule="cyclic", grid_layout="slice_major")
        return constructor(**kwargs)

    monkeypatch.setattr(fused, "DSAFusedIndexerKernel", configured_kernel)
    test_mxfp4_topology_mutable_capture(1, 2, 1, False, False, 64, 32768, False, 4)
