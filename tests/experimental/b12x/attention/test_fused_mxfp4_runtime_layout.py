"""Fused MXFP4 graph replay with runtime dimensions and paged storage layouts."""

import dataclasses
import importlib.util
import pathlib
import sys

import pytest
import torch
from cutlass import BFloat16, Float32, Int32, Uint8

from b12x.attention.dsa_indexer.kernel import _to_kernel_tensor

spec = importlib.util.spec_from_file_location(
    f"{__name__}._packed_fixture",
    pathlib.Path(__file__).with_name("test_fused_mxfp4.py"),
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="fused MXFP4 runtime layouts require SM120 or SM121",
)


def _kernel_tensor(value, dtype):
    return _to_kernel_tensor(value, dtype, assumed_align=1)


def _mutate_case(case, step, visible, kind):
    a = case.args
    saved = {
        name: a.pop(name)
        for name in ("candidate_indices", "candidate_lengths")
        if name in a
    }
    try:
        case.mutate(step, visible, kind)
    finally:
        a.update(saved)
    if saved:
        rows, capacity = a["candidate_indices"].shape
        generator = torch.Generator().manual_seed(91271 + step)
        ids = torch.full((rows, capacity), -1, dtype=torch.int32)
        count = min(capacity, case.width)
        for row in range(rows):
            ids[row, :count] = torch.randperm(case.width, generator=generator)[
                :count
            ].int()
        if step % 2:
            invalid = torch.tensor(
                [-1, case.width, 2**31 - 1, -(2**31)], dtype=torch.int32
            )
            ids[:, : min(4, capacity)] = invalid[: min(4, capacity)]
        a["candidate_indices"].copy_(ids)
        lengths = torch.tensor(
            [capacity - row * 13 for row in range(rows)], dtype=torch.int32
        )
        if visible <= 0:
            lengths.zero_()
        a["candidate_lengths"].copy_(lengths)
    if step == 4:
        a["page_table"][:, 1].copy_(a["page_table"][:, 3])
    return case.reference(
        a["q_mxfp4"].cpu(),
        a["q_scales"].cpu(),
        a["query_weights"].cpu(),
        a["page_table"].cpu(),
        a["cache_lengths"].cpu(),
        visible,
    )


def _bind_frame(base, case, pool, table, rows, width):
    a = case.args
    args = list(base.args)
    args[0] = _kernel_tensor(a["q_mxfp4"][:rows], Uint8)
    args[1] = _kernel_tensor(a["query_weights"][:rows], BFloat16)
    args[2] = _kernel_tensor(pool, Uint8)
    args[3] = _kernel_tensor(a["q_scales"][:rows], Uint8)
    args[4] = _kernel_tensor(table if table.shape[0] == 1 else table[:rows], Int32)
    args[5] = _kernel_tensor(a["cache_lengths"][:rows], Int32)
    args[6] = args[5]
    args[7] = args[5]
    args[8] = _kernel_tensor(base.scratch[0][:rows], Int32)
    args[9] = _kernel_tensor(base.scratch[1][:rows], Float32)
    candidates = a.get("candidate_indices", a["output_indices"])
    candidate_lengths = a.get("candidate_lengths", a["cache_lengths"])
    args[14] = (
        args[3],
        base.args[14][1],
        _kernel_tensor(candidates[:rows], Int32),
        _kernel_tensor(candidate_lengths[:rows], Int32),
        Int32(width),
    )
    return dataclasses.replace(
        base, args=tuple(args), rows=rows, owners=(*base.owners, pool, table)
    )


def _run_layout_case(
    page_size=256,
    gathered=False,
    shared=True,
    ctas=64,
    high=False,
    candidate_capacity=1024,
    rows=4,
):
    width = max(2048, page_size * 4, candidate_capacity if gathered else 0)
    width = (width + page_size - 1) // page_size * page_size
    case = m._case(
        page_size=page_size,
        gathered=gathered,
        shared=shared,
        high=high,
        offset=True,
        width=width,
    )
    a = case.args
    if rows != 4:
        for name, dtype, shape in [
            ("q_mxfp4", torch.uint8, (rows, 32, 64)),
            ("q_scales", torch.uint8, (rows, 32, 4)),
            ("query_weights", torch.bfloat16, (rows, 32)),
            ("cache_lengths", torch.int32, (rows,)),
            ("output_indices", torch.int32, (rows, 512)),
            ("output_scores", torch.float32, (rows, 512)),
        ]:
            a[name] = torch.empty(shape, dtype=dtype, device="cuda")
        if not shared:
            backing = torch.empty(
                (rows, width // page_size + 3), dtype=torch.int32, device="cuda"
            )
            a["page_table"] = backing[:, : width // page_size]
    if gathered:
        a["candidate_indices"] = torch.empty(
            (rows, candidate_capacity), dtype=torch.int32, device="cuda"
        )
        a["candidate_lengths"] = torch.empty((rows,), dtype=torch.int32, device="cuda")
    a["ctas_per_row"] = ctas
    pool_a = a["index_k_cache"]
    pages = width // page_size
    pool_backing = torch.empty(
        (pool_a.shape[0] - 1, pool_a.stride(0) + 16), dtype=torch.uint8, device="cuda"
    )
    pool_b = pool_backing[:, : page_size * 68]
    count = pages - 1
    pool_b[case.physical_base :, : page_size * 64].copy_(
        case.host_keys[: count * page_size].reshape(count, page_size * 64)
    )
    pool_b[case.physical_base :, page_size * 64 :].copy_(
        case.host_key_scales[: count * page_size].reshape(count, page_size * 4)
    )
    pool_empty = torch.empty((0, page_size * 68), dtype=torch.uint8, device="cuda")
    _mutate_case(case, 0, width, "mixed")
    base = m.mxfp4.prepare_fused_mxfp4(**a)
    scenarios = [
        (rows, candidate_capacity if gathered else width, pool_a, pages),
        (1, candidate_capacity if gathered else 1, pool_b, pages),
        (
            min(3, rows),
            candidate_capacity if gathered else 129,
            pool_b,
            max(1, pages - 2),
        ),
        (rows, candidate_capacity if gathered else min(width, 513), pool_a, pages),
        (min(3, rows), candidate_capacity if gathered else width, pool_b, 0),
        (1, candidate_capacity if gathered else width, pool_empty, pages),
    ]
    steps = [
        (0, width, "mixed"),
        (1, 73, "negative"),
        (2, 0, "zero"),
        (3, width - 5, "zero"),
        (4, width, "negative"),
    ]
    if rows > 4:
        scenarios = scenarios[:3]
        steps = steps[:3]
    with m.kernel_resolution_guard():
        for live_rows, bound, pool, columns in scenarios:
            a["index_k_cache"] = pool
            table = a["page_table"][:, :columns]
            call = _bind_frame(base, case, pool, table, live_rows, bound)
            call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    call()
                for step, visible, kind in steps:
                    expected, logical = _mutate_case(case, step, visible, kind)
                    expected = expected.masked_fill(
                        logical // page_size >= columns, -torch.inf
                    )
                    if not gathered:
                        expected = expected.masked_fill(logical >= bound, -torch.inf)
                    for scratch in call.scratch:
                        scratch.fill_(17)
                    a["output_indices"].fill_(-17)
                    a["output_scores"].fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in [
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ]:
                        assert before[key] == after[key], key
                    assert not bool(call.scratch[-1].count_nonzero())
                    selected_case = dataclasses.replace(
                        case,
                        args={
                            **a,
                            "output_indices": a["output_indices"][:live_rows],
                            "output_scores": a["output_scores"][:live_rows],
                        },
                    )
                    m._assert_selected(
                        selected_case, expected[:live_rows], logical[:live_rows]
                    )
                    assert bool((a["output_indices"][live_rows:] == -17).all())
                    assert bool(a["output_scores"][live_rows:].isnan().all())
            finally:
                graph.reset()


@pytest.mark.parametrize(
    "page_size,gathered,shared,ctas,capacity",
    [
        (8, False, False, 2, 513),
        (24, False, True, 64, 513),
        (64, True, False, 2, 128),
        (128, False, False, 2, 513),
        (256, True, True, 64, 513),
        (1024, True, False, 2, 2051),
    ],
)
def test_runtime_page_layouts_after_freeze(page_size, gathered, shared, ctas, capacity):
    _run_layout_case(
        page_size=page_size,
        gathered=gathered,
        shared=shared,
        ctas=ctas,
        candidate_capacity=capacity,
    )


@pytest.mark.parametrize("capacity", [1, 7, 128, 513, 2051, 16384])
def test_arbitrary_candidate_capacities_after_freeze(capacity):
    _run_layout_case(
        page_size=256, gathered=True, shared=True, ctas=64, candidate_capacity=capacity
    )


def test_prefill_row_capacity_rebinding_after_freeze():
    _run_layout_case(page_size=256, gathered=False, shared=True, ctas=2, rows=129)


@pytest.mark.parametrize("gathered", [False, True])
def test_runtime_pool_stride_and_physical_offset_above_int32(gathered):
    page_size, width = 256, 4096
    case = m._case(
        page_size=page_size, gathered=gathered, shared=True, offset=True, width=width
    )
    a = case.args
    a["ctas_per_row"] = 64
    _mutate_case(case, 0, width, "mixed")
    base = m.mxfp4.prepare_fused_mxfp4(**a)
    stride = 2**31 + 16
    pool = torch.empty_strided(
        (2, page_size * 68), (stride, 1), dtype=torch.uint8, device="cuda"
    )
    pool[:, : page_size * 64].copy_(
        case.host_keys[: 2 * page_size].reshape(2, page_size * 64)
    )
    pool[:, page_size * 64 :].copy_(
        case.host_key_scales[: 2 * page_size].reshape(2, page_size * 4)
    )
    a["index_k_cache"] = pool
    rows = 3
    bound = a["candidate_indices"].shape[1] if gathered else 2051
    with m.kernel_resolution_guard():
        call = _bind_frame(base, case, pool, a["page_table"], rows, bound)
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                call()
            for step, visible, kind in [
                (0, width, "mixed"),
                (1, 73, "negative"),
                (2, 0, "zero"),
                (4, width, "negative"),
            ]:
                _mutate_case(case, step, visible, kind)
                if step == 0:
                    a["page_table"].fill_(2)
                    a["page_table"][:, 0] = 0
                    a["page_table"][:, -1] = 1
                elif step == 4:
                    a["page_table"].zero_()
                    a["page_table"][:, -1] = 1
                expected, logical = case.reference(
                    a["q_mxfp4"].cpu(),
                    a["q_scales"].cpu(),
                    a["query_weights"].cpu(),
                    a["page_table"].cpu(),
                    a["cache_lengths"].cpu(),
                    visible,
                )
                if not gathered:
                    expected = expected.masked_fill(logical >= bound, -torch.inf)
                for scratch in call.scratch:
                    scratch.fill_(17)
                a["output_indices"].fill_(-17)
                a["output_scores"].fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in [
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ]:
                    assert before[key] == after[key], key
                assert not bool(call.scratch[-1].count_nonzero())
                selected = dataclasses.replace(
                    case,
                    args={
                        **a,
                        "output_indices": a["output_indices"][:rows],
                        "output_scores": a["output_scores"][:rows],
                    },
                )
                m._assert_selected(selected, expected[:rows], logical[:rows])
                assert bool((a["output_indices"][rows:] == -17).all())
                assert bool(a["output_scores"][rows:].isnan().all())
        finally:
            graph.reset()


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("unsigned", [False, True])
def test_query_address_and_weight_forms_after_freeze(monkeypatch, mode, unsigned):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(
            mxfp4_query_offset_mode=mode,
            mxfp4_unsigned_candidate_pages=unsigned,
            mxfp4_warp_weights=unsigned,
            mxfp4_aligned_q=False,
        )
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)
    _run_layout_case(
        page_size=24,
        gathered=True,
        shared=True,
        ctas=2,
        candidate_capacity=513,
    )
