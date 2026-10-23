"""Hierarchical MXFP4 publication checked against independent packed arithmetic."""

from dataclasses import dataclass
from functools import lru_cache

import cutlass.cute as cute
import pytest
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.attention.dsa_indexer.hierarchical_fused import (
    HierarchicalFusedMXFP4,
    hierarchical_scratch_shapes,
)
from b12x.attention.dsa_indexer import mxfp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="hierarchical MXFP4 requires SM120 or SM121",
)

_LOCAL_TYPES = (
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
_MERGE_TYPES = (Int32, Int32, Int32, Float32, Int32, Float32, Int32, Float32, Int32)


@lru_cache(None)
def _programs(heads, page_size, ctas):
    obj = HierarchicalFusedMXFP4(heads, page_size, ctas)
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in _LOCAL_TYPES
    )
    local = b12x_compile(
        obj.local,
        *pointers,
        Int32(1),
        Int32(1),
        Int32(1),
        Int64(1),
        Int64(1),
        Int64(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "attention.indexer.hierarchical_independent_test",
            1,
            "local",
            heads,
            page_size,
            ctas,
        ),
    )
    merge = None
    if ctas > 1:
        pointers = tuple(
            make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1)
            for t in _MERGE_TYPES
        )
        merge = b12x_compile(
            obj.merge,
            *pointers,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "attention.indexer.hierarchical_independent_test",
                1,
                "merge",
                heads,
                page_size,
                ctas,
            ),
        )
    return (
        local,
        merge,
        mxfp4._compile("sort", (2048, True), torch.cuda.current_device()),
    )


def _ptr(tensor, dtype):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)


def _decode(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)[
        (codes & 7).long()
    ]
    sign = torch.where((codes & 8) == 0, 1.0, -1.0)
    exponent = (scales.to(torch.int32) - 127).repeat_interleave(32, dim=-1)
    return torch.ldexp(magnitude * sign, exponent).bfloat16()


@dataclass
class _Producer:
    width: int
    page_size: int
    ctas: int
    q: torch.Tensor
    qs: torch.Tensor
    weights: torch.Tensor
    pool: torch.Tensor
    pages: torch.Tensor
    lengths: torch.Tensor
    active: torch.Tensor
    output: torch.Tensor
    values: torch.Tensor
    candidates: torch.Tensor
    candidate_lengths: torch.Tensor
    scratch: dict
    host_q: torch.Tensor
    host_qs: torch.Tensor
    host_weights: torch.Tensor
    host_keys: torch.Tensor
    host_key_scales: torch.Tensor
    host_pages: torch.Tensor
    local: object
    merge: object
    expand: object

    def __call__(self):
        rows = self.q.shape[0]
        s = self.scratch
        tensors = (
            self.q,
            self.qs,
            self.weights,
            self.pool,
            self.pages,
            self.lengths,
            self.active,
            self.output,
            self.values,
            s["block_indices"],
            s["block_values"],
            s["partial_indices"],
            s["partial_values"],
            s["partial_counts"],
            s["state"],
        )
        self.local(
            *(_ptr(t, d) for t, d in zip(tensors, _LOCAL_TYPES, strict=True)),
            Int32(rows),
            Int32(self.width),
            Int32(self.pages.shape[1]),
            Int64(0 if self.pages.shape[0] == 1 else self.pages.stride(0)),
            Int64(self.pool.stride(0)),
            Int64(self.pool.shape[0]),
            current_cuda_stream(),
        )
        if self.merge is not None:
            tensors = (
                self.lengths,
                self.active,
                self.output,
                self.values,
                s["block_indices"],
                s["block_values"],
                s["partial_indices"],
                s["partial_values"],
                s["partial_counts"],
            )
            self.merge(
                *(_ptr(t, d) for t, d in zip(tensors, _MERGE_TYPES, strict=True)),
                Int32(rows),
                current_cuda_stream(),
            )
        mxfp4._launch(
            "sort",
            (2048, True),
            (
                s["block_indices"],
                s["block_values"],
                self.candidates,
                s["block_values"],
                self.lengths,
                self.active,
                self.candidate_lengths,
            ),
            (rows,),
            launcher=self.expand,
        )

    def poison(self):
        for tensor in self.scratch.values():
            tensor.fill_(17)
        self.output.fill_(-17)
        self.values.fill_(float("nan"))
        self.candidates.fill_(-17)
        self.candidate_lengths.fill_(17)

    def scores(self, q=None, qs=None, weights=None):
        query = _decode(
            self.host_q if q is None else q, self.host_qs if qs is None else qs
        ).double()
        weights = self.host_weights if weights is None else weights
        lengths = self.lengths.cpu()
        active = int(self.active.cpu()[0])
        tables = self.pages.cpu().long()
        result = torch.full((query.shape[0], self.width), -torch.inf)
        for start in range(0, self.width, 2048):
            positions = torch.arange(start, min(start + 2048, self.width))
            columns = positions // self.page_size
            for row in range(query.shape[0]):
                valid = (positions < lengths[row]) & (positions < active)
                valid &= columns < tables.shape[1]
                physical = tables[
                    0 if tables.shape[0] == 1 else row,
                    columns.clamp(max=tables.shape[1] - 1),
                ]
                valid &= (physical >= 0) & (physical < self.pool.shape[0])
                slots = (
                    physical.clamp(0, self.pool.shape[0] - 1) * self.page_size
                    + positions % self.page_size
                )
                keys = _decode(
                    self.host_keys[slots], self.host_key_scales[slots]
                ).double()
                dots = (query[row] @ keys.T).bfloat16().relu()
                products = (
                    (dots.float() * weights[row].float()[:, None]).bfloat16().float()
                )
                total = torch.zeros(positions.numel(), dtype=torch.float32)
                for head in range(query.shape[1]):
                    total += products[head]
                result[row, positions] = (
                    total.bfloat16().float().masked_fill(~valid, -torch.inf)
                )
        return result


def _case(
    width,
    *,
    page_size=128,
    ctas=1,
    rows=2,
    shared=False,
    bright=False,
    large_stride=False,
):
    heads = 32
    pages = (width + page_size - 1) // page_size
    generator = torch.Generator().manual_seed(72491)
    physical_pages = 2 if large_stride else pages
    host_keys = torch.randint(
        0, 256, (physical_pages * page_size, 64), dtype=torch.uint8, generator=generator
    )
    host_sf = (
        torch.tensor([127, 126, 127, 128], dtype=torch.uint8)
        .expand(physical_pages * page_size, 4)
        .clone()
    )
    host_q = torch.randint(
        0, 256, (rows, heads, 64), dtype=torch.uint8, generator=generator
    )
    host_qs = (
        torch.tensor([126, 127, 128, 127], dtype=torch.uint8)
        .expand(rows, heads, 4)
        .clone()
    )
    host_weights = (
        torch.randint(-8, 9, (rows, heads), generator=generator).float() / 8
    ).bfloat16()
    if bright:
        host_keys.zero_()
        host_keys[: 2047 * 8].fill_(0x22)
        host_sf.fill_(127)
        host_q.fill_(0x22)
        host_qs.fill_(127)
        host_weights.fill_(1 / 32)
    if large_stride:
        pool = torch.empty_strided(
            (2, page_size * 68), (2**31 + 16, 1), dtype=torch.uint8, device="cuda"
        )
    else:
        backing = torch.empty(
            (pages, page_size * 68 + 16), dtype=torch.uint8, device="cuda"
        )
        pool = backing[:, : page_size * 68]
    pool[:, : page_size * 64].copy_(host_keys.reshape(physical_pages, page_size * 64))
    pool[:, page_size * 64 :].copy_(host_sf.reshape(physical_pages, page_size * 4))
    if large_stride:
        host_pages = torch.ones((1 if shared else rows, pages), dtype=torch.int32)
    elif bright:
        host_pages = (
            torch.arange(pages, dtype=torch.int32)[None]
            .expand(1 if shared else rows, -1)
            .clone()
        )
    else:
        host_pages = torch.stack(
            [
                torch.randperm(pages, generator=generator).int()
                for _ in range(1 if shared else rows)
            ]
        )
    table_backing = torch.empty(
        (host_pages.shape[0], pages + 3), dtype=torch.int32, device="cuda"
    )
    table = table_backing[:, :pages]
    table.copy_(host_pages)
    scratch = {
        name: torch.empty(shape, dtype=dtype, device="cuda")
        for name, (dtype, shape) in hierarchical_scratch_shapes(rows, ctas).items()
    }
    local, merge, expand = _programs(heads, page_size, ctas)
    return _Producer(
        width,
        page_size,
        ctas,
        host_q.cuda(),
        host_qs.cuda(),
        host_weights.cuda(),
        pool,
        table,
        torch.full((rows,), width, dtype=torch.int32, device="cuda"),
        torch.tensor([width], dtype=torch.int32, device="cuda"),
        torch.empty((rows, 512), dtype=torch.int32, device="cuda"),
        scratch["selected_values"],
        torch.full((rows, 16384), -1, dtype=torch.int32, device="cuda"),
        torch.zeros((rows,), dtype=torch.int32, device="cuda"),
        scratch,
        host_q,
        host_qs,
        host_weights,
        host_keys,
        host_sf,
        host_pages,
        local,
        merge,
        expand,
    )


def _assert_tokens(indices, values, expected, logical=None):
    indices, values = indices.cpu().long(), values.cpu()
    for row in range(expected.shape[0]):
        live = expected[row] > -torch.inf
        count = min(512, int(live.sum()))
        ids = indices[row, :count]
        assert bool((indices[row, count:] == -1).all())
        assert bool(torch.isneginf(values[row, count:]).all())
        assert ids.unique().numel() == count
        assert bool((ids[1:] > ids[:-1]).all())
        torch.testing.assert_close(
            values[row, :count].sort().values,
            expected[row, live].topk(count).values.sort().values,
            rtol=0,
            atol=0,
        )
        if logical is None:
            torch.testing.assert_close(
                values[row, :count], expected[row, ids], rtol=0, atol=0
            )
        else:
            for slot, position in enumerate(ids):
                matches = logical[row] == position
                assert bool(matches.any())
                assert bool((expected[row, matches] == values[row, slot]).any())


def _assert_publication(case, scores):
    candidate = case.candidates.cpu().long()
    lengths = case.candidate_lengths.cpu()
    visible_rows = (
        case.lengths.cpu().clamp_min(0).clamp_max(max(0, int(case.active.cpu()[0])))
    )
    for row in range(scores.shape[0]):
        visible = int(visible_rows[row])
        extent = min(visible, case.width)
        padded = torch.full(((extent + 7) // 8 * 8,), -torch.inf)
        padded[:extent] = scores[row, :extent]
        block_scores = padded.view(-1, 8).amax(1)
        newest = (visible - 1) // 8 if visible else -1
        if 0 <= newest < block_scores.numel():
            block_scores[newest] = torch.inf
        valid = block_scores > -torch.inf
        count = min(2048, int(valid.sum()))
        n = int(lengths[row])
        ids = candidate[row, :n]
        assert bool((candidate[row, n:] == -1).all())
        assert bool((ids[1:] > ids[:-1]).all())
        assert bool(((ids >= 0) & (ids < visible)).all())
        selected = torch.unique_consecutive(ids // 8)
        assert selected.numel() == count
        torch.testing.assert_close(
            block_scores[selected].sort().values,
            block_scores[valid].topk(count).values.sort().values,
            rtol=0,
            atol=0,
        )
        if 0 <= newest < block_scores.numel():
            assert bool((selected == newest).any())
        expanded = (
            torch.cat(
                [
                    torch.arange(int(block) * 8, min(int(block) * 8 + 8, visible))
                    for block in selected
                ]
            )
            if count
            else torch.empty(0, dtype=torch.int64)
        )
        torch.testing.assert_close(ids, expanded, rtol=0, atol=0)
    assert not bool(case.scratch["state"].count_nonzero())


def _replay_checked(graph, case):
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


@pytest.mark.parametrize(
    "ctas,page_size,shared", [(1, 64, False), (3, 256, True), (8, 24, False)]
)
def test_hierarchical_packed_arithmetic_and_mutable_invalid_pages(
    ctas, page_size, shared
):
    case = _case(4103, page_size=page_size, ctas=ctas, shared=shared)
    original_weights = case.host_weights.clone()
    case()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        try:
            with torch.cuda.graph(graph):
                case()
            for step, visible in enumerate([4103, 79, 4103, 0, 4097]):
                case.host_q = case.host_q.roll(1, -1)
                case.q.copy_(case.host_q)
                case.lengths.copy_(
                    torch.tensor([visible, max(-7, visible - 13)], dtype=torch.int32)
                )
                case.active.fill_(visible)
                case.pages.copy_(case.host_pages)
                if step % 2:
                    case.pages[:, 0] = -1
                    case.pages[:, 2] = case.pool.shape[0]
                    case.pages[:, -1] = 2**31 - 1
                if step == 2:
                    case.weights.zero_()
                    case.host_weights.zero_()
                if step == 3:
                    case.host_weights = original_weights.clone()
                    case.weights.copy_(case.host_weights)
                    case.host_keys = case.host_keys.roll(1, 0)
                    case.pool[:, : page_size * 64].copy_(
                        case.host_keys.reshape(case.pool.shape[0], page_size * 64)
                    )
                    case.host_qs = case.host_qs.roll(1, -1)
                    case.qs.copy_(case.host_qs)
                case.poison()
                _replay_checked(graph, case)
                scores = case.scores()
                _assert_tokens(case.output, case.values, scores)
                _assert_publication(case, scores)
        finally:
            graph.reset()


@pytest.mark.parametrize("ctas", [1, 4])
def test_worst_ranked_partial_newest_block_is_published(ctas):
    case = _case(16448, page_size=128, ctas=ctas, rows=1, shared=True, bright=True)
    case.lengths.fill_(16441)
    case.active.fill_(16441)
    case()
    torch.cuda.synchronize()
    scores = case.scores()
    _assert_tokens(case.output, case.values, scores)
    _assert_publication(case, scores)
    expected = torch.cat((torch.arange(2047 * 8), torch.tensor([16440]))).int()
    assert case.candidate_lengths.item() == expected.numel()
    torch.testing.assert_close(
        case.candidates[0, : expected.numel()].cpu(), expected, rtol=0, atol=0
    )
    assert scores[0, 16440] == 0
    assert bool((scores[0, : 2047 * 8] == 128).all())


def test_distant_newest_block_and_frozen_empty_replay():
    case = _case(1048576, page_size=128, ctas=4, rows=1, shared=True, bright=True)
    case()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        try:
            with torch.cuda.graph(graph):
                case()
            for visible in [524293, 1048576, 0]:
                case.lengths.fill_(visible)
                case.active.fill_(visible)
                case.poison()
                _replay_checked(graph, case)
                scores = case.scores()
                _assert_tokens(case.output, case.values, scores)
                _assert_publication(case, scores)
                if visible:
                    newest = (visible - 1) // 8 * 8
                    expected = torch.cat(
                        (torch.arange(2047 * 8), torch.arange(newest, visible))
                    ).int()
                    torch.testing.assert_close(
                        case.candidates[0, : expected.numel()].cpu(),
                        expected,
                        rtol=0,
                        atol=0,
                    )
        finally:
            graph.reset()


def test_hierarchical_source_to_consumer_frozen_replay():
    case = _case(17001, page_size=256, ctas=3, shared=False)
    generator = torch.Generator().manual_seed(49531)
    query = torch.randint(
        0, 256, case.host_q.shape, dtype=torch.uint8, generator=generator
    )
    scales = case.host_qs.clone()
    weights = (
        torch.randint(-8, 9, case.host_weights.shape, generator=generator).float() / 16
    ).bfloat16()
    output = torch.empty_like(case.output)
    values = torch.empty_like(case.values)
    consumer = mxfp4.prepare_fused_mxfp4(
        q_mxfp4=query.cuda(),
        q_scales=scales.cuda(),
        query_weights=weights.cuda(),
        index_k_cache=case.pool,
        page_table=case.pages,
        cache_lengths=case.lengths,
        active_width=case.active,
        output_indices=output,
        output_scores=values,
        candidate_indices=case.candidates,
        candidate_lengths=case.candidate_lengths,
        page_size=case.page_size,
        ctas_per_row=2,
    )
    case()
    consumer()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        try:
            with torch.cuda.graph(graph):
                case()
                consumer()
            for visible in [17001, 16441, 73, 0]:
                case.lengths.copy_(
                    torch.tensor([visible, max(0, visible - 19)], dtype=torch.int32)
                )
                case.active.fill_(visible)
                case.poison()
                for scratch in consumer.scratch:
                    scratch.fill_(17)
                output.fill_(-17)
                values.fill_(float("nan"))
                _replay_checked(graph, case)
                source_scores = case.scores()
                _assert_tokens(case.output, case.values, source_scores)
                _assert_publication(case, source_scores)
                dense = case.scores(query, scales, weights)
                logical = case.candidates.cpu().long()
                selected = dense.gather(1, logical.clamp(0, case.width - 1))
                selected.masked_fill_(logical < 0, -torch.inf)
                _assert_tokens(output, values, selected, logical)
                assert not bool(consumer.scratch[-1].count_nonzero())
        finally:
            graph.reset()


def test_hierarchical_pool_stride_above_int32():
    case = _case(2048, page_size=256, ctas=3, rows=1, shared=True, large_stride=True)
    assert case.pool.stride(0) > 2**31
    case()
    torch.cuda.synchronize()
    scores = case.scores()
    _assert_tokens(case.output, case.values, scores)
    _assert_publication(case, scores)
