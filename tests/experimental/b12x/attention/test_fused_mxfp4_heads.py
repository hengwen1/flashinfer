"""Fused MXFP4 score stages for production heads and byte-aligned views."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils.device import current_cuda_stream
from b12x.attention.dsa_indexer.mxfp4 import _TensorCorePagedScore

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="MXFP4 fused scoring requires SM12x",
)


@cute.jit
def _flat(tensor: cute.Tensor):
    return cute.make_tensor(tensor.iterator, cute.make_layout((1 << 40,)))


class _ScoreProbe:
    def __init__(self, heads, page_size, gathered, warp_weights, aligned_q):
        self.scorer = _TensorCorePagedScore(heads, gathered, page_size)
        self.warp_weights = warp_weights
        self.aligned_q = aligned_q

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        sf: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        counts: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q, sf, weights, pool, pages, lengths, active, candidates, counts, output
        ).launch(grid=(1, q.shape[0], 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        sf: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        counts: cute.Tensor,
        output: cute.Tensor,
    ):
        _, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.warp_weights):
            weight_pairs = self.scorer._load_weight_pairs(weights, cutlass.Int32(row))
        else:
            weight_pairs = None
        self.scorer._score_tile_fused(
            _flat(q),
            _flat(sf),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(lengths),
            _flat(active),
            _flat(candidates),
            _flat(counts),
            _flat(output),
            cutlass.Int32(q.shape[0]),
            cutlass.Int32(16),
            cutlass.Int32(pages.shape[1]),
            cutlass.Int64(pages.stride[0]),
            cutlass.Int64(pool.stride[0]),
            cutlass.Int64(pool.shape[0]),
            cutlass.Int32(0),
            weight_pairs=weight_pairs,
            hardware_scale_select=self.scorer.candidates,
            aligned_q=self.aligned_q,
        )


def _decode(packed, sf):
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(*packed.shape[:-1], 128)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    scale = (sf.int() << 23).view(torch.float32).repeat_interleave(32, -1)
    return (
        lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1, 1) * scale
    ).bfloat16()


def _expected(
    q, sf, keys, key_sf, weights, lengths, pages, candidates, counts, gathered
):
    dots = torch.einsum(
        "rhd,kd->rhk", _decode(q, sf).double(), _decode(keys, key_sf).double()
    ).bfloat16()
    products = (dots.relu() * weights[:, :, None]).bfloat16().float()
    total = torch.zeros((q.shape[0], 16), dtype=torch.float32)
    for head in range(q.shape[1]):
        total += products[:, head]
    logical = candidates.long() if gathered else torch.arange(16).expand(q.shape[0], 16)
    valid = (logical >= 0) & (logical < lengths[:, None]) & (logical < 16)
    valid &= pages[:, :1] == 0
    if gathered:
        total = total.gather(1, logical.clamp(0, 15))
        valid &= torch.arange(16)[None] < counts[:, None]
    return total.bfloat16().masked_fill(~valid, -torch.inf)


def _byte_view(shape, offset):
    storage = torch.empty(
        torch.Size(shape).numel() + offset, dtype=torch.uint8, device="cuda"
    )
    return storage[offset:].view(shape)


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("page_size,gathered", [(64, False), (128, True), (256, False)])
@pytest.mark.parametrize("warp_weights", [False, True])
def test_fused_score_heads_and_byte_views(heads, page_size, gathered, warp_weights):
    generator = torch.Generator().manual_seed(93071 + heads)
    q_host = torch.randint(
        0, 256, (3, heads, 64), dtype=torch.uint8, generator=generator
    )
    keys = torch.randint(0, 256, (16, 64), dtype=torch.uint8, generator=generator)
    q_host.flatten()[:16] = torch.arange(16, dtype=torch.uint8) * 17
    keys.flatten()[:16] = torch.arange(15, -1, -1, dtype=torch.uint8) * 17
    weights_host = torch.randn((3, heads), generator=generator).bfloat16()
    weights_host[:, 0] = torch.tensor([0.75, -0.5, 1], dtype=torch.bfloat16)
    weight_storage = torch.empty(
        weights_host.numel() + 1, dtype=torch.bfloat16, device="cuda"
    )
    weights = weight_storage[1:].view_as(weights_host)
    weights.copy_(weights_host)
    assert weights.data_ptr() % 4 == 2
    pool_storage = torch.zeros(
        (1, page_size * 68 + 16), dtype=torch.uint8, device="cuda"
    )
    pool = pool_storage[:, : page_size * 68]
    pool[0, : 16 * 64].copy_(keys.flatten())
    pages_storage = torch.zeros((3, 4), dtype=torch.int32, device="cuda")
    pages = pages_storage[:, 1:2]
    lengths_host = torch.tensor([16, 11, 0], dtype=torch.int32)
    lengths = lengths_host.cuda()
    active = torch.tensor([16], dtype=torch.int32, device="cuda")
    candidates_host = torch.stack(
        [torch.randperm(16, generator=generator) for _ in range(3)]
    ).int()
    candidates_host[0, -1] = -1
    candidates_host[1, 0] = 2147483647
    candidates = candidates_host.cuda()
    counts_host = torch.tensor([16, 12, 0], dtype=torch.int32)
    counts = counts_host.cuda()
    output = torch.empty((3, 16), dtype=torch.bfloat16, device="cuda")
    stream = current_cuda_stream()
    programs = {}
    for q_exp, k_exp in [
        ([124, 125, 126, 127], [127, 128, 129, 130]),
        ([87, 123, 133, 167], [167, 131, 121, 87]),
    ]:
        sf_host = torch.tensor(q_exp, dtype=torch.uint8).expand(3, heads, 4).clone()
        key_sf = torch.tensor(k_exp, dtype=torch.uint8).expand(16, 4).clone()
        pool[0, page_size * 64 : page_size * 64 + 16 * 4].copy_(key_sf.flatten())
        for q_offset, sf_offset in [(0, 0), (1, 0), (0, 1), (2, 3), (3, 2)]:
            q = _byte_view(q_host.shape, q_offset)
            sf = _byte_view(sf_host.shape, sf_offset)
            q.copy_(q_host)
            sf.copy_(sf_host)
            aligned_q = q.data_ptr() % 4 == 0 and sf.data_ptr() % 4 == 0
            tensors = (
                q,
                sf,
                weights,
                pool,
                pages,
                lengths,
                active,
                candidates,
                counts,
                output,
            )
            args = tuple(
                from_dlpack(t, assumed_align=a)
                for t, a in zip(tensors, [1, 1, 2, 16, 4, 4, 4, 4, 4, 2], strict=True)
            )
            if aligned_q not in programs:
                programs[aligned_q] = cute.compile(
                    _ScoreProbe(heads, page_size, gathered, warp_weights, aligned_q),
                    *args,
                    stream,
                )
            call = programs[aligned_q]
            output.fill_(float("nan"))
            call(*args, stream)
            torch.cuda.synchronize()
            expected = _expected(
                q_host,
                sf_host,
                keys,
                key_sf,
                weights_host,
                lengths_host,
                pages.cpu(),
                candidates_host,
                counts_host,
                gathered,
            )
            torch.testing.assert_close(
                output.cpu().view(torch.int16),
                expected.view(torch.int16),
                rtol=0,
                atol=0,
            )
            if heads in (1, 4, 64) and q_offset == 2 and sf_offset == 3:
                graph = torch.cuda.CUDAGraph()
                with kernel_resolution_guard():
                    with torch.cuda.graph(graph):
                        call(*args, current_cuda_stream())
                    changed_q = q_host ^ 0x88
                    changed_sf = sf_host + 1
                    changed_weights = -weights_host
                    q.copy_(changed_q)
                    sf.copy_(changed_sf)
                    weights.copy_(changed_weights)
                    output.fill_(float("nan"))
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
                    expected = _expected(
                        changed_q,
                        changed_sf,
                        keys,
                        key_sf,
                        changed_weights,
                        lengths_host,
                        pages.cpu(),
                        candidates_host,
                        counts_host,
                        gathered,
                    )
                    torch.testing.assert_close(
                        output.cpu().view(torch.int16),
                        expected.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                graph.reset()
                weights.copy_(weights_host)


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize(
    "variant", ["small256", "small512", "stream512-p1", "stream512-p4"]
)
def test_fused_head_variants_mutable_capture(monkeypatch, heads, variant):
    from types import SimpleNamespace

    from .test_fused_mxfp4_topology import _compile_topology
    from b12x.attention.dsa_indexer import fused_indexer

    generator = torch.Generator().manual_seed(93071 + heads)
    rows, logical_width, page_size = 3, 1024, 64
    gathered = variant.startswith("small")
    width = int(variant[5:]) if gathered else logical_width
    q_host = torch.randint(
        0, 256, (rows, heads, 64), dtype=torch.uint8, generator=generator
    )
    keys = torch.randint(0, 256, (16, 64), dtype=torch.uint8, generator=generator)
    weights_host = torch.randn((rows, heads), generator=generator).bfloat16()
    weights_host[:, 0] = torch.tensor([0.75, -0.5, 1], dtype=torch.bfloat16)
    sf_host = (
        torch.tensor([124, 125, 126, 127], dtype=torch.uint8)
        .expand(rows, heads, 4)
        .clone()
    )
    key_sf = torch.tensor([127, 128, 129, 130], dtype=torch.uint8).expand(16, 4).clone()
    q = _byte_view(q_host.shape, 1)
    sf = _byte_view(sf_host.shape, 3)
    weight_storage = torch.empty(
        weights_host.numel() + 1, dtype=torch.bfloat16, device="cuda"
    )
    weights = weight_storage[1:].view_as(weights_host)
    pool_storage = torch.zeros(
        (1, page_size * 68 + 16), dtype=torch.uint8, device="cuda"
    )
    pool = pool_storage[:, : page_size * 68]
    pool[0, : page_size * 64].copy_(keys.repeat(4, 1).flatten())
    pool[0, page_size * 64 :].copy_(key_sf.repeat(4, 1).flatten())
    table_storage = torch.zeros(
        (rows, logical_width // page_size + 2), dtype=torch.int32, device="cuda"
    )
    pages = table_storage[:, 1:-1]
    lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    active = torch.empty(1, dtype=torch.int32, device="cuda")
    candidate_host = torch.stack(
        [
            torch.randperm(logical_width, generator=generator)[:width]
            for _ in range(rows)
        ]
    ).int()
    candidate_host[0, 0] = -1
    candidate_host[1, 0] = 2147483647
    candidates = candidate_host.cuda()
    counts = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    scores = torch.empty((rows, 512), dtype=torch.float32, device="cuda")
    args = dict(
        q_mxfp4=q,
        q_scales=sf,
        query_weights=weights,
        index_k_cache=pool,
        page_table=pages,
        cache_lengths=lengths,
        active_width=active,
        output_indices=indices,
        output_scores=scores,
    )
    if gathered:
        args.update(candidate_indices=candidates, candidate_lengths=counts)
    case = SimpleNamespace(args=args, page_size=page_size, width=logical_width)
    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(mxfp4_small_output=gathered, mxfp4_aligned_q=False)
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)

    def mutate(step):
        q_values = q_host if step == 0 else q_host ^ 0x88
        sf_values = sf_host + (step % 2)
        weight_values = weights_host if step == 0 else -weights_host
        if step == 3:
            weight_values = torch.zeros_like(weights_host)
        q.copy_(q_values)
        sf.copy_(sf_values)
        weights.copy_(weight_values)
        live = torch.tensor([logical_width, logical_width - 11, 0], dtype=torch.int32)
        if step == 1:
            live = torch.tensor([73, 0, logical_width], dtype=torch.int32)
        visible = 0 if step == 2 else logical_width
        lengths.copy_(live)
        active.fill_(visible)
        base = _expected(
            q_values,
            sf_values,
            keys,
            key_sf,
            weight_values,
            torch.full((rows,), 16, dtype=torch.int32),
            torch.zeros((rows, 1), dtype=torch.int32),
            torch.arange(16).expand(rows, -1),
            torch.full((rows,), 16, dtype=torch.int32),
            False,
        ).float()
        logical = (
            candidate_host.long()
            if gathered
            else torch.arange(logical_width).expand(rows, -1)
        )
        valid = (logical >= 0) & (logical < live[:, None]) & (logical < visible)
        expected = base.gather(1, logical.clamp_min(0).remainder(16)).masked_fill(
            ~valid, -torch.inf
        )
        return base, logical, expected

    mutate(0)
    passes = 4 if variant.endswith("p4") else 1
    kernel, prepared, sms = _compile_topology(
        case,
        threads=512,
        passes=passes,
        residency=1,
        compact=not gathered,
        ctas=1,
        serial=True,
        vectors=False,
    )
    assert kernel.num_heads_static == heads
    assert kernel.mxfp4_small_output == gathered
    fused_indexer._check_mxfp4_cooperative_residency(
        kernel, prepared.launcher, num_rows=rows, num_sms=sms
    )
    prepared()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            prepared()
        for step in range(4):
            base, logical, expected = mutate(step)
            for scratch in prepared.scratch:
                scratch.fill_(17)
            indices.fill_(-17)
            scores.fill_(float("nan"))
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
            actual_indices, actual_scores = indices.cpu().long(), scores.cpu()
            for row in range(rows):
                count = min(512, int(torch.isfinite(expected[row]).sum()))
                chosen = actual_indices[row, :count]
                assert bool((chosen >= 0).all())
                assert bool((chosen[1:] > chosen[:-1]).all())
                assert bool((chosen[:, None] == logical[row]).any(-1).all())
                torch.testing.assert_close(
                    actual_scores[row, :count].view(torch.int32),
                    base[row, chosen.remainder(16)].view(torch.int32),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_scores[row, :count].sort().values.view(torch.int32),
                    expected[row].topk(count).values.sort().values.view(torch.int32),
                    rtol=0,
                    atol=0,
                )
                assert bool((actual_indices[row, count:] == -1).all())
                assert bool((actual_scores[row, count:] == -torch.inf).all())
    graph.reset()
