"""Prepared MSA selection preserves signed scores, causal tails and graph inputs."""

import pytest
import torch

from b12x.attention.dsa_indexer import MSACaps, plan_msa, run_msa
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "pattern", ["normal", "exponents", "cancellation", "nan", "near_ties"]
)
def test_msa_score_canonical_selected_indices(pattern):
    from b12x.attention.dsa_indexer.msa import _score, _select
    import triton

    torch.manual_seed(934)
    rows, heads, blocks, page_size = 19, 2, 37, 128
    query = torch.randn(rows, heads, 128, device="cuda")
    keys = torch.randn(2 * blocks, page_size, 128, device="cuda")
    if pattern == "exponents":
        query *= torch.exp2(torch.arange(128, device="cuda") % 17 - 8)
        keys *= torch.exp2(torch.arange(128, device="cuda") % 19 - 9)
    elif pattern == "cancellation":
        query[..., 1::2] = query[..., ::2]
        keys[..., 1::2] = -keys[..., ::2]
        keys[..., -1] += 0.0625
    elif pattern == "nan":
        query[0, :, 0] = torch.nan
        keys[0, ::2, 0] = torch.nan
    elif pattern == "near_ties":
        keys[:] = keys[0, 0].clone()
        keys[..., -1] = torch.arange(2 * blocks, device="cuda")[:, None] * 2.0**-9
        query[..., -1] = 2.0**-9
    query = query.to(torch.float8_e4m3fn)
    keys = keys.to(torch.float8_e4m3fn)
    pages = torch.randperm(2 * blocks, device="cuda").int().view(2, blocks)
    lengths = torch.tensor(
        [blocks * 128 - 1, blocks * 128 - 65], device="cuda", dtype=torch.int32
    )
    cu = torch.tensor([0, 7, 18], device="cuda", dtype=torch.int32)
    positions = torch.tensor(
        [0, 127, 128, 255, 2047, 4096, blocks * 128 - 2]
        + list(range(blocks * 128 - 76, blocks * 128 - 65))
        + [-1],
        device="cuda",
    )
    scores = torch.full((heads, rows, blocks), -torch.inf, device="cuda")
    actual = torch.empty(heads, rows, 16, device="cuda", dtype=torch.int32)

    def run():
        _score[(1, 2, heads * blocks)](
            query,
            keys,
            pages,
            lengths,
            cu,
            positions,
            scores,
            rows,
            heads,
            blocks,
            page_size,
            blocks,
            *keys.stride()[:2],
            *query.stride()[:2],
            16,
            blocks,
            num_warps=4,
        )
        _select[(rows, heads)](
            scores,
            positions,
            cu,
            actual,
            rows,
            heads,
            blocks,
            2,
            triton.next_power_of_2(blocks),
            num_warps=4,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for turn in range(3):
        if turn:
            query.view(torch.uint8).copy_(query.view(torch.uint8).roll(1, dims=0))
            keys.view(torch.uint8).copy_(keys.view(torch.uint8).roll(1, dims=1))
        scores.fill_(-torch.inf)
        graph.replay()
        expected = torch.full_like(actual, -1)
        for request in range(2):
            key = keys[pages[request].long()].float().flatten(0, 1)
            tokens = torch.arange(key.shape[0], device="cuda")
            for row in range(int(cu[request]), int(cu[request + 1])):
                for head in range(heads):
                    values = query[row, head].float() @ key.T
                    values.masked_fill_(
                        (tokens > positions[row]) | (tokens >= lengths[request]),
                        -torch.inf,
                    )
                    ranked = (
                        values.nan_to_num(
                            nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
                        )
                        .view(blocks, 128)
                        .amax(-1)
                    )
                    ranked[int(positions[row]) // 128] = torch.inf
                    chosen = []
                    for _ in range(16):
                        best = ranked.max()
                        if best == -torch.inf:
                            break
                        index = int(torch.nonzero(ranked == best).flatten().min())
                        chosen.append(index)
                        ranked[index] = -torch.inf
                    chosen.sort()
                    expected[head, row, : len(chosen)] = torch.tensor(
                        chosen, device="cuda", dtype=torch.int32
                    )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("blocks", [1, 15, 37, 129, 8192])
def test_msa_selector_exact_order_and_mutable_graph(blocks):
    from b12x.attention.dsa_indexer.msa import _select
    import triton

    rows, heads = 12, 2
    scores = torch.empty(heads, rows, blocks, device="cuda", dtype=torch.float32)
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)
    cu = torch.empty(3, device="cuda", dtype=torch.int32)
    output = torch.empty(heads, rows, 16, device="cuda", dtype=torch.int32)

    def run():
        _select[(rows, heads)](
            scores,
            positions,
            cu,
            output,
            rows,
            heads,
            blocks,
            2,
            triton.next_power_of_2(blocks),
            num_warps=4,
        )

    scores.zero_()
    positions.fill_(127)
    cu.copy_(torch.tensor([0, 6, rows], device="cuda", dtype=torch.int32))
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for turn, total in enumerate((rows, rows - 3, rows)):
        values = (
            torch.arange(heads * rows * blocks).reshape(heads, rows, blocks) % 7
        ).float() - 5
        values[0, 4, 0] = torch.inf
        values[0, 5, :] = -torch.inf
        values[1, 6, min(1, blocks - 1)] = torch.nan
        values[0, 8, :] = torch.nan
        values[0, 9, :] = torch.nan
        values[0, 7, ::2] = -0.0
        values[0, 7, 1::2] = 0.0
        values[1, 7, ::2] = torch.inf
        values[1, 8, ::128] = torch.inf
        values = values.roll(turn * 3, dims=-1)
        pos = torch.tensor(
            [
                -1,
                0,
                127,
                128,
                255,
                2047,
                2304,
                4095,
                blocks * 128 - 1,
                (triton.next_power_of_2(blocks) + 3) * 128,
                256,
                0,
            ]
        )
        pos.clamp_max_(blocks * 128 - 1)
        pos[9] = (triton.next_power_of_2(blocks) + 3) * 128
        pos[3:] = (pos[3:] - turn * 128).clamp_min(-1)
        scores.copy_(values)
        positions.copy_(pos)
        cu.copy_(
            torch.tensor([0, min(6, total), total], device="cuda", dtype=torch.int32)
        )
        output.fill_(-77)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        expected = torch.full((heads, rows, 16), -1, dtype=torch.int32)
        ids = torch.arange(blocks)
        for head in range(heads):
            for row in range(rows):
                live = row < total and pos[row] >= 0
                ranked = values[head, row].clone()
                ranked.masked_fill_(
                    ~((ids <= pos[row] // 128) & (row < total)), -torch.inf
                )
                if live and pos[row] // 128 < blocks:
                    ranked[pos[row] // 128] = torch.inf
                ranked.nan_to_num_(nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
                selected = []
                for _ in range(16):
                    best = ranked.max()
                    equal = torch.nonzero(ranked == best).flatten()
                    if live and best != -torch.inf and equal.numel():
                        index = int(equal.min())
                        selected.append(index)
                        ranked[index] = -torch.inf
                selected.sort()
                expected[head, row, : len(selected)] = torch.tensor(
                    selected, dtype=torch.int32
                )
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("capacity", [4096, 1048576])
def test_prepared_msa_unit_scale_fp8_graph(page_size, capacity):
    device = torch.device("cuda", 0)
    torch.manual_seed(410)
    rows, heads, batch, tokens = 19, 1, 2, 2305
    width = (tokens + page_size - 1) // page_size
    table_width = capacity // page_size
    query = torch.randn(rows, heads, 128, device=device).to(torch.float8_e4m3fn)
    backing = torch.randn(batch * width, page_size * 128 + 64, device=device).to(
        torch.float8_e4m3fn
    )
    cache = backing[:, : page_size * 128].view(batch * width, page_size, 128)
    table = torch.full((batch, table_width), -1, dtype=torch.int32, device=device)
    table[:, :width] = (
        torch.randperm(batch * width, device=device).int().view(batch, width)
    )
    lengths = torch.tensor([tokens, 257], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 7, 18], device=device, dtype=torch.int32)
    positions = torch.cat(
        (
            torch.arange(tokens - 7, tokens, device=device),
            torch.arange(246, 257, device=device),
            torch.tensor([-1], device=device),
        )
    )
    output = torch.empty(heads, rows, 16, device=device, dtype=torch.int32)
    args = dict(
        query=query,
        cache=cache,
        page_table=table,
        cache_lengths=lengths,
        cu_seqlens_q=cu,
        positions=positions,
    )
    caps = MSACaps(
        device=device,
        max_q_rows=rows,
        max_batch=batch,
        max_page_table_width=table_width,
        page_size=page_size,
    )
    plan = plan_msa(caps, **args)
    scratch = None
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            nonlocal scratch
            (spec,) = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            return PreparedCall(
                run=lambda: state.run(
                    query, cache, table, lengths, cu, positions, output, scratch
                ),
                owners=(scratch,),
            )

        session.prepare((plan.request(name="msa", prepare_call=prepare),))
        run_msa(plan, **args, output=output, scratch=scratch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_msa(plan, **args, output=output, scratch=scratch)
        for shift in (0, -1, -2048):
            positions[:7].add_(shift)
            if shift != -2048:
                positions[7:18].add_(shift)
            scratch.fill_(255)
            graph.replay()
            torch.cuda.synchronize()
            for request in range(batch):
                start, end = int(cu[request]), int(cu[request + 1])
                key = cache[table[request, :width].long()].float().flatten(0, 1)
                for row in range(start, end):
                    score = query[row, 0].float() @ key.T
                    valid = torch.arange(key.shape[0], device=device) <= positions[row]
                    score.masked_fill_(~valid, -torch.inf)
                    padded = torch.nn.functional.pad(
                        score, (0, (-score.numel()) % 128), value=-torch.inf
                    )
                    blocks = padded.view(-1, 128).amax(-1)
                    blocks[int(positions[row]) // 128] = torch.inf
                    ids = blocks.topk(min(16, blocks.numel())).indices
                    ids = ids[blocks[ids] != -torch.inf].sort().values
                    expected = torch.full((16,), -1, device=device, dtype=torch.int32)
                    expected[: ids.numel()] = ids.int()
                    torch.testing.assert_close(output[0, row], expected, rtol=0, atol=0)
            assert bool((output[:, 18] == -1).all())
        graph.reset()
