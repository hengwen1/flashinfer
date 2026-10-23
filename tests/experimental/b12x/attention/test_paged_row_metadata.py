"""Split attention schedules preserve integer offsets under changing graph inputs."""

import pytest
import torch

from b12x.attention.paged.graph_replay import update_prefill_graph_chunk_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "page,msa,window,adaptive",
    [
        (64, False, -1, False),
        (128, False, -1, False),
        (64, True, -1, False),
        (128, True, -1, False),
        (64, False, 257, False),
        (128, False, 257, False),
        (64, False, -1, True),
        (128, False, -1, True),
    ],
)
@pytest.mark.parametrize("batch,max_rows", [(1, 4096), (3, 129), (8, 6)])
@pytest.mark.parametrize("fixed", [False, True])
def test_split_prefill_metadata_integer_graph(
    page, msa, window, adaptive, batch, max_rows, fixed
):
    tile, group, max_chunks = 16, 16, 64
    slots = max_rows * max_chunks
    work_capacity = batch * slots + 17
    valid_capacity = work_capacity + 5
    lengths = torch.empty(batch, device="cuda", dtype=torch.int32)
    cu = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
    fields = {
        "request_indices": torch.empty(work_capacity, device="cuda", dtype=torch.int32),
        "qo_tile_indices": torch.empty(work_capacity, device="cuda", dtype=torch.int32),
        "kv_tile_indices": torch.empty(work_capacity, device="cuda", dtype=torch.int32),
        "merge_indptr": torch.empty(
            batch * max_rows + 1, device="cuda", dtype=torch.int32
        ),
        "o_indptr": torch.empty(batch + 1, device="cuda", dtype=torch.int32),
        "block_valid_mask": torch.empty(
            valid_capacity, device="cuda", dtype=torch.int32
        ),
        "kv_chunk_size_ptr": torch.tensor([128], device="cuda", dtype=torch.int32),
        "kv_window_start_tokens": torch.empty(batch, device="cuda", dtype=torch.int32),
        "total_num_rows_ptr": torch.empty(1, device="cuda", dtype=torch.int32),
    }

    def run():
        update_prefill_graph_chunk_metadata(
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            batch=batch,
            max_q_tiles_per_req=max_rows,
            max_chunks_per_q_tile=max_chunks,
            max_q_rows_per_req=max_rows,
            cta_tile_q=tile,
            gqa_group_size=group,
            page_size=page,
            window_left=window,
            split_kv=True,
            adaptive_chunking=adaptive,
            msa_block_sparse=msa,
            fixed_kv_chunk_size=128 if fixed else 0,
            **fields,
        )

    lengths.fill_(4097)
    cu.copy_(torch.arange(batch + 1, device="cuda", dtype=torch.int32) * max_rows)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for turn in range(3):
        qlens = [max_rows if turn != 1 else 0] + [
            max(0, max_rows - i * 3 - turn) for i in range(1, batch)
        ]
        caches = [4097 if q else 0 for q in qlens]
        if batch > 1:
            qlens[-1] = 0 if turn == 0 else min(1, max_rows)
            caches[-1] = 127 if qlens[-1] else 0
        boundaries = [0]
        for q in qlens:
            boundaries.append(boundaries[-1] + q)
        lengths.copy_(torch.tensor(caches, device="cuda", dtype=torch.int32))
        cu.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        for name, tensor in fields.items():
            tensor.fill_(128 if name == "kv_chunk_size_ptr" and not fixed else -123)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        chunk_size = 128
        if adaptive:
            policy_pages = max(max((c + page - 1) // page, 1) for c in caches)
            chunk_size = max((policy_pages + max_chunks - 1) // max_chunks, 1) * page
        chunks, starts = [], []
        for q, c in zip(qlens, caches, strict=True):
            pages = max((c + page - 1) // page, 1)
            first = max(c - max(q, 1), 0)
            start = min(max(first - window, 0) // page, pages - 1) if window >= 0 else 0
            effective = max(pages - start, 1)
            if msa:
                effective = min(effective, 16 * (128 // page))
            cp = max((chunk_size + page - 1) // page, 1)
            chunks.append(max((effective + cp - 1) // cp, 1))
            starts.append(start * page)
        prefix = [0]
        merge = torch.full((batch * max_rows + 1,), -123, dtype=torch.int32)
        merge[0] = 0
        for request, (q, count) in enumerate(zip(qlens, chunks, strict=True)):
            merge[boundaries[request] + 1 : boundaries[request + 1] + 1] = (
                prefix[-1] + torch.arange(1, q + 1) * count
            )
            prefix.append(prefix[-1] + q * count)
        expected = {
            "o_indptr": torch.tensor(prefix, dtype=torch.int32),
            "merge_indptr": merge,
            "kv_window_start_tokens": torch.tensor(starts, dtype=torch.int32),
            "kv_chunk_size_ptr": torch.tensor([chunk_size], dtype=torch.int32),
            "total_num_rows_ptr": torch.tensor([boundaries[-1]], dtype=torch.int32),
        }
        offsets = torch.arange(valid_capacity)
        req = offsets // slots
        local = offsets % slots
        qt, kt = local // max_chunks, local % max_chunks
        safe = req.clamp_max(batch - 1)
        active = (
            (req < batch)
            & (offsets < work_capacity)
            & (qt < torch.tensor(qlens)[safe])
            & (kt < torch.tensor(chunks)[safe])
            & (torch.tensor(qlens)[safe] > 0)
            & (torch.tensor(caches)[safe] > 0)
        )
        expected.update(
            request_indices=req[:work_capacity].int(),
            qo_tile_indices=qt[:work_capacity].int(),
            kv_tile_indices=kt[:work_capacity].int(),
            block_valid_mask=active.int(),
        )
        for name, value in expected.items():
            torch.testing.assert_close(
                fields[name].cpu(), value, rtol=0, atol=0, msg=name
            )
    graph.reset()
