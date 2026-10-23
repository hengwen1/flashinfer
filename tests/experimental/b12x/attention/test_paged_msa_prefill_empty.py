"""Empty selected-block lists produce zero attention from reused union scratch."""

import pytest
import torch

from b12x.attention import paged
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rows", [33, 4096])
@torch.no_grad()
def test_empty_prefill_unions_clear_consumed_sentinel_under_graph_replay(
    page, dtype, rows
):
    require_b12x()
    device = torch.device("cuda:0")
    context = 4099
    physical = (context + page - 1) // page
    key = torch.zeros((physical + 1, page, 1, 128), dtype=dtype, device=device)
    value = torch.ones_like(key)
    query = torch.zeros((rows, 16, 128), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    table = torch.arange(1, physical + 1, dtype=torch.int32, device=device)[
        None
    ].contiguous()
    lengths = torch.tensor([context], dtype=torch.int32, device=device)
    cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
    selected = torch.full((1, rows, 16), -1, dtype=torch.int32, device=device)
    scale = torch.ones(1, device=device) if dtype == torch.float8_e4m3fn else None
    geometry = dict(
        device=device,
        q_dtype=query.dtype,
        kv_dtype=dtype,
        num_q_heads=16,
        num_kv_heads=1,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=page,
        batch=1,
        max_cache_page_count=table.shape[1],
        window_left=-1,
    )
    capacity = paged.extend_graph_capacity(**geometry, total_q_capacity=rows)
    caps = paged.Caps(
        device=device,
        mode="extend",
        dtype=query.dtype,
        kv_dtype=dtype,
        num_q_heads=16,
        num_kv_heads=1,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=page,
        max_total_q=rows,
        max_batch=1,
        max_page_table_width=table.shape[1],
        num_cache_pages=key.shape[0],
        max_work_items=capacity.max_work_items,
        max_partial_rows=getattr(capacity, "max_partial_rows", 0),
        use_cuda_graph=True,
        msa_block_sparse=True,
        msa_union_tile=True,
    )
    arguments = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        output=output,
        q2k_indices=selected,
        k_descale=scale,
        v_descale=scale,
    )
    plan = paged.plan(caps, invocation=paged.invocation_from_tensors(caps, **arguments))
    scratch = None
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            nonlocal scratch
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in state.scratch_plan.scratch_specs()
            ]
            binding = state.bind(scratch=scratch, **arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="msa.empty_prefill", prepare_call=prepare),))

        def run():
            binding = paged.bind(plan, scratch=scratch, **arguments)
            binding.scratch.msa_union_blocks.fill_(0x55555555)
            binding.scratch.msa_union_masks.fill_(0x55555555)
            paged.run(plan=plan, binding=binding)

        run()
        torch.cuda.synchronize()
        assert not output.count_nonzero()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for count in (rows, 0, min(rows, 17), rows):
            lengths.fill_(context if count else 0)
            cu.copy_(torch.tensor([0, count], dtype=torch.int32, device=device))
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            assert not output[:count].count_nonzero()
            assert torch.isnan(output[count:]).all()
        graph.reset()
