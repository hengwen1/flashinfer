"""Paged softmax keeps FP32 statistics independently of BF16 PV probabilities."""

import math

import pytest
import torch

from b12x.attention import paged, paged_decode
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("engine", ["paged", "paged_decode"])
@pytest.mark.parametrize("query_rows", [1, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_paged_unrounded_softmax_statistics_graph(engine, query_rows, dtype):
    require_b12x()
    device = torch.device("cuda", 0)
    heads, dim, page, batch, width = 16, 128, 64, 2, 5
    query = torch.zeros(
        batch * query_rows, heads, dim, device=device, dtype=torch.bfloat16
    )
    query[..., 0] = -2 * torch.arange(1, heads + 1, device=device)
    key = torch.zeros(batch * width, page, 1, dim, device=device)
    key[..., 0] = 1
    value = torch.zeros_like(key)
    for request in range(batch):
        key[request * width, 0, 0, 0] = 0
        value[request * width, 0] = 1
    key, value = key.to(dtype), value.to(dtype)
    table = torch.arange(batch * width, device=device, dtype=torch.int32).view(
        batch, width
    )
    lengths = torch.tensor([257, 129], device=device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * query_rows
    output = torch.empty_like(query)
    scale = torch.ones(1, device=device)
    arguments = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        k_descale=scale if dtype == torch.float8_e4m3fn else None,
        v_descale=scale if dtype == torch.float8_e4m3fn else None,
    )
    if engine == "paged_decode":
        plan = paged_decode.plan(
            paged_decode.Caps(
                device=device,
                num_q_heads=heads,
                num_kv_heads=1,
                head_dim_qk=dim,
                head_dim_vo=dim,
                page_size=page,
                max_batch=batch,
                max_q_per_req=query_rows,
                kv_dtype=dtype,
            )
        )
    else:
        caps = paged.Caps(
            device=device,
            mode="decode" if query_rows == 1 else "verify",
            dtype=query.dtype,
            kv_dtype=dtype,
            num_q_heads=heads,
            num_kv_heads=1,
            head_dim_qk=dim,
            head_dim_vo=dim,
            page_size=page,
            max_total_q=query.shape[0],
            max_batch=batch,
            max_page_table_width=width,
            num_cache_pages=key.shape[0],
            max_work_items=2048,
            max_partial_rows=2048,
            use_cuda_graph=True,
        )
        invocation = dict(
            paged.invocation_from_tensors(caps, **arguments), query_len=query_rows
        )
        plan = paged.plan(caps, invocation=invocation)
    binding = scratch = None
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            nonlocal binding, scratch
            specs = (
                state.scratch_plan.scratch_specs()
                if engine == "paged"
                else plan.scratch_specs()
            )
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device) for s in specs
            ]
            binding = state.bind(
                scratch=scratch,
                **arguments,
                **({"plan": plan} if engine == "paged_decode" else {}),
            )
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="statistics", prepare_call=prepare),))

        def run():
            if engine == "paged":
                return paged.run(
                    binding=paged.bind(plan, scratch=scratch, **arguments), plan=plan
                )
            return paged_decode.run(binding)

        result = run()
        lse = result[1] if engine == "paged" else None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for first, second in ((257, 129), (193, 65), (257, 129)):
            lengths.copy_(
                torch.tensor([first, second], device=device, dtype=torch.int32)
            )
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            visible = (
                lengths[:, None]
                - query_rows
                + torch.arange(1, query_rows + 1, device=device)
            )
            denominator = 1 + (visible.flatten()[:, None] - 1) * torch.exp(
                query[..., 0].float() / math.sqrt(dim)
            )
            expected = denominator.reciprocal().bfloat16()[..., None].expand_as(output)
            if lse is not None:
                torch.testing.assert_close(
                    lse * math.log(2), denominator.log(), rtol=0, atol=3e-6
                )
            # Primary split attention rounds normalized partials and merge shared vectors to BF16.
            torch.testing.assert_close(
                output, expected, rtol=1 / 128 if engine == "paged" else 0, atol=0
            )
        graph.reset()
