"""Verifier schedule capacity is independent of physical packed query rows."""

import pytest
import torch

from b12x.attention import paged
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_verify_schedule_envelope_preserves_physical_query_capacity(
    selected, metadata_dtype, kv_dtype
):
    device = require_b12x()
    batch, width, rows, heads, dimension = 2, 7, 5, 16, 128
    query = (
        torch.randn((rows, heads, dimension), device=device, dtype=torch.bfloat16) * 0.1
    )
    key = torch.randn((4, 128, 1, dimension), device=device, dtype=torch.bfloat16) * 0.1
    value = torch.randn_like(key)
    key, value = key.to(kv_dtype), value.to(kv_dtype)
    output = torch.empty_like(query)
    table = torch.arange(4, device=device, dtype=metadata_dtype).reshape(batch, 2)
    lengths = torch.full((batch,), 137, device=device, dtype=metadata_dtype)
    cu = torch.tensor([0, 2, rows], device=device, dtype=metadata_dtype)
    indices = None
    if selected:
        indices = torch.full((1, rows, 16), -1, device=device, dtype=torch.int32)
        indices[:, :, :2] = torch.tensor([0, 1], device=device, dtype=torch.int32)
    arguments = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        q2k_indices=indices,
    )
    if kv_dtype == torch.float8_e4m3fn:
        scale = torch.ones(1, device=device, dtype=torch.float32)
        arguments.update(k_descale=scale, v_descale=scale)
    geometry = dict(
        device=device,
        q_dtype=query.dtype,
        kv_dtype=key.dtype,
        num_q_heads=heads,
        num_kv_heads=1,
        head_dim_qk=dimension,
        head_dim_vo=dimension,
        page_size=128,
        batch=batch,
        max_cache_page_count=2,
    )
    capacity = paged.verify_graph_capacity(**geometry, query_len=width)
    caps = paged.Caps(
        device=device,
        mode="verify",
        dtype=query.dtype,
        kv_dtype=key.dtype,
        num_q_heads=heads,
        num_kv_heads=1,
        head_dim_qk=dimension,
        head_dim_vo=dimension,
        page_size=128,
        max_total_q=batch * width,
        max_batch=batch,
        max_page_table_width=2,
        num_cache_pages=4,
        max_work_items=(
            max(capacity.max_work_items, batch * width * 32)
            if selected
            else capacity.max_work_items
        ),
        max_partial_rows=(
            max(capacity.max_partial_rows, batch * width * 32)
            if selected
            else capacity.max_partial_rows
        ),
        use_cuda_graph=True,
        copy_runtime_metadata=metadata_dtype != torch.int32,
        msa_block_sparse=selected,
    )
    invocation = dict(paged.invocation_from_tensors(caps, **arguments), query_len=width)
    if selected:
        invocation["force_split_kv"] = True
    plan = paged.plan(caps, invocation=invocation)
    rectangular_query = query.new_zeros((batch * width, heads, dimension))
    rectangular_output = torch.empty_like(rectangular_query)
    rectangular_arguments = dict(
        arguments, q=rectangular_query, output=rectangular_output
    )
    if selected:
        rectangular_indices = torch.full(
            (1, batch * width, 16), -1, device=device, dtype=torch.int32
        )
        rectangular_indices[:, :rows].copy_(indices)
        rectangular_arguments["q2k_indices"] = rectangular_indices
    rectangular_invocation = dict(
        paged.invocation_from_tensors(caps, **rectangular_arguments), query_len=width
    )
    if selected:
        rectangular_invocation["force_split_kv"] = True
    rectangular_plan = paged.plan(caps, invocation=rectangular_invocation)
    scratch = None
    rectangular_scratch = None
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

        def prepare_rectangular(state):
            nonlocal rectangular_scratch
            rectangular_scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in state.scratch_plan.scratch_specs()
            ]
            binding = state.bind(scratch=rectangular_scratch, **rectangular_arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(rectangular_scratch, binding)
            )

        session.prepare(
            (
                plan.request(name="verify.capacity", prepare_call=prepare),
                rectangular_plan.request(
                    name="verify.rectangular", prepare_call=prepare_rectangular
                ),
            )
        )
        if selected:
            invalid = dict(arguments, q2k_indices=indices[:, :-1].contiguous())
            with pytest.raises(ValueError, match="smaller than the query tensor"):
                paged.bind(plan, scratch=scratch, **invalid)

        def run():
            return paged.run(
                plan=plan, binding=paged.bind(plan, scratch=scratch, **arguments)
            )

        def run_rectangular():
            return paged.run(
                plan=rectangular_plan,
                binding=paged.bind(
                    rectangular_plan,
                    scratch=rectangular_scratch,
                    **rectangular_arguments,
                ),
            )

        actual, _ = run()
        assert actual.shape == query.shape
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
            run_rectangular()
        for counts in ((2, 3), (4, 1), (0, 5), (5, 0), (2, 3)):
            cu.copy_(
                torch.tensor([0, counts[0], rows], device=device, dtype=metadata_dtype)
            )
            query.normal_(std=0.1)
            rectangular_query[:rows].copy_(query)
            rectangular_output.fill_(float("nan"))
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"]
                == allocations
            )
            torch.testing.assert_close(
                output, rectangular_output[:rows], rtol=0, atol=0
            )
            begin = 0
            for request, count in enumerate(counts):
                for local in range(count):
                    end = 137 - count + local + 1
                    k = (
                        key[table[request].long()]
                        .reshape(256, 1, dimension)[:end, 0]
                        .float()
                    )
                    v = (
                        value[table[request].long()]
                        .reshape(256, 1, dimension)[:end, 0]
                        .float()
                    )
                    scores = query[begin + local].float() @ k.T / dimension**0.5
                    expected = scores.softmax(-1) @ v
                    weights = (scores - scores.amax(-1, keepdim=True)).exp()
                    denominator = weights.sum(-1, keepdim=True)
                    probability_error = (
                        (weights.bfloat16().float() - weights).abs() @ v.abs()
                    ) / denominator
                    output_error = (expected.abs() + probability_error) * 2**-8
                    arithmetic_error = (
                        16 * torch.finfo(torch.float32).eps * (weights @ v.abs())
                    ) / denominator
                    error = (output[begin + local].float() - expected).abs()
                    assert bool(
                        (
                            error <= probability_error + output_error + arithmetic_error
                        ).all()
                    )
                begin += count
        graph.reset()
