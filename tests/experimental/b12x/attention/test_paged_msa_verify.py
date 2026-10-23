"""Packed selected-block verification preserves query-local sets and split state."""

import math
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import paged
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("rows,msa", [(17, False), (16, False), (13, True)])
def test_verify_rejects_inconsistent_capacity(rows, msa):
    from b12x.attention.paged._preparation import _install_graph_replay_state

    caps = SimpleNamespace(
        use_cuda_graph=True,
        mode="verify",
        max_batch=2,
        max_total_q=rows,
        msa_block_sparse=msa,
    )
    with pytest.raises(ValueError, match="prepared row capacity"):
        _install_graph_replay_state(SimpleNamespace(caps=caps), -1, 7)


def _update(
    q,
    key,
    value,
    table,
    lengths,
    cu,
    selected,
    counts,
    context,
    turn,
    four_key_probability_rounding=False,
    duplicate_selected=False,
):
    q.normal_(std=0.2)
    lengths.fill_(max(context - turn * 127, max(counts), 1))
    cu.copy_(
        torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device=q.device)
    )
    selected.fill_(-1)
    expected = torch.empty_like(q)
    rounding_bound = torch.zeros_like(q, dtype=torch.float32)
    row = 0
    for request, count in enumerate(counts):
        for local in range(count):
            position = int(lengths[request]) - count + local
            blocks = position // 128 + 1
            for head in range(key.shape[2]):
                choices = torch.randperm(blocks - 1, device=q.device)[:15]
                choices = (
                    torch.cat((choices, choices.new_tensor([blocks - 1]))).sort().values
                )
                if duplicate_selected and turn == 1 and choices.numel() > 1:
                    choices[0] = choices[1]
                selected[head, row, : choices.numel()] = choices.int()
                tokens = (
                    choices[:, None] * 128 + torch.arange(128, device=q.device)
                ).flatten()
                tokens = tokens[tokens <= position]
                physical = table[request, tokens // key.shape[1]].long()
                keys = key[physical, tokens % key.shape[1], head].float()
                values = value[physical, tokens % key.shape[1], head].float()
                heads = slice(head * 16, (head + 1) * 16)
                score = q[row, heads].float() @ keys.T / math.sqrt(128)
                reference = score.softmax(-1) @ values
                expected[row, heads] = reference.bfloat16()
                if (
                    four_key_probability_rounding
                    and tokens.numel() == 4
                    and choices.tolist() == [0]
                ):
                    assert key.dtype == torch.float8_e4m3fn
                    # PV consumes BF16 unnormalized probabilities; normalization
                    # retains the FP32 denominator. Bound that rounding and the
                    # final BF16 cast against the independent FP32 reference.
                    weights = (score - score.amax(-1, keepdim=True)).exp()
                    denominator = weights.sum(-1, keepdim=True)
                    numerator_error = (
                        (weights.bfloat16().float() - weights).abs() @ values.abs()
                    ) / denominator
                    output_error = (reference.abs() + numerator_error) * 2**-8
                    reference_error = (expected[row, heads].float() - reference).abs()
                    arithmetic_error = (
                        16
                        * torch.finfo(torch.float32).eps
                        * (weights @ values.abs())
                        / denominator
                    )
                    rounding_bound[row, heads] = (
                        numerator_error
                        + output_error
                        + reference_error
                        + arithmetic_error
                    ).clamp_max(0.0005)
            row += 1
    return expected, row, rounding_bound


@pytest.mark.parametrize(
    "batch,width,kv_heads,page,dtype,padding,context,four_key_probability_rounding,duplicate_selected",
    [
        (1, 2, 1, 128, torch.float8_e4m3fn, 0, 4099, False, False),
        (8, 6, 1, 128, torch.float8_e4m3fn, 0, 4099, False, False),
        (1, 7, 1, 64, torch.float8_e4m3fn, 0, 4099, False, False),
        (2, 7, 1, 128, torch.float8_e4m3fn, 3, 4099, False, False),
        (2, 4, 2, 64, torch.bfloat16, 0, 4099, False, False),
        (4, 8, 2, 128, torch.float8_e4m3fn, 0, 4099, False, False),
        (2, 6, 1, 128, torch.bfloat16, 0, 4099, False, False),
        (1, 6, 1, 128, torch.float8_e4m3fn, 0, 257, True, False),
        (1, 6, 1, 128, torch.float8_e4m3fn, 0, 4099, False, True),
    ],
)
@torch.no_grad()
@pytest.mark.parametrize("external_schedule", [False, True])
def test_msa_verify_query_sets_ragged_replay(
    batch,
    width,
    kv_heads,
    page,
    dtype,
    padding,
    context,
    four_key_probability_rounding,
    duplicate_selected,
    external_schedule,
):
    require_b12x()
    device = torch.device("cuda", 0)
    torch.manual_seed(7123)
    virtual = 1048576
    physical = (context + page - 1) // page
    key = (
        torch.randn(1 + batch * physical, page, kv_heads, 128, device=device) * 0.2
    ).to(dtype)
    value = (torch.randn_like(key.float()) * 0.2).to(dtype)
    table = torch.zeros(batch, virtual // page, dtype=torch.int32, device=device)
    for request in range(batch):
        table[request, :physical] = torch.arange(
            1 + request * physical, 1 + (request + 1) * physical, device=device
        )
    rows = batch * width + padding
    q = torch.empty(rows, kv_heads * 16, 128, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)
    lengths = torch.empty(batch, dtype=torch.int32, device=device)
    cu = torch.empty(batch + 1, dtype=torch.int32, device=device)
    selected = torch.empty(kv_heads, rows, 16, dtype=torch.int32, device=device)
    scale = torch.ones(1, device=device) if dtype == torch.float8_e4m3fn else None
    _update(q, key, value, table, lengths, cu, selected, [width] * batch, context, 0)
    arguments = dict(
        q=q,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        q2k_indices=selected,
        k_descale=scale,
        v_descale=scale,
    )
    caps = paged.Caps(
        device=device,
        mode="verify",
        dtype=q.dtype,
        kv_dtype=dtype,
        num_q_heads=q.shape[1],
        num_kv_heads=kv_heads,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=page,
        max_total_q=rows,
        max_batch=batch,
        max_page_table_width=table.shape[1],
        num_cache_pages=key.shape[0],
        max_work_items=rows * 32,
        max_partial_rows=rows * 32,
        use_cuda_graph=True,
        msa_block_sparse=True,
    )
    invocation = dict(paged.invocation_from_tensors(caps, **arguments), query_len=width)
    plan = paged.plan(caps, invocation=invocation)
    scratch = None
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            nonlocal scratch
            assert state.config.query_tiles_per_request == width
            assert state.config.max_effective_kv_pages == 16 * (128 // page)
            assert state.scratch_plan._plan.cta_tile_q == 16
            assert (
                state.scratch_plan._plan.kv_chunk_size
                == state.config.chunk_pages_lut()[-1] * page
            )
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch_plan.scratch_specs()
            ]
            binding = state.bind(scratch=scratch, **arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="msa.verify", prepare_call=prepare),))
        schedule = schedule_storage = independent_scratch = None
        if external_schedule:
            from b12x.attention.paged.schedule import PrefillSchedule, schedule_nbytes

            schedule = PrefillSchedule.from_plan(plan)
            schedule_storage = torch.empty(
                schedule_nbytes(caps), dtype=torch.uint8, device=device
            )
            independent_scratch = [torch.empty_like(tensor) for tensor in scratch]

        def run():
            extra = {}
            if schedule is not None:
                schedule.update(schedule_storage, lengths, cu)
                extra["external_schedule"] = schedule.bind(schedule_storage)
            for _ in range(2 if schedule is not None else 1):
                if schedule is not None:
                    for tensor in scratch:
                        tensor.fill_(197)
                paged.run(
                    plan=plan,
                    binding=paged.bind(plan, scratch=scratch, **arguments, **extra),
                )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        patterns = (
            [width] * batch,
            [
                max(1, width - request - 1) if request % 2 == 0 else 0
                for request in range(batch)
            ],
            [
                width if request == batch - 1 else max(1, width - 2)
                for request in range(batch)
            ],
            [0] * batch,
            [width] * batch,
        )
        for turn, counts in enumerate(patterns):
            expected, live, rounding_bound = _update(
                q,
                key,
                value,
                table,
                lengths,
                cu,
                selected,
                counts,
                context,
                turn,
                four_key_probability_rounding=four_key_probability_rounding,
                duplicate_selected=duplicate_selected,
            )
            for tensor in scratch:
                tensor.fill_(197)
            if schedule_storage is not None:
                schedule_storage.fill_(197)
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            if four_key_probability_rounding:
                error = (output[:live].float() - expected[:live].float()).abs()
                original_limit = 0.0003 + 0.025 * expected[:live].float().abs()
                needs_allowance = error > original_limit
                torch.testing.assert_close(
                    output[:live][~needs_allowance],
                    expected[:live][~needs_allowance],
                    rtol=0.025,
                    atol=0.0003,
                )
                assert bool(
                    (
                        error[needs_allowance] <= rounding_bound[:live][needs_allowance]
                    ).all()
                )
                assert bool(torch.isfinite(output[:live]).all())
            else:
                torch.testing.assert_close(
                    output[:live], expected[:live], rtol=0.025, atol=0.0003
                )
            if live == 0:
                assert bool(torch.isnan(output).all())
            if schedule is not None:
                for tensor in independent_scratch:
                    tensor.fill_(197)
                canonical = paged.bind(plan, scratch=independent_scratch, **arguments)
                for name, actual in schedule.views(schedule_storage).items():
                    torch.testing.assert_close(
                        actual, getattr(canonical.scratch, name), rtol=0, atol=0
                    )
        selected.fill_(-1)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert output[:live].count_nonzero() == 0


@pytest.mark.parametrize("msa", [False, True])
@torch.no_grad()
def test_verify_omitted_width_uses_uniform_capacity(msa):
    require_b12x()
    device = torch.device("cuda", 0)
    batch, width, page = 2, 7, 128
    q = torch.zeros(batch * width, 16, 128, device=device, dtype=torch.bfloat16)
    key = torch.zeros(4, page, 1, 128, device=device, dtype=torch.bfloat16)
    value = torch.ones_like(key)
    table = torch.arange(4, device=device, dtype=torch.int32).view(batch, 2)
    lengths = torch.tensor([129, 255], device=device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    selected = torch.tensor([0, 1] + [-1] * 14, device=device, dtype=torch.int32)
    selected = selected.expand(1, q.shape[0], 16).contiguous() if msa else None
    output = torch.empty_like(q)
    args = dict(
        q=q,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        q2k_indices=selected,
    )
    caps = paged.Caps(
        device=device,
        mode="verify",
        dtype=q.dtype,
        kv_dtype=key.dtype,
        num_q_heads=16,
        num_kv_heads=1,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=page,
        max_total_q=q.shape[0],
        max_batch=batch,
        max_page_table_width=2,
        num_cache_pages=4,
        max_work_items=1024,
        max_partial_rows=1024,
        use_cuda_graph=True,
        msa_block_sparse=msa,
    )
    plan = paged.plan(caps, invocation=paged.invocation_from_tensors(caps, **args))
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch_plan.scratch_specs()
            ]
            binding = state.bind(scratch=scratch, **args)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="verify.width", prepare_call=prepare),))
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
