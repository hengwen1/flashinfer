"""Internal grouped routing preserves exact expert prefixes across live replays."""

import pytest
import torch

from benchmarks.experimental.b12x.benchmark_ds4_moe import make_synthetic_mxfp4_moe
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
from b12x.moe.fused_moe._impl import _map_core_workspace_views
from b12x.moe.fused_moe._preparation import _dynamic_program_arguments
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


_TOPK = 6
_TILE_M = 16


@pytest.fixture(scope="module")
def internal_routing_case(request):
    num_experts, hidden, intermediate, boundary_ids = request.param
    device = require_b12x()
    weights = make_synthetic_mxfp4_moe(
        num_experts, hidden, intermediate, seed=7319, device=device
    )
    gen = torch.Generator(device=device).manual_seed(7320)
    original_x = (torch.randn(24, hidden, device=device, generator=gen) * 2).to(
        torch.bfloat16
    )
    restored_x = (original_x.float() * 0.75 + 0.125).to(torch.bfloat16)
    boundary = torch.tensor(boundary_ids, device=device, dtype=torch.int32)
    columns = (
        torch.arange(24, device=device)[:, None]
        + torch.arange(_TOPK, device=device)[None, :]
    ) % _TOPK
    original_ids = boundary[columns].contiguous()
    mixed_ids = original_ids.clone()
    mixed_ids[::2, 1] = -1
    mixed_ids[1::3, 2] = -7
    mixed_ids[::3, 3] = num_experts
    mixed_ids[1::4, 4] = torch.iinfo(torch.int32).max
    mixed_ids[::5, 0] = torch.iinfo(torch.int32).min
    inactive_ids = (
        torch.tensor(
            [-1, -7, num_experts, num_experts + 1, torch.iinfo(torch.int32).max, -1],
            device=device,
            dtype=torch.int32,
        )
        .expand(24, _TOPK)
        .contiguous()
    )
    original_weights = torch.randn(24, _TOPK, device=device, generator=gen).softmax(-1)
    scenarios = (
        (original_x, original_ids, original_weights),
        (original_x, mixed_ids, original_weights),
        (original_x, inactive_ids, original_weights),
        (restored_x, original_ids.roll(1, dims=1), original_weights.flip(1)),
    )
    references = {}
    for capacity in (8, 16, 24):
        for index, (x, ids, route_weights) in enumerate(scenarios):
            references[capacity, index] = moe_reference_w4a8_mx(
                x[:capacity].float(),
                weights["w13_fp4"],
                weights["w13_mx"],
                None,
                weights["alphas"],
                weights["w2_fp4"],
                weights["w2_mx"],
                None,
                weights["alphas"],
                ids[:capacity],
                route_weights[:capacity],
                num_experts,
                hidden,
                intermediate,
                activation="silu",
                swiglu_limit=10.0,
            )
    declaration = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=declaration,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"],
            w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"],
            w2_block_scales=weights["w2_mx"],
            w13_global_scales=weights["alphas"],
            w2_global_scales=weights["alphas"],
            input_scale=weights["input_scale"],
            intermediate_scale=weights["input_scale"],
        ),
    )
    return device, experts, scenarios, references, num_experts, hidden


@pytest.mark.parametrize(
    "internal_routing_case,capacity,max_active_clusters,ids_dtype",
    [
        pytest.param(
            shape,
            capacity,
            grid,
            dtype,
            id=f"e{shape[0]}-{str(dtype).removeprefix('torch.')}-m{capacity}-grid{grid}",
        )
        for shape, capacities, dtypes in (
            (
                (384, 5120, 576, (0, 31, 32, 63, 64, 383)),
                (8, 16, 24),
                (torch.int32, torch.int64),
            ),
            ((33, 256, 192, (0, 1, 15, 16, 31, 32)), (24,), (torch.int32, torch.int64)),
        )
        for capacity in capacities
        for grid in (None, 1)
        for dtype in dtypes
    ],
    indirect=("internal_routing_case",),
)
def test_internal_grouped_routing_prefix_is_exact_and_live(
    internal_routing_case, capacity, max_active_clusters, ids_dtype
):
    device, experts, scenarios, references, num_experts, hidden = internal_routing_case
    x = scenarios[0][0][:capacity].clone()
    ids = scenarios[0][1][:capacity].to(ids_dtype).clone()
    route_weights = scenarios[0][2][:capacity].clone()
    output = torch.full(
        (capacity + 3, hidden), torch.nan, device=device, dtype=torch.bfloat16
    )
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=_TOPK),
        invocation={"fast_math": False},
        override=MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            dynamic_tile_m=_TILE_M,
            dynamic_route_mode="grouped",
            max_active_clusters=max_active_clusters,
        ),
    )
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations

    def prepare(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=ids,
            topk_weights=route_weights,
            output=output[:capacity],
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding),
            output=output[:capacity],
            owners=(scratch, binding),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="internal-routing", prepare_call=prepare),))
        scratch_plan = plan.prepared.state.scratch
        launch_plan = scratch_plan.launch_plan
        assert launch_plan.implementation == "dynamic"
        assert launch_plan.execution.tile_m == _TILE_M
        arguments = _dynamic_program_arguments(launch_plan, scratch_plan.caps)
        assert arguments["w4a8_repacked"] and arguments["w4a8_n64_repacked"]
        assert not arguments["direct_routing"] and not arguments["external_route_plan"]
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        views = _map_core_workspace_views(
            scratch_plan._core_workspace_plan,
            scratch[0],
            offset_bytes=scratch_plan.layout.route_workspace_nbytes,
            capacity_nbytes=scratch_plan.layout.core_workspace_nbytes,
            do_init=False,
        )
        assert views["materialized_intermediate"].numel() > 0
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            topk_ids=ids,
            topk_weights=route_weights,
            output=output[:capacity],
            input_scales_static=True,
        )
        storage = (*scratch, x, ids, route_weights, output)
        pointers = tuple(t.data_ptr() for t in storage)
        session.freeze()
        fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        for index, (source_x, source_ids, source_weights) in enumerate(scenarios):
            x.copy_(source_x[:capacity])
            ids.copy_(source_ids[:capacity])
            route_weights.copy_(source_weights[:capacity])
            views["row_counts"].fill_(0x35353535)
            views["expert_tile_base"].fill_(0x35353535)
            views["expert_write_rows"].fill_(0x35353535)
            for name in (
                "pair_head",
                "producers_done_count",
                "all_work_published",
                "task_head",
                "task_tail",
            ):
                views[name].fill_(0x35353535)
            views["token_map"].fill_(-17)
            views["token_weights"].fill_(torch.nan)
            views["materialized_intermediate"].view(torch.uint8).fill_(0xA5)
            output.fill_(torch.nan)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert tuple(t.data_ptr() for t in storage) == pointers
            active = ids[(ids >= 0) & (ids < num_experts)]
            counts = torch.bincount(active.long(), minlength=num_experts).to(
                torch.int32
            )
            prefix = torch.cat(
                (
                    torch.zeros(1, device=device, dtype=torch.int32),
                    ((counts + _TILE_M - 1) // _TILE_M).cumsum(0).to(torch.int32),
                )
            )
            for name in ("row_counts", "expert_write_rows"):
                torch.testing.assert_close(
                    views[name][:num_experts], counts, rtol=0, atol=0
                )
            torch.testing.assert_close(
                views["expert_tile_base"][: num_experts + 1], prefix, rtol=0, atol=0
            )
            assert views["barrier_count"][0].item() == 0
            actual = output[:capacity]
            assert actual.isfinite().all()
            if active.numel() == 0:
                assert torch.count_nonzero(actual).item() == 0
            else:
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(),
                    references[capacity, index].flatten(),
                    dim=0,
                ).item()
                assert cosine > 0.998, (capacity, index, cosine)
                assert torch.count_nonzero(actual).item() > 0
            assert output[capacity:].isnan().all()
        graph.reset()
