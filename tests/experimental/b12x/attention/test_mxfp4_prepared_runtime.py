"""Prepared fused MXFP4 workspace alignment and runtime capacity bindings."""

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from .test_fused_mxfp4 import _case, _assert_selected

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="MXFP4 fused selection requires SM120",
)


@pytest.mark.parametrize("page_size", [64, 128])
def test_fused_mxfp4_plan_aligned_workspace_and_runtime_bounds(page_size):
    from b12x.attention import dsa_indexer as api
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x._lib.compile_plan import program_keys

    case = _case(page_size=page_size, shared=True, width=32768)
    case.mutate(0, case.width, "mixed")
    device = case.args["q_mxfp4"].device
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        plan = api.plan(
            api.Caps(
                device=device,
                num_q_heads=32,
                max_q_rows=4,
                max_page_table_width=case.args["page_table"].shape[1],
                topk=512,
                page_size=page_size,
                cache_format="mxfp4",
            )
        )
        retained = []

        def prepare(state):
            (spec,) = state.layout.scratch_specs()
            backing = torch.empty(spec.shape[0] + 1, dtype=spec.dtype, device=device)
            case.args["scratch"] = backing[1:]
            binding = state.bind(**case.args)
            retained.append((state, binding))
            return PreparedCall(
                run=lambda: state.run(binding), owners=(binding, backing)
            )

        session.prepare(
            (plan.request(name="fused-mxfp4-bounds", prepare_call=prepare),)
        )
        state, _ = retained[0]
        assert program_keys(state)
        session.freeze()
        original = dict(case.args)
        for rows in (4, 1):
            for name in (
                "q_mxfp4",
                "q_scales",
                "query_weights",
                "cache_lengths",
                "output_indices",
                "output_scores",
            ):
                case.args[name] = original[name][:rows]
            for score_width in (case.width, 73):
                binding = api.bind(
                    plan,
                    score_width=score_width,
                    **{
                        name: value
                        for name, value in case.args.items()
                        if name not in ("page_size", "ctas_per_row")
                    },
                )
                runtime = binding.runtime.runtime
                for view in runtime.scratch.values():
                    assert view.data_ptr() % 256 == 0
                graph = torch.cuda.CUDAGraph()
                with kernel_resolution_guard():
                    with torch.cuda.graph(graph):
                        api.run(binding)
                    for step, visible, kind in (
                        (0, case.width, "mixed"),
                        (1, 73, "negative"),
                        (2, 0, "zero"),
                    ):
                        expected, logical = case.mutate(step, visible, kind)
                        expected[:, score_width:] = -float("inf")
                        case.args["scratch"].fill_(0xA5)
                        case.args["output_indices"].fill_(-17)
                        case.args["output_scores"].fill_(float("nan"))
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
                        assert not bool(runtime.scratch["merge_state"].count_nonzero())
                        _assert_selected(case, expected, logical)
                graph.reset()


def test_fused_mxfp4_production_plan_races_complete_warm_graphs(tmp_path):
    from b12x.attention import dsa_indexer as api
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation._measurement import capture_benchmark_call
    from .test_fused_mxfp4 import _decode

    case = _case(page_size=64, shared=True, width=1024)
    case.mutate(0, case.width, "mixed")
    a = case.args
    device = a["q_mxfp4"].device
    source = _decode(a["q_mxfp4"].cpu(), a["q_scales"].cpu()).to(device)
    activation = torch.empty_like(source)
    retained = []
    caps = api.Caps(
        device=device,
        num_q_heads=32,
        max_q_rows=4,
        max_page_table_width=16,
        topk=512,
        page_size=64,
        cache_format="mxfp4",
    )
    plan = api.plan(caps)

    def prepare(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(**{**a, "scratch": scratch})

        def run():
            state.quantize_query(
                activation, q_mxfp4=a["q_mxfp4"], q_scales=a["q_scales"]
            )
            state.run(binding)

        call = PreparedCall(
            run=run,
            produce=lambda: activation.copy_(source),
            owners=(binding, scratch, source, activation),
        )
        retained.append((state, binding, call))
        return call

    with PreparationSession(
        device=device,
        autotune=True,
        cache_policy="warm",
        cache_dir=tmp_path,
        compile_workers=2,
        rounds=1,
        samples=1,
    ) as session:
        request = plan.request(
            name="fused-warm-race",
            prepare_call=prepare,
            benchmark_call=lambda state: capture_benchmark_call(prepare(state)),
        )
        envelope = session.candidate_memory_envelope((request,))
        result = session.prepare((request,))
        assert envelope.scratch_nbytes >= plan.memory_requirements().scratch_nbytes
        assert result.benchmarked_candidates >= 2
        assert plan.selection.source == "tuned"
        assert result.coverage["fused-warm-race"]["measured_count"] >= 2
        session.freeze()
        binding = api.bind(
            plan,
            **{
                name: value
                for name, value in {
                    **a,
                    "scratch": torch.empty(
                        api.scratch_specs(plan, device=device)[0].shape,
                        dtype=torch.uint8,
                        device=device,
                    ),
                }.items()
                if name not in ("page_size", "ctas_per_row")
            },
        )
        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard():
            with torch.cuda.graph(graph):
                api.quantize_q_mxfp4(
                    plan, activation, q_mxfp4=a["q_mxfp4"], q_scales=a["q_scales"]
                )
                api.run(binding)
            for step, visible, kind in (
                (0, case.width, "mixed"),
                (1, 73, "negative"),
                (2, 0, "zero"),
                (3, case.width - 5, "zero"),
            ):
                case.mutate(step, visible, kind)
                activation.copy_(source)
                for view in binding.runtime.runtime.scratch.values():
                    view.fill_(17)
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
                assert not bool(
                    binding.runtime.runtime.scratch["merge_state"].count_nonzero()
                )
                expected, logical = case.reference(
                    a["q_mxfp4"].cpu(),
                    a["q_scales"].cpu(),
                    a["query_weights"].cpu(),
                    a["page_table"].cpu(),
                    a["cache_lengths"].cpu(),
                    visible,
                )
                _assert_selected(case, expected, logical)
        print(
            {
                "selected": plan.selection.config.to_dict(),
                "benchmarked_candidates": result.benchmarked_candidates,
            }
        )
        graph.reset()
