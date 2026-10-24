"""MXFP8 precision selection validates and times complete prepared calls."""

from dataclasses import replace

import pytest
import torch

from b12x.gemm import blockscaled
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    quantize_mxfp8_rows_torch,
)
from b12x.gemm.blockscaled import _a16
from b12x.gemm.blockscaled._tuning import TUNING
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.device import detect_device
from b12x.testing.reference.helpers import require_b12x
from .test_blockscaled_a16 import assert_close, make_weight, prepared_execution


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("n,k", [(1792, 5120), (8192, 1280)])
def test_auto_mxfp8_races_dense_configurations_and_replays(tmp_path, m, n, k):
    require_b12x()
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    weight, decoded, _ = make_weight("mxfp8", n, k)
    output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    query = replace(
        blockscaled.query_from_call(source, weight, out=output, activation_mode="auto"),
        workspace_form="provided",
    )
    eligible = {
        config
        for _, config in TUNING.choices(query, detect_device(source.device).identity)
    }
    assert sum(config.mode == "quantized" for config in eligible) > 1
    values, scales, gain, _ = _a16._weight_parts(weight)
    quantized = quantize_mxfp8_rows_torch(source)
    references = {
        "a16": source.float() @ decoded.T,
        "quantized": dequantize_mxfp8_rows_torch(quantized.values, quantized.scale_rows)
        @ decoded.T,
    }
    seen = set()

    def call(state):
        trial_source = source.clone()
        trial_output = torch.empty_like(output)
        scratch = torch.empty(
            state.required_workspace, dtype=torch.uint8, device=source.device
        )

        def run():
            return state.run(
                trial_source, values, scales, gain, out=trial_output, workspace=scratch
            )

        run()
        assert_close(trial_output, references[state.config.mode])
        seen.add(state.config)

        return PreparedCall(
            run=run,
            produce=lambda: trial_source.copy_(source),
            output=trial_output,
            owners=(trial_source, trial_output, scratch),
        )

    plan = blockscaled.plan(query)
    with PreparationSession(
        device=source.device, compile_workers=4, cache_dir=tmp_path, race_batch=16
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="mxfp8-precision", prepare_call=call, benchmark_call=call
                ),
            )
        )
        assert plan.selection.source == "tuned"
        assert seen == eligible
        print(
            f"M{m} N{n} K{k}: {len(seen)} validated candidates; selected {plan.selection.config}"
        )
        scratch = torch.empty(
            plan.prepared.state.required_workspace,
            dtype=torch.uint8,
            device=source.device,
        )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            blockscaled.mm(source, weight, out=output, workspace=scratch, plan=plan)
        for sign in (-1, 1, -1):
            source.neg_()
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert_close(output, references[plan.selection.config.mode] * sign)
        graph.reset()
        selected = plan.selection.config
        source.neg_()

    restored = blockscaled.plan(query)
    with PreparationSession(
        device=source.device, compile_workers=4, cache_dir=tmp_path
    ) as session:
        session.prepare(
            (
                restored.request(
                    name="mxfp8-precision", prepare_call=call, benchmark_call=call
                ),
            )
        )
        assert restored.selection.source == "cached"
        assert restored.selection.config == selected


@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("variant", ["swap", "split", "unroll"])
def test_quantized_configuration_graph_output_and_scratch(provided, variant):
    require_b12x()
    m, n, k = 8, 1792, 5120
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    weight, decoded, _ = make_weight("mxfp8", n, k)
    output = (
        torch.empty(m, n, device="cuda", dtype=torch.bfloat16) if provided else None
    )
    query = replace(
        blockscaled.query_from_call(
            source, weight, out=output, activation_mode="quantized"
        ),
        autotune_effort="high",
    )

    def matches(config):
        dense = config.dense_config
        if dense is None:
            return False
        if variant == "swap":
            return dense["swap_ab"]
        if variant == "split":
            return dense["split_k_slices"] == 2
        return dense["large_m_unroll"] and dense["split_k_slices"] == 1

    config = next(
        config
        for _, config in TUNING.choices(query, detect_device(source.device).identity)
        if matches(config)
    )
    quantized = quantize_mxfp8_rows_torch(source)
    reference = (
        dequantize_mxfp8_rows_torch(quantized.values, quantized.scale_rows) @ decoded.T
    )
    with prepared_execution(
        source, weight, activation_mode="quantized", out=output, config=config
    ) as (_, plan):
        if variant == "split" and not query.codegen["split_k_atomic"]:
            assert plan.prepared.state.required_workspace >= 2 * m * n * 4
        graph = torch.cuda.CUDAGraph()
        with (
            kernel_resolution_guard("quantized configuration capture"),
            torch.cuda.graph(graph),
        ):
            result = blockscaled.mm(source, weight, out=output, plan=plan)
        for sign in (-1, 1):
            source.neg_()
            result.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert_close(result, reference * sign)
        graph.reset()
