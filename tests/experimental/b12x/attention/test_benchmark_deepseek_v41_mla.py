"""Model-derived V4.1 benchmark geometry and prepared ragged replay."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.experimental.b12x import benchmark_deepseek_v41_mla as benchmark
from b12x.preparation import PreparationSession
from tests.experimental.b12x.conftest import require_b12x


@pytest.fixture
def model_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "num_attention_heads": 64,
                    "head_dim": 512,
                    "sliding_window": 128,
                    "index_topk": 512,
                    "num_hidden_layers": 40,
                    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
                    "num_nextn_predict_layers": 3,
                },
            }
        )
    )
    return path


def arguments(model_config, *extra):
    return benchmark.parse_args(
        [
            "--model-config",
            str(model_config),
            "--context-length",
            "512",
            "--context-capacity",
            "1048576",
            *extra,
        ]
    )


def test_import_uses_only_standard_library():
    root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from benchmarks.experimental.b12x import benchmark_deepseek_v41_mla; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "entrypoint", ["benchmark_compressed_sparse_mla.py", "benchmark_mla.py"]
)
@pytest.mark.parametrize(
    "preset", [["--preset", "flare-dsv41-tp4"], ["--preset=flare-dsv41-tp4"]]
)
def test_preset_describes_physical_geometry_without_loading_torch(
    model_config, entrypoint, preset
):
    root = Path(__file__).resolve().parents[4]
    script = root / "benchmarks/experimental/b12x" / entrypoint
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; path = sys.argv.pop(1); "
            "sys.argv[0] = path; "
            "\ntry: runpy.run_path(path, run_name='__main__')"
            "\nexcept SystemExit as error: assert error.code == 0"
            "\nassert 'torch' not in sys.modules",
            str(script),
            *preset,
            "--model-config",
            str(model_config),
            "--modes",
            "verify",
            "--forms",
            "c1",
            "--verify-shapes",
            "1x8x5",
            "--describe",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    description, output = [json.loads(line) for line in result.stdout.splitlines()]
    assert description["benchmark"] == "flare-dsv41-tp4"
    contract = output["contract"]
    assert (contract["batch"], contract["query_width"], contract["rows"]) == (1, 8, 5)
    assert contract["swa_page_size"] == 136
    assert contract["indexed_page_size"] == 128
    assert contract["num_q_heads"] == 16
    assert contract["compute_mode"] == "bf16"
    assert contract["native_mode"] == "decode"


def test_cases_keep_physical_target_rows_and_full_draft_blocks_separate(model_config):
    cases = benchmark.build_cases(
        arguments(
            model_config,
            "--modes",
            "decode,verify,prefill,draft",
            "--forms",
            "swa,c2,c1,draft-swa",
            "--decode-batches",
            "1,4",
            "--verify-shapes",
            "1x8x2,4x8x5",
            "--draft-batches",
            "1,4",
            "--prefill-rows",
            "4096",
        )
    )
    assert len(cases) == 17
    assert {case.num_q_heads for case in cases} == {16}
    assert {case.compute_mode for case in cases} == {"bf16"}
    assert {case.draft_depth for case in cases} == {7}
    for case in cases:
        assert case.context_length == 512 and case.context_capacity == 1048576
        assert case.native_mode == ("extend" if case.mode == "prefill" else "decode")
        if case.mode == "decode":
            assert case.rows == case.batch
            assert case.swa_width == case.swa_page_size == 128
        elif case.mode == "verify":
            assert (case.batch, case.query_width, case.rows) in {(1, 8, 2), (4, 8, 5)}
            assert case.max_query_tokens == 8
            assert case.swa_width == 128 and case.swa_page_size == 136
        elif case.mode == "draft":
            assert case.form == "draft-swa"
            assert case.rows == case.batch * 7
            assert case.max_query_tokens == 7
            assert case.swa_width == case.swa_page_size == 135
        else:
            rows = 128 if case.form == "c1" else 4096
            assert case.rows == case.max_query_tokens == rows
            assert case.swa_page_size == 128 + rows
        if case.form in ("swa", "draft-swa"):
            assert case.indexed_width == 0 and case.page_table_width == 1
        else:
            assert case.indexed_width == 512 and case.page_table_width == 8192
            assert case.indexed_page_size == {"c1": 128, "c2": 64}[case.form]


@pytest.mark.parametrize(
    "shape", ["0x8x1", "9x8x9", "1x0x1", "1x9x1", "4x8x3", "4x8x33"]
)
def test_cli_rejects_shapes_outside_physical_capacity(model_config, shape):
    with pytest.raises((ValueError, SystemExit)):
        benchmark.build_cases(
            arguments(model_config, "--modes", "verify", "--verify-shapes", shape)
        )


def test_width_one_verification_retains_decode_page_geometry(model_config):
    cases = benchmark.build_cases(
        arguments(
            model_config,
            "--modes",
            "verify",
            "--forms",
            "swa,c1,c2",
            "--verify-shapes",
            "4x1x4",
        )
    )
    assert len(cases) == 3
    assert all(case.rows == case.batch == 4 and case.query_width == 1 for case in cases)
    assert all(case.swa_width == case.swa_page_size == 128 for case in cases)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_attention_heads", 63),
        ("head_dim", 256),
        ("compress_ratios", [0, 1, 2]),
        ("compress_ratios", [4] * 40),
    ],
)
def test_profile_rejects_incompatible_geometry(model_config, field, value):
    config = json.loads(model_config.read_text())
    config["text_config"][field] = value
    model_config.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        benchmark.load_profile(model_config)


def test_profile_rejects_tensor_parallel_size_two(model_config):
    with pytest.raises(ValueError, match="TP size 4"):
        benchmark.load_profile(model_config, tp_size=2)


def test_profile_rejects_reordered_target_compression_layers(model_config):
    config = json.loads(model_config.read_text())
    ratios = config["text_config"]["compress_ratios"]
    ratios[2], ratios[20] = ratios[20], ratios[2]
    model_config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="40 target layers"):
        benchmark.load_profile(model_config)


@pytest.mark.parametrize("form", ["swa", "c1", "c2"])
def test_cuda_frozen_bf16_ragged_replay_uses_live_metadata(model_config, form):
    device = require_b12x()
    (case,) = benchmark.build_cases(
        arguments(
            model_config,
            "--modes",
            "verify",
            "--forms",
            form,
            "--verify-shapes",
            "4x8x5",
            "--cache-pages",
            "128",
        )
    )
    inputs = benchmark.make_inputs(case, device, seed=4100)
    assert inputs.q.shape == inputs.out.shape == (5, 16, 512)
    assert inputs.q.dtype == torch.bfloat16 and inputs.swa_cache.dtype == torch.uint8
    assert inputs.swa_cache.shape[1] == 136 * 528
    if form != "swa":
        assert inputs.indexed_cache.dtype == torch.uint8
        assert inputs.indexed_cache.shape[1] == case.indexed_page_size * 288
        assert inputs.page_table.shape == (5, 8192)
    else:
        assert inputs.indexed_cache is None and inputs.page_table is None
    caches = [inputs.swa_cache] + (
        [] if inputs.indexed_cache is None else [inputs.indexed_cache]
    )
    snapshots = [cache.clone() for cache in caches]
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        prepared = benchmark.prepare_case(case, inputs, session)
        assert prepared.plan.override is None
        assert prepared.plan.query.compute_mode == "bf16"
        assert prepared.plan.selection.config.v41_compute_mode == "bf16"
        assert prepared.plan.query.query_rows == 5
        assert prepared.plan.query.swa_page_size == 136
        assert prepared.plan.query.mode == "decode"
        benchmark.check_reference(prepared, inputs, case)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                prepared.run()
            pointers = (
                inputs.q.data_ptr(),
                inputs.out.data_ptr(),
                prepared.scratch.data_ptr(),
            )
            for active in (False, True, False):
                for cache, snapshot in zip(caches, snapshots):
                    if active:
                        cache.copy_(snapshot)
                    else:
                        # Empty rows must ignore poisoned FP8 payloads and scale bytes.
                        cache.fill_(127)
                inputs.swa_lengths.fill_(3 if active else 0)
                if form != "swa":
                    inputs.indexed_lengths.fill_(7 if active else 0)
                    first, second = inputs.indexed_physical_pages[:2]
                    table = inputs.page_table.clone()
                    remapped = torch.where(
                        table == first,
                        second,
                        torch.where(table == second, first, table),
                    )
                    inputs.page_table.copy_(remapped)
                expected = benchmark.reference(inputs, case)
                cache_before = [cache.clone() for cache in caches]
                inputs.out.fill_(float("nan"))
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                )
                torch.testing.assert_close(inputs.out, expected, atol=0.035, rtol=0.035)
                assert pointers == (
                    inputs.q.data_ptr(),
                    inputs.out.data_ptr(),
                    prepared.scratch.data_ptr(),
                )
                if not active:
                    assert inputs.out.eq(0).all()
                for cache, before in zip(caches, cache_before):
                    assert torch.equal(cache, before)
        finally:
            graph.reset()
