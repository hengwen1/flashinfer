"""Packed MXFP8 precision races include the native dense GEMM search."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.gemm._tuning import TUNING as DENSE_TUNING
from b12x.gemm.blockscaled._preparation import (
    _dense_lowering,
    _short_dense_lowering,
    _workspace_bytes,
    plan,
)
from b12x.gemm.blockscaled._tuning import (
    BlockscaledConfig,
    BlockscaledQuery,
    TUNING,
    quantized_expected_m,
    quantized_query,
)
from b12x.preparation import DeviceIdentity, FrozenMapping


IDENTITY = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell Max-Q")
DEVICE = SimpleNamespace(identity=IDENTITY)


def query(m=8, n=1792, k=5120, **kwargs):
    return BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=m,
        out_features=n,
        in_features=k,
        padded_in_features=k,
        output_mode="provided",
        workspace_form="provided",
        **kwargs,
    )


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("n,k", [(1792, 5120), (8192, 1280)])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_auto_includes_every_eligible_dense_configuration(m, n, k, effort):
    q = query(m, n, k, autotune_effort=effort)
    inner = quantized_query(q, expected_m=quantized_expected_m(q))
    expected = {
        DENSE_TUNING.config_payload(config)
        for _, config in DENSE_TUNING.choices(inner, IDENTITY)
    }
    configs = [config for _, config in TUNING.choices(q, IDENTITY)]
    quantized = [config for config in configs if config.mode == "quantized"]
    actual = {
        config.dense_config
        if config.dense_config is not None
        else DENSE_TUNING.config_payload(DENSE_TUNING.default_config(inner, IDENTITY))
        for config in quantized
    }
    assert expected <= actual
    assert len(actual) == len(quantized) > 1
    assert sum(config.mode == "a16" for config in configs) == 32
    for config in configs:
        assert TUNING.decode_config(FrozenMapping(config.to_dict())) == config
    for config in quantized:
        lowering = _dense_lowering(q, config, DEVICE)
        dense = (
            DENSE_TUNING.default_config(inner, IDENTITY)
            if config.dense_config is None
            else DENSE_TUNING.decode_config(config.dense_config)
        )
        assert lowering.mma_tiler_mn == (dense.tile_m, dense.tile_n)
        assert lowering.tile_k == dense.tile_k
        assert lowering.swap_ab == dense.swap_ab
        assert lowering.policy.split_k_slices == dense.split_k_slices
        assert lowering.policy.large_m_unroll == dense.large_m_unroll


@pytest.mark.parametrize("mode", ["a16", "quantized"])
def test_explicit_precision_constrains_search(mode):
    q = query(activation_mode=mode)
    configs = [config for _, config in TUNING.choices(q, IDENTITY)]
    assert {config.mode for config in configs} == {mode}
    assert len(configs) > 1
    if mode == "quantized":
        with pytest.raises(ValueError, match="conflicts"):
            TUNING.validate_config(
                replace(q, activation_mode="a16"), configs[-1], IDENTITY
            )


def test_disabled_autotuning_keeps_default_without_enumerating_dense(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled autotuning enumerated dense candidates")

    monkeypatch.setattr(type(DENSE_TUNING), "eligible_plan", unexpected)
    q = query(autotune_effort="off")
    configured = TUNING.configure(q, device=IDENTITY)
    assert configured.default.mode == "a16"


def test_pinned_dense_configuration_reaches_compile_payload():
    q = query()
    config = next(
        config
        for _, config in TUNING.choices(q, IDENTITY)
        if config.dense_config is not None and config.dense_config["swap_ab"]
    )
    declaration = plan(q, override=config)
    job = declaration._compile_jobs(config, DEVICE)[0]
    assert config.dense_config.to_dict() == job.args[1]["dense_config"]
    assert job.args[2]["swap_ab"]


def test_prefill_capacity_retains_short_row_lowering():
    q = query(m=4096)
    config = next(
        config
        for _, config in TUNING.choices(q, IDENTITY)
        if config.dense_config is not None
    )
    default = BlockscaledConfig(mode="quantized")
    assert _short_dense_lowering(q, config, DEVICE) == _short_dense_lowering(
        q, default, DEVICE
    )


def test_functional_split_k_has_preallocated_partials():
    q = replace(query(), output_mode="functional", workspace_form="owned")
    configs = [config for _, config in TUNING.choices(q, IDENTITY)]
    split = next(
        config
        for config in configs
        if config.dense_config is not None
        and config.dense_config["split_k_slices"] == 2
    )
    required = 0 if q.codegen["split_k_atomic"] else 2 * 8 * 1792 * 4
    assert _workspace_bytes(q, split) == required


def test_caller_workspace_limit_and_a16_knobs_are_validated():
    q = query()
    config = next(
        config
        for _, config in TUNING.choices(q, IDENTITY)
        if config.dense_config is not None
    )
    required = _workspace_bytes(q, config)
    TUNING.validate_config(replace(q, workspace_nbytes=required), config, IDENTITY)
    with pytest.raises(ValueError, match="workspace"):
        TUNING.validate_config(
            replace(q, workspace_nbytes=required - 1), config, IDENTITY
        )
    with pytest.raises(ValueError, match="dense_config"):
        TUNING.validate_config(
            replace(q, activation_mode="a16"), replace(config, mode="a16"), IDENTITY
        )


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("rows", [1, 8, 128])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_fp4_quantized_search_includes_dense_candidates(recipe, rows, effort):
    q = replace(
        query(m=rows, n=512, k=2560),
        recipe=recipe,
        activation_scale_available=recipe == "nvfp4",
        global_scale_kind="multiplier" if recipe == "nvfp4" else "none",
        autotune_effort=effort,
    )
    inner = quantized_query(q, expected_m=quantized_expected_m(q))
    expected = {
        DENSE_TUNING.config_payload(c) for _, c in DENSE_TUNING.choices(inner, IDENTITY)
    }
    configs = [c for _, c in TUNING.choices(q, IDENTITY)]
    quantized = [c for c in configs if c.mode == "quantized"]
    actual = {
        c.dense_config
        if c.dense_config is not None
        else DENSE_TUNING.config_payload(DENSE_TUNING.default_config(inner, IDENTITY))
        for c in quantized
    }
    assert expected <= actual
    assert len(actual) == len(quantized) > 1
    for mode in {c.mode for c in configs}:
        assert {
            c for _, c in TUNING.choices(replace(q, activation_mode=mode), IDENTITY)
        } == {c for c in configs if c.mode == mode}
    for c in quantized:
        lowered = _dense_lowering(q, c, DEVICE)
        dense = (
            DENSE_TUNING.default_config(inner, IDENTITY)
            if c.dense_config is None
            else DENSE_TUNING.decode_config(c.dense_config)
        )
        assert lowered.mma_tiler_mn == (dense.tile_m, dense.tile_n)
        assert lowered.policy.split_k_slices == (dense.split_k_slices or 1)
        required = _workspace_bytes(q, c)
        TUNING.validate_config(replace(q, workspace_nbytes=required), c, IDENTITY)
        with pytest.raises(ValueError, match="workspace"):
            TUNING.validate_config(
                replace(q, workspace_nbytes=required - 1), c, IDENTITY
            )


@pytest.mark.parametrize("rows", [128, 4096])
def test_mxfp8_prefill_races_packed_tiles_beside_expanded_baseline(rows):
    from b12x.gemm.blockscaled._tuning import mxfp8_a16_dequant

    q = query(m=rows)
    configs = [c for _, c in TUNING.choices(q, IDENTITY)]
    expanded = [c for c in configs if mxfp8_a16_dequant(q, c)]
    packed = [c for c in configs if c.mode == "a16" and not mxfp8_a16_dequant(q, c)]
    assert len(expanded) == 1
    assert len(packed) == 32
    assert all(c.a16_backend == "packed" for c in packed)
    assert {
        c for _, c in TUNING.choices(replace(q, activation_mode="a16"), IDENTITY)
    } == set(expanded + packed)
