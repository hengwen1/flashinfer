"""Fused MoE search effort preserves launch legality and heuristic defaults."""

from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity
from b12x.moe.fused_moe._tuning import TUNING
from .test_precision_choices import _nvfp4_query


DEVICES = (
    DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10"),
    DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell"),
)


def query(tokens=4, experts=384, top_k=6, recipe="w4a8_mx", **kwargs):
    return replace(
        _nvfp4_query(),
        quant_mode=recipe,
        quant_modes=(recipe,),
        source_format="fp4_e8m0_k32" if recipe == "w4a8_mx" else "modelopt_nvfp4",
        num_experts=experts,
        route_num_experts=experts,
        hidden_size=5120,
        intermediate_size=2304,
        num_tokens=tokens,
        top_k=top_k,
        routed_rows=tokens * top_k,
        shared_input_scales=recipe == "nvfp4",
        **kwargs,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("experts,top_k", ((384, 6), (128, 3)))
def test_nested_candidates_and_default_membership(device, experts, top_k):
    sets = []
    for effort in ("low", "medium", "high", "max"):
        q = query(experts=experts, top_k=top_k, autotune_effort=effort)
        configuration = TUNING.configure(q, device=device)
        default_payload = TUNING.config_payload(configuration.default)
        configuration.space.validate(default_payload)
        assert TUNING.lower(q, device, default_payload) == configuration.default
        candidates = {config for _, config in TUNING.iterate(configuration)}
        assert configuration.default in candidates
        sets.append(candidates)
    assert sets[0] <= sets[1] <= sets[2] < sets[3]
    assert any(c.max_active_clusters == 3 for c in sets[3])
    q = query(experts=experts, top_k=top_k, autotune_effort="off")
    configuration = TUNING.configure(q, device=device)
    assert list(TUNING.iterate(configuration)) == [
        (TUNING.config_payload(configuration.default), configuration.default)
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("experts,top_k,tokens", ((384, 6, 1025), (128, 3, 683)))
def test_inline_density_is_efficiency_only(device, experts, top_k, tokens):
    q = query(tokens, experts, top_k, csf_inline_scales=True)
    config = replace(
        TUNING.default_config(q, device),
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
        w4a16_route_mode=None,
    )
    payload = TUNING.config_payload(config)
    for effort in ("low", "medium"):
        with pytest.raises(ValueError, match="efficiency"):
            TUNING.lower(replace(q, autotune_effort=effort), device, payload)
    for effort in ("high", "max"):
        assert (
            TUNING.lower(replace(q, autotune_effort=effort), device, payload) == config
        )
    invalid = replace(config, dynamic_tile_m=64)
    with pytest.raises(ValueError, match="Inline scales"):
        TUNING.lower(
            replace(q, autotune_effort="max"), device, TUNING.config_payload(invalid)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_max_retains_residency_and_determinism_constraints(device):
    q = query(autotune_effort="max")
    config = replace(
        TUNING.default_config(q, device),
        backend="dynamic",
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        max_active_clusters=2 * device.sm_count + 1,
    )
    with pytest.raises(ValueError, match="outside|resident"):
        TUNING.lower(q, device, TUNING.config_payload(config))
    q = replace(q, deterministic_output=True)
    config = replace(config, max_active_clusters=None, dynamic_route_mode="direct")
    with pytest.raises(ValueError, match="deterministic"):
        TUNING.lower(q, device, TUNING.config_payload(config))


def test_effort_is_captured_and_not_a_program_identity(monkeypatch):
    from b12x.moe.fused_moe._preparation import _decode_query, _program_query_payload
    from b12x.preparation._efficiency import capture_autotune_effort

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "high")
    q = query(autotune_effort=capture_autotune_effort().value)
    assert _decode_query(TUNING.encode_query(q)).autotune_effort == "high"
    assert _program_query_payload(q) == _program_query_payload(
        replace(q, autotune_effort="max")
    )
    assert _decode_query(_program_query_payload(q)).autotune_effort == "medium"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("recipe", ("nvfp4", "w4a8_mx", "w4a16"))
def test_filters_naturally_preserve_defaults_across_shapes(device, recipe):
    for tokens in (1, 4, 8, 16, 64, 683, 1025, 2049):
        for experts, top_k in ((384, 6), (128, 3)):
            for intermediate in (2304, 1152, 576):
                for effort in ("low", "medium", "high", "max"):
                    q = replace(
                        query(tokens, experts, top_k, recipe),
                        intermediate_size=intermediate,
                        autotune_effort=effort,
                    )
                    if recipe == "w4a16":
                        q = replace(
                            q,
                            source_format="fp4_e8m0_k32",
                            w4a16_weight_layout="modelopt",
                            w4a16_scale_format="e8m0_k32",
                        )
                    config = TUNING.default_config(q, device)
                    payload = TUNING.config_payload(config)
                    TUNING.parameter_space(q, device).validate(payload)
                    assert TUNING.lower(q, device, payload) == config


@pytest.mark.parametrize("device", DEVICES)
def test_materialized_fc1_cta_budget_preserves_baseline_choices(device):
    q = replace(query(24, csf_inline_scales=True), intermediate_size=576)
    configuration = TUNING.configure(q, device=device)
    candidates = [config for _, config in TUNING.iterate(configuration)]
    groups = {2: set(), 3: set()}
    for config in candidates:
        if (
            config.backend != "dynamic"
            or config.dynamic_tile_m != 16
            or not config.csf_inline_scales
        ):
            assert config.materialized_fc1_cta_budget == 2
            continue
        payload = config.to_dict()
        budget = payload.pop("materialized_fc1_cta_budget")
        groups[budget].add(tuple(sorted(payload.items())))
    assert groups[2] and groups[3]
    assert groups[2] == groups[3]


def test_moe_config_roundtrip_requires_explicit_controls():
    from b12x.preparation import FrozenMapping
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    config = TUNING.default_config(
        replace(query(24, csf_inline_scales=True), intermediate_size=576), DEVICES[1]
    )
    defaults = MoeDecodeConfig("micro", "internal", None)
    assert defaults.materialized_fc1_cta_budget == 2
    assert defaults.compact_fuse_input_quantization is False
    for budget in (2, 3):
        value = replace(config, materialized_fc1_cta_budget=budget)
        assert MoeDecodeConfig.from_config(FrozenMapping(value.to_dict())) == value
    for fuse in (False, True):
        value = replace(defaults, compact_fuse_input_quantization=fuse)
        assert MoeDecodeConfig.from_config(FrozenMapping(value.to_dict())) == value
    for field in ("materialized_fc1_cta_budget", "compact_fuse_input_quantization"):
        payload = config.to_dict()
        payload.pop(field)
        with pytest.raises(ValueError, match="config fields must be exactly"):
            MoeDecodeConfig.from_config(FrozenMapping(payload))


@pytest.mark.parametrize(
    "changes",
    (
        {"activation": "situ"},
        {"intermediate_size": 512},
        {"source_format": "modelopt_nvfp4"},
        {"csf_inline_scales": False},
    ),
)
def test_materialized_fc1_cta_budget_requires_operand_geometry(changes):
    q = replace(query(24, csf_inline_scales=True), intermediate_size=576)
    config = replace(
        TUNING.default_config(q, DEVICES[1]), materialized_fc1_cta_budget=3
    )
    q = replace(q, **changes)
    with pytest.raises(ValueError):
        TUNING.lower(q, DEVICES[1], TUNING.config_payload(config))


@pytest.mark.parametrize("mac", (-1, 0, 128, 188))
def test_materialized_fc1_cta_budget_preserves_cluster_policy(mac):
    from b12x.moe.fused_moe._impl import (
        _decode_dynamic_launch_policy,
        _encode_dynamic_launch_policy,
    )

    kwargs = dict(
        volatile_launch_state=False,
        external_route_plan_requested=True,
        policy_max_active_clusters=mac,
        planned_tile_m=16,
        planned_direct_routing=False,
        w4a8_n64_repacked=True,
        csf_inline_scales=True,
    )
    baseline = _encode_dynamic_launch_policy(**kwargs)
    assert baseline == _encode_dynamic_launch_policy(
        **kwargs, materialized_fc1_cta_budget=2
    )
    alternate = _encode_dynamic_launch_policy(**kwargs, materialized_fc1_cta_budget=3)
    assert baseline != alternate
    assert _decode_dynamic_launch_policy(baseline) == _decode_dynamic_launch_policy(
        alternate
    )
    assert _decode_dynamic_launch_policy(alternate)[-1] == mac


def test_materialized_fc1_cta_budget_reaches_only_fc1():
    from b12x.moe._shared.kernels.silu import MoEDynamicKernelSilu

    kernels = [
        MoEDynamicKernelSilu(
            sf_vec_size=32,
            mma_tiler_mn=(16, 128),
            quant_recipe="w4a8_mx",
            w4a8_repacked=True,
            w4a8_n64_repacked=True,
            w4a8_n64_tail=True,
            materialize_intermediate=True,
            share_input_across_experts=True,
            csf_inline_scales=True,
            materialized_fc1_cta_budget=budget,
        )
        for budget in (2, 3)
    ]
    assert [k.materialized_phase1_kernel.cta_budget for k in kernels] == [2, 3]
    assert kernels[0].threads_per_cta == kernels[1].threads_per_cta
    assert (
        kernels[0].materialized_phase1_kernel.shared_bytes
        == kernels[1].materialized_phase1_kernel.shared_bytes
    )
    assert (
        kernels[0].materialized_phase2_kernel.shared_bytes
        == kernels[1].materialized_phase2_kernel.shared_bytes
    )
    assert (
        kernels[0].materialized_phase2_kernel.num_warps
        == kernels[1].materialized_phase2_kernel.num_warps
    )
