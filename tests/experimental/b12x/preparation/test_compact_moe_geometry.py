"""Native compact MoE preparation selects FC1 output tile geometry."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
from b12x.preparation import FrozenMapping
from .test_fused_moe_effort import DEVICES, query


@pytest.mark.parametrize("device", DEVICES)
def test_native_compact_search_preserves_default_and_explores_fc1_tiles(device):
    q = replace(query(tokens=8), intermediate_size=576)
    configuration = TUNING.configure(q, device=device)
    assert configuration.default.micro_fc1_tile_n == 128
    candidates = [config for _, config in TUNING.iterate(configuration)]
    assert {c.micro_fc1_tile_n for c in candidates if c.backend == "micro"} == {64, 128}
    assert all(c.micro_fc1_tile_n == 128 for c in candidates if c.backend != "micro")
    for config in candidates:
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config


@pytest.mark.parametrize("device", DEVICES)
def test_native_compact_search_explores_inline_and_expanded_fc1_tiles(device):
    q = replace(query(tokens=8), intermediate_size=576, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    choices = {
        (config.micro_fc1_tile_n, config.csf_inline_scales)
        for _, config in TUNING.iterate(configuration)
        if config.backend == "micro"
    }
    assert choices == {(64, False), (64, True), (128, False), (128, True)}


@pytest.mark.parametrize("device", DEVICES)
def test_n64_fc1_choice_rejects_other_storage_and_backends(device):
    q = replace(query(tokens=8), intermediate_size=576)
    config = MoeDecodeConfig("micro", "internal", None, micro_fc1_tile_n=64)
    assert TUNING.configure(q, device=device, override=config).pinned == config
    with pytest.raises(ValueError):
        TUNING.configure(
            replace(q, intermediate_size=256), device=device, override=config
        )
    with pytest.raises(ValueError):
        TUNING.configure(
            q,
            device=device,
            override=replace(
                config,
                backend="dynamic",
                dynamic_tile_m=16,
                dynamic_route_mode="grouped",
            ),
        )


def test_fc1_tile_config_codec_requires_explicit_geometry():
    config = MoeDecodeConfig("micro", "internal", None)
    payload = config.to_dict()
    assert payload["micro_fc1_tile_n"] == 128
    del payload["micro_fc1_tile_n"]
    with pytest.raises(ValueError):
        MoeDecodeConfig.from_config(FrozenMapping(payload))
    payload["micro_fc1_tile_n"] = True
    with pytest.raises(TypeError):
        MoeDecodeConfig.from_config(FrozenMapping(payload))


def test_compact_geometry_and_materialized_budget_remain_independent():
    device = DEVICES[1]
    q = replace(query(tokens=8), intermediate_size=576, csf_inline_scales=True)
    compact = MoeDecodeConfig(
        "micro", "internal", None, micro_fc1_tile_n=64, csf_inline_scales=True
    )
    assert compact.materialized_fc1_cta_budget == 2
    assert TUNING.configure(q, device=device, override=compact).pinned == compact
    assert MoeDecodeConfig.from_config(FrozenMapping(compact.to_dict())) == compact
    with pytest.raises(ValueError, match="three-CTA materialized FC1"):
        TUNING.configure(
            q, device=device, override=replace(compact, materialized_fc1_cta_budget=3)
        )
    materialized = MoeDecodeConfig(
        "dynamic",
        "internal",
        None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
        materialized_fc1_cta_budget=3,
    )
    q = replace(q, num_tokens=24, routed_rows=144)
    assert materialized.micro_fc1_tile_n == 128
    assert (
        TUNING.configure(q, device=device, override=materialized).pinned == materialized
    )
    assert (
        MoeDecodeConfig.from_config(FrozenMapping(materialized.to_dict()))
        == materialized
    )
    with pytest.raises(ValueError, match="N64 FC1 tiles"):
        TUNING.configure(
            q, device=device, override=replace(materialized, micro_fc1_tile_n=64)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_compact_copy_group_choices_preserve_default_and_roundtrip(device):
    q = replace(query(tokens=8), intermediate_size=576, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    assert configuration.default.micro_fc1_max_copy_groups == 2
    configs = [c for _, c in TUNING.iterate(configuration)]
    assert {c.micro_fc1_max_copy_groups for c in configs if c.backend == "micro"} == {
        1,
        2,
    }
    assert all(
        c.micro_fc1_max_copy_groups == 2 for c in configs if c.backend != "micro"
    )
    assert {
        (c.micro_fc1_tile_n, c.csf_inline_scales, c.micro_fc1_max_copy_groups)
        for c in configs
        if c.backend == "micro"
    } == {
        (n, inline, groups)
        for n in (64, 128)
        for inline in (False, True)
        for groups in (1, 2)
    }
    for config in configs:
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config


def test_copy_group_config_codec_requires_explicit_integer():
    config = MoeDecodeConfig("micro", "internal", None)
    assert config.micro_fc1_max_copy_groups == 2
    payload = config.to_dict()
    del payload["micro_fc1_max_copy_groups"]
    with pytest.raises(ValueError):
        MoeDecodeConfig.from_config(FrozenMapping(payload))
    for value in (True, 1.0, None):
        payload["micro_fc1_max_copy_groups"] = value
        with pytest.raises(TypeError):
            MoeDecodeConfig.from_config(FrozenMapping(payload))


@pytest.mark.parametrize("value", (0, 3, True))
def test_copy_group_override_rejects_invalid_modes(value):
    q = replace(query(tokens=8), intermediate_size=576)
    config = MoeDecodeConfig("micro", "internal", None, micro_fc1_max_copy_groups=value)
    with pytest.raises((TypeError, ValueError)):
        TUNING.configure(q, device=DEVICES[1], override=config)


def test_one_copy_group_rejects_other_kernel_families():
    q = replace(query(tokens=24), intermediate_size=576)
    dynamic = MoeDecodeConfig(
        "dynamic",
        "internal",
        None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        micro_fc1_max_copy_groups=1,
    )
    with pytest.raises(ValueError, match="one pending FC1 copy group"):
        TUNING.configure(q, device=DEVICES[1], override=dynamic)
    compact = MoeDecodeConfig("micro", "internal", None, micro_fc1_max_copy_groups=1)
    with pytest.raises(ValueError, match="one pending FC1 copy group"):
        TUNING.configure(
            replace(q, num_tokens=8, routed_rows=48, intermediate_size=256),
            device=DEVICES[1],
            override=compact,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_compact_geometry_copy_and_input_quantization_choices(device):
    q = replace(query(tokens=8), intermediate_size=576, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    choices = {
        (
            config.micro_fc1_tile_n,
            config.micro_fc1_max_copy_groups,
            config.micro_fc1_pipeline_stages,
            config.compact_fuse_input_quantization,
            config.csf_inline_scales,
        )
        for _, config in TUNING.iterate(configuration)
        if config.backend == "micro"
    }
    assert choices == {
        (tile, copies, stages, fuse, inline)
        for tile in (64, 128)
        for copies, stages in ((1, 1), (1, 2), (2, 2))
        for fuse in ((False, True) if tile == 128 else (False,))
        for inline in (False, True)
    }
    assert configuration.default.micro_fc1_tile_n == 128
    assert configuration.default.micro_fc1_max_copy_groups == 2
    assert configuration.default.compact_fuse_input_quantization is False
    for _, config in TUNING.iterate(configuration):
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config


def test_compact_input_quantization_rejects_unpaired_n64_tiles():
    from b12x.moe._shared.kernels.w4a8_compact_micro import _DirectW4A8CompactLaunch

    q = replace(query(tokens=8), intermediate_size=576)
    with pytest.raises(ValueError, match="paired N128"):
        TUNING.configure(
            q,
            device=DEVICES[1],
            override=MoeDecodeConfig(
                "micro",
                "internal",
                None,
                micro_fc1_tile_n=64,
                compact_fuse_input_quantization=True,
            ),
        )
    with pytest.raises(ValueError, match="paired compact projections"):
        _DirectW4A8CompactLaunch(
            max_tokens=8,
            num_topk=6,
            k=5120,
            n=576,
            experts=384,
            input_scale_count=384,
            down_scale_count=384,
            swiglu_limit=10.0,
            fast_math=True,
            fc1_tile_n=64,
            fuse_input_quantization=True,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_compact_pipeline_stage_choices_preserve_default_and_roundtrip(device):
    q = replace(query(tokens=8), intermediate_size=576, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    assert configuration.default.micro_fc1_pipeline_stages == 2
    configs = [config for _, config in TUNING.iterate(configuration)]
    assert {
        (
            c.micro_fc1_tile_n,
            c.csf_inline_scales,
            c.micro_fc1_max_copy_groups,
            c.micro_fc1_pipeline_stages,
        )
        for c in configs
        if c.backend == "micro"
    } == {
        (n, inline, groups, stages)
        for n in (64, 128)
        for inline in (False, True)
        for groups, stages in ((1, 1), (1, 2), (2, 2))
    }
    assert all(
        c.micro_fc1_pipeline_stages == 2 for c in configs if c.backend != "micro"
    )
    for config in configs:
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config


def test_pipeline_stage_codec_requires_explicit_integer():
    payload = MoeDecodeConfig("micro", "internal", None).to_dict()
    del payload["micro_fc1_pipeline_stages"]
    with pytest.raises(ValueError):
        MoeDecodeConfig.from_config(FrozenMapping(payload))
    for value in (True, 1.0, None):
        payload["micro_fc1_pipeline_stages"] = value
        with pytest.raises(TypeError):
            MoeDecodeConfig.from_config(FrozenMapping(payload))


@pytest.mark.parametrize("stages", (0, 3, True))
def test_pipeline_stage_override_rejects_invalid_modes(stages):
    config = MoeDecodeConfig(
        "micro",
        "internal",
        None,
        micro_fc1_max_copy_groups=1,
        micro_fc1_pipeline_stages=stages,
    )
    with pytest.raises((TypeError, ValueError)):
        TUNING.configure(
            replace(query(tokens=8), intermediate_size=576),
            device=DEVICES[1],
            override=config,
        )


def test_single_pipeline_stage_rejects_two_copy_groups_and_other_families():
    q = replace(query(tokens=8), intermediate_size=576)
    config = MoeDecodeConfig("micro", "internal", None, micro_fc1_pipeline_stages=1)
    with pytest.raises(ValueError, match="one FC1 pipeline stage"):
        TUNING.configure(q, device=DEVICES[1], override=config)
    with pytest.raises(ValueError, match="one FC1 pipeline stage"):
        TUNING.configure(
            q,
            device=DEVICES[1],
            override=replace(config, backend="dynamic", dynamic_tile_m=16),
        )
    with pytest.raises(ValueError):
        TUNING.configure(
            replace(q, intermediate_size=256),
            device=DEVICES[1],
            override=replace(config, micro_fc1_max_copy_groups=1),
        )
