"""Typed constructor choices and their explicit compilation facts."""

from dataclasses import replace

import pytest

from b12x.attention.dsa_indexer._mxfp4_tuning import FusedMXFP4Config, default_ctas
from b12x.attention.dsa_indexer.fused_indexer import DSAFusedIndexerKernel
from b12x.preparation import FrozenMapping


def test_config_codec_preserves_each_concrete_choice():
    config = FusedMXFP4Config(
        ctas_per_row=8,
        tile_schedule="cyclic",
        grid_layout="slice_major",
        state_reset_pdl=True,
        mxfp4_warp_weights=True,
        mxfp4_aligned_q=True,
        mxfp4_query_offset_mode=2,
        mxfp4_unsigned_candidate_pages=True,
        mxfp4_merge_bits=16,
        mxfp4_histogram_count_bits=32,
        mxfp4_threads=512,
        mxfp4_score_passes=4,
        mxfp4_coop_residency=2,
        mxfp4_compact_shared=True,
    )
    assert FusedMXFP4Config.from_config(FrozenMapping(config.to_dict())) == config
    with pytest.raises(ValueError, match="fields"):
        FusedMXFP4Config.from_config({**config.to_dict(), "unknown": True})


@pytest.mark.parametrize(
    "field,value",
    (
        ("ctas_per_row", False),
        ("grid_layout", "unknown"),
        ("tile_schedule", "unknown"),
        ("ordered_output", 1),
        ("mxfp4_query_offset_mode", True),
        ("mxfp4_merge_bits", 12),
    ),
)
def test_invalid_constructor_choices_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        FusedMXFP4Config(**{**dict(ctas_per_row=1), field: value})


def test_constructor_facts_include_resolved_geometry_and_load_policy():
    base = FusedMXFP4Config(ctas_per_row=2)
    values = []
    for config in (
        base,
        replace(base, mxfp4_warp_weights=True),
        replace(base, grid_layout="slice_major"),
        replace(
            base, mxfp4_threads=512, mxfp4_score_passes=4, mxfp4_compact_shared=True
        ),
        replace(
            base,
            mxfp4_partition_count=4,
            mxfp4_score_batch_tiles=2,
            mxfp4_cutoff_strategy="shared_compact",
        ),
    ):
        kernel = DSAFusedIndexerKernel(
            num_heads_static=32,
            topk=512,
            num_sms=188,
            max_seq_capacity=32768,
            mxfp4=True,
            page_size=128,
            score_width=32768,
            **config.constructor_options(),
        )
        facts = dict(kernel.constructor_config)
        for name, value in config.constructor_options().items():
            assert facts[name] == value
        values.append(kernel.constructor_config)
    assert len(set(values)) == 5


def test_wave_backup_preserves_four_and_eight_tile_partitions():
    assert default_ctas(64, 32768, 188) == 8
    assert default_ctas(128, 32768, 188) == 16
    assert default_ctas(128, 16384, 188) == 4


def test_pinned_cta_backup_resolves_state_for_the_same_geometry():
    from types import SimpleNamespace
    from b12x.attention.dsa_indexer._mxfp4_tuning import default_fused_config

    query = SimpleNamespace(
        max_candidates=0,
        max_page_table_width=256,
        page_size=128,
        max_q_rows=64,
        operands=FrozenMapping(),
    )
    device = SimpleNamespace(sm_count=188)
    serial = default_fused_config(query, device, ctas_per_row=8)
    resident = default_fused_config(query, device, ctas_per_row=2)
    assert serial.mxfp4_serial_state_words == 2
    assert resident.mxfp4_serial_state_words == 8968
    assert not resident.mxfp4_aligned_q


def test_large_row_capacity_uses_an_eligible_grid_dimension():
    from types import SimpleNamespace
    from b12x.attention.dsa_indexer._mxfp4_tuning import default_fused_config
    from b12x.attention.dsa_indexer._tuning import DsaIndexerConfig, _validate_config

    query = SimpleNamespace(
        num_q_heads=32,
        cache_format="mxfp4",
        max_candidates=0,
        max_page_table_width=256,
        page_size=128,
        max_q_rows=65536,
        max_k_rows=32768,
        candidate_topk_blocks=0,
        operands=FrozenMapping(),
    )
    device = SimpleNamespace(sm_count=188)
    fused = default_fused_config(query, device)
    assert fused.grid_layout == "slice_major"
    _validate_config(query, DsaIndexerConfig(backend="native", mxfp4=fused), device)
    with pytest.raises(ValueError, match="dimension"):
        _validate_config(
            query,
            DsaIndexerConfig(
                backend="native", mxfp4=replace(fused, grid_layout="row_major")
            ),
            device,
        )


def test_inactive_cooperative_histogram_pins_are_rejected():
    from types import SimpleNamespace
    from b12x.attention.dsa_indexer._tuning import DsaIndexerConfig, _validate_config

    query = SimpleNamespace(
        num_q_heads=32,
        cache_format="mxfp4",
        max_candidates=0,
        max_page_table_width=256,
        page_size=128,
        max_q_rows=4,
        max_k_rows=32768,
        candidate_topk_blocks=0,
        operands=FrozenMapping(),
    )
    device = SimpleNamespace(sm_count=188)
    for fused in (
        FusedMXFP4Config(ctas_per_row=1, mxfp4_merge_bits=16),
        FusedMXFP4Config(ctas_per_row=2, mxfp4_histogram_count_bits=32),
    ):
        with pytest.raises(ValueError, match="requires|require"):
            _validate_config(
                query, DsaIndexerConfig(backend="native", mxfp4=fused), device
            )


@pytest.mark.parametrize("invocation_page_size", (64, 128))
def test_mxfp4_page_geometry_matches_the_binding_capacity(invocation_page_size):
    from b12x.attention import dsa_indexer

    caps = dsa_indexer.Caps(
        device="cpu",
        num_q_heads=32,
        max_q_rows=1,
        max_page_table_width=4,
        topk=512,
        page_size=64,
        cache_format="mxfp4",
    )
    invocation = FrozenMapping({"page_size": invocation_page_size})
    if invocation_page_size != caps.page_size:
        with pytest.raises(ValueError, match="page_size must match Caps.page_size"):
            dsa_indexer.plan(caps, invocation=invocation)
    else:
        assert (
            dsa_indexer.plan(caps, invocation=invocation).query.page_size
            == caps.page_size
        )


@pytest.mark.parametrize("width", (256, 512))
def test_small_output_candidates_record_only_executed_options(width):
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer._mxfp4_tuning import small_output_config
    from b12x.attention.dsa_indexer._tuning import TUNING
    from b12x.preparation import DeviceIdentity

    query = dsa_indexer.plan(
        dsa_indexer.Caps(
            device="cpu",
            num_q_heads=32,
            max_q_rows=4,
            max_page_table_width=width // 64,
            topk=512,
            page_size=64,
            cache_format="mxfp4",
        )
    ).query
    device = DeviceIdentity("nvidia", (12, 0), 188, "Blackwell")
    small = [
        config
        for _, config in TUNING.eligible_plan(query, device).candidates
        if config.mxfp4.mxfp4_small_output
    ]
    assert len(small) == 1
    assert small[0].mxfp4 == small_output_config(width)
    assert small[0].mxfp4.mxfp4_threads == (512 if width == 512 else 1024)
    for name, value in (
        ("mxfp4_threads", 1024 if width == 512 else 512),
        ("grid_layout", "row_major"),
        ("vector_output_pairs", True),
        ("mxfp4_warp_weights", True),
        ("mxfp4_query_offset_mode", 1),
        ("mxfp4_unsigned_candidate_pages", True),
        ("state_reset_pdl", True),
        ("mxfp4_score_passes", 2),
        ("mxfp4_compact_shared", True),
    ):
        invalid = replace(small[0], mxfp4=replace(small[0].mxfp4, **{name: value}))
        with pytest.raises(ValueError, match="canonical fixed geometry"):
            TUNING.configure(query, device=device, override=invalid)
    sorted_composition = replace(
        small[0], mxfp4=replace(small[0].mxfp4, ordered_output=False)
    )
    assert (
        TUNING.configure(query, device=device, override=sorted_composition).pinned
        == sorted_composition
    )


@pytest.mark.parametrize(
    "route", ("auto", "paged_fused", "paged_tiled", "packed_contiguous")
)
def test_mxfp4_plan_rejects_split_routes_and_preserves_fp8_routes(route):
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer._tuning import _validate_query

    capacities = dict(
        device="cpu",
        num_q_heads=32,
        max_q_rows=1,
        max_page_table_width=4,
        topk=512,
        page_size=64,
    )
    caps = dsa_indexer.Caps(**capacities, cache_format="mxfp4")
    query = dsa_indexer.plan(caps).query
    if route in ("auto", "paged_fused"):
        declared = dsa_indexer.plan(
            dsa_indexer.Caps(**capacities, cache_format="mxfp4", route=route)
        )
        assert declared.query.route == route
        invoked = dsa_indexer.plan(caps, invocation=FrozenMapping({"route": route}))
        assert invoked.query.route == route
        _validate_query(replace(query, route=route), None)
    else:
        with pytest.raises(ValueError, match="MXFP4 route"):
            dsa_indexer.Caps(**capacities, cache_format="mxfp4", route=route)
        with pytest.raises(ValueError, match="MXFP4 route"):
            dsa_indexer.plan(caps, invocation=FrozenMapping({"route": route}))
        with pytest.raises(ValueError, match="MXFP4 route"):
            _validate_query(replace(query, route=route), None)
    assert (
        dsa_indexer.Caps(**capacities, cache_format="fp8", route=route).route == route
    )
    _validate_query(replace(query, cache_format="fp8", route=route), None)
