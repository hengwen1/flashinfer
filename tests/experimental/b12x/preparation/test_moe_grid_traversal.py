"""Sparse MoE grid traversal preserves the complete ordered search domain."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import DeviceIdentity, ParameterSpace
from .test_fused_moe_effort import query
from .test_precision_choices import _nvfp4_a16_query


def _dense(space):
    return ParameterSpace(
        knobs=space.knobs,
        predicates=space.predicates,
        efficiency_predicates=space.efficiency_predicates,
        exhaustive=space.exhaustive,
        autotune_effort=space.autotune_effort,
    )


CASES = (
    *((tokens, 576, "w4a8_mx") for tokens in (1, 4, 8, 16, 24, 64, 1025)),
    (8, 512, "w4a8_mx"),
    (24, 512, "w4a8_mx"),
    (8, 512, "nvfp4"),
    (1025, 512, "nvfp4"),
)


@pytest.mark.parametrize("sm_count", (7, 48, 96, 188))
@pytest.mark.parametrize("tokens,width,recipe", CASES)
@pytest.mark.parametrize("effort", ("off", "low", "medium", "high", "max"))
def test_ordered_lowered_domain_and_default(sm_count, tokens, width, recipe, effort):
    device = DeviceIdentity("nvidia", (12, 0), sm_count, "Blackwell")
    q = replace(
        query(tokens=tokens, recipe=recipe, autotune_effort=effort),
        intermediate_size=width,
        csf_inline_scales=True,
    )
    configuration = TUNING.configure(q, device=device)
    sparse = configuration.space
    dense = _dense(sparse)
    expected = list(TUNING.iterate(replace(configuration, space=dense)))
    actual = list(TUNING.iterate(configuration))
    assert actual == expected
    assert actual
    assert configuration.default in [config for _, config in actual]
    sparse.validate(TUNING.config_payload(configuration.default))
    grid = next(k for k in sparse.knobs if k.name == "max_active_clusters")
    assert grid.values == (None, *range(1, max(grid.values[1:], default=0) + 1))


@pytest.mark.parametrize("sm_count", (7, 48, 188))
@pytest.mark.parametrize("tokens,width", ((8, 576), (24, 576), (8, 512)))
def test_built_space_promotion_preserves_dense_max_and_validation(
    sm_count, tokens, width
):
    device = DeviceIdentity("nvidia", (12, 0), sm_count, "Blackwell")
    q = replace(query(tokens=tokens), intermediate_size=width, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    medium = configuration.space
    maximum = TUNING.parameter_space(replace(q, autotune_effort="max"), device)
    expected = list(_dense(maximum).configurations())
    for promoted in (
        replace(medium, exhaustive=True),
        replace(medium, autotune_effort="max"),
        maximum,
    ):
        assert list(promoted.configurations()) == expected
    choice = dict(TUNING.config_payload(configuration.default))
    choice["max_active_clusters"] = next(
        value for value in range(1, sm_count + 1) if value not in medium.sparse_grids
    )
    if next(k for k in medium.knobs if k.name == "max_active_clusters").active(choice):
        with pytest.raises(ValueError, match="efficiency"):
            medium.validate(choice)
        replace(medium, exhaustive=True).validate(choice)
        maximum.validate(choice)
    choice["max_active_clusters"] = 2 * sm_count + 1
    with pytest.raises(ValueError, match="outside"):
        maximum.validate(choice)


@pytest.mark.parametrize("multi", (False, True))
@pytest.mark.parametrize("effort", ("medium", "max"))
def test_w4a16_ordered_domain_with_conditional_grid(multi, effort):
    device = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
    q = replace(_nvfp4_a16_query(8), autotune_effort=effort)
    if multi:
        q = replace(
            q,
            quant_mode="multi",
            quant_modes=("nvfp4", "w4a16"),
            weight_layouts=("mma_view", "source_native"),
        )
    configuration = TUNING.configure(q, device=device)
    assert list(TUNING.iterate(configuration)) == list(
        TUNING.iterate(replace(configuration, space=_dense(configuration.space)))
    )


def test_sparse_traversal_skips_only_rejected_grid_products():
    device = DeviceIdentity("nvidia", (12, 0), 188, "Blackwell")
    for tokens, expected_raw, expected_legal in (
        (8, 2256, 300),
        (16, 2112, 228),
        (24, 2112, 84),
    ):
        q = replace(query(tokens=tokens), intermediate_size=576, csf_inline_scales=True)
        configuration = TUNING.configure(q, device=device)
        sparse = TUNING.iterate(configuration)
        dense = TUNING.iterate(
            replace(configuration, space=_dense(configuration.space))
        )
        assert list(sparse) == list(dense)
        assert sparse.cartesian_count == expected_raw
        assert sparse.legal_count == dense.legal_count == expected_legal
        assert sparse.cartesian_count < dense.cartesian_count
