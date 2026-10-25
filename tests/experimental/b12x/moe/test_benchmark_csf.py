"""Benchmark checkpoint encoding preserves every native scale byte."""

import pytest
import torch

from b12x._lib.quant._csf_prepare import expand_source_scales, validate_source
from benchmarks.experimental.b12x.moe_preparation import encode_csf_32x2_source
from ..conftest import require_b12x
from ..quantization.test_csf_encode import _scales


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("rows,columns", [(64, 3), (384, 18), (1152, 160)])
def test_block_checkpoint_roundtrip(quantization, rows, columns):
    device = require_b12x()
    logical = _scales(
        3, rows, columns, window=16 if quantization == "nvfp4" else 2, seed=71
    ).to(device)
    source = encode_csf_32x2_source(logical, quantization=quantization)
    assert source.encoding == "csf_32x2_v1"
    validate_source(source)
    restored = torch.full_like(logical, 0x7F)
    expand_source_scales(source, restored, swizzled=False)
    torch.testing.assert_close(restored, logical, rtol=0, atol=0)


def test_encoded_mxfp4_inline_override_uses_prepared_planes(monkeypatch):
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
    from b12x.preparation import detect_device
    from .test_mxfp4_csf import _prepare_arms

    device = require_b12x()
    (native, compressed), _ = _prepare_arms(5120, 576, "w31", monkeypatch)
    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="triton",
        max_active_clusters=188,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
        materialized_fc1_cta_budget=3,
    )
    for owner, encoded in ((compressed, True), (native, False)):
        declaration = moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=24, top_k=6),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=False),
        )
        assert declaration.query.csf_inline_scales is encoded
        if encoded:
            TUNING.configure(
                declaration.query,
                device=detect_device(device).identity,
                override=config,
            )
        else:
            with pytest.raises(ValueError, match="inline CSF scales"):
                TUNING.configure(
                    declaration.query,
                    device=detect_device(device).identity,
                    override=config,
                )
