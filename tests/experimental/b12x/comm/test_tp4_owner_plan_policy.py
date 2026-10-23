"""Owner publication preserves fixed transport and reduction rounding contracts."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie.pcie_oneshot import _CuTeOneshotBackend, _eager_storage_shards


@pytest.mark.parametrize(
    "world,dtype,rounding,sharded,push,staged,rows,hidden,expected",
    [
        (
            4,
            torch.bfloat16,
            torch.bfloat16,
            True,
            False,
            True,
            48,
            6144,
            "stage_tp4_owner",
        ),
        (
            4,
            torch.float16,
            torch.float16,
            True,
            False,
            True,
            48,
            6144,
            "stage_tp4_owner",
        ),
        (4, torch.bfloat16, None, True, False, True, 48, 6144, "stage_pull"),
        (4, torch.bfloat16, torch.float32, True, False, True, 48, 6144, "stage_pull"),
        (4, torch.float32, torch.bfloat16, True, False, True, 48, 6144, "stage_pull"),
        (4, torch.float16, torch.bfloat16, True, False, True, 48, 6144, "stage_pull"),
        (4, torch.bfloat16, torch.bfloat16, False, False, True, 48, 6144, "stage_pull"),
        (4, torch.bfloat16, torch.bfloat16, True, True, True, 48, 6144, "stage_push"),
        (4, torch.bfloat16, torch.bfloat16, True, False, False, 48, 6144, "registered"),
        (3, torch.bfloat16, torch.bfloat16, True, False, True, 48, 6144, "stage_pull"),
        (8, torch.bfloat16, torch.bfloat16, True, False, True, 48, 6144, "stage_pull"),
        (
            4,
            torch.bfloat16,
            torch.bfloat16,
            True,
            False,
            True,
            1,
            6144,
            "stage_remote_push",
        ),
        (
            4,
            torch.bfloat16,
            torch.bfloat16,
            True,
            False,
            True,
            1,
            520,
            "stage_tp4_owner",
        ),
    ],
)
def test_matching_rounding_and_existing_storage(
    world, dtype, rounding, sharded, push, staged, rows, hidden, expected
):
    policy = (push, True, True, sharded, True)
    state = SimpleNamespace(
        world_size=world,
        transport_policy=policy,
        eager_tables=object() if staged else None,
        eager_buffer_bytes=4096 * 6144 * 4 if staged else None,
    )
    source = torch.empty((rows, hidden), dtype=dtype, device="meta")
    mode, *_ = _CuTeOneshotBackend._fused_launch_config(
        state, source, reduction_dtype=rounding
    )
    assert mode == expected
    if mode == "stage_tp4_owner":
        assert _eager_storage_shards(world, policy) >= 2
