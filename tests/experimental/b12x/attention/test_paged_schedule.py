"""External replay schedules accept only exactly matching selected geometry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention.paged.schedule import PrefillSchedule, schedule_nbytes


def test_external_schedule_checks_each_geometry_field_and_tensor_contract():
    caps = SimpleNamespace(
        device=torch.device("cpu"),
        max_work_items=8,
        max_total_q=4,
        max_batch=2,
        page_size=128,
        mode="verify",
        use_cuda_graph=True,
    )
    plan = SimpleNamespace(
        split_kv=True,
        page_table_shape=(2, 32),
        qo_tile_indices=(0, 1),
        kv_tile_indices=(0, 1),
        block_valid_mask=(True, True),
        cta_tile_q=16,
        gqa_group_size=16,
        window_left=-1,
        msa_block_sparse=True,
        kv_chunk_size=128,
    )
    owner = SimpleNamespace(caps=caps, plan=plan)
    schedule = PrefillSchedule.from_scratch_plan(owner)
    size = schedule_nbytes(caps)
    backing = torch.empty(size + 255, dtype=torch.uint8)
    offset = (-backing.data_ptr()) % 256
    storage = backing[offset : offset + size]
    binding = schedule.bind(storage)
    scratch = SimpleNamespace(device=torch.device("cpu"))
    binding.install(scratch, owner)
    assert scratch._external_prefill_schedule
    for index, value in enumerate(schedule.geometry):
        geometry = list(schedule.geometry)
        geometry[index] = not value if isinstance(value, bool) else value + 1
        altered = replace(schedule, geometry=tuple(geometry)).bind(storage)
        with pytest.raises(ValueError, match="selected plan geometry"):
            altered.install(scratch, owner)
    with pytest.raises(ValueError, match="capacities"):
        schedule.bind(storage[:-1])
    views = dict(binding.views)
    views.pop("merge_indptr")
    with pytest.raises(ValueError, match="every scheduling field"):
        replace(binding, views=views).install(scratch, owner)
    views = dict(binding.views)
    views["merge_indptr"] = views["merge_indptr"].long()
    with pytest.raises(ValueError, match="tensor metadata"):
        replace(binding, views=views).install(scratch, owner)


@pytest.mark.parametrize(
    "mode,split", [("decode", True), ("verify", False), ("extend", False)]
)
def test_external_schedule_rejects_other_execution_paths(mode, split):
    owner = SimpleNamespace(
        caps=SimpleNamespace(mode=mode, use_cuda_graph=True),
        plan=SimpleNamespace(split_kv=split),
    )
    with pytest.raises(ValueError, match="split graph verify/extend"):
        PrefillSchedule.from_scratch_plan(owner)


def test_external_schedule_requires_vector_live_metadata(monkeypatch):
    import b12x.attention.paged.schedule as schedules

    schedule = PrefillSchedule(
        counts=(8, 8, 8, 8, 5, 3, 1, 2, 1),
        geometry=(2, 1, 1, 1, 16, 16, 128, True, -1, True, 128),
        device=torch.device("cpu"),
    )
    backing = torch.empty(9 * 256 + 255, dtype=torch.uint8)
    offset = (-backing.data_ptr()) % 256
    storage = backing[offset : offset + 9 * 256]
    monkeypatch.setattr(
        schedules, "update_prefill_graph_chunk_metadata", lambda **kwargs: None
    )
    lengths, cu = torch.empty(2, dtype=torch.int32), torch.empty(3, dtype=torch.int32)
    for bad_lengths, bad_cu in ((lengths[:, None], cu), (lengths, cu[:, None])):
        with pytest.raises(ValueError, match="vectors"):
            schedule.update(storage, bad_lengths, bad_cu)
