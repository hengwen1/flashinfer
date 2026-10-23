"""Caller-owned prefill schedules reusable by equivalent prepared attention plans."""

from dataclasses import dataclass

import torch

from b12x.preparation import require_prepared
from .graph_replay import update_prefill_graph_chunk_metadata


SCHEDULE_FIELDS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "block_valid_mask",
    "merge_indptr",
    "o_indptr",
    "kv_chunk_size_ptr",
    "kv_window_start_tokens",
    "total_num_rows_ptr",
)


def schedule_counts(caps):
    """Return fixed capacities independently of the selected attention geometry."""
    return (
        *(int(caps.max_work_items),) * 4,
        int(caps.max_total_q) + 1,
        int(caps.max_batch) + 1,
        1,
        int(caps.max_batch),
        1,
    )


def schedule_nbytes(caps):
    return sum(((count * 4 + 255) // 256) * 256 for count in schedule_counts(caps))


def schedule_views(storage, counts):
    required = sum(((count * 4 + 255) // 256) * 256 for count in counts)
    if storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous():
        raise ValueError(
            "prefill schedule requires contiguous one-dimensional uint8 storage"
        )
    if storage.numel() < required or storage.data_ptr() % 256:
        raise ValueError(
            "prefill schedule storage must fit its capacities and be 256-byte aligned"
        )
    offset, views = 0, {}
    for name, count in zip(SCHEDULE_FIELDS, counts, strict=True):
        views[name] = storage.narrow(0, offset, count * 4).view(torch.int32)
        offset += ((count * 4 + 255) // 256) * 256
    return views


def replay_shape(plan):
    active = [
        (q, kv)
        for q, kv, valid in zip(
            plan.qo_tile_indices,
            plan.kv_tile_indices,
            plan.block_valid_mask,
            strict=False,
        )
        if valid
    ]
    if not active:
        raise RuntimeError("prefill graph replay plan contains no active work")
    q_tiles = max(q for q, _ in active) + 1
    chunks = max(kv for _, kv in active) + 1
    rows = max(
        1,
        (q_tiles * int(plan.cta_tile_q) + int(plan.gqa_group_size) - 1)
        // int(plan.gqa_group_size),
    )
    return q_tiles, chunks, rows


@dataclass(frozen=True)
class PrefillSchedule:
    """Selected scheduling geometry, without numerical attention or KV storage."""

    counts: tuple
    geometry: tuple
    device: torch.device
    metadata: object = None

    @classmethod
    def from_scratch_plan(cls, scratch_plan, metadata=None):
        caps, plan = scratch_plan.caps, scratch_plan.plan
        if not (
            caps.use_cuda_graph and caps.mode in ("verify", "extend") and plan.split_kv
        ):
            raise ValueError(
                "external prefill schedules require split graph verify/extend attention"
            )
        q_tiles, chunks, rows = replay_shape(plan)
        geometry = (
            int(plan.page_table_shape[0]),
            q_tiles,
            chunks,
            rows,
            int(plan.cta_tile_q),
            int(plan.gqa_group_size),
            int(caps.page_size),
            bool(plan.split_kv),
            int(plan.window_left),
            bool(plan.msa_block_sparse),
            int(plan.kv_chunk_size),
        )
        return cls(schedule_counts(caps), geometry, caps.device, metadata)

    @classmethod
    def from_plan(cls, declaration):
        state = require_prepared(declaration, "attention.gqa")
        return cls.from_scratch_plan(state.scratch_plan, state.metadata)

    @property
    def key(self):
        return self.counts, self.geometry

    def views(self, storage):
        if storage.device != self.device:
            raise ValueError(
                "prefill schedule and selected plan must use the same device"
            )
        return schedule_views(storage, self.counts)

    def update(self, storage, cache_seqlens, cu_seqlens_q):
        if any(t.ndim != 1 for t in (cache_seqlens, cu_seqlens_q)):
            raise ValueError("prepared prefill schedules require live metadata vectors")
        if any(
            t.dtype != torch.int32 or not t.is_contiguous()
            for t in (cache_seqlens, cu_seqlens_q)
        ):
            raise ValueError(
                "prepared prefill schedules require contiguous int32 live metadata"
            )
        batch, q_tiles, chunks, rows, cta_q, gqa, page, split, window, msa, chunk = (
            self.geometry
        )
        update_prefill_graph_chunk_metadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            **self.views(storage),
            batch=batch,
            max_q_tiles_per_req=q_tiles,
            max_chunks_per_q_tile=chunks,
            max_q_rows_per_req=rows,
            cta_tile_q=cta_q,
            gqa_group_size=gqa,
            page_size=page,
            split_kv=split,
            window_left=window,
            msa_block_sparse=msa,
            fixed_kv_chunk_size=chunk,
            _prepared=self.metadata,
        )

    def bind(self, storage):
        return PrefillScheduleBinding(self.key, self.views(storage))


@dataclass(frozen=True)
class PrefillScheduleBinding:
    """Read-only scheduling views produced before every attention consumer."""

    key: tuple
    views: dict

    def install(self, scratch, owner):
        expected = PrefillSchedule.from_scratch_plan(owner)
        if self.key != expected.key:
            raise ValueError(
                "external prefill schedule does not match selected plan geometry"
            )
        if set(self.views) != set(SCHEDULE_FIELDS):
            raise ValueError(
                "external prefill schedule must supply every scheduling field"
            )
        for name, count in zip(SCHEDULE_FIELDS, expected.counts, strict=True):
            tensor = self.views[name]
            if (
                tensor.device != scratch.device
                or tensor.dtype != torch.int32
                or tensor.shape != (count,)
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "external prefill schedule has incompatible tensor metadata"
                )
            setattr(scratch, name, tensor)
        scratch._external_prefill_schedule = True
