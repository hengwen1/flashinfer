"""World-size-dispatched PCIe all-reduce runtime."""

from __future__ import annotations

import logging
import os

from contextlib import ExitStack, contextmanager
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation import FrozenMapping, Plan
from b12x.preparation.types import require_prepared
from ._tuning import PcieConfig

from .pcie_hierarchical import (
    SUPPORTED_WORLD_SIZES as HIERARCHICAL_WORLD_SIZES,
)
from .pcie_hierarchical import (
    PCIeHierarchicalAllReduce,
)
from .pcie_island_rs import (
    CROSSOVER_ELEMENTS as ISLAND_RS_CROSSOVER_ELEMENTS,
)
from .pcie_island_rs import (
    PREFERRED_ALIGNMENT_ELEMENTS as ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS,
)
from .pcie_island_rs import (
    SUPPORTED_WORLD_SIZES as ISLAND_RS_WORLD_SIZES,
)
from .pcie_island_rs import (
    PCIeIslandRSAllReduce,
)
from .pcie_oneshot import (
    DEFAULT_MAX_SIZE,
    DEFAULT_RANK_DATA_BYTES,
    SUPPORTED_WORLD_SIZES as ONESHOT_WORLD_SIZES,
    TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_BYTES,
    PCIeOneshotAllReducePool,
    _tp2_plain_remote_push_enabled,
    _broadcast_gather_object,
    _run_collective_preallocation_setup,
)
from .pcie_twoshot_bf16 import PCIeTwoShotBF16
from .pcie_twoshot import TWOSHOT_REQUIRED_SMS


logger = logging.getLogger(__name__)


# Maximum message capacity qualified for the TP16 island runtime. Callers use
# this shared policy instead of duplicating an implementation-specific limit.
ISLAND_RS_MAX_BYTES = 160 * 1024
# Partial islands use the leader path without the equal-quarter alternative.
# Limit default dispatch to decode vectors of at most 32 KiB.
PARTIAL_ISLAND_MAX_BYTES = 32 * 1024
# Serialized TP4 BF16 messages above this size favor single-rounding two-shot.
BF16_TWOSHOT_MIN_BYTES = 64 * 1024
# The push two-shot stays below one-shot through verification-sized batches;
# the cap bounds the channel's staging (two slots plus the reduce-scatter shard).
BF16_TWOSHOT_MAX_BYTES = 2 * 1024 * 1024


def _bf16_twoshot_supported(*, exchange_group, device, world_size) -> bool:
    """Coordinate capability-query failures before exchanging SM counts."""
    visible_sms = _run_collective_preallocation_setup(
        owner="PCIe twoshot-bf16 capability query",
        exchange_group=exchange_group,
        setup=lambda: int(
            torch.cuda.get_device_properties(device).multi_processor_count
        ),
    )
    counts = _broadcast_gather_object(visible_sms, exchange_group)
    if len(counts) != world_size or any(
        type(count) is not int or count <= 0 for count in counts
    ):
        raise RuntimeError("invalid peer PCIe twoshot-bf16 SM count")
    return all(count >= TWOSHOT_REQUIRED_SMS for count in counts)


def _algorithm_override() -> str:
    """Select the established runtime or enable size-routed island dispatch."""

    choice = os.getenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "auto").strip().lower()
    if choice not in ("auto", "hierarchical", "island_rs"):
        raise ValueError(
            "B12X_PCIE_ALLREDUCE_ALGORITHM must be auto, hierarchical or "
            f"island_rs, got {choice!r}"
        )
    return choice


def recommended_max_bytes(world_size: int, *, default: int = DEFAULT_MAX_SIZE) -> int:
    """Return the default dispatch capacity for the selected topology.

    Enabled TP2 graph peer-push needs at least 512 KiB. Callers must still use
    shape and execution-mode routing before selecting it over NCCL. Forced
    island reduce-scatter advertises its full supported capacity. Partial
    islands cap the default at 32 KiB; explicit caller overrides remain a
    caller policy, and larger native collectives remain supported.
    """

    if world_size in ISLAND_RS_WORLD_SIZES and _algorithm_override() == "island_rs":
        return max(default, ISLAND_RS_MAX_BYTES)
    if world_size == 2 and _tp2_plain_remote_push_enabled():
        return max(default, TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_BYTES)
    if world_size in (9, 10):
        return min(default, PARTIAL_ISLAND_MAX_BYTES)
    return default


MAX_DIRECT_WORLD_SIZE = 8
DIRECT_WORLD_SIZES = tuple(
    world_size
    for world_size in ONESHOT_WORLD_SIZES
    if world_size <= MAX_DIRECT_WORLD_SIZE
)
SUPPORTED_WORLD_SIZES = (*DIRECT_WORLD_SIZES, *HIERARCHICAL_WORLD_SIZES)


def _algorithm_for_world_size(world_size: int) -> str:
    if world_size in DIRECT_WORLD_SIZES:
        return "oneshot"
    if world_size in HIERARCHICAL_WORLD_SIZES:
        return "hierarchical"
    raise ValueError(
        f"unsupported PCIe all-reduce world size {world_size}; "
        f"supported world sizes are {SUPPORTED_WORLD_SIZES}"
    )


class PCIeAllReduce:
    """Select a peer-safe all-reduce implementation from the world size.

    Worlds through TP8 use the low-latency all-peer oneshot runtime. Serialized
    TP4 BF16 messages in the 64 KiB–2 MiB band use single-rounding two-shot
    when every rank has at least 64 visible SMs and supplied output storage
    is disjoint from the input. TP9,
    TP10, TP12 and TP16 use islands of up to four GPUs so no context maps more than
    six peers. Other worlds fail closed instead of exceeding the CUDA peer
    connection limit.
    """

    def __init__(
        self,
        runtime: Any,
        algorithm: str,
        island_rs: Any = None,
        *,
        algorithm_override: Optional[str] = None,
        bf16_twoshot: Any = None,
    ) -> None:
        self._runtime = runtime
        # Optional second implementation for the same world. When present the
        # dispatcher routes by message size instead of exposing a knob.
        self._island_rs = island_rs
        self._bf16_twoshot = bf16_twoshot
        resolved_override = (
            _algorithm_override()
            if algorithm_override is None
            else str(algorithm_override)
        )
        if resolved_override not in ("auto", "hierarchical", "island_rs"):
            raise ValueError(
                "algorithm_override must be auto, hierarchical or island_rs, "
                f"got {resolved_override!r}"
            )
        self._algorithm_override = resolved_override
        self.algorithm = algorithm
        self.rank = runtime.rank
        self.world_size = runtime.world_size
        self.device = runtime.device

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        ipc=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeAllReduce":
        world_size = dist.get_world_size(group=exchange_group)
        algorithm = _algorithm_for_world_size(world_size)
        algorithm_override = _algorithm_override()
        if algorithm == "oneshot":
            runtime = PCIeOneshotAllReducePool.from_exchange_group(
                exchange_group=exchange_group,
                device=device,
                eager_buffer_bytes=eager_buffer_bytes,
                max_size=max_size,
                rank_data_bytes=rank_data_bytes,
                ext_module=ext_module,
                ipc=ipc,
                single_channel=single_channel,
                max_concurrent_channels=max_concurrent_channels,
            )
        else:
            if int(max_concurrent_channels) != 1:
                raise ValueError(
                    "hierarchical all-reduce supports exactly one concurrent channel"
                )
            if max_size < torch.bfloat16.itemsize:
                raise ValueError("max_size must hold at least one BF16 element")
            runtime = PCIeHierarchicalAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=max_size // torch.bfloat16.itemsize,
                ext_module=ext_module,
                ipc=ipc,
            )
        island_rs = None
        bf16_twoshot = None
        try:
            island_rs = cls._maybe_island_rs(
                exchange_group=exchange_group,
                device=device,
                max_size=max_size,
                algorithm_override=algorithm_override,
                ipc=ipc,
            )
            if (
                world_size == 4
                and single_channel
                and max_size >= BF16_TWOSHOT_MIN_BYTES
                and _bf16_twoshot_supported(
                    exchange_group=exchange_group,
                    device=runtime.device,
                    world_size=world_size,
                )
            ):
                # Native rows contain one 16-byte pack; tensor dimensions need
                # not match the transport geometry. Each rank owns complete packs.
                capacity = min(int(max_size), BF16_TWOSHOT_MAX_BYTES)
                bf16_twoshot = PCIeTwoShotBF16.from_exchange_group(
                    exchange_group=exchange_group,
                    device=device,
                    max_rows=(capacity + 63) // 64 * 4,
                    row_elems=8,
                    ipc=ipc,
                )
        except BaseException:
            try:
                if island_rs is not None:
                    island_rs.close()
            finally:
                runtime.close()
            raise
        return cls(
            runtime,
            algorithm,
            island_rs,
            algorithm_override=algorithm_override,
            bf16_twoshot=bf16_twoshot,
        )

    @staticmethod
    def _maybe_island_rs(
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int,
        algorithm_override: str,
        ipc: object = None,
    ) -> Any:
        """Attach the equal-quarter runtime after an explicit policy opt-in.

        Opt-in capacity includes :data:`ISLAND_RS_MAX_BYTES` independently of
        the caller's ``max_size``. A coordinated construction failure leaves
        every rank on the hierarchical runtime and records the reason in the
        process log.
        """

        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in ISLAND_RS_WORLD_SIZES:
            return None
        # The equal-quarter runtime owns an additional CUDA IPC slab and has a
        # stricter CUDA-graph output contract than the hierarchical runtime.
        # Construct it only for an explicit opt-in so unrelated auxiliary IPC
        # collectives retain the established peer-mapping and setup behavior.
        if algorithm_override != "island_rs":
            return None
        capacity = max(int(max_size), ISLAND_RS_MAX_BYTES)
        elements = capacity // torch.bfloat16.itemsize
        try:
            return PCIeIslandRSAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=elements - (elements % 2),
                ipc=ipc,
            )
        except RuntimeError as exc:
            if dist.get_rank(group=exchange_group) == 0:
                logger.warning(
                    "PCIe island reduce-scatter is unavailable; using the "
                    "hierarchical all-reduce runtime: %s",
                    exc,
                )
            return None

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=(
                max_input_bytes if eager_buffer_bytes is None else eager_buffer_bytes
            ),
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
            max_concurrent_channels=max_concurrent_channels,
        )

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        """Whether another runtime may safely map every rank as a peer."""

        return self.algorithm == "oneshot"

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Prepare the runtime's semantic channel owners."""
        self._runtime.prepare_channels(channel_ids)
        if self._island_rs is not None:
            self._island_rs.prepare_channels(channel_ids)

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        if self._island_rs is not None:
            self._island_rs.for_stream(stream, channel_id=channel_id)
        return self._runtime.for_stream(stream, channel_id=channel_id)

    def _use_island_rs(
        self,
        inp: torch.Tensor,
    ) -> bool:
        """Route aligned large messages to the equal-quarter runtime.

        Small ones stay on the hierarchy, whose critical path is shorter for the
        ranks that are not the island leader. Unaligned quarters also stay on
        the hierarchy because the equal-quarter kernel's partial transfer group
        costs more than the leader path. Large aligned vectors would otherwise
        funnel the whole vector through the island leader's PCIe link.
        """

        if self._island_rs is None:
            return False
        override = self._algorithm_override
        if override == "hierarchical":
            return False
        island_accepts = self._island_rs.should_allreduce(inp)
        if not island_accepts:
            return False
        hierarchy_accepts = self._runtime.should_allreduce(inp)
        if not hierarchy_accepts:
            return True
        return (
            inp.numel() > ISLAND_RS_CROSSOVER_ELEMENTS
            and inp.numel() % ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS == 0
        )

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._runtime.should_allreduce(inp):
            return True
        return self._use_island_rs(inp)

    def _use_bf16_twoshot(self, inp: torch.Tensor, out: Optional[torch.Tensor]) -> bool:
        target = self._bf16_twoshot
        if (
            target is None
            or not BF16_TWOSHOT_MIN_BYTES
            <= inp.numel() * inp.element_size()
            <= min(self.max_size, BF16_TWOSHOT_MAX_BYTES)
            or not target.accepts(inp)
        ):
            return False
        if out is None:
            return True
        if not target.accepts(out) or inp.shape != out.shape:
            return False
        source_begin, output_begin = inp.data_ptr(), out.data_ptr()
        nbytes = inp.numel() * inp.element_size()
        return (
            source_begin + nbytes <= output_begin
            or output_begin + nbytes <= source_begin
        )

    def all_reduce_fused_add(
        self,
        inp,
        residual,
        *,
        plan,
        input_addend=None,
        group_weights=None,
        group_ids=None,
        group_id_limit=0,
    ):
        """Reduce into the residual without producing normalized activations."""
        if self.algorithm != "oneshot" or not self.single_channel:
            raise ValueError(
                "residual-only reduction requires a single-channel oneshot runtime"
            )
        channel = self._runtime._prepared_channel_for_stream(None, None)
        state = require_prepared(plan, "comm.pcie", inp.device)
        state.require_runtime(channel)
        channel._check_stream()
        if not state.query.surface.endswith(".all_reduce_fused_add"):
            raise ValueError("residual-only reduction requires its prepared operation")
        state.run_fused(
            inp,
            residual,
            None,
            None,
            residual,
            0.0,
            input_addend,
            group_weights,
            group_ids,
            group_id_limit,
        )
        return residual

    def plan(
        self,
        inp: torch.Tensor,
        *,
        operation: str = "all_reduce",
        stream: object = None,
        channel_id: Optional[str] = None,
        invocation: FrozenMapping = FrozenMapping(),
        override: PcieConfig | None = None,
        **call,
    ) -> Plan:
        """Select an existing native channel once, then declare its plan.

        Semantic channels must already exist. This does not construct transports,
        register inputs, allocate workspace, or compile any program.
        """
        from ._preparation import plan, query_from_runtime

        if operation not in (
            "all_reduce",
            "all_reduce_fused_add_rms_norm",
            "all_reduce_fused_add",
        ):
            raise ValueError(f"unsupported all-reduce operation {operation!r}")
        if operation == "all_reduce" and self._use_bf16_twoshot(inp, call.get("out")):
            query = query_from_runtime(
                self._bf16_twoshot,
                surface="PCIeTwoShotBF16.all_reduce",
                call={"inp": inp, **call},
            )
            return plan(
                query,
                runtime=self._bf16_twoshot,
                invocation=invocation,
                override=override,
            )
        if self.algorithm == "oneshot":
            target = self._runtime._prepared_channel_for_stream(stream, channel_id)
            surface = f"OneshotAllReduce.{operation}"
        else:
            if operation != "all_reduce":
                raise ValueError("fused RMSNorm requires an all-peer oneshot runtime")
            target = self._island_rs if self._use_island_rs(inp) else self._runtime
            surface = f"{type(target).__name__}.all_reduce"
        query = query_from_runtime(target, surface=surface, call={"inp": inp, **call})
        return plan(query, runtime=target, invocation=invocation, override=override)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        blocks: Optional[int] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        state = require_prepared(plan, "comm.pcie", inp.device)
        if self._bf16_twoshot is not None and state.runtime is self._bf16_twoshot:
            if peer_input_ptrs is not None or blocks is not None:
                raise ValueError(
                    "two-shot all-reduce uses its prepared staging and launch geometry"
                )
            if stream is not None:
                with torch.cuda.stream(stream):
                    return self._bf16_twoshot.all_reduce(inp, plan=plan, out=out)
            return self._bf16_twoshot.all_reduce(inp, plan=plan, out=out)
        if self.algorithm == "hierarchical":
            if peer_input_ptrs is not None:
                raise ValueError(
                    "peer_input_ptrs are unavailable for hierarchical all-reduce"
                )
            target = state.runtime
            if target is not self._runtime and target is not self._island_rs:
                raise ValueError("plan belongs to another all-reduce manager")
            return target.all_reduce(
                inp,
                plan=plan,
                out=out,
                blocks=blocks,
                stream=stream,
                channel_id=channel_id,
            )
        if blocks is not None:
            raise ValueError("blocks is only available for hierarchical all-reduce")
        return self._runtime.all_reduce(
            inp,
            plan=plan,
            out=out,
            peer_input_ptrs=peer_input_ptrs,
            stream=stream,
            channel_id=channel_id,
        )

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        zero_centered: bool = False,
        reduction_dtype: torch.dtype | None = None,
        input_addend: torch.Tensor | None = None,
        group_weights: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        group_id_limit: int = 0,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.algorithm != "oneshot":
            raise ValueError("fused RMSNorm requires an all-peer oneshot runtime")
        return self._runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            plan=plan,
            out=out,
            residual_out=residual_out,
            zero_centered=zero_centered,
            reduction_dtype=reduction_dtype,
            input_addend=input_addend,
            group_weights=group_weights,
            group_ids=group_ids,
            group_id_limit=group_id_limit,
            peer_input_ptrs=peer_input_ptrs,
            stream=stream,
            channel_id=channel_id,
        )

    @contextmanager
    def capture(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        with ExitStack() as stack:
            stack.enter_context(
                self._runtime.capture(stream=stream, channel_id=channel_id)
            )
            if self._island_rs is not None:
                stack.enter_context(
                    self._island_rs.capture(stream=stream, channel_id=channel_id)
                )
            if self._bf16_twoshot is not None:
                stack.enter_context(self._bf16_twoshot.capture())
            # Callers must retain message-size dispatch while recording a
            # graph; yielding the hierarchy would bypass the island runtime.
            yield self

    def close(self) -> None:
        if self._bf16_twoshot is not None:
            self._bf16_twoshot.close()
            self._bf16_twoshot = None
        if self._island_rs is not None:
            self._island_rs.close()
            self._island_rs = None
        self._runtime.close()

    def __getattr__(self, name: str):
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            raise AttributeError(name)
        return getattr(runtime, name)


__all__ = [
    "DIRECT_WORLD_SIZES",
    "MAX_DIRECT_WORLD_SIZE",
    "PCIeAllReduce",
    "SUPPORTED_WORLD_SIZES",
]
