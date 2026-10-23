"""Prepared PCIe oneshot all-reduce launchers.

The declaration stores only immutable launch metadata.  The corresponding
materialized state retains the caller's already-established native runtime,
registered buffers, and resolved CuTe launchers.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
)
import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.compile_plan import load_programs

from .pcie_oneshot import IPC_SLAB_ALIGNMENT, _align_up, _eager_storage_shards
from ._tuning import PcieConfig, PcieQuery, TUNING

_PLAIN_SURFACES = frozenset(
    {
        "OneshotAllReduce.all_reduce",
        "OneshotAllReducePool.all_reduce",
    }
)
_FUSED_SURFACES = frozenset(
    {
        "OneshotAllReduce.all_reduce_fused_add_rms_norm",
        "OneshotAllReduce.all_reduce_fused_add",
        "OneshotAllReducePool.all_reduce_fused_add_rms_norm",
    }
)


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[dtype]


def _fused_semantics(
    dtype,
    *,
    residual_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    zero_centered=False,
    reduction_dtype=None,
    input_addend=False,
    group_size=0,
    group_id_limit=0,
    group_ids_dtype="int32",
):
    if (
        type(group_size) is not int
        or group_size < 0
        or type(group_id_limit) is not int
        or group_id_limit < 0
        or (group_size > 0) != (group_id_limit > 0)
        or group_ids_dtype not in ("int32", "int64")
    ):
        raise ValueError(
            "weighted group geometry requires a positive size/ID bound and int32/int64 IDs"
        )
    supported = (torch.float16, torch.bfloat16, torch.float32)
    operands = {
        "residual_dtype": residual_dtype or dtype,
        "weight_dtype": weight_dtype or dtype,
        "output_dtype": output_dtype or dtype,
    }
    if any(
        value not in supported for value in operands.values()
    ) or reduction_dtype not in (None, *supported):
        raise ValueError(
            "fused RMSNorm operands and reduction rounding require FP16, BF16, or FP32"
        )
    if type(zero_centered) is not bool:
        raise TypeError("zero_centered must be a boolean")
    if type(input_addend) is not bool:
        raise TypeError("input_addend must be a boolean")
    return {
        "input_addend": input_addend,
        "group_size": int(group_size),
        "group_id_limit": int(group_id_limit),
        "group_ids_dtype": group_ids_dtype,
        **{key: _dtype_name(value) for key, value in operands.items()},
        "zero_centered": zero_centered,
        "reduction_dtype": (
            None if reduction_dtype is None else _dtype_name(reduction_dtype)
        ),
    }


def _check_input_addend(inp, input_addend):
    if input_addend is None:
        return False
    if not isinstance(input_addend, torch.Tensor) or (
        input_addend.shape != inp.shape
        or input_addend.dtype != inp.dtype
        or input_addend.device != inp.device
        or not input_addend.is_contiguous()
        or input_addend.data_ptr() % 16
    ):
        raise ValueError(
            "input addend must match source shape, dtype, device and packed contiguous layout"
        )
    return True


def _input_from_call(call: Mapping[str, object]) -> torch.Tensor:
    inp = call.get("inp")
    if not isinstance(inp, torch.Tensor):
        raise TypeError("oneshot preparation call requires its actual inp tensor")
    return inp


def _state(runtime):
    try:
        return runtime._ext._state(runtime._ptr)
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "oneshot preparation requires a live native oneshot runtime"
        ) from exc


@dataclass(frozen=True)
class _InvocationTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    _stride: tuple[int, ...]
    device: torch.device
    alignment: int = 16

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def element_size(self):
        return self.dtype.itemsize

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._stride == _contiguous_strides(self.shape)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _logical_input(inp, group_size=0):
    if not group_size:
        return inp
    if (
        inp.ndim != 3
        or inp.shape[1] != group_size
        or min(inp.shape) <= 0
        or not inp.is_contiguous()
    ):
        raise ValueError(
            "weighted source requires nonempty [rows, groups, hidden] storage"
        )
    shape = (int(inp.shape[0]), int(inp.shape[2]))
    return _InvocationTensor(shape, inp.dtype, _contiguous_strides(shape), inp.device)


def _grouped_inputs(inp, group_weights=None, group_ids=None, group_id_limit=0):
    if group_weights is None and group_ids is None and group_id_limit == 0:
        return inp, 0, 0, "int32"
    if inp.ndim != 3 or not inp.is_contiguous() or inp.data_ptr() % 16:
        raise ValueError(
            "weighted source requires aligned contiguous [rows, groups, hidden] storage"
        )
    if type(group_id_limit) is not int or group_id_limit <= 0:
        raise ValueError("weighted group ID bound must be positive")
    for name, tensor, dtypes in (
        ("group weights", group_weights, (torch.float32,)),
        ("group IDs", group_ids, (torch.int32, torch.int64)),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != inp.shape[:2]
            or tensor.dtype not in dtypes
            or tensor.device != inp.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % tensor.element_size()
        ):
            raise ValueError(
                f"{name} must match the source row/group geometry and device"
            )
    size = int(inp.shape[1])
    return (
        _logical_input(inp, size),
        size,
        group_id_limit,
        ("int64" if group_ids.dtype == torch.int64 else "int32"),
    )


def _owned_resident_layout(runtime, native) -> tuple[int, int, int]:
    """Return the native runtime's actual locally-owned allocation layout.

    The extension owns the signal metadata size; the Python runtime owns the
    rank-data tensor and retains each locally exported IPC slab in
    ``_owned_buffers``.  This intentionally does not infer ownership from
    imported peer pointers or a pool-only convenience attribute.
    """
    rank_data = runtime.rank_data
    rank_data_nbytes = int(rank_data.numel() * rank_data.element_size())
    eager_bytes = native.eager_buffer_bytes
    if eager_bytes is None:
        return rank_data_nbytes, 0, 0
    shards = _eager_storage_shards(
        int(runtime.world_size), tuple(bool(value) for value in native.transport_policy)
    )
    signal_nbytes = _align_up(int(runtime._ext.meta_size()), IPC_SLAB_ALIGNMENT)
    eager_nbytes = _align_up(int(eager_bytes) * shards, IPC_SLAB_ALIGNMENT)
    slab_nbytes = signal_nbytes + 2 * eager_nbytes
    return rank_data_nbytes, slab_nbytes, len(runtime._owned_buffers)


def _plain_launcher_metadata(backend, native, inp):
    graph = backend._plain_graph_plan(native, inp)
    eager_transport, eager_threads, eager_blocks = backend._plain_launch_config(
        native,
        inp,
        graph_channel=False,
    )
    return {
        "dtype": _dtype_name(inp.dtype),
        "transport": graph.transport,
        "threads": graph.threads,
        "blocks": graph.blocks,
        "eager_transport": eager_transport,
        "eager_threads": eager_threads,
        "eager_blocks": eager_blocks,
        "size_packs": graph.size_packs,
        "stage_input": graph.stage_input,
        "device_index": graph.device_index,
        "remote_push_region_packs": graph.remote_push_region_packs,
    }


def query_from_metadata(
    runtime,
    *,
    surface: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    strides: tuple[int, ...] | None = None,
    alignment: int = 16,
    residual_dtype: torch.dtype | None = None,
    weight_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    zero_centered: bool = False,
    reduction_dtype: torch.dtype | None = None,
    input_addend: bool = False,
    group_size: int = 0,
    group_id_limit: int = 0,
    group_ids_dtype: str = "int32",
) -> PcieQuery:
    """Extract a native declaration from immutable invocation metadata only."""
    tensor = _InvocationTensor(
        tuple(shape),
        dtype,
        _contiguous_strides(tuple(shape)) if strides is None else tuple(strides),
        runtime.device,
        alignment,
    )
    physical = tensor
    tensor = _logical_input(physical, group_size)
    if group_size and surface not in _FUSED_SURFACES:
        raise ValueError("weighted groups require fused collective staging")
    if surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {surface!r}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("oneshot invocation dtype is unsupported")
    if (
        tensor.numel() * tensor.element_size() > runtime.max_size
        or (tensor.numel() * tensor.element_size()) % 16
    ):
        raise ValueError("oneshot invocation exceeds its established runtime capacity")
    native = _state(runtime)
    backend = runtime._ext
    stage_input = native.eager_tables is not None
    variants = ((False, 0), (True, 0), (True, 1)) if stage_input else ((False, 0),)
    if surface in _PLAIN_SURFACES:
        launcher = _plain_launcher_metadata(backend, native, tensor)
        registered = not launcher["stage_input"]
    else:
        if (input_addend or group_size) and not stage_input:
            raise ValueError("input addend requires staged collective storage")
        if (
            tensor.ndim == 0
            or tensor.numel() == 0
            or tensor.shape[-1] * tensor.element_size() % 16
            or tensor.stride() != _contiguous_strides(tensor.shape)
        ):
            raise ValueError(
                "fused oneshot input must be nonempty and contiguous with 16-byte row packs"
            )
        mode, single_cta, register_normalize, threads = backend._fused_launch_config(
            native, tensor, reduction_dtype=reduction_dtype
        )
        pack_elems = 16 // tensor.element_size()
        hidden_packs = int(tensor.shape[-1]) // pack_elems
        rows = tensor.numel() // int(tensor.shape[-1])
        min_ctas = (hidden_packs + threads * 3 - 1) // (threads * 3)
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        ctas = override if override > 0 else max(max(1, 3 // rows), min_ctas)
        ctas = max(1, min(ctas, 36 // rows))
        launcher = {
            **_fused_semantics(
                dtype,
                residual_dtype=residual_dtype,
                weight_dtype=weight_dtype,
                output_dtype=output_dtype,
                zero_centered=zero_centered,
                reduction_dtype=reduction_dtype,
                input_addend=input_addend,
                group_size=group_size,
                group_id_limit=group_id_limit,
                group_ids_dtype=group_ids_dtype,
            ),
            "normalize": not surface.endswith(".all_reduce_fused_add"),
            "dtype": _dtype_name(dtype),
            "mode": mode,
            "single_cta": single_cta,
            "register_normalize": register_normalize,
            "threads": threads,
            "device_index": backend._device_index(runtime.device),
            "hidden_packs": hidden_packs,
            "rows": rows,
            "ctas_per_row": ctas,
            "blocks": min(rows * ctas, 36),
            "stage_input": stage_input,
        }
        registered = not stage_input
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    setup = {
        "shape": physical.shape,
        "stride": physical.stride(),
        "logical_shape": tensor.shape,
        "dtype": _dtype_name(dtype),
        "device_index": backend._device_index(runtime.device),
        "elements": tensor.numel(),
        "element_size": tensor.element_size(),
        "rows": tensor.numel() // tensor.shape[-1],
        "rank_data_bytes": int(runtime.rank_data_bytes),
        "rank_data_nbytes": rank_data_nbytes,
        "owned_slab_nbytes": slab_nbytes,
        "owned_slab_count": owned_slab_count,
        "eager_buffer_bytes": None
        if native.eager_buffer_bytes is None
        else int(native.eager_buffer_bytes),
        "eager_storage_shards": _eager_storage_shards(
            int(runtime.world_size),
            tuple(bool(value) for value in native.transport_policy),
        ),
        "registered": registered,
        "variants": variants,
    }
    return PcieQuery(
        surface=surface,
        world_size=int(runtime.world_size),
        rank=int(runtime.rank),
        topology="pcie_ipc",
        call=FrozenMapping(launcher),
        setup=FrozenMapping(setup),
    )


def query_from_runtime(runtime, *, surface, call) -> PcieQuery:
    """Extract the exact immutable launcher contract from a live channel."""
    if surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {surface!r}")
    values = dict(call)
    inp = _input_from_call(values)
    physical = inp
    inp, group_size, group_id_limit, group_ids_dtype = _grouped_inputs(
        physical,
        values.get("group_weights"),
        values.get("group_ids"),
        values.get("group_id_limit", 0),
    )
    if group_size and surface not in _FUSED_SURFACES:
        raise ValueError("weighted groups require fused collective staging")
    native = _state(runtime)
    if inp.device != runtime.device:
        raise ValueError("oneshot input device differs from its native runtime")
    if not runtime.should_allreduce(inp):
        raise ValueError("input does not satisfy PCIe oneshot requirements")
    if surface in _FUSED_SURFACES and (
        inp.ndim == 0
        or inp.numel() == 0
        or not inp.is_contiguous()
        or inp.shape[-1] * inp.element_size() % 16
    ):
        raise ValueError(
            "fused oneshot input must be nonempty and contiguous with 16-byte row packs"
        )
    backend = runtime._ext
    dtype = _dtype_name(inp.dtype)
    stage_input = native.eager_tables is not None
    variants = ((False, 0), (True, 0), (True, 1)) if stage_input else ((False, 0),)
    if surface in _PLAIN_SURFACES:
        launcher = _plain_launcher_metadata(backend, native, inp)
    else:
        input_addend = _check_input_addend(inp, values.get("input_addend"))
        if (input_addend or group_size) and not stage_input:
            raise ValueError("input addend requires staged collective storage")
        mode, single_cta, register_normalize, threads = backend._fused_launch_config(
            native, inp, reduction_dtype=values.get("reduction_dtype")
        )
        pack_elems = 16 // inp.element_size()
        hidden_packs = int(inp.shape[-1]) // pack_elems
        rows = inp.numel() // int(inp.shape[-1])
        min_ctas = (hidden_packs + threads * 3 - 1) // (threads * 3)
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        ctas = override if override > 0 else max(max(1, 3 // rows), min_ctas)
        ctas = max(1, min(ctas, 36 // rows))
        launcher = {
            **_fused_semantics(
                inp.dtype,
                residual_dtype=getattr(values.get("residual"), "dtype", inp.dtype),
                weight_dtype=getattr(values.get("weight"), "dtype", inp.dtype),
                output_dtype=getattr(values.get("out"), "dtype", inp.dtype),
                zero_centered=values.get("zero_centered", False),
                reduction_dtype=values.get("reduction_dtype"),
                input_addend=input_addend,
                group_size=group_size,
                group_id_limit=group_id_limit,
                group_ids_dtype=group_ids_dtype,
            ),
            "normalize": not surface.endswith(".all_reduce_fused_add"),
            "dtype": dtype,
            "mode": mode,
            "single_cta": single_cta,
            "register_normalize": register_normalize,
            "threads": threads,
            "device_index": backend._device_index(inp.device),
            "hidden_packs": hidden_packs,
            "rows": rows,
            "ctas_per_row": ctas,
            "blocks": min(rows * ctas, 36),
            "stage_input": stage_input,
        }
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    setup = {
        "shape": tuple(physical.shape),
        "stride": tuple(physical.stride()),
        "logical_shape": tuple(inp.shape),
        "dtype": dtype,
        "device_index": backend._device_index(inp.device),
        "elements": int(inp.numel()),
        "element_size": int(inp.element_size()),
        "rows": 1 if inp.ndim == 0 else int(inp.numel() // inp.shape[-1]),
        "rank_data_bytes": int(runtime.rank_data_bytes),
        "rank_data_nbytes": rank_data_nbytes,
        "owned_slab_nbytes": slab_nbytes,
        "owned_slab_count": owned_slab_count,
        "eager_buffer_bytes": None
        if native.eager_buffer_bytes is None
        else int(native.eager_buffer_bytes),
        "eager_storage_shards": _eager_storage_shards(
            int(runtime.world_size),
            tuple(bool(value) for value in native.transport_policy),
        ),
        "registered": not stage_input,
        "variants": variants,
    }
    return PcieQuery(
        surface=surface,
        world_size=int(runtime.world_size),
        rank=int(runtime.rank),
        topology="pcie_ipc",
        call=FrozenMapping(launcher),
        setup=FrozenMapping(setup),
    )


@program_cache(scope="preparation")
def compile_oneshot_surface(query_payload, ordinal):
    """Compile all fixed slot variants represented by metadata only."""
    payload = dict(query_payload)
    query = PcieQuery(
        surface=payload["surface"],
        world_size=payload["world_size"],
        rank=payload["rank"],
        topology=payload["topology"],
        call=FrozenMapping(payload["call"]),
        setup=FrozenMapping(payload["setup"]),
    )
    call, setup = dict(query.call), dict(query.setup)
    variants = tuple(tuple(value) for value in setup["variants"])
    with torch.cuda.device(int(ordinal)):
        if query.surface in _PLAIN_SURFACES:
            from ._oneshot_cute import get_oneshot_launcher

            return {
                variant: get_oneshot_launcher(
                    call["dtype"],
                    query.world_size,
                    query.rank,
                    call["stage_input"],
                    variant[0],
                    variant[1],
                    call["transport"] if variant[0] else call["eager_transport"],
                    call["threads"] if variant[0] else call["eager_threads"],
                    call["device_index"],
                )
                for variant in variants
            }
        if query.surface in _FUSED_SURFACES:
            from ._oneshot_cute import get_fused_oneshot_launcher

            return {
                variant: get_fused_oneshot_launcher(
                    call["dtype"],
                    query.world_size,
                    query.rank,
                    call["mode"],
                    call["single_cta"],
                    call["register_normalize"],
                    variant[0],
                    variant[1],
                    call["threads"],
                    call["device_index"],
                    residual_dtype=call["residual_dtype"],
                    weight_dtype=call["weight_dtype"],
                    output_dtype=call["output_dtype"],
                    zero_centered=call["zero_centered"],
                    reduction_dtype=call["reduction_dtype"],
                    input_addend=call["input_addend"],
                    group_size=call["group_size"],
                    group_id_limit=call["group_id_limit"],
                    group_ids_dtype=call["group_ids_dtype"],
                    normalize=call["normalize"],
                )
                for variant in variants
            }
    raise ValueError(f"unsupported oneshot preparation surface {query.surface!r}")


@dataclass
class _OneshotExecutionState:
    query: PcieQuery
    runtime: object
    launchers: Mapping[tuple[bool, int], object]
    _bound_input: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected = tuple(tuple(variant) for variant in self.query.setup["variants"])
        missing = tuple(
            variant for variant in expected if self.launchers.get(variant) is None
        )
        if missing:
            raise RuntimeError(
                f"oneshot preparation did not retain required slot variants: {missing}"
            )

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("PCIe plan belongs to a different native runtime owner")
        if (runtime.rank, runtime.world_size) != (
            self.query.rank,
            self.query.world_size,
        ):
            raise ValueError("PCIe runtime identity differs from preparation")

    def _check_input(self, inp):
        setup = self.query.setup
        if (
            tuple(inp.shape),
            tuple(inp.stride()),
            _dtype_name(inp.dtype),
            inp.device.index,
        ) != (
            tuple(setup["shape"]),
            tuple(setup["stride"]),
            setup["dtype"],
            setup["device_index"],
        ):
            raise ValueError("oneshot input metadata differs from the prepared plan")

    def bind_input(self, inp: torch.Tensor) -> None:
        self._check_input(inp)
        self.runtime._prepare_prepared_surface(self.query, inp)
        # Staged runtimes copy each serving input into their fixed IPC slabs;
        # retaining the disposable priming tensor would make it look like
        # serving depends on that tensor.  Direct registered runtimes instead
        # have a real producer-owned pointer contract and keep that binding.
        self._bound_input = inp if self.runtime._eager_ptrs is None else None

    def _require_bound_input(self, inp):
        self._check_input(inp)
        if self._bound_input is None:
            if self.runtime._eager_ptrs is not None:
                return
            raise RuntimeError("oneshot plan was not bound during preparation")

    def run_plain(self, inp, out):
        self._require_bound_input(inp)
        return self.runtime._run_prepared_plain(inp, out, self.query, self.launchers)

    def run_fused(
        self,
        inp,
        residual,
        weight,
        out,
        residual_out,
        epsilon,
        input_addend=None,
        group_weights=None,
        group_ids=None,
        group_id_limit=0,
    ):
        self._require_bound_input(inp)
        if self.query.surface not in _FUSED_SURFACES:
            raise ValueError("fused RMSNorm requires a fused preparation plan")
        logical, group_size, limit, ids_dtype = _grouped_inputs(
            inp, group_weights, group_ids, group_id_limit
        )
        if (group_size, limit, ids_dtype) != (
            self.query.call["group_size"],
            self.query.call["group_id_limit"],
            self.query.call["group_ids_dtype"],
        ):
            raise ValueError("weighted groups differ from the prepared plan")
        if (
            _check_input_addend(logical, input_addend)
            != self.query.call["input_addend"]
        ):
            raise ValueError("input addend differs from the prepared plan")
        operands = (
            ("residual", residual, logical.shape, self.query.call["residual_dtype"]),
            ("weight", weight, logical.shape[-1:], self.query.call["weight_dtype"]),
            ("output", out, logical.shape, self.query.call["output_dtype"]),
            (
                "residual output",
                residual_out,
                logical.shape,
                self.query.call["residual_dtype"],
            ),
        )
        if not self.query.call["normalize"]:
            if weight is not None or out is not None:
                raise ValueError(
                    "residual-only collective has no affine weight or normalized output"
                )
            operands = (operands[0], operands[-1])
            begin = residual_out.data_ptr()
            end = begin + residual_out.numel() * residual_out.element_size()
            for tensor in (inp, input_addend, group_weights, group_ids):
                if tensor is not None:
                    source_begin = tensor.data_ptr()
                    source_end = source_begin + tensor.numel() * tensor.element_size()
                    if begin < source_end and source_begin < end:
                        raise ValueError(
                            "residual output must not overlap collective contributions"
                        )
            residual_begin = residual.data_ptr()
            residual_end = residual_begin + residual.numel() * residual.element_size()
            if (
                begin != residual_begin
                and begin < residual_end
                and residual_begin < end
            ):
                raise ValueError(
                    "residual output may only alias the residual input exactly"
                )
        for name, tensor, shape, dtype in operands:
            if (
                tensor.shape != shape
                or tensor.dtype != getattr(torch, dtype)
                or tensor.device != inp.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"fused RMSNorm {name} metadata differs from the prepared plan"
                )
        if out is not None and out.data_ptr() == residual_out.data_ptr():
            raise ValueError("output and residual output must not alias")
        if not epsilon >= 0:
            raise ValueError("epsilon must be non-negative")
        return self.runtime._run_prepared_fused(
            inp,
            residual,
            weight,
            out,
            residual_out,
            epsilon,
            self.query,
            self.launchers,
            input_addend,
            group_weights,
            group_ids,
        )


def _prepare_plain_call(
    state: _OneshotExecutionState,
    *,
    inp: torch.Tensor,
    out: torch.Tensor,
    produce=None,
    reset=None,
    restore=None,
    owners=(),
) -> PreparedCall:
    """Bind and prime a caller-owned plain all-reduce without a public warmup API."""
    state.bind_input(inp)
    return PreparedCall(
        run=lambda: state.run_plain(inp, out),
        output=out,
        produce=produce,
        reset=reset,
        restore=restore,
        owners=tuple(owners),
    )


def _prepare_fused_call(
    state: _OneshotExecutionState,
    *,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    out: torch.Tensor | None,
    residual_out: torch.Tensor,
    epsilon: float,
    input_addend: torch.Tensor | None = None,
    group_weights: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    group_id_limit: int = 0,
    produce=None,
    reset=None,
    restore=None,
    owners=(),
) -> PreparedCall:
    """Bind and prime the complete fused arithmetic with its actual resources."""
    state.bind_input(inp)
    return PreparedCall(
        run=lambda: state.run_fused(
            inp,
            residual,
            weight,
            out,
            residual_out,
            epsilon,
            input_addend,
            group_weights,
            group_ids,
            group_id_limit,
        ),
        output=residual_out if out is None else (out, residual_out),
        produce=produce,
        reset=reset,
        restore=restore,
        owners=tuple(owners),
    )


def plan(
    query: PcieQuery,
    *,
    runtime,
    invocation=FrozenMapping(),
    override: PcieConfig | None = None,
) -> Plan:
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    if query.surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {query.surface!r}")
    if runtime is None:
        raise TypeError(
            "oneshot declarations require the existing native runtime owner"
        )
    if invocation:
        raise ValueError("oneshot invocation semantics belong in PcieQuery")

    def materialize(selection, device):
        launchers = load_programs(
            compile_oneshot_surface(TUNING.encode_query(query), device.ordinal)
        )
        return _OneshotExecutionState(query, runtime, MappingProxyType(launchers))

    native = _state(runtime)
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    resident = rank_data_nbytes + slab_nbytes * owned_slab_count

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._oneshot_preparation:compile_oneshot_surface",
                TUNING.encode_query(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(
                    ("pcie-oneshot-runtime", id(runtime)),
                    resident,
                    resident,
                ),
            ),
        ),
        _materialize=materialize,
        _device=runtime.device,
    )
