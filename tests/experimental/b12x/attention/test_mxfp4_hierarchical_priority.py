"""Mandatory newest-block priority remains strict among positive infinities."""

from functools import lru_cache

import cutlass.cute as cute
import pytest
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.attention.dsa_indexer.hierarchical_fused import (
    HierarchicalFusedMXFP4,
    hierarchical_scratch_shapes,
)
from b12x.attention.dsa_indexer.mxfp4 import _compile, _launch, _ptr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_LOCAL_TYPES = (
    Uint8,
    Uint8,
    BFloat16,
    Uint8,
    Int32,
    Int32,
    Int32,
    Int32,
    Float32,
    Int32,
    Float32,
    Int32,
    Float32,
    Int32,
    Int32,
)
_MERGE_TYPES = (Int32, Int32, Int32, Float32, Int32, Float32, Int32, Float32, Int32)


@lru_cache(None)
def _programs(ctas):
    obj = HierarchicalFusedMXFP4(32, 128, ctas)
    ptrs = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in _LOCAL_TYPES
    )
    local = b12x_compile(
        obj.local,
        *ptrs,
        Int32(1),
        Int32(1),
        Int32(1),
        Int64(1),
        Int64(1),
        Int64(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "attention.indexer.hierarchical_priority_test", 1, ("local", ctas)
        ),
    )
    merge = None
    if ctas > 1:
        ptrs = tuple(
            make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1)
            for t in _MERGE_TYPES
        )
        merge = b12x_compile(
            obj.merge,
            *ptrs,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.indexer.hierarchical_priority_test", 1, ("merge", ctas)
            ),
        )
    return local, merge, _compile("sort", (2048, True), torch.cuda.current_device())


@pytest.mark.parametrize("ctas", [1, 4])
@pytest.mark.parametrize("visible", [32768, 131072])
@pytest.mark.parametrize("unscored", [0, 7, 8])
def test_newest_block_is_reserved_among_infinite_scores(ctas, visible, unscored):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    width = visible - unscored
    npages = visible // 128
    q = torch.full((1, 32, 64), 0x22, dtype=torch.uint8, device="cuda")
    qs = torch.full((1, 32, 4), 127, dtype=torch.uint8, device="cuda")
    weights = torch.full((1, 32), float("inf"), dtype=torch.bfloat16, device="cuda")
    pool = torch.full((npages, 128 * 68), 0x22, dtype=torch.uint8, device="cuda")
    pool[:, 128 * 64 :] = 127
    pages = torch.arange(npages, dtype=torch.int32, device="cuda")[None, :]
    lengths = torch.tensor([visible], dtype=torch.int32, device="cuda")
    active = lengths.clone()
    output = torch.empty((1, 512), dtype=torch.int32, device="cuda")
    values = torch.empty((1, 512), device="cuda")
    scratch = {
        name: torch.empty(shape, dtype=dtype, device="cuda")
        for name, (dtype, shape) in hierarchical_scratch_shapes(1, ctas).items()
    }
    for tensor in scratch.values():
        tensor.fill_(17)
    local, merge, expand = _programs(ctas)
    tensors = (
        q,
        qs,
        weights,
        pool,
        pages,
        lengths,
        active,
        output,
        values,
        scratch["block_indices"],
        scratch["block_values"],
        scratch["partial_indices"],
        scratch["partial_values"],
        scratch["partial_counts"],
        scratch["state"],
    )
    local(
        *(_ptr(t, d) for t, d in zip(tensors, _LOCAL_TYPES, strict=True)),
        Int32(1),
        Int32(width),
        Int32(npages),
        Int64(npages),
        Int64(128 * 68),
        Int64(npages),
        current_cuda_stream(),
    )
    if merge is not None:
        tensors = (
            lengths,
            active,
            output,
            values,
            scratch["block_indices"],
            scratch["block_values"],
            scratch["partial_indices"],
            scratch["partial_values"],
            scratch["partial_counts"],
        )
        merge(
            *(_ptr(t, d) for t, d in zip(tensors, _MERGE_TYPES, strict=True)),
            Int32(1),
            current_cuda_stream(),
        )
    candidates = torch.empty((1, 16384), dtype=torch.int32, device="cuda")
    counts = torch.empty((1,), dtype=torch.int32, device="cuda")
    _launch(
        "sort",
        (2048, True),
        (
            scratch["block_indices"],
            scratch["block_values"],
            candidates,
            values,
            lengths,
            active,
            counts,
        ),
        (1,),
        launcher=expand,
    )
    torch.cuda.synchronize()
    assert bool(torch.isposinf(values).all())
    assert bool((output[:, 1:] > output[:, :-1]).all())
    assert bool(((output >= 0) & (output < width)).all())
    assert counts.item() == 16384
    assert bool((candidates[:, 1:] > candidates[:, :-1]).all())
    blocks = candidates[0, ::8] // 8
    newest = (visible - 1) // 8
    assert bool((blocks == newest).any()) == (newest < (width + 7) // 8)
    assert not bool(scratch["state"].any())
