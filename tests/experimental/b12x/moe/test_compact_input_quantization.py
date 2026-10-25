from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.w4a8_compact_projection import (
    _stage_compact_input_k128,
)
from b12x.testing.reference.helpers import require_b12x


class _CopyInputStage:
    def __init__(self, rows: int, k: int):
        self.rows = rows
        self.k = k

    @cute.jit
    def __call__(
        self, source_ptr: cute.Pointer, output_ptr: cute.Pointer, stream: cuda.CUstream
    ):
        source = cute.make_tensor(source_ptr, cute.make_layout((self.rows * self.k,)))
        output = cute.make_tensor(
            output_ptr, cute.make_layout((self.rows * (self.k // 128) * 64,))
        )
        self.kernel(source, output).launch(
            grid=(self.rows * (self.k // 128), 1, 1),
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        tile = Int32(bidx)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 64], 128]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        if tid < Int32(64):
            st_shared_u32(base + tid * Int32(4), Uint32(0x5A5A5A5A))
        cute.arch.sync_threads()
        _stage_compact_input_k128(
            source,
            base,
            base + Int32(128),
            tid,
            tile // Int32(self.k // 128),
            tile % Int32(self.k // 128),
            self.k,
        )
        cute.arch.sync_threads()
        if tid < Int32(64):
            output[Int64(tile) * Int64(64) + Int64(tid)] = ld_shared_u32(
                base + tid * Int32(4)
            )


@pytest.mark.parametrize("capacity", (4, 8))
@pytest.mark.parametrize("source_offset", (0, 1))
def test_compact_shared_input_matches_planned_quantization(capacity, source_offset):
    device = require_b12x()
    k = 65536 // capacity
    backing = torch.empty(65536 + source_offset, dtype=torch.bfloat16, device=device)
    source = backing[source_offset:].view(capacity, k)
    aligned_source = torch.empty_like(source)
    actual = torch.empty((capacity, k // 128, 256), dtype=torch.uint8, device=device)
    expected = empty_mxfp8_rows_for_dense_gemm(capacity, k, device=device)
    finite = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    finite = finite.view(torch.bfloat16)
    finite.masked_fill_(~torch.isfinite(finite), 0)
    patterns = (
        finite.view(capacity, k),
        finite.view(256, 256).T.contiguous().view(capacity, k),
        torch.zeros_like(source),
        torch.full_like(source, -0.0),
        (
            torch.linspace(-1.0, 1.0, 65536, dtype=torch.float32, device=device)
            * (2.0**-120)
        )
        .to(torch.bfloat16)
        .view(capacity, k),
    )

    def pointer(dtype, tensor, alignment):
        return make_ptr(
            dtype,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=alignment,
        )

    program = b12x_compile(
        _CopyInputStage(capacity, k),
        make_ptr(cutlass.BFloat16, 2, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "test.moe.compact_input_stage", 1, (capacity, k)
        ),
    )

    def run():
        program(
            pointer(cutlass.BFloat16, source, 2),
            pointer(cutlass.Uint32, actual, 16),
            current_cuda_stream(),
        )

    def reference():
        quantize_mxfp8_rows_cute(
            aligned_source,
            expected.values,
            expected.scale_rows,
            expected.scale_mma,
            expected_m=capacity,
        )

    source.copy_(patterns[0])
    aligned_source.copy_(source)
    run()
    reference()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    tensors = (
        source,
        aligned_source,
        actual,
        expected.values,
        expected.scale_rows,
        expected.scale_mma,
    )
    addresses = tuple(tensor.data_ptr() for tensor in tensors)
    with kernel_resolution_guard("shared compact input quantization replay"):
        for pattern in patterns:
            source.copy_(pattern)
            aligned_source.copy_(source)
            actual.fill_(0xA5)
            reference()
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert tuple(tensor.data_ptr() for tensor in tensors) == addresses
            torch.testing.assert_close(
                actual[:, :, :128].reshape(capacity, k),
                expected.values.view(torch.uint8).reshape(capacity, k),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual[:, :, 128:132].reshape(capacity, k // 32),
                expected.scale_rows.view(torch.uint8).reshape(capacity, k // 32),
                rtol=0,
                atol=0,
            )
            assert torch.all(actual[:, :, 132:] == 0x5A)
    graph.reset()
