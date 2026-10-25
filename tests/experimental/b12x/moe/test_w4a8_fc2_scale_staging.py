"""Native FC2 payload and scale transport preserves exact bytes."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import pytest
import torch

from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.utils.device import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel
from b12x.testing.reference.helpers import require_b12x


class StageNativeFC2Scales:
    """Publish native words through the production FC2 scale staging helper."""

    def __init__(self, intermediate_size, source_tile_m):
        self.total_k32 = intermediate_size // 32
        self.k128_tiles = (self.total_k32 + 3) // 4
        self.experts = 3
        self.output_tiles = 4
        direct = source_tile_m == 1
        self.reader = W4A8MaterializedPhase2Kernel(
            source_tile_m=source_tile_m,
            deterministic_output=True,
            direct_routes=direct,
            fused_topk=1 if direct else 0,
            n64_repacked=True,
            n64_tail=intermediate_size % 128 == 64,
        )

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream):
        self.kernel(source, output).launch(
            grid=(self.k128_tiles, self.output_tiles, self.experts),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        tid = Int32(cute.arch.thread_idx()[0])
        k128, output_tile, expert = cute.arch.block_idx()
        staging = SmemAllocator().allocate_tensor(Uint32, cute.make_layout(128), 16)
        staging[tid] = Uint32(0xDEADBEEF)
        cute.arch.sync_threads()
        self.reader._stage_n64_sfb(
            source,
            shared_ptr_to_u32(staging.iterator),
            Int32(expert),
            Int32(output_tile),
            Int32(k128),
            Int32(self.output_tiles),
            self.total_k32,
            tid,
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        output[expert, output_tile, k128, tid] = Int32(staging[tid])


def _expected_native_scale_bytes(source, total_k32, experts, output_tiles):
    slices = (total_k32 + 3) // 4
    packed = source.view(experts, output_tiles, 128 * total_k32)
    expected = torch.zeros(
        experts,
        output_tiles,
        slices,
        128,
        4,
        dtype=torch.uint8,
        device=source.device,
    )
    for k128 in range(slices):
        columns = min(4, total_k32 - k128 * 4)
        rows = packed[:, :, k128 * 512 : k128 * 512 + 128 * columns]
        expected[:, :, k128, :, :columns] = rows.reshape(
            experts, output_tiles, 128, columns
        )
    return expected


@pytest.mark.parametrize("intermediate_size", (64, 192, 320, 576))
@pytest.mark.parametrize("source_tile_m", (1, 16, 64))
@pytest.mark.parametrize("base_offset_bytes", (0, 4))
def test_native_fc2_scale_staging_is_byteexact_and_live(
    intermediate_size, source_tile_m, base_offset_bytes
):
    require_b12x()
    stager = StageNativeFC2Scales(intermediate_size, source_tile_m)
    size = stager.experts * stager.output_tiles * 128 * stager.total_k32
    generator = torch.Generator(device="cuda").manual_seed(197 + intermediate_size)
    storage = torch.randint(
        0,
        256,
        (size + base_offset_bytes,),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    source = storage.narrow(0, base_offset_bytes, size)
    output = torch.empty(
        stager.experts,
        stager.output_tiles,
        stager.k128_tiles,
        128,
        device="cuda",
        dtype=torch.int32,
    )
    tensors = (
        from_dlpack(source.view(torch.uint32), assumed_align=4),
        from_dlpack(output, assumed_align=4),
    )
    program = compile_kernel(stager, *tensors, current_cuda_stream())
    program(*tensors, current_cuda_stream())
    expected = _expected_native_scale_bytes(
        source, stager.total_k32, stager.experts, stager.output_tiles
    )
    assert torch.equal(output.view(torch.uint8).reshape_as(expected), expected)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            program(*tensors, current_cuda_stream())
        for _ in range(3):
            source.bitwise_xor_(0xA5)
            expected = _expected_native_scale_bytes(
                source, stager.total_k32, stager.experts, stager.output_tiles
            )
            output.fill_(0x5A5A5A5A)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output.view(torch.uint8).reshape_as(expected), expected)
    finally:
        graph.reset()


class StageNativeFC2Weights:
    """Expose every staged weight byte, including inactive K64 padding."""

    def __init__(self, intermediate_size, source_tile_m):
        self.total_k32 = intermediate_size // 32
        self.k128_tiles = (self.total_k32 + 3) // 4
        self.experts = 3
        self.output_tiles = 4
        direct = source_tile_m == 1
        self.reader = W4A8MaterializedPhase2Kernel(
            source_tile_m=source_tile_m,
            deterministic_output=True,
            direct_routes=direct,
            fused_topk=1 if direct else 0,
            n64_repacked=True,
            n64_tail=intermediate_size % 128 == 64,
        )

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream):
        self.kernel(source, output).launch(
            grid=(self.k128_tiles, self.output_tiles, self.experts),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        tid = Int32(cute.arch.thread_idx()[0])
        k128, output_tile, expert = cute.arch.block_idx()
        staging = SmemAllocator().allocate_tensor(Uint32, cute.make_layout(2048), 16)
        for offset in cutlass.range_constexpr(16):
            staging[tid + Int32(offset * 128)] = Uint32(0xDEADBEEF)
        cute.arch.sync_threads()
        self.reader._stage_n64_b(
            source,
            shared_ptr_to_u32(staging.iterator),
            Int32(expert),
            Int32(output_tile),
            Int32(k128),
            Int32(self.output_tiles),
            self.total_k32,
            tid,
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        for offset in cutlass.range_constexpr(16):
            word = tid + Int32(offset * 128)
            output[expert, output_tile, k128, word] = Int32(staging[word])


@pytest.mark.parametrize("intermediate_size", (64, 192, 320, 576))
@pytest.mark.parametrize("source_tile_m", (1, 16, 64))
@pytest.mark.parametrize("base_offset_bytes", (0, 16))
def test_native_fc2_weight_staging_is_byteexact_and_live(
    intermediate_size, source_tile_m, base_offset_bytes
):
    require_b12x()
    stager = StageNativeFC2Weights(intermediate_size, source_tile_m)
    size = stager.experts * stager.output_tiles * intermediate_size * 64
    source = torch.randint(
        0, 256, (size + base_offset_bytes,), dtype=torch.uint8, device="cuda"
    ).narrow(0, base_offset_bytes, size)
    output = torch.empty(
        stager.experts,
        stager.output_tiles,
        stager.k128_tiles,
        2048,
        dtype=torch.int32,
        device="cuda",
    )
    tensors = (
        from_dlpack(source.view(torch.uint32), assumed_align=16),
        from_dlpack(output, assumed_align=16),
    )
    program = compile_kernel(stager, *tensors, current_cuda_stream())

    def expected_bytes():
        expected = torch.zeros(
            stager.experts,
            stager.output_tiles,
            stager.k128_tiles,
            8192,
            dtype=torch.uint8,
            device="cuda",
        )
        packed = source.view(stager.experts, stager.output_tiles, -1)
        for tile in range(stager.k128_tiles):
            count = min(8192, packed.shape[-1] - tile * 8192)
            expected[:, :, tile, :count] = packed[
                :, :, tile * 8192 : tile * 8192 + count
            ]
        return expected

    program(*tensors, current_cuda_stream())
    assert torch.equal(output.view(torch.uint8), expected_bytes())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            program(*tensors, current_cuda_stream())
        for _ in range(3):
            source.bitwise_xor_(0xA5)
            expected = expected_bytes()
            output.fill_(0x5A5A5A5A)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output.view(torch.uint8), expected)
    finally:
        graph.reset()
