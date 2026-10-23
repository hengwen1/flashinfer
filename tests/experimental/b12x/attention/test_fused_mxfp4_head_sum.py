"""Ordered mixed BF16/FP32 head sums at finite precision boundaries."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from b12x._lib.intrinsics.memory import u32_as_f32
from b12x._lib.utils.device import current_cuda_stream
from b12x.attention.dsa_indexer.mxfp4 import _fused_bf16_pair_accumulate

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="MXFP4 fused selection requires SM120",
)


class _PackedHeadSumProbe:
    @cute.jit
    def __call__(
        self,
        totals: cute.Tensor,
        packed: cute.Tensor,
        actual: cute.Tensor,
        expected: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(totals, packed, actual, expected).launch(
            grid=((totals.shape[0] + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        totals: cute.Tensor,
        packed: cute.Tensor,
        actual: cute.Tensor,
        expected: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        i = cutlass.Int32(bx) * cutlass.Int32(256) + cutlass.Int32(tx)
        if i < totals.shape[0]:
            actual[i] = _fused_bf16_pair_accumulate(totals[i], packed[i])
            partial = totals[i] + u32_as_f32(packed[i] << cutlass.Uint32(16))
            expected[i] = partial + u32_as_f32(packed[i] & cutlass.Uint32(0xFFFF0000))


def test_packed_bf16_head_sum_matches_scalar_finite_boundaries():
    device = torch.device("cuda", 0)
    bits = torch.arange(65536, dtype=torch.int64, device=device)
    lo = bits.repeat(3)
    hi = torch.cat((bits ^ 0x8000, bits, bits.flip(0)))
    packed = (lo | (hi << 16)).to(torch.int32)
    total_bits = (
        torch.tensor(
            [
                0,
                0x80000000,
                1,
                0x80000001,
                0x007FFFFF,
                0x00800000,
                0x3F800000,
                0xBF800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
            ],
            dtype=torch.int64,
            device=device,
        )
        .to(torch.int32)
        .view(torch.float32)
    )
    totals = total_bits.repeat(packed.numel())
    packed = packed.repeat_interleave(total_bits.numel())
    actual = torch.empty_like(totals)
    expected = torch.empty_like(totals)
    args = tuple(
        from_dlpack(t, assumed_align=16) for t in (totals, packed, actual, expected)
    )
    stream = current_cuda_stream()
    compiled = cute.compile(_PackedHeadSumProbe(), *args, stream)
    compiled(*args, stream)
    torch.cuda.synchronize()
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual.view(torch.int32)[finite],
        expected.view(torch.int32)[finite],
        rtol=0,
        atol=0,
    )
    assert bool(
        ((expected != 0) & (expected.abs() < torch.finfo(torch.float32).tiny)).any()
    )
    assert bool(((expected == 0) & torch.signbit(expected)).any())
    print({"finite_ordered_sums": int(finite.sum())})
