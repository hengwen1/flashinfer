"""Literal DS41 activation-quantization semantics and an independent Torch oracle."""

import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset


@triton.jit
def reference_quant_kernel(
    X,
    Q,
    S,
    ROWS,
    M: tl.constexpr,
    K: tl.constexpr,
    CHUNKS: tl.constexpr = 16,
    STORE_ROWS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * CHUNKS + tl.arange(0, CHUNKS)
    column = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + column, column < K, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    ratio = amax * (1.0 / 448.0)
    bits = ratio.to(tl.uint32, bitcast=True)
    exponent = (
        ((bits >> 23) & 255).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    )
    scale = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -448.0), 448.0).to(
        tl.float8e4nv
    )
    tl.store(Q + row * K + column, quantized, column < K)
    byte = (exponent + 127).to(tl.uint8)
    if STORE_ROWS:
        tl.store(ROWS + row * (K // 32) + group, byte, group < K // 32)
    tl.store(S + _scale_offset(row, group, K // 32), byte, group < K // 32)
    if row == 0:
        pad = (M // 128) * 128 + tl.arange(0, 128)
        tl.store(
            S + _scale_offset(pad[:, None], group[None, :], K // 32),
            127,
            (pad[:, None] >= M)
            & (pad[:, None] < triton.cdiv(M, 128) * 128)
            & (group[None, :] < K // 32),
        )
