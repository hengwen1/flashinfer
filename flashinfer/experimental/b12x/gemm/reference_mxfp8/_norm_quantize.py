# SPDX-License-Identifier: Apache-2.0
"""Q/KV RMSNorm and reference FP8 Q quantization with BF16 rounding."""

import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset


@triton.jit
def norm_quant_kernel(
    X, QW, KW, QBF, KV, Q8, S, ROWS, M: tl.constexpr, PDL: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    if task == 0:
        c = tl.arange(0, 2048)
        w = tl.load(QW + c, c < 1280, 0).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        x = tl.load(X + row * 1792 + c, c < 1280, 0).to(tl.float32)
        variance = tl.sum(x * x, 0) / 1280.0
        y = (x * tl.rsqrt(variance + 1e-20) * w).to(QBF.dtype.element_ty)
        tl.store(QBF + row * 1280 + c, y, c < 1280)
        grouped = tl.reshape(y.to(tl.float32), (64, 32))
        amax = tl.maximum(tl.max(tl.abs(grouped), 1), 1e-4)
        bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        exponent = (
            ((bits >> 23) & 255).to(tl.int32)
            - 127
            + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        )
        scale = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        quant = tl.minimum(
            tl.maximum(tl.div_rn(grouped, scale[:, None]), -448.0), 448.0
        ).to(tl.float8e4nv)
        tl.store(Q8 + row * 1280 + c, tl.reshape(quant, (2048,)), c < 1280)
        group = tl.arange(0, 64)
        byte = (exponent + 127).to(tl.uint8)
        tl.store(ROWS + row * 40 + group, byte, group < 40)
        tl.store(S + _scale_offset(row, group, 40), byte, group < 40)
    else:
        kc = tl.arange(0, 512)
        kw = tl.load(KW + kc).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        kx = tl.load(X + row * 1792 + 1280 + kc).to(tl.float32)
        ky = kx * tl.rsqrt(tl.sum(kx * kx, 0) / 512.0 + 1e-20) * kw
        tl.store(KV + row * 512 + kc, ky)
