"""Source-built reference-precision FP8 attention projection support.

This research implementation uses E4M3 activations with per-32 UE8M0 scales,
including DeepSeek V4.1's absolute-maximum floor of 1e-4. Its CUDA projection
is derived from MoET revision 5d25cff77495; it accumulates in FP32 and writes
BF16. Preparation retains the module and caller buffers through graph drain.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import torch


@lru_cache(None)
def projection_module():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("reference CUDA module must be built before graph capture")
    from torch.utils.cpp_extension import load
    from flashinfer.jit.env import (
        CCCL_INCLUDE_DIRS,
        FLASHINFER_JIT_DIR,
    )

    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("reference MXFP8 GEMV source supports SM120")
    source = Path(__file__).parent / "csrc" / "projection.cu"
    digest = sha256(source.read_bytes() + torch.__version__.encode()).hexdigest()[:16]
    build = FLASHINFER_JIT_DIR / ("flare-reference-mxfp8-" + digest)
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="flare_reference_mxfp8_" + digest,
        sources=[str(source)],
        build_directory=str(build),
        extra_include_paths=[str(path) for path in CCCL_INCLUDE_DIRS],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "-gencode=arch=compute_120,code=sm_120",
        ],
    )


@lru_cache(None)
def quantizer(rows, width):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("reference quantizer must be prepared before graph capture")
    from ._quantize import reference_quant_kernel
    import triton

    compiled = reference_quant_kernel.warmup(
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.uint8,
        torch.uint8,
        rows,
        width,
        grid=(rows, triton.cdiv(width // 32, 16), 1),
        num_warps=4,
        enable_fp_fusion=False,
    )

    def run(source, values, scales):
        compiled[(rows, triton.cdiv(width // 32, 16), 1)](
            source, values, scales, scales, rows, width, 16, False
        )

    run.owner = compiled
    return run


@lru_cache(None)
def norm_quantizer(rows, weight_dtype, *, pdl):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "reference normalization must be prepared before graph capture"
        )
    from ._norm_quantize import norm_quant_kernel

    compiled = norm_quant_kernel.warmup(
        torch.bfloat16,
        weight_dtype,
        weight_dtype,
        torch.bfloat16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.uint8,
        torch.uint8,
        rows,
        pdl,
        grid=(rows, 2, 1),
        num_warps=8,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )

    def run(
        projected,
        query_weight,
        kv_weight,
        query_bf16,
        kv,
        query_fp8,
        scales,
        scale_rows,
    ):
        compiled[(rows, 2, 1)](
            projected,
            query_weight,
            kv_weight,
            query_bf16,
            kv,
            query_fp8,
            scales,
            scale_rows,
            rows,
            pdl,
        )

    run.owner = compiled
    return run


def compile_quantizer(rows, width, ordinal):
    """Describe the exact Triton program used by the prepared projection."""
    return quantizer(rows, width).owner
