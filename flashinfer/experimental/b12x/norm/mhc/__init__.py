"""mHC residual for SM12x: fused RMSNorm + hyper-connection mixing +
projection (DeepSeek-style), BF16 with TF32 projection paths.

Each declaration owns one operation and exact planned nonempty M:
``run_pre`` (residual -> mixed input + carry), ``run_post`` (output mix-back),
``run_post_pre`` (fused post+pre), or ``run_collapse`` (weighted/uniform stream
contraction). Sinkhorn-normalized mix matrices; mix constants are exposed as
``MIXES`` / ``MULT`` / ``PARTIALS``.

V4.1 lagged mixing is a paired prepared invocation: pass incoming FP32
``pre_mix[tokens,4]`` and a disjoint caller-owned FP32 ``pre_out[tokens,4]``
to ``run_pre`` or ``run_post_pre``. The retained native producer performs the
BF16 collapse and the finalizer then applies optional RMSNorm; each call still
returns exactly ``residual, post, comb, y`` while writing next-layer
coefficients to ``pre_out``.

For normalized caller-owned lagged invocations below 96 rows, declaring
``split_lagged=True`` permits ``run_input(binding, operands, ...)`` followed by
``run_coefficients(binding, scale, base)``. The first phase writes the BF16
residual and normalized input, retaining FP32 projection partials. The second
phase writes FP32 post, combination, and outgoing pre-mix coefficients without
reading residual or normalized data. Keep partials private until coefficient
completion and join before coefficient consumption or partial reuse. Ordinary
``run_pre``/``run_post_pre`` still execute the entire declared operation.

``plan(Caps(...), invocation=...)`` is allocation-free. ``PreparationSession``
selects and primes the complete plan before ``bind`` builds scratch views.
Functional operations carry the same opaque prepared reference through eager,
torch.compile and graph capture. Fully bound operations expose scratch and
output mutations to compilation without allocating buffers. Bind partials,
residual, normalized output, post and comb storage, plus ``pre_out`` for lagged
mixing. ``run_post`` also supports a caller-owned ``out`` through compilation.
Bindings carry their own prepared plan; there is no implicit runtime
configuration path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mhc",
    group="norm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "MhcConfig",
        "MhcQuery",
        "plan",
        "bind",
        "run_pre",
        "run_post",
        "run_post_pre",
        "run_collapse",
        "run_input",
        "run_coefficients",
        "MIXES",
        "MULT",
        "PARTIALS",
        "DEFAULT_SPLIT_K",
        "DEFAULT_BLOCK_K",
        "DEFAULT_BLOCK_H",
        "is_supported",
    ),
    dtypes=("bf16",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/integration/residual.py",
            "b12x/integration/residual_kernels.py",
        ),
    ),
    test_path="tests/experimental/b12x/norm/test_mhc.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        DEFAULT_BLOCK_H,
        DEFAULT_BLOCK_K,
        DEFAULT_SPLIT_K,
        MIXES,
        MULT,
        PARTIALS,
        Binding,
        Caps,
        MhcConfig,
        MhcQuery,
        Plan,
        bind,
        is_supported,
        plan,
        run_post,
        run_post_pre,
        run_pre,
        run_collapse,
        run_input,
        run_coefficients,
    )

install_lazy_api(globals(), META)
