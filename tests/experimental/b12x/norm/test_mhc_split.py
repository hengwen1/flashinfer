"""Lagged mHC phases preserve rounding and declare their independent lifetimes."""

from dataclasses import replace

import pytest
import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.mhc import make_inputs

from ..conftest import require_b12x


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@torch.no_grad()
def test_split_phases_preserve_outputs_and_release_producer_inputs(
    hidden, norm_dtype, operation
):
    device, rows = require_b12x(), 7
    residual, branch, fn, scale, base = make_inputs(
        tokens=rows, hidden_size=hidden, seed=4281, device=device
    )
    norm = torch.linspace(0.5, 1.5, hidden, device=device).to(norm_dtype)
    previous_post = torch.rand((rows, 4), device=device)
    previous_comb = torch.eye(4, device=device).expand(rows, -1, -1).contiguous()
    incoming = torch.rand((rows, 4), device=device)
    operands = (
        (residual,)
        if operation == "pre"
        else (branch, residual, previous_post, previous_comb)
    )
    options = dict(
        pre_mix=incoming,
        norm_weight=norm,
        rms_eps=1e-20,
        norm_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    bindings = []
    owners = []
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        for split in (False, True):
            plan = mhc.plan(
                mhc.Caps(
                    device=device,
                    max_tokens=rows,
                    hidden_size=hidden,
                    split_k=hidden // 64,
                ),
                invocation=dict(
                    operation=operation,
                    expanded_residual=operation == "pre",
                    output_mode="provided",
                    lagged_mix=True,
                    split_lagged=split,
                    has_norm_weight=True,
                    norm_weight_dtype=str(norm_dtype).removeprefix("torch."),
                    rms_eps=1e-20,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                    sinkhorn_iters=20,
                ),
            )

            def prepare(state, plan=plan):
                scratch = torch.empty(
                    state.scratch_specs()[0].shape, device=device, dtype=torch.uint8
                )
                binding = state.bind(
                    scratch=scratch,
                    tokens=rows,
                    out=torch.empty_like(residual),
                    y=torch.empty_like(branch),
                    post=torch.empty_like(incoming),
                    comb=torch.empty_like(previous_comb),
                    pre_out=torch.empty_like(incoming),
                    plan=plan,
                )
                bindings.append(binding)
                owners.append(scratch)
                entry = getattr(_impl, f"_b12x_mhc_{operation}_impl")
                return PreparedCall(
                    run=lambda: entry(
                        *operands,
                        fn,
                        scale,
                        base,
                        **options,
                        binding=binding,
                        _state=state,
                    )
                )

            session.prepare(
                (plan.request(name=f"{operation}.{split}", prepare_call=prepare),)
            )
        control, candidate = bindings
        entry = mhc.run_pre if operation == "pre" else mhc.run_post_pre
        entry(*operands, fn, scale, base, **options, binding=control)
        for value in (candidate.post_buffer, candidate.comb_buffer, candidate.pre_out):
            value.fill_(float("nan"))
        mhc.run_input(
            candidate, (*operands, fn, scale, base), norm_weight=norm, pre_mix=incoming
        )
        torch.testing.assert_close(candidate.out, control.out, rtol=0, atol=0)
        torch.testing.assert_close(candidate.y, control.y, rtol=0, atol=0)
        for value in (candidate.post_buffer, candidate.comb_buffer, candidate.pre_out):
            assert value.isnan().all()
        # Coefficient finalization consumes retained partials and scale/base only.
        for value in (*operands, fn, norm, incoming, candidate.out, candidate.y):
            value.fill_(float("nan"))
        mhc.run_coefficients(candidate, scale, base)
        assert candidate.out.isnan().all() and candidate.y.isnan().all()
        for actual, expected in (
            (candidate.post_buffer, control.post_buffer),
            (candidate.comb_buffer, control.comb_buffer),
            (candidate.pre_out, control.pre_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="mHC partials"):
            mhc.run_coefficients(
                replace(candidate, partials=candidate.partials[:, :-1]), scale, base
            )
