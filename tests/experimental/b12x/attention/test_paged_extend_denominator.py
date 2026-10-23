"""Paged extend normalizes with FP32 exponential sums before P conversion."""

import math

import pytest
import torch

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import (
    make_msa_q2k_indices,
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)

from .test_attention_paged_msa import _run_msa_extend
from .test_paged import _run_eager


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "tokens,context,page_size", [(17, 17, 128), (17, 257, 64), (4096, 4096, 128)]
)
def test_extend_preserves_unrounded_exponential_denominator(
    fp8, sparse, tokens, context, page_size
):
    require_b12x()
    q, key, value, pages, lengths, cu = make_paged_inputs(
        q_seqlens=[tokens],
        cache_seqlens=[context],
        page_size=page_size,
        q_heads=16,
        kv_heads=1,
        head_dim=128,
        seed=178,
    )
    descales = {}
    if fp8:
        key, value, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
            key, value, pages, lengths
        )
        descales = dict(k_descale=k_descale, v_descale=v_descale)
    selected = make_msa_q2k_indices(
        cache_seqlens=lengths, cu_seqlens_q=cu, num_kv_heads=1, seed=195
    )
    if sparse:
        _, actual = _run_msa_extend(
            q, key, value, pages, lengths, cu, selected, msa_union_tile=True, **descales
        )
    else:
        _, actual = _run_eager(
            q, key, value, pages, lengths, cu, mode="extend", **descales
        )
        actual = actual * math.log(2)
    rows = sorted(
        set((0, min(15, tokens - 1), min(127, tokens - 1), tokens // 2, tokens - 1))
    )
    reference = []
    for row in rows:
        limit = context - tokens + row
        visible = []
        for block in (
            selected[0, row].tolist() if sparse else range(math.ceil(context / 128))
        ):
            if block >= 0:
                visible.extend(range(block * 128, min((block + 1) * 128, limit + 1)))
        positions = torch.tensor(visible, dtype=torch.int64, device=q.device)
        physical = pages[0, positions // page_size].long()
        k = key[physical, positions % page_size, 0].float()
        if fp8:
            k = k * k_descale.float()
        scores = q[row].float() @ k.T / math.sqrt(128)
        reference.append(torch.logsumexp(scores, -1))
    expected = torch.stack(reference)
    error = (actual[rows] - expected).abs().max().item()
    print(
        f"GQA extend FP32 denominator fp8={fp8} sparse={sparse} tokens={tokens} context={context} page={page_size} max_lse_error={error}"
    )
    torch.testing.assert_close(actual[rows], expected, rtol=0, atol=1e-5)
