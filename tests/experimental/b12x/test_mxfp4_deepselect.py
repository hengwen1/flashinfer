"""Prepared BF16 selection preserves values, logical IDs, and mutable bounds."""

import gc

import pytest
import torch

from b12x.attention import dsa_indexer as api
from b12x.attention.dsa_indexer.mxfp4 import _compile
from b12x._lib.program_cache import evict_unretained
from b12x._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_guard,
)
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.parametrize("width", [512, 513, 4096, 16384])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize(
    "scratch_offset,external_offset", [(0, 0), (4, 0), (16, 0), (0, 1)]
)
def test_prepared_selection_values_and_replay(
    width, candidates, scratch_offset, external_offset
):
    rows, page_size = 4, 64
    device = torch.device("cuda", 0)

    def external(tensor):
        backing = torch.empty(
            tensor.numel() + external_offset, dtype=tensor.dtype, device=device
        )
        return backing[external_offset:].view_as(tensor).copy_(tensor)

    history = width * (2 if candidates else 1)
    pages = (history + 511) // 512 * (512 // page_size)
    declaration = api.plan(
        api.Caps(
            device=device,
            num_q_heads=1,
            max_q_rows=rows,
            max_page_table_width=pages,
            page_size=page_size,
            topk=512,
            cache_format="mxfp4",
            max_candidates=width if candidates else 0,
        )
    )
    (spec,) = declaration.scratch_specs()
    lengths = external(torch.full((rows,), history, device=device, dtype=torch.int32))
    active = external(torch.tensor([history], device=device, dtype=torch.int32))
    counts = external(torch.full((rows,), width, device=device, dtype=torch.int32))
    logical = external(
        torch.arange(width, device=device, dtype=torch.int32)[None].expand(rows, -1)
    )
    if candidates:
        logical.mul_(2)
    output = external(torch.empty((rows, 512), device=device, dtype=torch.int32))
    values = external(torch.empty((rows, 512), device=device, dtype=torch.float32))
    backing = torch.empty(
        spec.shape[0] + scratch_offset, dtype=spec.dtype, device=device
    )
    args = dict(
        scratch=backing[scratch_offset:],
        q_mxfp4=torch.zeros((rows, 1, 64), device=device, dtype=torch.uint8),
        q_scales=torch.full((rows, 1, 4), 127, device=device, dtype=torch.uint8),
        query_weights=torch.ones((rows, 1), device=device, dtype=torch.bfloat16),
        index_k_cache=torch.zeros(
            (pages, api.index_mxfp4_page_bytes(page_size)),
            device=device,
            dtype=torch.uint8,
        ),
        page_table=torch.arange(pages, device=device, dtype=torch.int32)[None],
        cache_lengths=lengths,
        active_width=active,
        output_indices=output,
        output_scores=values,
        score_width=width,
        candidate_indices=logical if candidates else None,
        candidate_lengths=counts if candidates else None,
    )

    def prime(state):
        binding = state.bind(**args)
        return PreparedCall(run=lambda: state.run(binding), owners=(binding, args))

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (declaration.request(name="bf16-selection", prepare_call=prime),)
        )
        binding = api.bind(declaration, **args)
        scores = binding.runtime.runtime.scratch["scores"]
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            api.select(binding)
        session.freeze()
        _compile.cache_clear()
        evict_unretained(())
        gc.collect()
        with kernel_resolution_guard("prepared BF16 selector"):
            api.select(binding)
        for seed in (481, 6191):
            torch.manual_seed(seed)
            for visible in (history, 513, 127, 0, history - 1):
                lengths.fill_(visible)
                lengths[0] = 0
                active.fill_(visible)
                counts.fill_(width)
                counts[-1] = width // 2
                logical.copy_(
                    torch.arange(width, device=device, dtype=torch.int32)[None]
                )
                if candidates:
                    logical.mul_(2)
                    logical[:, ::17] = -1
                args["scratch"].fill_(0xA5)
                source = torch.randn((rows, width), device=device, dtype=torch.bfloat16)
                source[:, ::3] = 1
                source[:, ::31] = torch.inf
                source[:, ::37] = -torch.inf
                live = (
                    (logical >= 0) & (logical < lengths[:, None]) & (logical < active)
                )
                if candidates:
                    live &= torch.arange(width, device=device)[None] < counts[:, None]
                source.masked_fill_(~live, -torch.inf)
                scores.copy_(source)
                output.fill_(-42)
                values.fill_(torch.nan)
                before = torch.cuda.memory_stats()
                graph.replay()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key]
                valid = output >= 0
                assert bool(((output[:, 1:] > output[:, :-1]) | ~valid[:, 1:]).all())
                assert torch.equal(
                    valid.sum(1), (live & (source > -torch.inf)).sum(1).clamp(max=512)
                )
                chosen = (output // (2 if candidates else 1)).clamp_min(0).long()
                assert bool(((logical.gather(1, chosen) == output) | ~valid).all())
                reference = (
                    source.float().gather(1, chosen).masked_fill(~valid, -torch.inf)
                )
                torch.testing.assert_close(values, reference, rtol=0, atol=0)
                torch.testing.assert_close(
                    values.sort().values,
                    source.float().topk(512, dim=1).values.sort().values,
                    rtol=0,
                    atol=0,
                )
        graph.reset()


@pytest.mark.parametrize("candidates", [False, True])
def test_deepselect_preparation_rejects_frozen_factory_miss(candidates):
    _compile.cache_clear()
    with kernel_resolution_guard("BF16 selector factory miss"):
        with pytest.raises(KernelResolutionFrozenError, match="DeepSelect preparation"):
            _compile("deepselect", (candidates,), 0)
