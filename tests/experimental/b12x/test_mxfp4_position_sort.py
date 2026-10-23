"""Fused selection and block expansion preserve storage and logical positions."""

import gc
import math
import weakref

import pytest
import torch

from b12x.attention.dsa_indexer.mxfp4 import _compile, _launch
from b12x._lib.compile_plan import load_programs, program_keys, retain_compiled_programs
from b12x._lib.program_cache import evict_unretained
from b12x._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_guard,
)


@pytest.mark.parametrize("rows", [1, 4, 128])
@pytest.mark.parametrize("offset", [0, 1])
def test_block_expansion_mutable_graph(rows, offset):
    topk, expand = 2048, True
    device = torch.device("cuda", 0)

    def empty(shape, dtype=torch.float32):
        return torch.empty(math.prod(shape) + offset, dtype=dtype, device=device)[
            offset:
        ].view(shape)

    indices = empty((rows, topk), torch.int32).zero_()
    values = empty((rows, topk)).zero_()
    lengths = empty((rows,), torch.int32).zero_()
    active = empty((1,), torch.int32).zero_()
    width = topk * 8
    output = empty((rows, width), torch.int32)
    output_values = empty((rows, topk))
    counts = empty((rows,), torch.int32).fill_(-42)
    with retain_compiled_programs() as retained:
        launcher = _compile("sort", (topk, expand), 0)
        load_programs(launcher)
    assert len(program_keys(launcher)) == 1
    assert program_keys(launcher)[0].dialect == "triton"
    raw_ref = weakref.ref(launcher.raw)

    def run(launcher):
        _launch(
            "sort",
            (topk, expand),
            (indices, values, output, output_values, lengths, active, counts),
            (rows,),
            launcher=launcher,
        )

    run(launcher)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(launcher)
    _compile.cache_clear()
    evict_unretained(())
    gc.collect()
    with kernel_resolution_guard("prepared position sort"):
        run(launcher)
    assert any(owner is launcher.raw for owner in retained.owners.values())
    del launcher
    gc.collect()
    assert raw_ref() is not None
    for seed in (41, 99271):
        torch.manual_seed(seed)
        for visible in (0, 1, 7, 8, 127, 511, 512, 513, 16441, 1048576):
            indices.copy_(
                torch.randint(
                    -topk, topk * 2, indices.shape, device=device, dtype=torch.int32
                )
            )
            values.copy_(indices)
            values[:, ::11] = -torch.inf
            values[:, ::31] = torch.nan
            values.masked_fill_(indices % 127 == 0, torch.inf)
            lengths.fill_(visible)
            lengths[-1] = 0
            active.fill_(max(visible - 1, 0))
            extent = lengths.clamp_min(0).minimum(active.clamp_min(0))
            valid = (
                (indices >= 0) & (values > -torch.inf) & (indices * 8 < extent[:, None])
            )
            order = indices.masked_fill(~valid, torch.iinfo(torch.int32).max).argsort(
                dim=1
            )
            sorted_indices = indices.gather(1, order)
            sorted_valid = valid.gather(1, order)
            tokens = sorted_indices.long()[:, :, None] * 8 + torch.arange(
                8, device=device
            )
            live = sorted_valid[:, :, None] & (tokens < extent[:, None, None])
            expected = tokens.masked_fill(~live, -1).reshape(rows, width).int()
            expected_counts = live.sum((1, 2)).int()
            output.fill_(-42)
            output_values.fill_(torch.nan)
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
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    graph.reset()


def test_block_expansion_rejects_frozen_factory_miss():
    topk, expand = 2048, True
    _compile.cache_clear()
    with kernel_resolution_guard("position sort cache miss"):
        with pytest.raises(
            KernelResolutionFrozenError, match="refusing triton.compile"
        ):
            _compile("sort", (topk, expand), 0)


def test_planned_candidate_expansion_addresses_beyond_signed_int32():
    from b12x.attention import dsa_indexer as api
    from b12x.preparation import PreparationSession, PreparedCall

    block_topk, width = 2048, 64
    boundary = 2**31 // (block_topk * 8)
    rows = boundary + 1
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info(device)[0] < 14 << 30:
        pytest.skip("candidate expansion regression requires 14 GiB free GPU memory")
    declaration = api.plan(
        api.Caps(
            device=device,
            num_q_heads=1,
            max_q_rows=rows,
            max_page_table_width=1,
            page_size=width,
            topk=512,
            cache_format="mxfp4",
            candidate_topk_blocks=block_topk,
        )
    )
    (spec,) = declaration.scratch_specs()
    lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    active = torch.tensor([width], device=device, dtype=torch.int32)
    output = torch.empty((rows, 512), device=device, dtype=torch.int32)
    candidates = torch.empty((rows, block_topk * 8), device=device, dtype=torch.int32)
    counts = torch.empty_like(lengths)
    args = dict(
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        q_mxfp4=torch.zeros((rows, 1, 64), device=device, dtype=torch.uint8),
        q_scales=torch.full((rows, 1, 4), 127, device=device, dtype=torch.uint8),
        query_weights=torch.ones((rows, 1), device=device, dtype=torch.bfloat16),
        index_k_cache=torch.zeros(
            (1, api.index_mxfp4_page_bytes(width)), device=device, dtype=torch.uint8
        ),
        page_table=torch.zeros((1, 1), device=device, dtype=torch.int32),
        cache_lengths=lengths,
        active_width=active,
        output_indices=output,
        candidate_output=candidates,
        candidate_output_lengths=counts,
    )

    def prime(state):
        binding = state.bind(**args)
        return PreparedCall(run=lambda: state.run(binding), owners=(binding, args))

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                declaration.request(
                    name="expanded-position-addresses", prepare_call=prime
                ),
            )
        )
        binding = api.bind(declaration, **args)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            api.run(binding)
        session.freeze()
        selected = torch.tensor([0, boundary - 1, boundary], device=device)
        positions = torch.arange(block_topk * 8, device=device, dtype=torch.int32)[
            None
        ].expand(3, -1)
        for visible in (width, 63, 0):
            lengths.fill_(visible)
            lengths[-1] = max(visible - 1, 0)
            active.fill_(visible)
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
            expected = positions.masked_fill(positions >= lengths[selected, None], -1)
            torch.testing.assert_close(candidates[selected], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                counts[selected], lengths[selected], rtol=0, atol=0
            )
        graph.reset()


@pytest.mark.parametrize("width", [512, 513, 4096, 16384])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize(
    "scratch_offset,external_offset", [(0, 0), (4, 0), (16, 0), (0, 1)]
)
def test_planned_selection_preserves_unaligned_storage(
    width, candidates, scratch_offset, external_offset
):
    from b12x.attention import dsa_indexer as api
    from b12x.preparation import PreparationSession, PreparedCall

    rows, page_size = 4, 64
    device = torch.device("cuda", 0)

    def external(value):
        storage = torch.empty(
            value.numel() + external_offset, dtype=value.dtype, device=device
        )
        return storage[external_offset:].view_as(value).copy_(value)

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
    lengths = torch.full((rows,), history, device=device, dtype=torch.int32)
    active = torch.tensor([history], device=device, dtype=torch.int32)
    counts = torch.full((rows,), width, device=device, dtype=torch.int32)
    logical = (
        torch.arange(width, device=device, dtype=torch.int32)[None]
        .expand(rows, -1)
        .clone()
    )
    if candidates:
        logical.mul_(2)
    output = torch.empty((rows, 512), device=device, dtype=torch.int32)
    values = torch.empty((rows, 512), device=device, dtype=torch.float32)
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

    for name in (
        "q_mxfp4",
        "q_scales",
        "query_weights",
        "page_table",
        "cache_lengths",
        "active_width",
        "output_indices",
        "output_scores",
        "candidate_indices",
        "candidate_lengths",
    ):
        if args[name] is not None:
            args[name] = external(args[name])
    lengths, active = args["cache_lengths"], args["active_width"]
    output, values = args["output_indices"], args["output_scores"]
    if candidates:
        logical, counts = args["candidate_indices"], args["candidate_lengths"]

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
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            api.run(binding)
        session.freeze()
        _compile.cache_clear()
        evict_unretained(())
        gc.collect()
        with kernel_resolution_guard("prepared fused MXFP4 indexer"):
            api.run(binding)
        for seed in (481, 6191):
            torch.manual_seed(seed)
            codes = torch.randint(0, 16, (history,), dtype=torch.uint8, device=device)
            positions = torch.arange(history, device=device)
            physical, token = positions // page_size, positions % page_size
            args["index_k_cache"][
                physical[:, None], token[:, None] * 64 + torch.arange(64, device=device)
            ] = (codes | (codes << 4))[:, None]
            args["index_k_cache"][
                physical[:, None],
                page_size * 64 + token[:, None] * 4 + torch.arange(4, device=device),
            ] = 127
            levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
            keys = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
            args["query_weights"].copy_(
                torch.tensor([[1], [-1], [0.5], [-0.5]], device=device)
            )
            for step, visible in enumerate((history, 513, 127, 0, history - 1)):
                query = 1 if step % 2 == 0 else -1
                args["q_mxfp4"].fill_(0x22 if query == 1 else 0xAA)
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
                dot = (keys * query * 128).bfloat16().float().clamp_min(0)
                full_scores = (dot[None] * args["query_weights"]).bfloat16()
                source = full_scores.gather(1, logical.clamp_min(0).long())
                live = (
                    (logical >= 0) & (logical < lengths[:, None]) & (logical < active)
                )
                if candidates:
                    live &= torch.arange(width, device=device)[None] < counts[:, None]
                source.masked_fill_(~live, -torch.inf)
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
                assert torch.equal(valid.sum(1), live.sum(1).clamp(max=512))
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
