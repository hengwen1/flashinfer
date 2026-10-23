"""Selected-position sorting preserves values, padding, and partial blocks."""

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


@pytest.mark.parametrize("topk,expand", [(512, False), (2048, True)])
@pytest.mark.parametrize("rows", [1, 4, 128])
@pytest.mark.parametrize("offset", [0, 1])
def test_position_sort_mutable_graph(topk, expand, rows, offset):
    device = torch.device("cuda", 0)

    def empty(shape, dtype=torch.float32):
        return torch.empty(math.prod(shape) + offset, dtype=dtype, device=device)[
            offset:
        ].view(shape)

    indices = empty((rows, topk), torch.int32).zero_()
    values = empty((rows, topk)).zero_()
    lengths = empty((rows,), torch.int32).zero_()
    active = empty((1,), torch.int32).zero_()
    width = topk * (8 if expand else 1)
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
                (indices >= 0)
                & (values > -torch.inf)
                & (indices * (8 if expand else 1) < extent[:, None])
            )
            order = indices.masked_fill(~valid, torch.iinfo(torch.int32).max).argsort(
                dim=1
            )
            sorted_indices = indices.gather(1, order)
            sorted_valid = valid.gather(1, order)
            if expand:
                tokens = sorted_indices.long()[:, :, None] * 8 + torch.arange(
                    8, device=device
                )
                live = sorted_valid[:, :, None] & (tokens < extent[:, None, None])
                expected = tokens.masked_fill(~live, -1).reshape(rows, width).int()
                expected_counts = live.sum((1, 2)).int()
            else:
                expected = sorted_indices.masked_fill(~sorted_valid, -1)
                expected_values = values.gather(1, order).masked_fill(
                    ~sorted_valid, -torch.inf
                )
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
            if expand:
                torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
            else:
                torch.testing.assert_close(
                    output_values, expected_values, rtol=0, atol=0
                )
                assert bool((counts == -42).all())
    graph.reset()


@pytest.mark.parametrize("topk,expand", [(512, False), (2048, True)])
def test_position_sort_rejects_frozen_factory_miss(topk, expand):
    _compile.cache_clear()
    with kernel_resolution_guard("position sort cache miss"):
        with pytest.raises(
            KernelResolutionFrozenError, match="refusing triton.compile"
        ):
            _compile("sort", (topk, expand), 0)


def test_position_sort_addresses_beyond_signed_int32():
    topk = 512
    boundary = 2**31 // topk
    rows = boundary + 1
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info(device)[0] < 36 << 30:
        pytest.skip("large row-address regression requires 36 GiB free GPU memory")
    indices = torch.empty((rows, topk), device=device, dtype=torch.int32)
    indices.copy_(torch.arange(topk - 1, -1, -1, device=device, dtype=torch.int32))
    values = torch.ones((rows, topk), device=device)
    values[-1].fill_(7)
    output = torch.empty_like(indices)
    output_values = torch.empty_like(values)
    lengths = torch.full((rows,), topk, device=device, dtype=torch.int32)
    active = torch.tensor([topk], device=device, dtype=torch.int32)
    counts = torch.full_like(lengths, -42)
    launcher = _compile("sort", (topk, False), 0)

    def run():
        _launch(
            "sort",
            (topk, False),
            (indices, values, output, output_values, lengths, active, counts),
            (rows,),
            launcher=launcher,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    selected = torch.tensor([0, boundary - 1, boundary], device=device)
    expected_positions = torch.arange(topk, device=device, dtype=torch.int32)[
        None
    ].expand(3, -1)
    expected_values = torch.tensor([1, 1, 7], device=device, dtype=torch.float32)[
        :, None
    ].expand(-1, topk)
    for visible in (topk, 7, 0):
        active.fill_(visible)
        before = torch.cuda.memory_stats()
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
        valid = expected_positions < visible
        torch.testing.assert_close(
            output[selected], expected_positions.masked_fill(~valid, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            output_values[selected],
            expected_values.masked_fill(~valid, -torch.inf),
            rtol=0,
            atol=0,
        )
        assert bool((counts[selected] == -42).all())
    graph.reset()


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

    def run_sort(binding):
        state, runtime = binding.plan, binding.runtime
        scratch = runtime.scratch
        _launch(
            "sort",
            (block_topk, True),
            (
                scratch["block_indices"],
                scratch["block_values"],
                runtime.candidate_output,
                scratch["block_values"],
                runtime.cache_lengths,
                runtime.active_width,
                runtime.candidate_output_lengths,
            ),
            (rows,),
            launcher=state._launchers[("sort", (block_topk, True))],
        )

    def prime(state):
        binding = state.bind(**args)
        binding.runtime.scratch["block_indices"].copy_(
            torch.arange(block_topk - 1, -1, -1, device=device, dtype=torch.int32)
        )
        binding.runtime.scratch["block_values"].fill_(1)
        return PreparedCall(run=lambda: run_sort(binding), owners=(binding, args))

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
        binding = api.bind(declaration, **args).runtime
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run_sort(binding)
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


@pytest.mark.parametrize("width", [512, 513])
@pytest.mark.parametrize("candidates", [False, True])
def test_planned_selection_preserves_unaligned_scratch(width, candidates):
    from b12x.attention import dsa_indexer as api
    from b12x.preparation import PreparationSession, PreparedCall

    scratch_offset = 4
    rows, page_size = 4, 64
    device = torch.device("cuda", 0)
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
