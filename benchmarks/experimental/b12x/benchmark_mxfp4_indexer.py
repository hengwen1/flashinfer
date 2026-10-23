"""Measure prepared production MXFP4 indexing under CUDA graph replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x._lib.runtime_control import kernel_resolution_guard
from benchmarks.experimental.b12x.deepseek_attention_profiles import (
    load_attention_profile,
)
from b12x.attention import dsa_indexer as indexer
from b12x.attention.dsa_indexer._mxfp4_tuning import default_fused_config
from benchmarks.experimental.b12x.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)


def make_plan(session, caps, arguments, args):
    invocation = indexer.invocation_from_tensors(caps, **arguments)
    plan = indexer.plan(caps, invocation=invocation)
    if args.preparation == "heuristic" or args.fused_ctas_per_row is not None:
        config = plan.contract.configure(
            plan.query, device=session.device.identity
        ).default
        if args.fused_ctas_per_row is not None:
            config = replace(
                config,
                mxfp4=default_fused_config(
                    plan.query,
                    session.device.identity,
                    ctas_per_row=args.fused_ctas_per_row,
                ),
            )
        plan = indexer.plan(caps, invocation=invocation, override=config)
    return plan


def prepare_mxfp4(session, plan, *, q, keys, slots, arguments, tuning_seed):
    generator = torch.Generator(device=q.device).manual_seed(tuning_seed)
    tuning_source = (
        torch.randn(q.shape, generator=generator, device=q.device) / 4
    ).bfloat16()

    def prepare(state):
        trial = dict(arguments)
        activation = torch.empty_like(q)
        (spec,) = state.layout.scratch_specs()
        trial["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
        for name in (
            "q_mxfp4",
            "q_scales",
            "output_indices",
            "output_scores",
            "candidate_output",
            "candidate_output_lengths",
        ):
            if trial.get(name) is not None:
                trial[name] = torch.empty_like(trial[name])
        state.write_index_keys(
            keys, index_k_cache=trial["index_k_cache"], slot_mapping=slots
        )
        binding = state.bind(**trial)

        def run():
            state.quantize_query(
                activation, q_mxfp4=trial["q_mxfp4"], q_scales=trial["q_scales"]
            )
            state.run(binding)

        return PreparedCall(
            run=run,
            produce=lambda: activation.copy_(tuning_source),
            owners=(trial, binding, activation, tuning_source),
        )

    session.prepare(
        (
            plan.request(
                name="mxfp4-indexer", prepare_call=prepare, benchmark_call=prepare
            ),
        )
    )
    indexer.quantize_q_mxfp4(
        plan, q, q_mxfp4=arguments["q_mxfp4"], q_scales=arguments["q_scales"]
    )


def quantized_reference(x):
    """Independent RN-even E2M1 encoding and BF16 dequantization oracle."""
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax / 6).contiguous().view(torch.int32)
    scales = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    factors = (scales.int() << 23).view(torch.float32)
    normalized = groups / factors[..., None]
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    choice = (normalized.abs()[..., None] - lut[order]).abs().argmin(-1)
    codes = order[choice].to(torch.uint8) | (
        torch.signbit(normalized).to(torch.uint8) << 3
    )
    decoded = (
        (
            lut[(codes & 7).long()]
            * torch.where((codes & 8) != 0, -1, 1)
            * factors[..., None]
        )
        .bfloat16()
        .reshape_as(x)
    )
    codes = codes.reshape(*x.shape[:-1], 128)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scales, decoded


def score_reference(decoded_q, decoded_keys, weights):
    """BF16 dot/product stages followed by increasing-head FP32 summation."""
    dots = torch.einsum("rhd,kd->rhk", decoded_q, decoded_keys)
    products = (dots.relu() * weights[:, :, None]).float()
    total = torch.zeros(
        (decoded_q.shape[0], decoded_keys.shape[0]),
        dtype=torch.float32,
        device=decoded_q.device,
    )
    for head in range(weights.shape[1]):
        total += products[:, head]
    return total.bfloat16()


def check_selection(scores, indices, values, candidates=None):
    for row in range(scores.shape[0]):
        valid = torch.isfinite(scores[row])
        count = min(512, int(valid.sum()))
        selected = indices[row, :count].long()
        assert bool((indices[row, count:] == -1).all())
        assert bool(torch.isneginf(values[row, count:]).all())
        assert bool((selected[1:] > selected[:-1]).all())
        expected_values = scores[row, valid].float().topk(count).values.sort().values
        torch.testing.assert_close(
            values[row, :count].sort().values, expected_values, rtol=0, atol=0
        )
        if candidates is None:
            torch.testing.assert_close(
                values[row, :count], scores[row, selected].float(), rtol=0, atol=0
            )
        else:
            logical, order = candidates[row].long().sort()
            offsets = torch.searchsorted(logical, selected)
            torch.testing.assert_close(logical[offsets], selected, rtol=0, atol=0)
            torch.testing.assert_close(
                values[row, :count], scores[row, order[offsets]].float(), rtol=0, atol=0
            )


@torch.inference_mode()
def run_mode(args, candidates_mode):
    device = torch.device("cuda", args.device)
    rows_capacity, context = max(args.rows), max(args.widths)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = (
        torch.randn(
            (rows_capacity, args.heads, 128), generator=generator, device=device
        )
        / 4
    ).bfloat16()
    keys = (
        torch.randn((context, 128), generator=generator, device=device) / 4
    ).bfloat16()
    weights = (
        torch.randn((rows_capacity, args.heads), generator=generator, device=device)
        / 32
    ).bfloat16()
    timing_q, timing_weights = q.clone(), weights.clone()
    q_reference, sf_reference, decoded_q = quantized_reference(q)
    key_reference, key_scales_reference, decoded_k = quantized_reference(keys)
    packed = torch.empty_like(q_reference)
    scales = torch.empty_like(sf_reference)
    page_bytes = indexer.index_mxfp4_page_bytes(args.page_size)
    if args.page_stride < page_bytes or args.page_stride % 16:
        raise ValueError(
            "page stride must be aligned and cover the native record planes"
        )
    page_count = (context + args.page_size - 1) // args.page_size
    base_pid = 2**31 // args.page_stride + 17 if args.high_pid else 7
    backing = torch.empty(
        (base_pid + page_count, args.page_stride), dtype=torch.uint8, device=device
    )
    pool = backing.as_strided(
        (base_pid + page_count, page_bytes), (args.page_stride, 1)
    )
    physical_pages = (
        torch.randperm(page_count, generator=generator, device=device).int() + base_pid
    )
    positions = torch.arange(context, device=device)
    slots = (
        physical_pages[positions // args.page_size].long() * args.page_size
        + positions % args.page_size
    )
    planned_pages = (args.capacity + args.page_size - 1) // args.page_size
    table = torch.full((1, planned_pages), -1, dtype=torch.int32, device=device)
    table[0, :page_count].copy_(physical_pages)
    lengths = torch.full((rows_capacity,), context, dtype=torch.int32, device=device)
    active = torch.tensor([context], dtype=torch.int32, device=device)
    candidate_capacity = args.candidate_capacity if candidates_mode else 0
    candidate_ids = None
    candidate_lengths = None
    if candidates_mode:
        if candidate_capacity not in (512, 16384):
            raise ValueError("consumer capacity must be 512 or 16384")
        candidate_ids = torch.empty(
            (rows_capacity, candidate_capacity), dtype=torch.int32, device=device
        )
        candidate_lengths = torch.zeros(
            (rows_capacity,), dtype=torch.int32, device=device
        )
        candidate_ids.fill_(-1)
    output_indices = torch.empty((rows_capacity, 512), dtype=torch.int32, device=device)
    output_scores = torch.empty(
        (rows_capacity, 512), dtype=torch.float32, device=device
    )
    arguments = dict(
        q_mxfp4=packed,
        q_scales=scales,
        query_weights=weights,
        index_k_cache=pool,
        page_table=table,
        cache_lengths=lengths,
        active_width=active,
        output_indices=output_indices,
        output_scores=output_scores,
        candidate_indices=candidate_ids,
        candidate_lengths=candidate_lengths,
    )
    session = PreparationSession(
        device=device,
        cache_policy="warm",
        compile_workers=2,
        namespace={
            "benchmark": "mxfp4_indexer",
            "tuning_query_seed": args.seed + 200003,
            "source_tuning_query_seed": args.seed + 200004,
        },
    )
    publisher = None
    publisher_scratch = None
    if candidates_mode:
        source_q = (
            torch.randn(q.shape, generator=generator, device=device) / 4
        ).bfloat16()
        timing_source_q = source_q.clone()
        source_weights = (
            torch.randn(weights.shape, generator=generator, device=device) / 32
        ).bfloat16()
        source_packed, source_scales, source_decoded = quantized_reference(source_q)
        source_output = torch.empty_like(output_indices)
        source_values = torch.empty_like(output_scores)
        source_caps = indexer.Caps(
            device=device,
            num_q_heads=args.heads,
            max_q_rows=rows_capacity,
            max_page_table_width=planned_pages,
            page_size=args.page_size,
            cache_format="mxfp4",
            topk=512,
            candidate_topk_blocks=2048 if candidate_capacity == 16384 else 0,
        )
        source_args = dict(
            q_mxfp4=source_packed,
            q_scales=source_scales,
            query_weights=source_weights,
            index_k_cache=pool,
            page_table=table,
            cache_lengths=lengths,
            active_width=active,
            output_indices=source_output,
            output_scores=source_values,
            candidate_output=candidate_ids if candidate_capacity == 16384 else None,
            candidate_output_lengths=candidate_lengths
            if candidate_capacity == 16384
            else None,
        )
        source_plan = make_plan(session, source_caps, source_args, args)
        prepare_mxfp4(
            session,
            source_plan,
            q=source_q,
            keys=keys,
            slots=slots,
            arguments=source_args,
            tuning_seed=args.seed + 200004,
        )
        (source_spec,) = source_plan.scratch_specs()
        publisher_scratch = torch.empty(
            source_spec.shape, dtype=source_spec.dtype, device=device
        )
        publisher = indexer.bind(source_plan, scratch=publisher_scratch, **source_args)

        def publish():
            indexer.quantize_q_mxfp4(
                source_plan, source_q, q_mxfp4=source_packed, q_scales=source_scales
            )
            indexer.run(publisher)
            if candidate_capacity == 512:
                candidate_ids.copy_(source_output)
                candidate_lengths.copy_(lengths.clamp(max=512))

        publish()

    plans = {}
    workspaces = {}
    row_arguments = {}
    row_operands = {
        "q_mxfp4",
        "q_scales",
        "query_weights",
        "cache_lengths",
        "output_indices",
        "output_scores",
        "candidate_indices",
        "candidate_lengths",
    }
    for rows in args.rows:
        trial = {
            name: value[:rows] if name in row_operands and value is not None else value
            for name, value in arguments.items()
        }
        row_arguments[rows] = trial
        plan = make_plan(
            session,
            indexer.Caps(
                device=device,
                num_q_heads=args.heads,
                max_q_rows=rows,
                max_page_table_width=planned_pages,
                page_size=args.page_size,
                cache_format="mxfp4",
                topk=512,
                max_candidates=candidate_capacity,
            ),
            trial,
            args,
        )
        prepare_mxfp4(
            session,
            plan,
            q=q[:rows],
            keys=keys,
            slots=slots,
            arguments=trial,
            tuning_seed=args.seed + 200003,
        )
        (spec,) = plan.scratch_specs()
        workspaces[rows] = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        plans[rows] = plan

    stored_pages = pool[physical_pages.long()]
    torch.testing.assert_close(
        stored_pages[:, : args.page_size * 64].reshape(-1, 64)[:context],
        key_reference,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        stored_pages[:, args.page_size * 64 :].reshape(-1, 4)[:context],
        key_scales_reference,
        rtol=0,
        atol=0,
    )
    del stored_pages
    torch.testing.assert_close(packed, q_reference, rtol=0, atol=0)
    torch.testing.assert_close(scales, sf_reference, rtol=0, atol=0)

    def bind(rows, visible):
        score_width = candidate_capacity if candidates_mode else visible
        if args.full_score_width and not candidates_mode:
            score_width = args.capacity
        return indexer.bind(
            plans[rows],
            scratch=workspaces[rows],
            score_width=score_width,
            **row_arguments[rows],
        )

    def dense_oracle(query, query_weights, rows, width):
        expected = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
        for begin in range(0, width, 4096):
            end = min(begin + 4096, width)
            expected[:, begin:end] = score_reference(
                query[:rows], decoded_k[begin:end], query_weights[:rows]
            )
        return expected

    def oracle(rows, visible):
        expected = dense_oracle(decoded_q, weights, rows, context)
        logical = positions[None]
        expected.masked_fill_(
            (logical >= lengths[:rows, None]) | (logical >= active[0]), -torch.inf
        )
        if candidates_mode:
            ids = candidate_ids[:rows].long()
            expected = expected.gather(1, ids.clamp_min(0))
            expected.masked_fill_(
                (
                    torch.arange(candidate_capacity, device=device)[None]
                    >= candidate_lengths[:rows, None]
                )
                | (ids < 0),
                -torch.inf,
            )
        elif args.full_score_width:
            full = torch.full(
                (rows, args.capacity), -torch.inf, dtype=expected.dtype, device=device
            )
            full[:, :context] = expected
            expected = full
        else:
            expected = expected[:, :visible]
        return expected

    for rows in args.rows:
        indexer.run(bind(rows, context))
    torch.cuda.synchronize(device)
    records = []
    session.freeze()
    graphs = {}
    try:
        for visible in args.widths:
            active.fill_(visible)
            lengths.fill_(visible)
            if publisher is not None:
                publish()
                _, _, source_decoded = quantized_reference(source_q)
                source_expected = dense_oracle(
                    source_decoded, source_weights, rows_capacity, context
                )
                source_expected[:, visible:] = -torch.inf
                check_selection(source_expected, source_output, source_values)
                if candidate_capacity == 16384:
                    from benchmarks.experimental.b12x.benchmark_dsa_indexer_profiles import (
                        _assert_published_blocks,
                    )

                    _assert_published_blocks(
                        source_expected[:, :visible],
                        candidate_ids,
                        candidate_lengths,
                        visible,
                    )
            for rows in args.rows:
                plan, scratch = plans[rows], workspaces[rows]
                binding = bind(rows, visible)
                indexer.quantize_q_mxfp4(
                    plan, q[:rows], q_mxfp4=packed[:rows], q_scales=scales[:rows]
                )
                expected = oracle(rows, visible)
                indexer.run(binding)
                selected_candidates = (
                    None if candidate_ids is None else candidate_ids[:rows]
                )
                check_selection(
                    expected,
                    output_indices[:rows],
                    output_scores[:rows],
                    selected_candidates,
                )

                def fused():
                    indexer.quantize_q_mxfp4(
                        plan, q[:rows], q_mxfp4=packed[:rows], q_scales=scales[:rows]
                    )
                    indexer.run(binding)

                calls = {"fused": fused}
                graphs = {}
                for name, call in calls.items():
                    with kernel_resolution_guard("MXFP4 complete benchmark"):
                        call()
                    check_selection(
                        expected,
                        output_indices[:rows],
                        output_scores[:rows],
                        selected_candidates,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        call()
                    graphs[name] = graph
                guards = []
                for check in range(args.checks):
                    check_generator = torch.Generator(device=device).manual_seed(
                        args.seed + 100003 + check
                    )
                    q.copy_(
                        (
                            torch.randn(
                                q.shape, generator=check_generator, device=device
                            )
                            / 4
                        ).bfloat16()
                    )
                    weights.copy_(
                        (
                            torch.randn(
                                weights.shape, generator=check_generator, device=device
                            )
                            / 32
                        ).bfloat16()
                    )
                    q_reference, sf_reference, decoded_q = quantized_reference(q)
                    bounds = (visible, max(0, visible - 1), min(31, visible), 0)
                    bound = bounds[check % len(bounds)]
                    active.fill_(bound)
                    lengths.fill_(bound)
                    if rows > 1:
                        lengths[0] = 0
                        lengths[1] = min(31, bound)
                    if publisher is not None:
                        source_q.copy_(
                            (
                                torch.randn(
                                    source_q.shape,
                                    generator=check_generator,
                                    device=device,
                                )
                                / 4
                            ).bfloat16()
                        )
                        publish()
                        candidate_lengths[rows - 1] = min(
                            int(candidate_lengths[rows - 1]), 17
                        )
                    expected_mutable = oracle(rows, visible)
                    for name, graph in graphs.items():
                        scratch.fill_(0xA5)
                        output_indices.fill_(-42)
                        output_scores.fill_(torch.nan)
                        packed.fill_(0xA5)
                        scales.fill_(0xA5)
                        torch.cuda.synchronize(device)
                        before = torch.cuda.memory_stats(device)
                        graph.replay()
                        graph.replay()
                        torch.cuda.synchronize(device)
                        after = torch.cuda.memory_stats(device)
                        for counter in (
                            "allocation.all.allocated",
                            "allocation.all.freed",
                            "allocated_bytes.all.allocated",
                            "allocated_bytes.all.freed",
                        ):
                            assert before[counter] == after[counter], (name, counter)
                        torch.testing.assert_close(
                            packed[:rows], q_reference[:rows], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            scales[:rows], sf_reference[:rows], rtol=0, atol=0
                        )
                        assert bool(output_indices[rows:].eq(-42).all())
                        assert bool(output_scores[rows:].isnan().all())
                        assert bool(packed[rows:].eq(0xA5).all())
                        assert bool(scales[rows:].eq(0xA5).all())
                        check_selection(
                            expected_mutable,
                            output_indices[:rows],
                            output_scores[:rows],
                            selected_candidates,
                        )
                    guards.append(
                        {"seed": args.seed + 100003 + check, "active_width": bound}
                    )
                active.fill_(visible)
                lengths.fill_(visible)
                q.copy_(timing_q)
                weights.copy_(timing_weights)
                q_reference, sf_reference, decoded_q = quantized_reference(q)
                if publisher is not None:
                    source_q.copy_(timing_source_q)
                    publish()
                expected = oracle(rows, visible)
                for graph in graphs.values():
                    graph.replay()
                    torch.cuda.synchronize(device)
                    check_selection(
                        expected,
                        output_indices[:rows],
                        output_scores[:rows],
                        selected_candidates,
                    )
                allocation_before_warmup = torch.cuda.memory_allocated(device)
                addresses = (
                    output_indices.data_ptr(),
                    scratch.data_ptr(),
                    pool.data_ptr(),
                )
                for graph in graphs.values():
                    for _ in range(args.warmup):
                        graph.replay()
                torch.cuda.synchronize(device)
                flush = make_l2_flush_fn(args.cold_l2)
                allocated = torch.cuda.memory_allocated(device)
                samples = {name: [] for name in graphs}
                for sample in range(args.samples):
                    for name in (
                        list(graphs) if sample % 2 == 0 else list(reversed(graphs))
                    ):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        if flush is not None:
                            flush()
                        begin.record()
                        for _ in range(args.replays):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(
                            begin.elapsed_time(end) * 1000 / args.replays
                        )
                allocation_after_timing = torch.cuda.memory_allocated(device)
                assert allocation_after_timing == allocated, (
                    allocation_before_warmup,
                    allocated,
                    allocation_after_timing,
                )
                assert addresses == (
                    output_indices.data_ptr(),
                    scratch.data_ptr(),
                    pool.data_ptr(),
                )
                record = {
                    "mode": "candidates" if candidates_mode else "dense",
                    "rows": rows,
                    "heads": args.heads,
                    "indexed_keys": visible,
                    "logical_context": visible * args.compression_ratio,
                    "profile_form": args.profile_form,
                    "operation": "candidate_consumer"
                    if candidates_mode
                    else "plain_top512",
                    "source_publication_timed": False,
                    "consumer_width": candidate_capacity,
                    "visible": visible,
                    "input_seed": args.seed,
                    "tuning_query_seed": args.seed + 200003,
                    "source_tuning_query_seed": args.seed + 200004
                    if candidates_mode
                    else None,
                    "mutable_checks": guards,
                    "cache_policy": "cold_each_replay"
                    if args.cold_l2
                    else "warm_graph",
                    "score_kind": "fused_mxfp4",
                    "configuration": plan.prepared.selection.config.to_dict(),
                    "preparation_cache_policy": "warm",
                    "score_width": candidate_capacity
                    if candidates_mode
                    else (args.capacity if args.full_score_width else visible),
                    "allocation_bytes": {
                        "before_warmup": allocation_before_warmup,
                        "after_warmup": allocated,
                        "after_timing": allocation_after_timing,
                    },
                    "planned_capacity": args.capacity,
                    "scratch_bytes": scratch.numel(),
                    "high_pid_byte_offset": base_pid * args.page_stride,
                    "samples_us": samples,
                    "median_us": {
                        name: median(values) for name, values in samples.items()
                    },
                    "correctness": "Exact quantization and BF16-stage selected-score multiset; sorted logical IDs and candidate association; captured run mutates inputs and bounds under captured replay with poisoned scratch, frozen resolution, stable addresses, and unchanged allocation counters.",
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ("mode", "rows", "visible", "median_us")
                        }
                    ),
                    flush=True,
                )
                for graph in graphs.values():
                    graph.reset()
    finally:
        for graph in graphs.values():
            graph.reset()
        session.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", default="1,8,128")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=410831)
    parser.add_argument("--checks", type=int, default=4)
    parser.add_argument(
        "--preparation", choices=("autotune", "heuristic"), default="autotune"
    )
    parser.add_argument("--fused-ctas-per-row", type=int)
    parser.add_argument("--arms", choices=("fused",), default="fused")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--profile-form", choices=("c2-dense", "c1-reindex"))
    parser.add_argument("--compression-ratio", type=int, default=1)
    parser.add_argument("--widths", default="32768")
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument(
        "--full-score-width",
        action="store_true",
        help="Replay dense scoring at its full reserved width with only a live prefix populated",
    )
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--page-stride", type=int, default=8704)
    parser.add_argument("--candidate-capacity", type=int, default=16384)
    parser.add_argument("--modes", default="dense,candidates")
    parser.add_argument(
        "--high-pid", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Read-evict L2 before each timed graph; requires --replays 1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fused_ctas_per_row is not None and args.fused_ctas_per_row <= 0:
        parser.error("--fused-ctas-per-row must be positive")
    if args.profile_form:
        profile = load_attention_profile(
            "deepseek-v4.1-flash", args.model_path, block_size=128
        )
        form = next(
            form for form in profile.indexer_forms if form.name == args.profile_form
        )
        if args.heads != form.heads or args.page_size != form.page_size:
            parser.error("heads/page-size do not match the selected model profile")
        if args.profile_form == "c2-dense" and args.modes != "dense":
            parser.error("c2-dense requires --modes dense")
        if args.profile_form == "c1-reindex" and (
            args.modes != "candidates" or args.candidate_capacity != form.max_candidates
        ):
            parser.error(
                "c1-reindex requires --modes candidates --candidate-capacity 16384"
            )
        args.compression_ratio = form.ratio
    args.rows = [int(value) for value in args.rows.split(",")]
    args.widths = [int(value) for value in args.widths.split(",")]
    args.modes = args.modes.split(",")
    if args.cold_l2 and args.replays != 1:
        parser.error("--cold-l2 requires --replays 1 so every measured replay is cold")
    if (
        min(
            *args.rows,
            *args.widths,
            args.samples,
            args.replays,
            args.warmup,
            args.checks,
        )
        <= 0
        or max(args.widths) > args.capacity
    ):
        parser.error(
            "counts must be positive and visible widths must fit planned capacity"
        )
    if any(mode not in ("dense", "candidates") for mode in args.modes):
        parser.error("modes must be dense and/or candidates")
    if args.full_score_width and "candidates" in args.modes:
        parser.error("--full-score-width applies only to dense source scoring")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {
        "command": [sys.executable, *sys.argv],
        "arguments": {
            **vars(args),
            "output": str(args.output),
            "model_path": str(args.model_path) if args.model_path else None,
        },
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl")
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "CUTE_", "B12X_", "OMP_"))
        },
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "scope": "Prepared MXFP4 component calls include query quantization, scoring, selection, and logical-position ordering. The prepared production Plan selects and retains the fused configuration before capture. Timings exclude source candidate publication, TP collective, and whole-model execution.",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        result["results"].extend(run_mode(args, mode == "candidates"))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
