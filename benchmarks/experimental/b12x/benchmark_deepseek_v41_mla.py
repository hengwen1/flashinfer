#!/usr/bin/env python3
"""Benchmark prepared DeepSeek V4.1 MLA contracts used by Flare TP4.

Queries, cache payloads, sinks, and selections are synthetic. The benchmark
measures attention alone, including logical-to-physical indexed-page mapping;
it does not measure projections, index selection, cache writes, or serving.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


PRESET = "flare-dsv41-tp4"
FORMS = ("swa", "c2", "c1", "draft-swa")
MODES = ("decode", "verify", "prefill", "draft")


@dataclass(frozen=True)
class Shape:
    batch: int
    width: int
    rows: int

    def __post_init__(self):
        if not 1 <= self.batch <= 8 or not 1 <= self.width <= 8:
            raise ValueError("B and W must be within the Flare capacities B<=8, W<=8")
        if not self.batch <= self.rows <= self.batch * self.width:
            raise ValueError("physical P must satisfy B <= P <= B*W")

    @classmethod
    def parse(cls, value):
        try:
            return cls(*(int(part) for part in value.split("x")))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"invalid BxWxP shape {value!r}: {exc}"
            ) from exc


@dataclass(frozen=True)
class Profile:
    model_config: str
    num_q_heads: int
    target_layers: tuple[int, ...]
    draft_layers: int
    window: int
    indexed_width: int


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    form: str
    batch: int
    query_width: int
    rows: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    page_table_width: int
    num_q_heads: int
    context_length: int
    context_capacity: int
    compute_mode: str = "bf16"
    draft_depth: int = 7
    cache_pages: int = 895489
    prefill_rows: int = 4096

    @property
    def active_cache_pages(self):
        return min(self.cache_pages, 180225)

    @property
    def native_mode(self):
        return "extend" if self.mode == "prefill" else "decode"

    @property
    def max_query_tokens(self):
        return self.query_width

    @property
    def ratio(self):
        return {"c1": 1, "c2": 2}.get(self.form, 0)


def load_profile(path, tp_size=4):
    path = Path(path)
    raw = json.loads(path.read_text())
    text = raw.get("text_config", raw)
    if (
        raw.get("model_type") != "deepseek_v41"
        or text.get("model_type") != "deepseek_v41_text"
    ):
        raise ValueError(
            "model config must describe DeepSeek V4.1 root and text models"
        )
    if tp_size != 4 or text["num_attention_heads"] != 64:
        raise ValueError(
            "Flare TP4 profile requires TP size 4 and 64 global attention heads"
        )
    if (
        text["head_dim"] != 512
        or text["sliding_window"] != 128
        or text["index_topk"] != 512
    ):
        raise ValueError(
            "Flare profile requires head_dim=512, sliding_window=128, index_topk=512"
        )
    layers = tuple(text["compress_ratios"][: text["num_hidden_layers"]])
    if (
        text["num_hidden_layers"] != 40
        or layers != (0, 0) + (2,) * 18 + (1,) * 20
        or int(text.get("num_nextn_predict_layers", 0)) != 3
    ):
        raise ValueError(
            "Flare TP4 storage requires 40 target layers (2 SWA, 18 C2, 20 C1) and 3 draft layers"
        )
    return Profile(
        str(path.resolve()),
        text["num_attention_heads"] // tp_size,
        layers,
        int(text.get("num_nextn_predict_layers", 0)),
        128,
        512,
    )


def _csv(value):
    return tuple(part for part in value.split(",") if part)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--context-capacity", type=int, default=1048576)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--cache-pages",
        type=int,
        default=895489,
        help="indexed descriptor rows; changing this overrides the package geometry",
    )
    parser.add_argument("--forms", type=_csv, default=FORMS)
    parser.add_argument("--modes", type=_csv, default=MODES)
    parser.add_argument(
        "--decode-batches", type=lambda s: tuple(map(int, _csv(s))), default=(1, 4)
    )
    parser.add_argument(
        "--verify-shapes",
        type=lambda s: tuple(Shape.parse(v) for v in _csv(s)),
        default=tuple(
            Shape.parse(v) for v in ("1x8x2", "1x8x4", "1x8x5", "1x8x8", "8x8x64")
        ),
    )
    parser.add_argument(
        "--draft-batches", type=lambda s: tuple(map(int, _csv(s))), default=(1, 4)
    )
    parser.add_argument("--draft-depth", type=int, default=7)
    parser.add_argument("--prefill-rows", type=int, default=4096)
    parser.add_argument("--compute-mode", choices=("bf16",), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--compile-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction bytes; zero selects twice the detected L2 capacity",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print resolved contracts without importing GPU libraries",
    )
    return parser.parse_args(argv)


def build_cases(args):
    profile = load_profile(args.model_config, args.tp_size)
    if (
        not args.forms
        or set(args.forms) - set(FORMS)
        or not args.modes
        or set(args.modes) - set(MODES)
    ):
        raise ValueError("forms/modes must be nonempty lists of supported names")
    if args.context_capacity <= 0 or args.context_capacity % 128:
        raise ValueError("context capacity must be positive and divisible by 128")
    if not 0 < args.context_length <= args.context_capacity:
        raise ValueError("context length must fit the context capacity")
    if args.cache_pages < 2:
        raise ValueError(
            "cache pages must include a null page and at least one physical page"
        )
    if not 1 <= args.draft_depth <= 7 or args.prefill_rows < 1:
        raise ValueError(
            "draft depth must be within 1..7 and prefill rows must be positive"
        )
    if args.warmup < 1 or args.replays < 1 or args.compile_workers < 0:
        raise ValueError(
            "warmup/replays must be positive and compile workers nonnegative"
        )
    if args.compute_mode != "bf16":
        raise ValueError("Flare V4.1 profile requires BF16 attention arithmetic")
    if args.l2_flush_bytes < 0:
        raise ValueError("L2 flush bytes must be nonnegative")
    cases = []

    def add(mode, form, batch, width, rows):
        if form not in args.forms:
            return
        if form == "draft-swa" and profile.draft_layers == 0:
            raise ValueError("draft-swa requires next-token prediction layers")
        ratio = {"c1": 1, "c2": 2}.get(form, 0)
        if form != "draft-swa" and ratio not in profile.target_layers:
            raise ValueError(f"model has no {form} target layers")
        if mode == "decode" or mode == "verify" and width == 1:
            page, swa_width = 128, 128
        elif mode == "draft":
            page = swa_width = 128 + args.draft_depth
        else:
            page, swa_width = 128 + width, 128
        query_extent = args.prefill_rows if mode == "prefill" else width
        if args.context_length + query_extent > args.context_capacity:
            raise ValueError(
                "context plus query extent exceeds the prepared context capacity"
            )
        cases.append(
            Case(
                f"{mode}-{form}-b{batch}-w{width}-p{rows}",
                mode,
                form,
                batch,
                width,
                rows,
                swa_width,
                page,
                512 if ratio else 0,
                128 // ratio if ratio else 64,
                args.context_capacity // 128 if ratio else 1,
                profile.num_q_heads,
                args.context_length,
                args.context_capacity,
                args.compute_mode,
                args.draft_depth,
                args.cache_pages,
                args.prefill_rows,
            )
        )

    for mode in args.modes:
        before = len(cases)
        if mode == "decode":
            for batch in args.decode_batches:
                Shape(batch, 1, batch)
                for form in FORMS[:3]:
                    add(mode, form, batch, 1, batch)
        elif mode == "verify":
            for shape in args.verify_shapes:
                for form in FORMS[:3]:
                    add(mode, form, shape.batch, shape.width, shape.rows)
        elif mode == "draft":
            for batch in args.draft_batches:
                Shape(batch, 1, batch)
                add(
                    mode, "draft-swa", batch, args.draft_depth, batch * args.draft_depth
                )
        else:
            for form in FORMS[:3]:
                rows = (
                    min(128, args.prefill_rows) if form == "c1" else args.prefill_rows
                )
                add(mode, form, 1, rows, rows)
        if len(cases) == before:
            raise ValueError(f"mode {mode} has no selected compatible forms/shapes")
    return cases


@dataclass
class Inputs:
    q: object
    out: object
    swa_cache: object
    indexed_cache: object
    swa_indices: object
    swa_lengths: object
    indexed_indices: object
    indexed_lengths: object
    page_table: object
    sink: object
    table_storage: object
    indexed_physical_pages: object
    swa_physical_pages: object
    query_seed: object

    def binding_args(self):
        return dict(
            q=self.q,
            swa_indices=self.swa_indices,
            swa_lengths=self.swa_lengths,
            indexed_indices=self.indexed_indices,
            indexed_lengths=self.indexed_lengths,
            indexed_page_table=self.page_table,
        )

    def run_args(self):
        return dict(
            swa_k_cache=self.swa_cache,
            indexed_k_cache=self.indexed_cache,
            attn_sink=self.sink,
            sm_scale=512**-0.5,
            out=self.out,
        )

    def storage(self):
        def describe(tensor):
            if tensor is None:
                return None
            return dict(
                shape=list(tensor.shape),
                strides=list(tensor.stride()),
                dtype=str(tensor.dtype),
                allocated_bytes=tensor.untyped_storage().nbytes(),
            )

        return {
            name: describe(getattr(self, name))
            for name in (
                "q",
                "out",
                "swa_cache",
                "indexed_cache",
                "swa_indices",
                "swa_lengths",
                "indexed_indices",
                "indexed_lengths",
                "page_table",
                "sink",
            )
        }


def make_inputs(case, device, seed=4100):
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        pack_deepseek_v41_cache_reference,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    q = (
        torch.randn(
            (case.rows, case.num_q_heads, 512), device=device, generator=generator
        )
        * 0.2
    ).bfloat16()
    q[:, 0, :448].zero_()
    q[:, 0, 448:].mul_(8)
    scales = torch.linspace(0.03, 0.7, 32, device=device).repeat_interleave(16)
    sink = torch.linspace(
        -0.25, 0.25, case.num_q_heads, dtype=torch.float32, device=device
    )
    if case.mode in ("prefill", "draft"):
        count = case.rows // case.batch
        requests = torch.arange(case.batch, device=device).repeat_interleave(count)
        local = torch.arange(count, device=device).repeat(case.batch)
    else:
        remaining = case.rows
        counts = []
        for lane in range(case.batch):
            count = min(case.query_width, remaining - (case.batch - lane - 1))
            counts.append(count)
            remaining -= count
        requests = torch.tensor(
            [lane for lane, count in enumerate(counts) for _ in range(count)],
            dtype=torch.long,
            device=device,
        )
        local = torch.tensor(
            [position for count in counts for position in range(count)],
            dtype=torch.long,
            device=device,
        )
    start = case.context_length
    if case.mode == "prefill" and case.form == "c1":
        start += case.prefill_rows - case.rows
    positions = local + start
    offsets = torch.arange(case.swa_width, device=device)[None]
    decode = case.mode == "decode" or case.mode == "verify" and case.query_width == 1
    if decode:
        swa_cache = torch.empty_strided(
            (129, 128 * 528), (3043840, 1), dtype=torch.uint8, device=device
        )
        swa_physical = torch.arange(1, case.batch + 1, device=device)
    else:
        swa_cache = torch.empty(
            (case.batch, case.swa_page_size * 528), dtype=torch.uint8, device=device
        )
        swa_physical = torch.arange(case.batch, device=device)
    values = (
        torch.randn(
            (case.batch * case.swa_page_size, 512), device=device, generator=generator
        )
        * scales
    ).bfloat16()
    swa_cache[swa_physical] = pack_deepseek_v41_cache_reference(
        values, page_size=case.swa_page_size, cache_kind="swa"
    )
    if decode:
        swa_cache[0].fill_(127)
    if case.mode == "draft":
        context = min(start, 128)
        lengths = torch.full(
            (case.rows,), context + case.draft_depth, device=device, dtype=torch.int32
        )
        indices = torch.where(offsets < context, offsets, 128 + offsets - context)
        indices = indices + requests[:, None] * case.swa_page_size
    else:
        left = (positions - 127).clamp_min(0)
        columns = left[:, None] + offsets
        lengths = (positions + 1).clamp(max=128).int()
        if decode:
            indices = (requests[:, None] + 1) * 128 + columns.remainder(128)
        else:
            indices = torch.where(
                columns < start, columns.remainder(128), 128 + columns - start
            )
            indices = indices + requests[:, None] * case.swa_page_size
    indices = torch.where(offsets < lengths[:, None], indices, -1).int().contiguous()
    indexed_cache = indexed_indices = indexed_lengths = page_table = table = (
        physical
    ) = None
    if case.ratio:
        visible = torch.div(positions + 1, case.ratio, rounding_mode="floor")
        indexed_lengths = visible.clamp(max=case.indexed_width).int()
        columns = torch.arange(case.indexed_width, device=device)[None]
        logical = (
            columns + torch.arange(case.rows, device=device)[:, None] * 17
        ) % visible[:, None].clamp_min(1)
        valid = columns < indexed_lengths[:, None]
        indexed_indices = torch.where(valid, logical, -1).int().contiguous()
        table = torch.full(
            (case.batch, case.page_table_width), -1, device=device, dtype=torch.int32
        )
        touched = torch.unique(
            (
                requests[:, None] * case.page_table_width
                + logical // case.indexed_page_size
            )[valid]
        )
        count = touched.numel()
        if count + 1 > case.active_cache_pages:
            raise ValueError(
                f"indexed fixture needs {count + 1} physical pages, has {case.active_cache_pages}"
            )
        physical = (
            torch.randperm(count, generator=generator, device=device)
            + case.active_cache_pages
            - count
        )
        table.view(-1)[touched] = physical.int()
        page_table = (
            table.expand(case.rows, -1)
            if case.batch == 1
            else table[requests].contiguous()
        )
        indexed_cache = torch.empty(
            (case.cache_pages, case.indexed_page_size * 288),
            dtype=torch.uint8,
            device=device,
        )
        indexed_cache[0].fill_(127)
        values = (
            torch.randn(
                (count * case.indexed_page_size, 512),
                device=device,
                generator=generator,
            )
            * scales.flip(0)
            * 3
        ).bfloat16()
        indexed_cache[physical] = pack_deepseek_v41_cache_reference(
            values, page_size=case.indexed_page_size, cache_kind="indexed"
        )
    return Inputs(
        q,
        torch.empty_like(q),
        swa_cache,
        indexed_cache,
        indices,
        lengths,
        indexed_indices,
        indexed_lengths,
        page_table,
        sink,
        table,
        physical,
        swa_physical,
        q.clone(),
    )


@dataclass
class PreparedCase:
    plan: object
    binding: object
    scratch: object
    inputs: Inputs

    def run(self):
        from b12x.attention import compressed_sparse_mla as mla

        return mla.run(plan=self.plan, binding=self.binding, **self.inputs.run_args())


def prepare_case(case, inputs, session):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.preparation import PreparedCall

    caps = mla.Caps(
        device=inputs.q.device,
        num_q_heads=case.num_q_heads,
        max_q_rows=case.rows,
        max_width=case.swa_width + case.indexed_width,
        swa_width=case.swa_width,
        indexed_width=case.indexed_width,
        max_page_table_width=case.page_table_width,
        swa_page_size=case.swa_page_size,
        indexed_page_size=case.indexed_page_size,
        cache_format="deepseek_v41",
        compute_mode=case.compute_mode,
        mode=case.native_mode,
        decode_row_capacity=case.rows if case.native_mode == "decode" else None,
        use_cuda_graph=True,
    )
    plan = mla.plan(
        caps,
        invocation=mla.invocation_from_tensors(
            q=inputs.q,
            swa_k_cache=inputs.swa_cache,
            indexed_k_cache=inputs.indexed_cache,
            attn_sink=inputs.sink,
            out=inputs.out,
        ),
    )

    def candidate(state, *, benchmark):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=inputs.q.device)
        output = torch.empty_like(inputs.out)
        binding = state.bind_for_preparation(scratch=scratch, **inputs.binding_args())
        arguments = dict(inputs.run_args(), out=output)
        return PreparedCall(
            run=lambda: state.run(binding, **arguments),
            produce=lambda: inputs.q.copy_(inputs.query_seed),
            output=output if benchmark else None,
            owners=(inputs, scratch, binding, output),
        )

    def prepare(state):
        return candidate(state, benchmark=False)

    def benchmark(state):
        return candidate(state, benchmark=True)

    session.prepare(
        (plan.request(name=case.name, prepare_call=prepare, benchmark_call=benchmark),)
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=inputs.q.device)
    return PreparedCase(
        plan, mla.bind(plan, scratch=scratch, **inputs.binding_args()), scratch, inputs
    )


def reference(inputs, case):
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        compressed_sparse_mla_reference,
    )

    physical = None
    if inputs.indexed_indices is not None:
        logical = inputs.indexed_indices.long().clamp_min(0)
        pages = inputs.page_table.long().gather(1, logical // case.indexed_page_size)
        physical = pages * case.indexed_page_size + logical % case.indexed_page_size
        physical = torch.where(inputs.indexed_indices >= 0, physical, -1).int()
    return compressed_sparse_mla_reference(
        inputs.q,
        inputs.swa_cache,
        inputs.swa_indices,
        inputs.swa_lengths,
        sm_scale=512**-0.5,
        attn_sink=inputs.sink,
        extra_k_cache=inputs.indexed_cache,
        extra_indices=physical,
        extra_topk_lengths=inputs.indexed_lengths,
        swa_page_size=case.swa_page_size,
        extra_page_size=case.indexed_page_size if case.ratio else None,
        cache_format="deepseek_v41",
    )


def check_reference(prepared, inputs, case, *, run=True):
    import torch

    if run:
        prepared.run()
    expected = reference(inputs, case)
    torch.testing.assert_close(inputs.out, expected, atol=0.035, rtol=0.035)
    difference = inputs.out.float() - expected.float()
    expected_rms = expected.float().square().mean().sqrt().item()
    rmse = difference.square().mean().sqrt().item()
    cosine = None
    if expected_rms:
        cosine = torch.nn.functional.cosine_similarity(
            inputs.out.float().reshape(-1), expected.float().reshape(-1), dim=0
        ).item()
        if not math.isfinite(cosine) or cosine < 0.995:
            raise RuntimeError(f"compressed MLA cosine {cosine} is below 0.995")
    elif not bool(inputs.out.eq(0).all()):
        raise RuntimeError("empty attention reference requires exact zero output")
    return dict(
        max_abs=difference.abs().max().item(),
        rmse=rmse,
        reference_rms=expected_rms,
        cosine=cosine,
        relative_rmse=rmse / expected_rms if expected_rms else 0.0,
    )


def validate_replay(prepared, inputs, case, graph):
    """Exercise runtime lengths and physical page remapping on the captured graph."""
    import torch

    mutable_table = inputs.table_storage if case.batch == 1 else inputs.page_table
    metadata = [
        inputs.swa_indices,
        inputs.swa_lengths,
        inputs.indexed_lengths,
        mutable_table,
    ]
    saved = [(tensor, tensor.clone()) for tensor in metadata if tensor is not None]
    caches = [(inputs.swa_cache, inputs.swa_physical_pages)]
    if inputs.indexed_cache is not None:
        caches.append((inputs.indexed_cache, inputs.indexed_physical_pages))
    snapshots = [(cache, pages, cache[pages].clone()) for cache, pages in caches]
    pointers = (inputs.q.data_ptr(), inputs.out.data_ptr(), prepared.scratch.data_ptr())
    try:
        row = torch.arange(case.rows, device=inputs.q.device)
        inputs.swa_lengths.copy_(
            torch.where(
                row % 3 == 0, 0, torch.where(row % 3 == 1, 1, inputs.swa_lengths)
            )
        )
        if inputs.indexed_lengths is not None:
            inputs.indexed_lengths.copy_(
                torch.where(
                    row % 3 == 0,
                    0,
                    torch.where(row % 3 == 1, 1, inputs.indexed_lengths),
                )
            )
            physical = inputs.indexed_physical_pages.sort().values
            valid = mutable_table >= 0
            if physical.numel() > 1:
                positions = torch.searchsorted(physical, mutable_table[valid].long())
                mutable_table[valid] = physical.roll(1)[positions].int()
        inputs.swa_indices[0].fill_(0)
        inputs.out.fill_(float("nan"))
        torch.cuda.synchronize(inputs.q.device)
        allocated = torch.cuda.memory_stats(inputs.q.device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(inputs.q.device)
        if (
            torch.cuda.memory_stats(inputs.q.device)["allocation.all.allocated"]
            != allocated
        ):
            raise RuntimeError("captured attention replay allocated CUDA storage")
        check_reference(prepared, inputs, case, run=False)
        if not bool(inputs.out[0].eq(0).all()):
            raise RuntimeError("inactive query row did not produce zero output")
        if pointers != (
            inputs.q.data_ptr(),
            inputs.out.data_ptr(),
            prepared.scratch.data_ptr(),
        ):
            raise RuntimeError("captured attention replay changed bound storage")
        for cache, pages, expected in snapshots:
            if not torch.equal(cache[pages], expected):
                raise RuntimeError("attention replay modified packed cache bytes")
    finally:
        for tensor, original in saved:
            tensor.copy_(original)
    graph.replay()
    torch.cuda.synchronize(inputs.q.device)
    check_reference(prepared, inputs, case, run=False)


def benchmark_case(case, args):
    import torch
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.testing.benchmark import measure_calls
    from benchmarks.experimental.b12x.common import (
        capture_cuda_graph,
        make_l2_flush_fn,
        resolve_l2_flush_bytes,
    )

    device = torch.device("cuda", args.device)
    inputs = make_inputs(case, device, args.seed)
    storage = inputs.storage()
    flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0
    eviction = make_l2_flush_fn(args.flush_l2, bytes_hint=flush_bytes)
    print(
        json.dumps(
            dict(
                case=case_contract(case),
                storage=storage,
                active_cache_pages=case.active_cache_pages,
                touched_indexed_pages=(
                    0
                    if inputs.indexed_physical_pages is None
                    else inputs.indexed_physical_pages.numel()
                ),
                storage_scope="Ordinary PyTorch backing for the declared view; not Flare's VMM residency.",
            )
        ),
        flush=True,
    )
    with PreparationSession(
        device=device, autotune=True, compile_workers=args.compile_workers
    ) as session:
        prepared = prepare_case(case, inputs, session)
        numerical = check_reference(prepared, inputs, case)
        session.freeze()
        graph = capture_cuda_graph(prepared.run, warmup=args.warmup)
        try:
            validate_replay(prepared, inputs, case, graph)
            measured = measure_calls(
                {
                    "attention": PreparedCall(
                        run=graph.replay,
                        produce=lambda: None,
                        output=inputs.out,
                        owners=(graph, prepared, inputs),
                    )
                },
                device=device,
                warmup=args.warmup,
                samples=args.replays,
                eviction=eviction if eviction is not None else lambda: None,
            )
            raw = measured.raw_samples("attention")
            check_reference(prepared, inputs, case, run=False)
            config = asdict(prepared.plan.selection.config)
            return dict(
                case=case_contract(case),
                storage=storage,
                cache_policy="L2 eviction outside timed replay"
                if args.flush_l2
                else "fixed-input replay without L2 eviction",
                l2_flush_bytes=flush_bytes,
                selected_config=config,
                selection_source=prepared.plan.selection.source,
                timing_method=measured.method,
                rounds=measured.rounds,
                median_us=statistics.median(raw),
                p90_us=sorted(raw)[math.ceil(len(raw) * 0.9) - 1],
                raw_samples_us=raw,
                reference=numerical,
            )
        finally:
            torch.cuda.synchronize(device)
            graph.reset()


def main(argv=None):
    args = parse_args(argv)
    try:
        cases = build_cases(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    description = dict(
        benchmark=PRESET,
        contents="Synthetic post-RoPE Q, packed cache values, attention sinks, and logical selections.",
        scope="Prepared attention and indexed-page mapping only; no recorded tensor values or whole-model timing.",
        storage="Kernel-visible package descriptors backed by ordinary PyTorch allocations; Flare uses VMM.",
        cache_locality="High-address compact physical page placement and banded synthetic logical selections; not recorded serving locality.",
        cache_pages_override=args.cache_pages != 895489,
        configuration={
            key: (
                str(value)
                if isinstance(value, Path)
                else [asdict(shape) for shape in value]
                if key == "verify_shapes"
                else value
            )
            for key, value in vars(args).items()
        },
    )
    print(json.dumps(description), flush=True)
    if args.describe:
        for case in cases:
            print(
                json.dumps(dict(contract=case_contract(case))),
                flush=True,
            )
        return 0
    import torch

    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability not in ((12, 0), (12, 1)):
        raise SystemExit(f"SM120/SM121 required, found {capability}")
    print(
        json.dumps(
            dict(device=args.device, gpu=torch.cuda.get_device_name(args.device))
        ),
        flush=True,
    )
    output = args.output_csv.open("x", newline="") if args.output_csv else None
    fields = (
        "case",
        "storage",
        "cache_policy",
        "l2_flush_bytes",
        "median_us",
        "p90_us",
        "selected_config",
        "selection_source",
        "timing_method",
        "rounds",
        "reference",
        "raw_samples_us",
    )
    writer = csv.DictWriter(output, fieldnames=fields) if output else None
    if writer:
        writer.writeheader()
    try:
        for case in cases:
            result = benchmark_case(case, args)
            print(json.dumps(result), flush=True)
            if writer:
                writer.writerow(
                    {
                        key: json.dumps(value)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in result.items()
                    }
                )
                output.flush()
    finally:
        if output:
            output.close()
    return 0


def case_contract(case):
    contract = asdict(case)
    contract["configured_draft_depth"] = contract.pop("draft_depth")
    contract["native_mode"] = case.native_mode
    return contract


def dispatch_preset(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    matches = []
    for index, value in enumerate(arguments):
        if value == "--preset" and index + 1 < len(arguments):
            matches.append((index, 2, arguments[index + 1]))
        elif value.startswith("--preset="):
            matches.append((index, 1, value.split("=", 1)[1]))
    if not any(value == PRESET for _, _, value in matches):
        return None
    if len(matches) != 1:
        raise SystemExit("specify the MLA preset once")
    index, count, _ = matches[0]
    return main(arguments[:index] + arguments[index + count :])


if __name__ == "__main__":
    raise SystemExit(main())
