"""Prepared signed bilinear block selection over a unit-scale FP8 paged cache."""

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x.preparation import Knob, MemoryRequirements, Plan, TuningContract
from b12x.preparation.types import plan_from_handle, require_prepared


@triton.jit
def _score(
    Q,
    K,
    Pages,
    Lengths,
    Cu,
    Positions,
    Scores,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    BQ: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    tile, request, hb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head, block_slot = hb // BLOCK_SLOTS, hb % BLOCK_SLOTS
    begin, end = tl.load(Cu + request), tl.load(Cu + request + 1)
    if begin + tile * BQ < end:
        rows = begin + tile * BQ + tl.arange(0, BQ)
        live = (rows < end) & (rows < ROWS)
        pos = tl.load(Positions + rows, live, other=-1)
        dim = tl.arange(0, 128)
        length = tl.load(Lengths + request)
        query = tl.load(
            Q + rows[:, None] * Q_ROW + head * Q_HEAD + dim[None, :],
            live[:, None],
            other=0.0,
        )
        extent = tl.minimum(length, tl.max(pos, 0) + 1)
        for block in range(block_slot, tl.cdiv(extent, 128), BLOCK_SLOTS):
            tokens = block * 128 + tl.arange(0, 128)
            pages = tl.load(
                Pages + request * TABLE_WIDTH + tokens // PAGE_SIZE,
                tokens < length,
                other=0,
            )
            valid = (tokens < length) & (pages >= 0)
            keys = tl.load(
                K
                + pages[None, :].to(tl.int64) * K_PAGE
                + (tokens[None, :] % PAGE_SIZE) * K_TOKEN
                + dim[:, None],
                valid[None, :],
                other=0.0,
            )
            scores = tl.dot(query, keys, out_dtype=tl.float32)
            scores = tl.where(
                live[:, None] & valid[None, :] & (tokens[None, :] <= pos[:, None]),
                scores,
                -float("inf"),
            )
            maximum = tl.max(scores, axis=1)
            tl.store(Scores + (head * ROWS + rows) * BLOCKS + block, maximum, live)


@triton.jit
def _select(
    Scores,
    Positions,
    Cu,
    Output,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    total = tl.load(Cu + BATCH)
    pos = tl.load(Positions + row)
    live = (row < total) & (pos >= 0)
    visible = tl.where(live, tl.minimum(BLOCKS, pos // 128 + 1), 0)
    SEGMENT: tl.constexpr = max(16, min(WIDTH, 128))
    retained = tl.full((16,), 0, tl.uint64)
    for begin in range(0, visible, SEGMENT):
        blocks = begin + tl.arange(0, SEGMENT)
        scores = tl.load(
            Scores + (head * ROWS + row) * BLOCKS + blocks,
            blocks < visible,
            other=-float("inf"),
        )
        scores = tl.where(blocks == pos // 128, float("inf"), scores)
        scores = tl.where(scores == 0.0, 0.0, scores)
        bits = scores.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        # Unsigned keys order equal scores by the lowest block id.
        keys = (ordered.to(tl.uint64) << 32) | (~blocks.to(tl.uint32)).to(tl.uint64)
        keys = tl.where((blocks < visible) & (scores > -float("inf")), keys, 0)
        local = tl.topk(keys, 16)
        if begin == 0:
            retained = local
        else:
            retained = tl.topk(tl.cat(retained, local, can_reorder=True), 16)
    chosen = tl.where(
        retained != 0,
        (~retained.to(tl.uint32)).to(tl.int32),
        2147483647,
    )
    chosen = tl.sort(chosen, descending=False)
    tl.store(
        Output + (head * ROWS + row) * 16 + tl.arange(0, 16),
        tl.where(chosen == 2147483647, -1, chosen),
    )


@dataclass(frozen=True, kw_only=True)
class MSACaps:
    device: object
    max_q_rows: int
    max_batch: int
    max_page_table_width: int
    num_index_heads: int = 1
    page_size: int = 128
    max_q_per_request: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(
            self,
            "max_q_per_request",
            self.max_q_rows
            if self.max_q_per_request is None
            else self.max_q_per_request,
        )
        if (
            min(
                self.max_q_rows,
                self.max_batch,
                self.max_page_table_width,
                self.num_index_heads,
                self.max_q_per_request,
            )
            <= 0
        ):
            raise ValueError("MSA capacities must be positive")
        if self.max_q_per_request > self.max_q_rows:
            raise ValueError(
                "per-request MSA query capacity exceeds total query capacity"
            )
        if self.page_size not in (64, 128):
            raise ValueError("MSA index pages must contain 64 or 128 tokens")


@dataclass(frozen=True)
class Query:
    rows: int
    batch: int
    heads: int
    pages: int
    page_size: int
    cache_stride: tuple
    query_stride: tuple
    position_dtype: str
    cu_dtype: str
    length_dtype: str
    table_dtype: str
    max_q_per_request: int

    @property
    def block_slots(self):
        query_tiles = triton.cdiv(self.max_q_per_request, 16) * self.batch * self.heads
        return min(self.blocks, max(32, triton.cdiv(512, query_tiles)))

    @property
    def blocks(self):
        return triton.cdiv(self.pages * self.page_size, 128)


@dataclass(frozen=True)
class Config:
    warps: int = 4

    @classmethod
    def from_config(cls, payload):
        return cls(**dict(payload))


def _validate(query, config, device):
    if config.warps not in (4, 8):
        raise ValueError("MSA scoring supports four or eight warps")


CONTRACT = TuningContract(
    component_id="attention.msa_indexer",
    query_schema_version=3,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(Query)),
    config_fields=frozenset({"warps"}),
    encode_query=lambda q: {f.name: getattr(q, f.name) for f in fields(Query)},
    encode_config=lambda c: {"warps": c.warps},
    decode_config=Config.from_config,
    validate_query=lambda q, d: None,
    validate_config=_validate,
    default_config=lambda q, d: Config(),
    knobs=(Knob(name="warps", values=(4, 8)),),
    parameters=lambda q, d: {"warps": (4, 8)},
)


@dataclass(frozen=True)
class _Pointer:
    dtype: object

    def data_ptr(self):
        return 16


def _constants(q):
    return (
        q.rows,
        q.heads,
        q.blocks,
        q.page_size,
        q.pages,
        *q.cache_stride[:2],
        *q.query_stride[:2],
        16,
        q.block_slots,
    )


@program_cache(scope="preparation")
def compile_msa(payload, warps, ordinal):
    q = Query(**dict(payload))
    fp8 = _Pointer(torch.float8_e4m3fn)
    f32 = _Pointer(torch.float32)
    integer = lambda dtype: _Pointer(getattr(torch, dtype))
    with torch.cuda.device(ordinal):
        score = _score.warmup(
            fp8,
            fp8,
            integer(q.table_dtype),
            integer(q.length_dtype),
            integer(q.cu_dtype),
            integer(q.position_dtype),
            f32,
            *_constants(q),
            grid=(
                triton.cdiv(q.max_q_per_request, 16),
                q.batch,
                q.heads * q.block_slots,
            ),
            num_warps=warps,
        )
        select = _select.warmup(
            f32,
            integer(q.position_dtype),
            integer(q.cu_dtype),
            _Pointer(torch.int32),
            q.rows,
            q.heads,
            q.blocks,
            q.batch,
            triton.next_power_of_2(q.blocks),
            grid=(q.rows, q.heads),
            num_warps=4,
        )
    return score, select


@dataclass(frozen=True)
class State:
    query: Query
    warps: int
    programs: tuple

    def run(self, query, cache, pages, lengths, cu, positions, output, scratch):
        q = self.query
        scores = scratch.view(torch.float32)[: q.heads * q.rows * q.blocks]
        self.programs[0][
            (triton.cdiv(q.max_q_per_request, 16), q.batch, q.heads * q.block_slots)
        ](query, cache, pages, lengths, cu, positions, scores, *_constants(q))
        self.programs[1][(q.rows, q.heads, 1)](
            scores,
            positions,
            cu,
            output,
            q.rows,
            q.heads,
            q.blocks,
            q.batch,
            triton.next_power_of_2(q.blocks),
        )


def plan_msa(caps, *, query, cache, page_table, cache_lengths, cu_seqlens_q, positions):
    """Declare fixed-address MSA scoring and exact ascending block selection."""
    if query.dtype != torch.float8_e4m3fn or query.shape != (
        caps.max_q_rows,
        caps.num_index_heads,
        128,
    ):
        raise ValueError("MSA query must be unit-scale E4M3 [rows, index_heads, 128]")
    if (
        cache.dtype != torch.float8_e4m3fn
        or cache.ndim != 3
        or cache.shape[1:] != (caps.page_size, 128)
    ):
        raise ValueError("MSA cache must be unit-scale E4M3 [pages, page_size, 128]")
    if cache.stride(-1) != 1 or query.stride(-1) != 1:
        raise ValueError("MSA head dimensions must be contiguous")
    if (
        page_table.shape != (caps.max_batch, caps.max_page_table_width)
        or not page_table.is_contiguous()
    ):
        raise ValueError("MSA page table must match the declared batch and width")
    if (
        cache_lengths.shape != (caps.max_batch,)
        or cu_seqlens_q.shape != (caps.max_batch + 1,)
        or positions.shape != (caps.max_q_rows,)
    ):
        raise ValueError(
            "MSA length, query-start and position metadata must match capacities"
        )
    tensors = query, cache, page_table, cache_lengths, cu_seqlens_q, positions
    if any(t.device != query.device for t in tensors):
        raise ValueError("MSA operands must share one device")
    if any(
        t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
        for t in tensors[2:]
    ):
        raise ValueError("MSA metadata must use contiguous integer tensors")
    q = Query(
        caps.max_q_rows,
        caps.max_batch,
        caps.num_index_heads,
        caps.max_page_table_width,
        caps.page_size,
        tuple(cache.stride()),
        tuple(query.stride()),
        *(
            str(t.dtype).removeprefix("torch.")
            for t in (positions, cu_seqlens_q, cache_lengths, page_table)
        ),
        caps.max_q_per_request,
    )
    payload = CONTRACT.encode_query(q)

    def materialize(selection, device):
        programs = compile_msa(payload, selection.config.warps, device.ordinal)
        load_programs(programs)
        return attach_programs(State(q, selection.config.warps, programs), programs)

    return Plan(
        contract=CONTRACT,
        query=q,
        _device=caps.device,
        _compile_jobs=lambda config, d: (
            CompileJob.create(
                "b12x.attention.dsa_indexer.msa:compile_msa",
                payload,
                config.warps,
                d.ordinal,
            ),
        ),
        _memory_requirements=lambda c, d: MemoryRequirements(
            scratch=(
                scratch_buffer_spec(
                    "msa_indexer",
                    nbytes=q.heads * q.rows * q.blocks * 4,
                    device=caps.device,
                ),
            )
        ),
        _materialize=materialize,
    )


@torch.library.custom_op("b12x::msa_indexer", mutates_args=("output", "scratch"))
def _run(
    query: torch.Tensor,
    cache: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    cu: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "attention.msa_indexer", query.device
    )
    state.run(query, cache, pages, lengths, cu, positions, output, scratch)


@_run.register_fake
def _fake(query, cache, pages, lengths, cu, positions, output, scratch, handle):
    pass


def run_msa(
    plan,
    *,
    query,
    cache,
    page_table,
    cache_lengths,
    cu_seqlens_q,
    positions,
    output,
    scratch,
):
    """Execute prepared MSA selection without allocations or host metadata reads."""
    state = require_prepared(plan, "attention.msa_indexer", query.device)
    q = state.query
    if (
        query.shape != (q.rows, q.heads, 128)
        or query.dtype != torch.float8_e4m3fn
        or tuple(query.stride()) != q.query_stride
    ):
        raise ValueError(
            "MSA query shape, precision and strides must match preparation"
        )
    if (
        cache.ndim != 3
        or cache.shape[1:] != (q.page_size, 128)
        or cache.dtype != torch.float8_e4m3fn
        or tuple(cache.stride()) != q.cache_stride
    ):
        raise ValueError("MSA index cache precision and layout must match preparation")
    metadata = (page_table, cache_lengths, cu_seqlens_q, positions)
    expected = (
        ((q.batch, q.pages), q.table_dtype),
        ((q.batch,), q.length_dtype),
        ((q.batch + 1,), q.cu_dtype),
        ((q.rows,), q.position_dtype),
    )
    for tensor, (shape, dtype) in zip(metadata, expected, strict=True):
        if (
            tensor.shape != shape
            or tensor.dtype != getattr(torch, dtype)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "MSA metadata geometry, integer precision and strides must match preparation"
            )
    if any(t.device != query.device for t in (cache, *metadata, output)):
        raise ValueError("MSA bound tensors must remain on the prepared device")
    if (
        output.shape != (q.heads, q.rows, 16)
        or output.dtype != torch.int32
        or not output.is_contiguous()
    ):
        raise ValueError(
            "MSA selected blocks must be contiguous int32 [heads, rows, 16]"
        )
    storage = scratch_tensor(scratch, plan.scratch_specs(), owner="MSA indexer")
    _run(
        query,
        cache,
        page_table,
        cache_lengths,
        cu_seqlens_q,
        positions,
        output,
        storage,
        plan.handle,
    )
    return output
