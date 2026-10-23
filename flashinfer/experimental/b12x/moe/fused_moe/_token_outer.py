"""Prepared NVFP4 MoE with independently scaled input and intermediate rows."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from types import MappingProxyType

import torch
import triton
import triton.language as tl
from triton.runtime.jit import MockTensor

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.quant.nvfp4_csf import decode_nvfp4_csf_pair
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x.preparation import (
    FrozenMapping,
    Knob,
    MemoryRequirements,
    ParameterSpace,
    ParameterBinding,
    Plan,
    TuningContract,
)
from b12x.preparation.types import _CompositePlan

from .execution import RoutingSpec
from ._preparation import (
    _FusedMoeCapacityState,
    _RouteTopKState,
    compile_route_topk,
)
from ._tuning import MoeRouteQuery, ROUTE_TUNING
from .._shared.routing import _route_topk_values
from .._shared.kernels.activations import (
    normalize_swiglu_limit_for_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
)


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["M"])
def _quantize(
    X,
    Q,
    S,
    Outer,
    Valid,
    M,
    K: tl.constexpr,
    B: tl.constexpr,
    MaskRoutes: tl.constexpr = False,
):
    row = tl.program_id(0)
    live = tl.full((), True, tl.int1)
    if MaskRoutes:
        live = tl.load(Valid + row) != 0
    pairs = tl.arange(0, B // 2)
    lo = tl.load(X + row * K + 2 * pairs, (2 * pairs < K) & live, 0).to(tl.float32)
    hi = tl.load(X + row * K + 2 * pairs + 1, (2 * pairs + 1 < K) & live, 0).to(
        tl.float32
    )
    pair_max = tl.maximum(tl.abs(lo), tl.abs(hi))
    row_max = tl.max(pair_max, 0)
    outer = row_max * (1.0 / (448.0 * 6.0))
    encode = tl.where(row_max == 0, 3.4028234663852886e38, _rcp(outer))
    blocks = tl.reshape(pair_max, (B // 16, 8))
    block_max = tl.max(blocks, 1)
    scale = (encode * (block_max * _rcp(6.0))).to(tl.float8e4nv)
    sf = scale.to(tl.float32)
    inverse = tl.where(sf == 0, 0.0, _rcp(sf * _rcp(encode)))
    inverse = tl.reshape(tl.broadcast_to(inverse[:, None], (B // 16, 8)), (B // 2,))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        constraints="=r,f,f",
        args=[lo * inverse, hi * inverse],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(Q + row * (K // 2) + pairs, packed.to(tl.uint8), pairs < K // 2)
    cols = tl.arange(0, B // 16)
    tl.store(S + row * (K // 16) + cols, scale, cols < K // 16)
    tl.store(Outer + row, outer)


@triton.jit
def _clear(Counts, TaskExpert, E: tl.constexpr, Tasks: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Counts + idx, 0, idx < E)
    tl.store(TaskExpert + idx, -1, idx < Tasks)


@triton.jit(do_not_specialize=["R"])
def _pack(
    Ids,
    Counts,
    Rows,
    Valid,
    R,
    E: tl.constexpr,
    Capacity: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    e = tl.load(Ids + r, r < R, -1).to(tl.int32)
    valid = (r < R) & (e >= 0) & (e < E)
    tl.store(Valid + r, valid.to(tl.uint8), r < R)
    slot = tl.atomic_add(Counts + tl.maximum(e, 0), 1, valid, sem="relaxed")
    tl.store(Rows + e * Capacity + slot, r, valid)


@triton.jit
def _route_quant_pack(
    X,
    Q,
    S,
    Outer,
    Logits,
    Selected,
    Ids,
    Weights,
    Correction,
    ImageBias,
    ImageMask,
    LiveCount,
    Counts,
    Rows,
    Valid,
    K: tl.constexpr,
    E: tl.constexpr,
    TopK: tl.constexpr,
    Capacity: tl.constexpr,
    Score: tl.constexpr,
    Renormalize: tl.constexpr,
    Scale: tl.constexpr,
    HasCorrection: tl.constexpr,
    HasImageBias: tl.constexpr,
    HasLiveCount: tl.constexpr,
):
    row = tl.program_id(0)
    live = tl.full((), True, tl.int1)
    if HasLiveCount:
        live = row < tl.load(LiveCount)
    if live:
        selected, ids, weights = _route_topk_values(
            Logits,
            Correction,
            ImageBias,
            ImageMask,
            row.to(tl.int64),
            E,
            1,
            1,
            1,
            Scale,
            E,
            triton.next_power_of_2(E),
            TopK,
            triton.next_power_of_2(TopK),
            Renormalize,
            Score,
            HasCorrection,
            HasImageBias,
        )
    else:
        selected = tl.full((triton.next_power_of_2(TopK),), 0, tl.float32)
        ids = tl.full((triton.next_power_of_2(TopK),), -1, tl.int32)
        weights = tl.full((triton.next_power_of_2(TopK),), 0, tl.float32)
    slots = tl.arange(0, triton.next_power_of_2(TopK))
    routes = row * TopK + slots
    mask = slots < TopK
    tl.store(Selected + routes, selected, mask)
    tl.store(Ids + routes, ids, mask)
    tl.store(Weights + routes, weights, mask)
    valid = mask & (ids >= 0) & (ids < E)
    tl.store(Valid + routes, valid.to(tl.uint8), mask)
    packed = tl.atomic_add(Counts + tl.maximum(ids, 0), 1, valid, sem="relaxed")
    tl.store(Rows + ids * Capacity + packed, routes, valid)
    _quantize(X, Q, S, Outer, Valid, 1, K, triton.next_power_of_2(K))


@triton.jit
def _tasks(
    Counts,
    TaskExpert,
    TaskBegin,
    E: tl.constexpr,
    BM: tl.constexpr,
    BE: tl.constexpr,
    BT: tl.constexpr,
):
    e = tl.program_id(0)
    es = tl.arange(0, BE)
    counts = tl.load(Counts + es, es < E, 0)
    tiles = tl.cdiv(counts, BM)
    start = tl.sum(tl.where(es < e, tiles, 0), 0)
    count = tl.load(Counts + e)
    t = tl.arange(0, BT)
    tl.store(TaskExpert + start + t, e, t < tl.cdiv(count, BM))
    tl.store(TaskBegin + start + t, t * BM, t < tl.cdiv(count, BM))


@triton.jit
def _sf_offset(n, k, N: tl.constexpr, K: tl.constexpr):
    cols: tl.constexpr = triton.cdiv(K // 16, 4) * 4
    return (
        (n // 128) * 128 * cols
        + (k // 4) * 512
        + (n % 32) * 16
        + ((n % 128) // 32) * 4
        + k % 4
    )


@triton.jit(do_not_specialize=["M"])
def _gemm(
    A,
    AS,
    AO,
    W,
    WS,
    Alpha,
    Rows,
    Counts,
    TaskExpert,
    TaskBegin,
    Y,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    TopK: tl.constexpr,
    Capacity: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FC1: tl.constexpr,
    GateFirst: tl.constexpr,
    Activation: tl.constexpr,
    Limit: tl.constexpr,
    SiluAlpha: tl.constexpr,
    SiluBeta: tl.constexpr,
    ClampAfter: tl.constexpr,
):
    task = tl.program_id(0)
    expert = tl.load(TaskExpert + task)
    if expert >= 0:
        begin = tl.load(TaskBegin + task)
        count = tl.load(Counts + expert)
        r = begin + tl.arange(0, BM)
        route = tl.load(Rows + expert * Capacity + r, r < count, 0)
        valid = (r < count) & (route < M * TopK)
        if FC1:
            ar = route // TopK
        else:
            ar = route
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 16)
        acc = tl.full((BM, BN), 0, tl.float32)
        if FC1:
            up = tl.full((BM, BN), 0, tl.float32)
            gn = n + (0 if GateFirst else N)
            un = n + (N if GateFirst else 0)
            wn: tl.constexpr = N * 2
        else:
            gn = n
            wn: tl.constexpr = N
        for base in range(triton.cdiv(K, BK)):
            pk = base * (BK // 2) + kp
            sk = base * (BK // 16) + ks
            a = tl.load(
                A + ar[:, None] * (K // 2) + pk[None, :],
                valid[:, None] & (pk[None, :] < K // 2),
                0,
            )
            sa = tl.load(
                AS + ar[:, None] * (K // 16) + sk[None, :],
                valid[:, None] & (sk[None, :] < K // 16),
                0.0,
            )
            w = tl.load(
                W + expert * wn * (K // 2) + gn[None, :] * (K // 2) + pk[:, None],
                (n[None, :] < N) & (pk[:, None] < K // 2),
                0,
            )
            ws = tl.load(
                WS
                + expert * triton.cdiv(wn, 128) * 128 * triton.cdiv(K // 16, 4) * 4
                + _sf_offset(gn[:, None], sk[None, :], wn, K),
                (n[:, None] < N) & (sk[None, :] < K // 16),
                0.0,
            )
            acc = tl.dot_scaled(a, sa, "e2m1", w, ws, "e2m1", acc)
            if FC1:
                wu = tl.load(
                    W + expert * wn * (K // 2) + un[None, :] * (K // 2) + pk[:, None],
                    (n[None, :] < N) & (pk[:, None] < K // 2),
                    0,
                )
                su = tl.load(
                    WS
                    + expert * triton.cdiv(wn, 128) * 128 * triton.cdiv(K // 16, 4) * 4
                    + _sf_offset(un[:, None], sk[None, :], wn, K),
                    (n[:, None] < N) & (sk[None, :] < K // 16),
                    0.0,
                )
                up = tl.dot_scaled(a, sa, "e2m1", wu, su, "e2m1", up)
        gain = tl.load(Alpha + expert) * tl.load(AO + ar, valid, 0)
        acc = acc * gain[:, None]
        if FC1:
            gate = acc
            up = up * gain[:, None]
            if Limit is not None:
                up = tl.minimum(tl.maximum(up, -Limit), Limit)
                if not ClampAfter:
                    gate = tl.minimum(gate, Limit)
            if Activation == "swigluoai_uninterleave":
                gate = gate * tl.div_rn(1.0, 1.0 + tl.exp(-SiluAlpha * gate))
                up = up + SiluBeta
            else:
                gate = gate * tl.div_rn(1.0, 1.0 + tl.exp(-gate))
            if ClampAfter and Limit is not None:
                gate = tl.minimum(gate, Limit)
            acc = gate * up
        tl.store(
            Y + route[:, None] * N + n[None, :], acc, valid[:, None] & (n[None, :] < N)
        )


@triton.jit(do_not_specialize=["M"])
def _sum(
    Y,
    Weights,
    Ids,
    O,
    M,
    H: tl.constexpr,
    TopK: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in range(TopK):
        route = token * TopK + k
        expert = tl.load(Ids + route)
        value = tl.load(
            Y + route * H + cols, (cols < H) & (expert >= 0) & (expert < E), 0
        )
        weight = tl.load(Weights + route)
        acc += value.to(tl.float32) * weight
    tl.store(O + token * H + cols, acc, cols < H)


@dataclass(frozen=True)
class Query:
    tokens: int
    top_k: int
    experts: int
    hidden: int
    intermediate: int
    dtype: str
    activation: str
    gate_first: bool
    limit: float | None
    alpha: float
    beta: float
    clamp_after: bool
    outer_scale: str = "token"
    compressed_scales: bool = False
    route_experts: int = 0
    route_dtype: str = "float32"
    route_score: str = "softmax"
    route_renormalize: bool = True
    route_scale: float = 1.0
    route_correction: bool = False
    route_image_bias: bool = False
    route_image_mask: bool = False
    route_valid_tokens: bool = False
    output_layout: str = "token"


@dataclass(frozen=True)
class Config:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 128
    warps: int = 4
    fuse_routing: bool = False
    fuse_tasks: bool = False


def _route_query(q):
    return MoeRouteQuery(
        num_tokens=q.tokens,
        num_experts=q.route_experts,
        top_k=q.top_k,
        logits_dtype=q.route_dtype,
        score_func=q.route_score,
        renormalize=q.route_renormalize,
        routed_scaling_factor=q.route_scale,
        has_correction_bias=q.route_correction,
        has_image_correction_bias=q.route_image_bias,
        has_image_mask=q.route_image_mask,
        has_valid_tokens=q.route_valid_tokens,
        logits_row_stride=q.route_experts,
        topk_row_stride=q.top_k,
        ids_row_stride=q.top_k,
        weights_row_stride=q.top_k,
    )


def _parameters(q, device):
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "fuse_routing": (False, True) if q.route_experts else (False,),
            "fuse_tasks": (False, True) if q.compressed_scales else (False,),
        },
    )


def _validate_query(q, device):
    if q.output_layout not in ("token", "routes"):
        raise ValueError("MoE output layout must be token or routes")
    if q.route_experts:
        ROUTE_TUNING.validate_query(_route_query(q), device)
    if min(q.tokens, q.top_k, q.experts, q.hidden, q.intermediate) <= 0:
        raise ValueError("MoE geometry must be positive")
    if q.hidden % 128 or q.intermediate % 128:
        raise ValueError("token-scaled NVFP4 dimensions must be multiples of 128")
    if device is not None and device.compute_capability not in ((12, 0), (12, 1)):
        raise ValueError("token-scaled NVFP4 MoE requires SM120 or SM121")


def _validate_config(q, c, device):
    if c.fuse_tasks and not q.compressed_scales:
        raise ValueError("Task fusion requires compressed expert scales")
    if c.fuse_routing and not q.route_experts:
        raise ValueError("routing fusion requires integrated router inputs")
    if (
        c.block_m not in (16, 32)
        or c.block_n not in (64, 128)
        or c.block_k not in (128, 256)
        or c.warps not in (4, 8)
    ):
        raise ValueError("invalid token-scaled NVFP4 tile configuration")


TUNING = TuningContract(
    component_id="moe.decode",
    query_schema_version=5,
    config_schema_version=4,
    semantic_version=5,
    query_fields=frozenset(Query.__dataclass_fields__),
    config_fields=frozenset(Config.__dataclass_fields__),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda p: Config(**p),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda q, d: Config(),
    knobs=(
        Knob(name="block_m", values=(16, 32)),
        Knob(name="block_n", values=(64, 128)),
        Knob(name="block_k", values=(128, 256)),
        Knob(name="warps", values=(4, 8)),
        Knob(name="fuse_routing", values=None),
        Knob(name="fuse_tasks", values=(False, True), binding=ParameterBinding.RUNTIME),
    ),
    parameters=_parameters,
)


def _layout(q, c):
    rows = q.tokens * q.top_k
    tasks = triton.cdiv(rows, c.block_m) + q.experts
    specs = (
        ("input", (q.tokens, q.hidden // 2), torch.uint8),
        ("input_sf", (q.tokens, q.hidden // 16), torch.float8_e4m3fn),
        ("input_outer", (q.tokens,), torch.float32),
        ("counts", (q.experts,), torch.int32),
        ("valid", (rows,), torch.uint8),
        ("rows", (q.experts, rows), torch.int32),
        ("task_expert", (tasks,), torch.int32),
        ("task_begin", (tasks,), torch.int32),
        ("intermediate", (rows, q.intermediate), getattr(torch, q.dtype)),
        ("intermediate_q", (rows, q.intermediate // 2), torch.uint8),
        ("intermediate_sf", (rows, q.intermediate // 16), torch.float8_e4m3fn),
        ("intermediate_outer", (rows,), torch.float32),
        ("routes", (rows, q.hidden), getattr(torch, q.dtype)),
    )
    if q.output_layout == "routes":
        specs = tuple(spec for spec in specs if spec[0] != "routes")
    offset, result = 0, []
    for name, shape, dtype in specs:
        offset = triton.cdiv(offset, 256) * 256
        size = dtype.itemsize
        for dim in shape:
            size *= dim
        result.append((name, shape, dtype, offset, size))
        offset += size
    return tuple(result), triton.cdiv(offset, 256) * 256


@program_cache(scope="preparation")
def compile_programs(query, config, ordinal):
    q, c = Query(**query), Config(**config)
    rows = q.tokens * q.top_k
    tasks = triton.cdiv(rows, c.block_m) + q.experts
    mock = lambda dtype: MockTensor(dtype, (1,))
    byte, sf, fp32, i32, io = map(
        mock,
        (
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float32,
            torch.int32,
            getattr(torch, q.dtype),
        ),
    )
    with torch.cuda.device(ordinal):
        # Triton 3.7.1's scaled-MMA pass recognizes SM120 only. Select that lowering
        # for SM121; PTX and cubin emission still use the physical target.
        gemm_options = (
            {"arch": "sm120"}
            if torch.cuda.get_device_capability(ordinal) == (12, 1)
            else {}
        )
        quant = tuple(
            _quantize.warmup(
                io,
                byte,
                sf,
                fp32,
                byte,
                1,
                K=k,
                B=triton.next_power_of_2(k),
                MaskRoutes=index == 1,
                grid=(1,),
                num_warps=4,
                enable_fp_fusion=False,
            )
            for index, k in enumerate((q.hidden, q.intermediate))
        )
        clear = _clear.warmup(i32, i32, q.experts, tasks, 256, grid=(1,))
        pack = tuple(
            _pack.warmup(
                mock(dtype), i32, i32, byte, 1, q.experts, rows, 256, grid=(1,)
            )
            for dtype in (torch.int32, torch.int64)
        )
        task = _tasks.warmup(
            i32,
            i32,
            i32,
            q.experts,
            c.block_m,
            triton.next_power_of_2(q.experts),
            triton.next_power_of_2(triton.cdiv(rows, c.block_m)),
            grid=(1,),
        )
        gemms = tuple(
            _gemm.warmup(
                byte,
                sf,
                fp32,
                byte,
                sf,
                fp32,
                i32,
                i32,
                i32,
                i32,
                io,
                1,
                k,
                n,
                q.top_k,
                rows,
                c.block_m,
                c.block_n,
                c.block_k,
                fc1,
                q.gate_first,
                q.activation,
                q.limit,
                q.alpha,
                q.beta,
                q.clamp_after,
                grid=(1, 1),
                num_warps=c.warps,
                enable_fp_fusion=False,
                **gemm_options,
            )
            for fc1, k, n in (
                (True, q.hidden, q.intermediate),
                (False, q.intermediate, q.hidden),
            )
        )
        sums = (
            ()
            if q.output_layout == "routes"
            else tuple(
                _sum.warmup(
                    io,
                    fp32,
                    mock(dtype),
                    io,
                    1,
                    q.hidden,
                    q.top_k,
                    q.experts,
                    256,
                    grid=(1, 1),
                    enable_fp_fusion=False,
                )
                for dtype in (torch.int32, torch.int64)
            )
        )
        route = None
        if q.route_experts:
            if c.fuse_routing:
                route = _route_quant_pack.warmup(
                    io,
                    byte,
                    sf,
                    fp32,
                    mock(getattr(torch, q.route_dtype)),
                    fp32,
                    i32,
                    fp32,
                    fp32,
                    fp32,
                    mock(torch.bool),
                    i32,
                    i32,
                    i32,
                    byte,
                    K=q.hidden,
                    E=q.route_experts,
                    TopK=q.top_k,
                    Capacity=rows,
                    Score=q.route_score,
                    Renormalize=q.route_renormalize,
                    Scale=q.route_scale,
                    HasCorrection=q.route_correction,
                    HasImageBias=q.route_image_bias,
                    HasLiveCount=q.route_valid_tokens,
                    grid=(1,),
                    num_warps=4 if q.route_experts <= 256 else 8,
                    enable_fp_fusion=False,
                )
            else:
                route = compile_route_topk(asdict(_route_query(q)), ordinal)
    return quant, clear, pack, task, gemms, sums, route


@dataclass(frozen=True)
class TokenOuterBinding:
    state: object
    a: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    output: torch.Tensor
    views: object
    plan: object = None
    compiler_scratch: torch.Tensor | None = None
    compiler_a4_prefill: bool | None = None
    output_cast_target: object = None
    input_scales_static: bool = False
    fast_math: bool | None = None
    activation_amax: object = None
    layer_idx: object = None
    route_expert_map: object = None
    output_expert_map: object = None
    scales_expanded: bool = False
    router_logits: torch.Tensor | None = None
    selected_logits: torch.Tensor | None = None
    correction_bias: torch.Tensor | None = None
    image_correction_bias: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None
    num_valid_tokens: torch.Tensor | None = None

    def run(self):
        s, v = self.state, self.views
        q, c, w = s.query, s.config, s.experts._impl
        m = self.a.shape[0]
        rows = q.tokens * q.top_k
        tasks = triton.cdiv(rows, c.block_m) + q.experts
        quant, clear, pack, task, gemms, sums, route = s.programs
        clear[(triton.cdiv(max(q.experts, tasks), 256), 1, 1)](
            v["counts"], v["task_expert"]
        )
        id_type = 0 if self.topk_ids.dtype == torch.int32 else 1
        if q.route_experts and c.fuse_routing:
            route[(m, 1, 1)](
                self.a,
                v["input"],
                v["input_sf"],
                v["input_outer"],
                self.router_logits,
                self.selected_logits,
                self.topk_ids,
                self.topk_weights,
                self.correction_bias,
                self.image_correction_bias,
                self.image_mask,
                self.num_valid_tokens,
                v["counts"],
                v["rows"],
                v["valid"],
            )
        else:
            if q.route_experts:
                _RouteTopKState(_route_query(q), route).run(
                    self.router_logits,
                    self.selected_logits,
                    self.topk_ids,
                    self.topk_weights,
                    correction_bias=self.correction_bias,
                    image_correction_bias=self.image_correction_bias,
                    image_mask=self.image_mask,
                    num_valid_tokens=self.num_valid_tokens,
                )
            quant[0][(m, 1, 1)](
                self.a, v["input"], v["input_sf"], v["input_outer"], v["valid"], m
            )
            pack[id_type][(triton.cdiv(m * q.top_k, 256), 1, 1)](
                self.topk_ids, v["counts"], v["rows"], v["valid"], m * q.top_k
            )
        fuse_tasks = c.fuse_tasks and not self.scales_expanded
        if not fuse_tasks:
            task[(q.experts, 1, 1)](v["counts"], v["task_expert"], v["task_begin"])
        if s.scale_decoder is not None and not self.scales_expanded:
            decoder = s.scale_decoder
            # Packing counts already identify every expert whose scales are read.
            if fuse_tasks:
                ids, mode = self.topk_ids, 0
                if m * q.top_k >= 16 * q.experts:
                    mode = 3
                elif m * q.top_k >= 64 and not decoder.direct_routes:
                    ids, mode = v["counts"], 2
                decode_nvfp4_csf_pair(
                    decoder.first,
                    decoder.second,
                    ids,
                    w.w1_blockscale,
                    w.w2_blockscale,
                    mode=mode,
                    program=s.task_programs[int(ids.dtype == torch.int64)],
                    task_metadata=(v["counts"], v["task_expert"], v["task_begin"]),
                    task_block_m=c.block_m,
                )
            elif 64 <= m * q.top_k < 16 * q.experts and not decoder.direct_routes:
                decode_nvfp4_csf_pair(
                    decoder.first,
                    decoder.second,
                    v["counts"],
                    w.w1_blockscale,
                    w.w2_blockscale,
                    mode=2,
                    program=decoder.programs[0],
                )
            else:
                decoder.decode(self.topk_ids, w.w1_blockscale, w.w2_blockscale)
        common = (v["rows"], v["counts"], v["task_expert"], v["task_begin"])
        gemms[0][(tasks, triton.cdiv(q.intermediate, c.block_n), 1)](
            v["input"],
            v["input_sf"],
            v["input_outer"],
            w.w1_fp4,
            w.w1_blockscale.view(torch.float8_e4m3fn),
            w.w1_alphas,
            *common,
            v["intermediate"],
            m,
        )
        quant[1][(m * q.top_k, 1, 1)](
            v["intermediate"],
            v["intermediate_q"],
            v["intermediate_sf"],
            v["intermediate_outer"],
            v["valid"],
            m * q.top_k,
        )
        gemms[1][(tasks, triton.cdiv(q.hidden, c.block_n), 1)](
            v["intermediate_q"],
            v["intermediate_sf"],
            v["intermediate_outer"],
            w.w2_fp4,
            w.w2_blockscale.view(torch.float8_e4m3fn),
            w.w2_alphas,
            *common,
            self.output if q.output_layout == "routes" else v["routes"],
            m,
        )
        if q.output_layout == "token":
            sums[id_type][(m, triton.cdiv(q.hidden, 256), 1)](
                v["routes"], self.topk_weights, self.topk_ids, self.output, m
            )
        return self.output


@dataclass(frozen=True)
class State:
    experts: object
    query: Query
    config: Config
    programs: object
    scale_decoder: object = None
    task_programs: tuple = ()

    def scratch_specs(self):
        return (
            scratch_buffer_spec(
                "tp_moe.scratch",
                nbytes=_layout(self.query, self.config)[1],
                device=self.experts.device,
            ),
        )

    @property
    def scratch(self):
        return self

    def _precision_choice(self, requested):
        if requested is not None:
            raise ValueError("token outer scaling does not support a4_prefill")
        return None

    def uses_expanded_nvfp4_scales(self, **kwargs):
        return self.scale_decoder is not None

    def bind(self, *, a, topk_ids, topk_weights, output, scratch, **kwargs):
        q = self.query
        if (
            a.ndim != 2
            or a.shape[1] != q.hidden
            or not 0 < a.shape[0] <= q.tokens
            or not a.is_contiguous()
        ):
            raise ValueError(
                "activation shape or layout exceeds prepared token-scaled MoE geometry"
            )
        if a.dtype != getattr(torch, q.dtype) or a.device != self.experts.device:
            raise ValueError("activation dtype or device differs from prepared experts")
        if topk_ids.shape != (a.shape[0], q.top_k) or topk_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("route IDs differ from prepared top-k geometry")
        if topk_weights.shape != topk_ids.shape or topk_weights.dtype != torch.float32:
            raise ValueError("route weights must be FP32 with the prepared top-k shape")
        output_shape = (
            (a.shape[0], q.top_k, q.hidden) if q.output_layout == "routes" else a.shape
        )
        if output.shape != output_shape or output.dtype != a.dtype:
            raise ValueError(
                "output must match the prepared MoE layout and activation dtype"
            )
        if any(
            not t.is_contiguous() or t.device != a.device
            for t in (topk_ids, topk_weights, output)
        ):
            raise ValueError("MoE operands must be contiguous on the experts' device")
        route_inputs = {}
        if q.route_experts:
            if a.shape[0] != q.tokens:
                raise ValueError(
                    "integrated routing requires its exact prepared token count"
                )
            if topk_ids.dtype != torch.int32:
                raise ValueError("integrated routing produces int32 route IDs")
            for name, shape, dtype in (
                (
                    "router_logits",
                    (a.shape[0], q.route_experts),
                    getattr(torch, q.route_dtype),
                ),
                ("selected_logits", topk_ids.shape, torch.float32),
            ):
                tensor = kwargs.get(name)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != shape
                    or tensor.dtype != dtype
                    or tensor.device != a.device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} differs from prepared integrated routing geometry"
                    )
                route_inputs[name] = tensor
            for name, enabled, shape, dtype in (
                (
                    "correction_bias",
                    q.route_correction,
                    (q.route_experts,),
                    torch.float32,
                ),
                (
                    "image_correction_bias",
                    q.route_image_bias,
                    (q.route_experts,),
                    torch.float32,
                ),
                ("image_mask", q.route_image_mask, (a.shape[0],), torch.bool),
                ("num_valid_tokens", q.route_valid_tokens, (1,), torch.int32),
            ):
                tensor = kwargs.get(name)
                if (tensor is not None) != enabled:
                    raise ValueError(
                        f"{name} differs from prepared integrated routing controls"
                    )
                if tensor is not None and (
                    tensor.shape != shape
                    or tensor.dtype != dtype
                    or tensor.device != a.device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} requires contiguous {dtype} {shape} on the activation device"
                    )
                route_inputs[name] = tensor
        for name in (
            "activation_amax",
            "route_expert_map",
            "output_expert_map",
            "a4_prefill",
        ):
            if kwargs.get(name) is not None:
                raise ValueError(f"token outer scaling does not support {name}")
        if kwargs.get("experts", self.experts) is not self.experts:
            raise ValueError("prepared experts differ from declaration")
        raw = scratch_tensor(scratch, self.scratch_specs(), owner="token-scaled MoE")
        views = {
            name: raw[offset : offset + size].view(dtype).view(shape)
            for name, shape, dtype, offset, size in _layout(q, self.config)[0]
        }
        return TokenOuterBinding(
            self,
            a,
            topk_weights,
            topk_ids,
            output,
            views,
            scales_expanded=kwargs.get("scales_expanded", False),
            **route_inputs,
        )

    def run(self, binding):
        return binding.run()


def plan(
    experts,
    *,
    capacity,
    routing=None,
    invocation=FrozenMapping(),
    override=None,
    scale_expansion=None,
):
    routing = routing or RoutingSpec()
    if routing.apply_router_weight_on_input or routing.collect_activation_amax:
        raise ValueError("token outer scaling requires output-weighted NVFP4 experts")
    activation, geometry = experts.plan.activation, experts.plan.geometry
    if activation.nonlinearity not in ("silu", "swigluoai_uninterleave"):
        raise ValueError("token outer scaling requires a gated SiLU activation")
    decoder = experts._impl.nvfp4_csf
    if capacity.route_num_experts not in (None, 0, geometry.num_experts):
        raise ValueError("integrated routing requires one logit per prepared expert")
    counts = tuple(sorted({capacity.max_tokens, *capacity.warmup_token_counts}))
    expansions = (
        {}
        if scale_expansion is None
        else dict(scale_expansion)
        if isinstance(scale_expansion, Mapping)
        else {counts[0]: scale_expansion}
    )
    if decoder is not None and set(expansions) != set(counts):
        raise ValueError(
            "Scale expansion plans must cover every compressed MoE variant"
        )
    if expansions and decoder is None:
        raise ValueError("Scale expansion requires compressed expert scales")
    for tokens, expansion in expansions.items():
        if (
            not isinstance(expansion, Plan)
            or expansion.contract.component_id != "quantization.csf_block_expansion"
            or expansion.query.routes != tokens * capacity.top_k
            or expansion.query.first
            != decoder.first.geometry[:7] + decoder.first.geometry[9:]
            or expansion.query.second
            != decoder.second.geometry[:7] + decoder.second.geometry[9:]
        ):
            raise ValueError("Scale expansion geometry differs from its MoE variant")

    def child(tokens):
        q = Query(
            tokens,
            capacity.top_k,
            geometry.num_experts,
            geometry.hidden_size,
            geometry.intermediate_size,
            str(activation.io_dtype).removeprefix("torch."),
            activation.nonlinearity,
            experts.plan._impl.w13_layout == "w31",
            normalize_swiglu_limit_for_activation(
                activation.nonlinearity, activation.swiglu_limit
            ),
            normalize_swiglu_alpha_for_activation(
                activation.nonlinearity, activation.swiglu_alpha
            ),
            normalize_swiglu_beta_for_activation(
                activation.nonlinearity, activation.swiglu_beta
            ),
            activation.swiglu_clamp_after_silu,
            compressed_scales=decoder is not None,
            route_experts=capacity.route_num_experts or 0,
            route_dtype=str(routing.logits_dtype or torch.float32).removeprefix(
                "torch."
            ),
            route_score=routing.score_func,
            route_renormalize=routing.renormalize,
            route_scale=routing.routed_scaling_factor,
            route_correction=routing.has_correction_bias,
            route_image_bias=routing.has_image_correction_bias,
            route_image_mask=routing.has_image_mask,
            route_valid_tokens=routing.has_valid_tokens,
            output_layout=capacity.output_layout,
        )

        def jobs(c, d):
            return (
                CompileJob.create(
                    "b12x.moe.fused_moe._token_outer:compile_programs",
                    asdict(q),
                    asdict(c),
                    d.ordinal,
                ),
            )

        def memory(c, d):
            return MemoryRequirements(
                scratch=(
                    scratch_buffer_spec(
                        "tp_moe.scratch", nbytes=_layout(q, c)[1], device=experts.device
                    ),
                )
            )

        def materialize(selection, d):
            programs = compile_programs(asdict(q), asdict(selection.config), d.ordinal)
            scale_decoder = (
                None
                if decoder is None
                else expansions[tokens].prepared.state.decoder(decoder.active)
            )
            task_programs = (
                expansions[tokens].prepared.state.task_programs
                if selection.config.fuse_tasks
                else ()
            )
            state = State(
                experts, q, selection.config, programs, scale_decoder, task_programs
            )
            scale_programs = (
                ()
                if scale_decoder is None
                else (*scale_decoder.programs, *scale_decoder.routing_programs)
            )
            load_programs(
                attach_programs(state, programs, *scale_programs, *task_programs)
            )
            return state

        return Plan(
            contract=TUNING,
            query=q,
            invocation=FrozenMapping(
                {
                    **dict(invocation),
                    **(
                        {
                            "csf_scale_geometry": (
                                decoder.first.geometry,
                                decoder.second.geometry,
                            )
                        }
                        if decoder is not None
                        else {}
                    ),
                }
            ),
            override=override,
            _compile_jobs=jobs,
            _memory_requirements=memory,
            _materialize=materialize,
            _device=experts.device,
        )

    children = {n: child(n) for n in counts}
    if len(children) == 1:
        return next(iter(children.values()))
    return _CompositePlan(
        component_id="moe.decode",
        capacity_metadata=FrozenMapping(
            {"token_counts": counts, "outer_scale": "token"}
        ),
        variants=children,
        _assemble=lambda states, d: _FusedMoeCapacityState(
            MappingProxyType(dict(states))
        ),
    )
