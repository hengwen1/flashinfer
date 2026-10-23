from dataclasses import asdict, replace
from itertools import product

import torch
import pytest
import triton
import triton.language as tl
from b12x.testing.reference.helpers import require_b12x


@triton.jit
def _native_dot(A, B, SA, SB, O):
    m = tl.arange(0, 16)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    s = tl.arange(0, 8)
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    b = tl.load(B + n[None, :] * 64 + k[:, None])
    sa = tl.load(SA + m[:, None] * 8 + s[None, :])
    sb = tl.load(SB + n[:, None] * 8 + s[None, :])
    out = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(O + m[:, None] * 64 + n[None, :], out)


def test_native_dot():
    require_b12x()
    a = torch.full((16, 64), 0x22, dtype=torch.uint8, device="cuda")
    b = torch.full((64, 64), 0x22, dtype=torch.uint8, device="cuda")
    sa = torch.ones((16, 8), dtype=torch.float8_e4m3fn, device="cuda")
    sb = torch.ones((64, 8), dtype=torch.float8_e4m3fn, device="cuda")
    out = torch.empty((16, 64), device="cuda")
    options = {"arch": "sm120"} if torch.cuda.get_device_capability() == (12, 1) else {}
    kernel = _native_dot[(1,)](a, b, sa, sb, out, **options)
    torch.testing.assert_close(out, torch.full_like(out, 128))
    _assert_native_nvfp4_kernel(kernel)


def _assert_native_nvfp4_kernel(kernel):
    assert "kind::mxf4nvf4.block_scale.scale_vec::4X" in kernel.asm["ptx"]
    major, minor = torch.cuda.get_device_capability()
    assert f".target sm_{major}{minor}a" in kernel.asm["ptx"]


_TILES = tuple(product((16, 32), (64, 128), (128, 256), (4, 8)))


@pytest.mark.parametrize("tokens", [1, 4096])
@pytest.mark.parametrize("bm,bn,bk,warps", _TILES)
def test_real_geometry_compiles_every_candidate(tokens, bm, bn, bk, warps):
    require_b12x()
    from b12x.moe.fused_moe._token_outer import Config, Query, compile_programs

    query = Query(
        tokens=tokens,
        top_k=4,
        experts=128,
        hidden=6144,
        intermediate=768,
        dtype="bfloat16",
        activation="swigluoai_uninterleave",
        gate_first=True,
        limit=7.0,
        alpha=1.702,
        beta=1.0,
        clamp_after=False,
    )
    programs = compile_programs(
        asdict(query), asdict(Config(bm, bn, bk, warps)), torch.cuda.current_device()
    )
    for kernel in programs[4]:
        _assert_native_nvfp4_kernel(kernel)


def _weights(e=4, h=128, i=128, activation="swigluoai_uninterleave", csf=False):
    from b12x.moe import fused_moe as moe
    from b12x._lib.intrinsics.reference import swizzle_block_scale

    def packed(n, k):
        # Both signs and every finite E2M1 magnitude occur.
        w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
        sf = (torch.rand(e, n, k // 16, device="cuda") * 0.15 + 0.02).to(
            torch.float8_e4m3fn
        )
        return w, swizzle_block_scale(sf.view(torch.uint8)), sf

    w1, s1, l1 = packed(2 * i, h)
    w2, s2, l2 = packed(h, i)
    alpha1 = torch.linspace(0.15, 0.3, e, device="cuda")
    alpha2 = torch.linspace(0.1, 0.2, e, device="cuda")
    wp = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a4",
            nonlinearity=activation,
            io_dtype=torch.bfloat16,
            outer_scale="token",
            swiglu_limit=7.0,
            swiglu_alpha=1.702 if activation == "swigluoai_uninterleave" else 1.0,
            swiglu_beta=1.0 if activation == "swigluoai_uninterleave" else 0.0,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(scale_compression="csf" if csf else None),
    )
    experts = moe.prepare_weights(
        plan=wp,
        weights=moe.PackedWeights(
            w13=w1,
            w2=w2,
            w13_block_scales=s1,
            w2_block_scales=s2,
            w13_global_scales=alpha1,
            w2_global_scales=alpha2,
        ),
        **(
            {"scale_scratch": (torch.empty_like(s1), torch.empty_like(s2))}
            if csf
            else {}
        ),
    )

    def decode(w, s, alpha):
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device="cuda",
        )
        code = torch.stack((w & 15, w >> 4), -1).reshape(e, w.shape[1], -1)
        return lut[code.long()] * s.float().repeat_interleave(16, -1)

    return experts, decode(w1, l1, alpha1), decode(w2, l2, alpha2), alpha1, alpha2


def _qd(x, normalized=False):
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_per_token_cute_dsl,
    )

    q, sf, outer = nvfp4_quantize_per_token_cute_dsl(
        x, 1.0 / (448 * 6), sf_layout=2, enable_pdl=False, nvfp4_4over6=None
    )
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=x.device,
    )
    code = torch.stack((q & 15, q >> 4), -1).reshape_as(x)
    decoded = lut[code.long()] * sf.view(torch.float8_e4m3fn).float().repeat_interleave(
        16, -1
    )
    if not normalized:
        decoded *= outer[:, None]
    return decoded, (q, sf, outer)


def test_quantizer_matches_reference():
    require_b12x()
    from b12x.moe.fused_moe._token_outer import _quantize

    for k in (128, 384, 6144):
        x = torch.randn(11, k, dtype=torch.bfloat16, device="cuda")
        x[0].zero_()
        x[1] *= 1e-5
        x[2] *= 1e4
        _, expected = _qd(x)
        q, sf, outer = (torch.empty_like(t) for t in expected)
        _quantize[(11,)](
            x,
            q,
            sf.view(torch.float8_e4m3fn),
            outer,
            q,
            11,
            K=k,
            B=triton.next_power_of_2(k),
            enable_fp_fusion=False,
        )
        torch.testing.assert_close(q, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(sf, expected[1], rtol=0, atol=0)
        torch.testing.assert_close(outer, expected[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,h,i,e,topk,tile",
    [
        (1, 128, 128, 4, 2, None),
        (17, 128, 128, 4, 2, None),
        (65, 128, 128, 4, 2, None),
        (4096, 128, 128, 4, 4, None),
        (17, 6144, 768, 128, 4, None),
        (4096, 6144, 768, 128, 4, (32, 128, 128, 4)),
    ]
    + [(17, 384, 128, 4, 2, tile) for tile in _TILES],
)
def test_prepared_moe(m, h, i, e, topk, tile):
    _check_prepared_moe(m, h, i, e, topk, tile)


@pytest.mark.parametrize("tile", [(16, 64, 128, 4), (32, 128, 256, 8)])
def test_prepared_token_scaled_silu(tile):
    _check_prepared_moe(17, 384, 128, 4, 2, tile, activation="silu")


def _check_prepared_moe(
    m,
    h,
    i,
    e,
    topk,
    tile,
    activation="swigluoai_uninterleave",
    csf=False,
    ids_dtype=torch.int32,
    fuse_tasks=False,
):
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.moe.fused_moe._token_outer import Config

    torch.manual_seed(871)
    experts, w1, w2, alpha1, alpha2 = _weights(e, h, i, activation, csf)
    a = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
    if m > 1:
        a[0].zero_()
    if tile is not None and m > 2:
        a[1] *= 1e-4
        a[2] *= 1e4
    ids = torch.randint(0, e, (m, topk), dtype=ids_dtype, device="cuda")
    ids[-1, -1] = -1
    if tile is not None and m > 1:
        ids[-1].fill_(-1)
        ids[-2, 0] = e
    weights = torch.rand(m, topk, device="cuda")
    out = torch.empty_like(a)
    expansion = (
        moe.plan_scale_expansion(experts, routes=m * topk, ids_dtype=ids_dtype)
        if csf
        else None
    )
    plan = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(max_tokens=m, top_k=topk),
        override=None if tile is None else Config(*tile, fuse_tasks=fuse_tasks),
        scale_expansion=expansion,
    )

    def bind(state):
        buffers = tuple(
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in state.scratch_specs()
        )
        b = state.bind(
            a=a, topk_ids=ids, topk_weights=weights, output=out, scratch=buffers
        )
        return PreparedCall(run=b.run, output=out, owners=(buffers, b))

    def reference():
        x, (_, _, xouter) = _qd(a, normalized=True)
        routes = torch.zeros(m, topk, h, dtype=torch.float32, device=a.device)
        for expert in range(e):
            token, branch = torch.where(ids == expert)
            if not token.numel():
                continue
            projected = (x[token] @ w1[expert].T) * (xouter[token] * alpha1[expert])[
                :, None
            ]
            gate, up = projected.chunk(2, -1)
            gate = gate.clamp(max=7)
            up = up.clamp(-7, 7)
            alpha = 1.702 if activation == "swigluoai_uninterleave" else 1.0
            beta = 1.0 if activation == "swigluoai_uninterleave" else 0.0
            hidden = (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(
                torch.bfloat16
            )
            hidden, (_, _, houter) = _qd(hidden, normalized=True)
            y = (
                ((hidden @ w2[expert].T) * (houter * alpha2[expert])[:, None])
                .to(torch.bfloat16)
                .float()
            )
            routes[token, branch] = y * weights[token, branch, None]
        return routes.sum(1).to(torch.bfloat16)

    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        expansion_requests = ()
        if expansion is not None:

            def bind_scales(state):
                active = torch.empty(e, dtype=torch.int32, device="cuda")
                b = state.bind(
                    ids=ids,
                    out13=experts._impl.w1_blockscale,
                    out2=experts._impl.w2_blockscale,
                    scratch=active,
                )
                return PreparedCall(run=b.run, owners=(b, active))

            expansion_requests = (
                expansion.request(name="scales", prepare_call=bind_scales),
            )
        session.prepare(
            (
                *expansion_requests,
                plan.request(
                    name="token-scaled-moe",
                    prepare_call=bind,
                    dependencies=tuple(r.name for r in expansion_requests),
                ),
            )
        )
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in plan.scratch_specs()
        )
        binding = moe.bind(
            plan, a=a, topk_ids=ids, topk_weights=weights, output=out, scratch=scratch
        )
        moe.run(binding=binding)
        torch.testing.assert_close(out, reference(), rtol=0.015, atol=0.001)
        if csf:
            _assert_route_decoder_equivalent(binding)
        assert (
            moe.uses_expanded_nvfp4_scales(
                plan, num_tokens=m, route_ids_dtype=ids_dtype
            )
            == csf
        )
        if csf:
            assert moe.expand_scales(experts)
            expanded = moe.bind(
                plan,
                a=a,
                topk_ids=ids,
                topk_weights=weights,
                output=out,
                scratch=scratch,
                scales_expanded=True,
            )
            expected = out.clone()
            moe.run(binding=expanded)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.run(binding=binding)
        for iteration in range(3):
            a.neg_()
            ids.add_(1).remainder_(e)
            if csf and iteration == 1:
                ids.zero_()
            if csf and iteration == 2:
                ids.fill_(-1)
            if tile is not None and m > 1:
                ids[-1].fill_(-1)
                ids[-2, 0] = e
            for buf in scratch:
                buf.fill_(0x7F)
            if csf:
                experts._impl.w1_blockscale.fill_(0x7F)
                experts._impl.w2_blockscale.fill_(0x7F)
            out.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(out, reference(), rtol=0.015, atol=0.001)
            if csf:
                _assert_route_decoder_equivalent(binding)


def _assert_route_decoder_equivalent(binding):
    """Compare count-routed scale expansion with independent route decoding."""
    state = binding.state
    weights = state.experts._impl
    ids = binding.topk_ids.flatten()
    live = ids[(ids >= 0) & (ids < state.query.experts)].long()
    active = torch.zeros(state.query.experts, dtype=torch.bool, device=ids.device)
    active[live] = True
    scales = (weights.w1_blockscale, weights.w2_blockscale)
    expected = [value.view(state.query.experts, -1)[active].clone() for value in scales]
    expected_output = binding.output.clone()
    for value in scales:
        value.fill_(0x7F)
    state.scale_decoder.decode(binding.topk_ids, *scales)
    reference = replace(binding, scales_expanded=True)
    reference.run()
    torch.testing.assert_close(binding.output, expected_output, rtol=0, atol=0)
    for value, original in zip(scales, expected, strict=True):
        torch.testing.assert_close(
            value.view(state.query.experts, -1)[active], original, rtol=0, atol=0
        )


def test_compiled_autotuned_capacity_family():
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    experts, *_ = _weights()
    plan = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=33, top_k=2, warmup_token_counts=(1, 17)
        ),
    )

    def operands(tokens):
        a = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, 4, (tokens, 2), dtype=torch.int64, device="cuda")
        weights = torch.rand(tokens, 2, device="cuda")
        return a, ids, weights, torch.empty_like(a)

    def prepare(state):
        a, ids, weights, out = operands(state.query.tokens)
        fixture = a.clone()
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in state.scratch_specs()
        )
        binding = state.bind(
            a=a, topk_ids=ids, topk_weights=weights, output=out, scratch=scratch
        )
        return PreparedCall(
            run=binding.run,
            output=out,
            produce=lambda: a.copy_(fixture),
            owners=(binding, scratch, fixture),
        )

    with PreparationSession(device="cuda", autotune=True, compile_workers=1) as session:
        session.prepare(
            (
                plan.request(
                    name="token-scaled-capacities",
                    prepare_calls={n: prepare for n in plan.token_counts},
                    benchmark_calls={n: prepare for n in plan.token_counts},
                ),
            )
        )
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in plan.scratch_specs()
        )
        bindings = []
        for tokens in (1, 17, 23, 33):
            a, ids, weights, out = operands(tokens)
            binding = moe.bind(
                plan,
                a=a,
                topk_ids=ids,
                topk_weights=weights,
                output=out,
                scratch=scratch,
            )
            bindings.append(binding)
        compiled = [
            torch.compile(lambda b=b: moe.run(binding=b), fullgraph=True)
            for b in bindings
        ]
        for binding, function in zip(bindings, compiled, strict=True):
            moe.run(binding=binding)
            expected = binding.output.clone()
            function()
            torch.testing.assert_close(binding.output, expected, rtol=0, atol=0)
            del expected
        session.freeze()
        graphs = []
        for function in compiled:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                function()
            graphs.append(graph)
        allocated = torch.cuda.memory_allocated()
        for index in (3, 0, 2, 1, 3, 0):
            binding = bindings[index]
            binding.topk_ids.zero_()
            binding.a.neg_()
            for buffer in scratch:
                buffer.fill_(0x7F)
            graphs[index].replay()
            expected = binding.output.clone()
            moe.run(binding=binding)
            torch.testing.assert_close(binding.output, expected, rtol=0, atol=0)
            del expected
        assert torch.cuda.memory_allocated() == allocated


def test_outer_scale_contract():
    from b12x.moe import fused_moe as moe

    with pytest.raises(ValueError, match="outer_scale"):
        moe.ActivationSpec(
            mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, outer_scale="row"
        )
    with pytest.raises(ValueError, match="requires A4"):
        moe.ActivationSpec(
            mode="a16",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            outer_scale="token",
        )
    with pytest.raises(ValueError, match="without an A16 cutoff"):
        moe.ActivationSpec(
            mode="a4",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            outer_scale="token",
            a16_max_tokens=16,
        )


@pytest.mark.parametrize(
    "m,h,i,e,topk",
    [
        (1, 128, 128, 4, 2),
        (17, 384, 128, 4, 2),
        (48, 6144, 768, 128, 4),
        (4096, 128, 128, 4, 4),
    ],
)
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_compressed_token_scales(m, h, i, e, topk, ids_dtype):
    _check_prepared_moe(
        m, h, i, e, topk, (16, 64, 256, 4), csf=True, ids_dtype=ids_dtype
    )


def test_compressed_variant_expansion_geometry():
    from types import SimpleNamespace
    from dataclasses import replace
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._token_outer import plan
    from b12x._lib.quant.csf_blocks_tuning import TUNING, CsfExpansionQuery
    from b12x.preparation import Plan

    geometry = (256, 8, 4, 4, 0, 0, 4, 8, 1, 0, 0)
    decoder = SimpleNamespace(
        first=SimpleNamespace(geometry=geometry),
        second=SimpleNamespace(geometry=geometry),
    )
    experts = SimpleNamespace(
        device=torch.device("cuda:0"),
        _impl=SimpleNamespace(nvfp4_csf=decoder),
        plan=SimpleNamespace(
            activation=moe.ActivationSpec(
                mode="a4",
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
                outer_scale="token",
            ),
            geometry=moe.MoEGeometry(
                num_experts=4, hidden_size=128, intermediate_size=128
            ),
            _impl=SimpleNamespace(w13_layout="w31"),
        ),
    )

    def expansion(routes):
        return Plan(
            contract=TUNING,
            query=CsfExpansionQuery(
                geometry[:7] + geometry[9:],
                geometry[:7] + geometry[9:],
                routes,
                "int64",
            ),
            _compile_jobs=lambda *args: (),
            _memory_requirements=lambda *args: None,
            _materialize=lambda *args: None,
        )

    capacity = moe.ExecutionCapacity(max_tokens=17, top_k=2, warmup_token_counts=(1,))
    scales = {1: expansion(2), 17: expansion(34)}
    declaration = plan(experts, capacity=capacity, scale_expansion=scales)
    assert set(declaration.variants) == {1, 17}
    assert all(child.query.compressed_scales for child in declaration.variants.values())
    with pytest.raises(ValueError, match="cover every"):
        plan(experts, capacity=capacity, scale_expansion=scales[1])
    with pytest.raises(ValueError, match="geometry differs"):
        plan(experts, capacity=capacity, scale_expansion={1: scales[17], 17: scales[1]})
    with pytest.raises(ValueError, match="geometry differs"):
        plan(
            experts,
            capacity=capacity,
            scale_expansion={
                1: scales[1],
                17: replace(
                    scales[17],
                    query=replace(
                        scales[17].query,
                        second=tuple(reversed(scales[17].query.second)),
                    ),
                ),
            },
        )


@pytest.mark.parametrize("score", ["sigmoid", "softmax", "sqrtsoftplus"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("experts", [128, 1024])
@pytest.mark.parametrize("topk", [3, 4])
def test_route_quant_pack_exact(score, renormalize, experts, topk):
    require_b12x()
    from b12x.moe._shared.routing import route_topk
    from b12x.moe.fused_moe._token_outer import _route_quant_pack, _quantize

    m, k = 48, 6144
    torch.manual_seed(431)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(m, experts, device="cuda")
    logits[0].zero_()
    bias = torch.randn(experts, device="cuda") * 0.1
    image_bias = -bias
    image_mask = torch.arange(m, device="cuda") % 2 == 0
    count = torch.tensor([m], device="cuda", dtype=torch.int32)
    shape = (m, topk)
    expected = (
        torch.empty(shape, device="cuda"),
        torch.empty(shape, device="cuda", dtype=torch.int32),
        torch.empty(shape, device="cuda"),
    )
    actual = tuple(torch.empty_like(value) for value in expected)
    quant = (
        torch.empty((m, k // 2), device="cuda", dtype=torch.uint8),
        torch.empty((m, k // 16), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(m, device="cuda"),
    )
    expected_quant = tuple(torch.empty_like(value) for value in quant)
    counts = torch.empty(experts, device="cuda", dtype=torch.int32)
    rows = torch.empty((experts, m * topk), device="cuda", dtype=torch.int32)
    valid = torch.empty(m * topk, device="cuda", dtype=torch.uint8)
    options = dict(
        score_func=score,
        renormalize=renormalize,
        routed_scaling_factor=1.7,
        correction_bias=bias,
        image_correction_bias=image_bias,
        image_mask=image_mask,
        num_valid_tokens=count,
    )
    for live in (m, 17, 0, 1, m):
        count.fill_(live)
        source.neg_()
        logits.neg_()
        logits[live:].fill_(float("nan"))
        route_topk(logits, *expected, **options)
        counts.zero_()
        rows.fill_(0x7F7F7F7F)
        valid.fill_(255)
        _route_quant_pack[(m,)](
            source,
            *quant,
            logits,
            *actual,
            bias,
            image_bias,
            image_mask,
            count,
            counts,
            rows,
            valid,
            K=k,
            E=experts,
            TopK=topk,
            Capacity=m * topk,
            Score=score,
            Renormalize=renormalize,
            Scale=1.7,
            HasCorrection=True,
            HasImageBias=True,
            HasLiveCount=True,
            num_warps=4 if experts <= 256 else 8,
            enable_fp_fusion=False,
        )
        _quantize[(m,)](
            source,
            *expected_quant,
            valid,
            m,
            K=k,
            B=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
        for got, want in zip(actual + quant, expected + expected_quant, strict=True):
            torch.testing.assert_close(
                got.view(torch.uint8), want.view(torch.uint8), rtol=0, atol=0
            )
        flat = actual[1].flatten()
        torch.testing.assert_close(valid, (flat >= 0).to(torch.uint8), rtol=0, atol=0)
        histogram = torch.bincount(flat[flat >= 0].long(), minlength=experts)
        torch.testing.assert_close(counts.long(), histogram, rtol=0, atol=0)
        counts_cpu, rows_cpu, flat_cpu = histogram.cpu(), rows.cpu(), flat.cpu()
        for expert in range(experts):
            n = int(counts_cpu[expert])
            torch.testing.assert_close(
                rows_cpu[expert, :n].sort().values,
                torch.where(flat_cpu == expert)[0].int(),
                rtol=0,
                atol=0,
            )
        logits.normal_()


@pytest.mark.parametrize(
    "m,h,i,e",
    [
        (1, 128, 128, 4),
        (17, 128, 128, 4),
        (48, 6144, 768, 128),
        (4096, 128, 128, 4),
        (4096, 6144, 768, 128),
    ],
)
@pytest.mark.parametrize("csf", [False, True])
def test_prepared_integrated_route_exact(m, h, i, e, csf):
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._token_outer import Config
    from b12x.preparation import PreparationSession, PreparedCall

    torch.manual_seed(942)
    experts, *_ = _weights(e, h, i, csf=csf)
    a = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(m, e, device="cuda")
    bias = torch.randn(e, device="cuda") * 0.1
    count = torch.tensor([m], device="cuda", dtype=torch.int32)
    topk = min(e, 4)
    expansion = moe.plan_scale_expansion(experts, routes=m * topk) if csf else None
    plans = [
        moe.plan_execution(
            experts=experts,
            capacity=moe.ExecutionCapacity(
                max_tokens=m, top_k=topk, route_num_experts=e
            ),
            routing=moe.RoutingSpec(
                logits_dtype=torch.float32,
                score_func="sigmoid",
                has_correction_bias=True,
                has_valid_tokens=True,
            ),
            override=Config(32, 64, 256, 4, fuse),
            scale_expansion=expansion,
        )
        for fuse in (False, True)
    ]
    bindings = []

    def bind(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            for spec in state.scratch_specs()
        )
        ids = torch.empty((m, topk), device="cuda", dtype=torch.int32)
        selected = torch.empty((m, topk), device="cuda")
        weights = torch.empty_like(selected)
        output = torch.empty_like(a)
        binding = state.bind(
            a=a,
            router_logits=logits,
            selected_logits=selected,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            correction_bias=bias,
            num_valid_tokens=count,
        )
        bindings.append(binding)
        return PreparedCall(run=binding.run, owners=(binding, scratch))

    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        requests = []
        if expansion is not None:

            def bind_scales(state):
                active = torch.empty(e, device="cuda", dtype=torch.int32)
                ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
                binding = state.bind(
                    ids=ids,
                    out13=experts._impl.w1_blockscale,
                    out2=experts._impl.w2_blockscale,
                    scratch=active,
                )
                return PreparedCall(run=binding.run, owners=(binding, active, ids))

            requests.append(expansion.request(name="scales", prepare_call=bind_scales))
        for index, plan in enumerate(plans):
            requests.append(
                plan.request(
                    name=f"integrated-{index}",
                    prepare_call=bind,
                    dependencies=("scales",) if expansion else (),
                )
            )
        session.prepare(tuple(requests))
        # Materialization may prime a selected plan more than once.
        bindings = []
        for plan in plans:
            from b12x.preparation.types import require_prepared

            call = bind(require_prepared(plan, "moe.decode"))
            bindings[-1] = replace(
                bindings[-1], plan=plan, compiler_scratch=call.owners[1][0]
            )
        control, candidate = bindings
        session.freeze()
        compiled = torch.compile(lambda: moe.run(binding=candidate), fullgraph=True)
        compiled()
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        allocated = torch.cuda.memory_allocated()
        for live in (m, min(17, m), 0, 1, m):
            count.fill_(live)
            logits.normal_()
            logits[live:].fill_(float("nan"))
            a.neg_()
            bias.neg_()
            for binding in bindings:
                for value in binding.views.values():
                    value.view(torch.uint8).fill_(0x7F)
            control.run()
            expected = tuple(
                value.clone()
                for value in (
                    control.output,
                    control.selected_logits,
                    control.topk_ids,
                    control.topk_weights,
                    control.views["input"],
                    control.views["input_sf"],
                    control.views["input_outer"],
                    control.views["counts"],
                    control.views["valid"],
                    control.views["intermediate"],
                    control.views["intermediate_q"],
                    control.views["intermediate_sf"],
                    control.views["intermediate_outer"],
                    control.views["routes"],
                )
            )
            graph.replay()
            actual = (
                candidate.output,
                candidate.selected_logits,
                candidate.topk_ids,
                candidate.topk_weights,
                candidate.views["input"],
                candidate.views["input_sf"],
                candidate.views["input_outer"],
                candidate.views["counts"],
                candidate.views["valid"],
                candidate.views["intermediate"],
                candidate.views["intermediate_q"],
                candidate.views["intermediate_sf"],
                candidate.views["intermediate_outer"],
                candidate.views["routes"],
            )
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    got.view(torch.uint8), want.view(torch.uint8), rtol=0, atol=0
                )
            del expected, actual, got, want
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated


@pytest.mark.parametrize("csf", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "m,h,i,e",
    [(1, 128, 128, 4), (17, 128, 128, 4), (48, 6144, 768, 128), (4096, 128, 128, 4)],
)
def test_caller_owned_route_output_matches_staged_chain(
    csf, ids_dtype, m, h, i, e, fuse_tasks=False
):
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._token_outer import Config, _sum
    from b12x.preparation import PreparationSession, PreparedCall

    torch.manual_seed(3419)
    experts, *_ = _weights(e, h, i, csf=csf)
    source = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (m, 4), dtype=ids_dtype, device="cuda")
    weights = torch.rand(m, 4, device="cuda")
    token_output = torch.empty_like(source)
    route_output = torch.empty(m, 4, h, dtype=source.dtype, device=source.device)
    actual = torch.empty_like(source)
    expansion = (
        moe.plan_scale_expansion(experts, routes=ids.numel(), ids_dtype=ids_dtype)
        if csf
        else None
    )
    plans = [
        moe.plan_execution(
            experts=experts,
            capacity=moe.ExecutionCapacity(max_tokens=m, top_k=4, output_layout=layout),
            override=Config(
                32, 64, 256, 4, fuse_tasks=fuse_tasks and layout == "routes"
            ),
            scale_expansion=expansion,
        )
        for layout in ("token", "routes")
    ]
    outputs = (token_output, route_output)
    bindings = []

    def prepare(state, output):
        scratch = tuple(
            torch.empty(spec.shape, device=source.device, dtype=spec.dtype)
            for spec in state.scratch_specs()
        )
        binding = state.bind(
            a=source, topk_ids=ids, topk_weights=weights, output=output, scratch=scratch
        )
        bindings.append(binding)
        return PreparedCall(run=binding.run, owners=(binding, scratch, output))

    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        requests = []
        if expansion is not None:

            def scales(state):
                active = torch.empty(e, dtype=torch.int32, device=source.device)
                binding = state.bind(
                    ids=ids,
                    out13=experts._impl.w1_blockscale,
                    out2=experts._impl.w2_blockscale,
                    scratch=active,
                )
                return PreparedCall(run=binding.run, owners=(binding, active))

            requests.append(expansion.request(name="scales", prepare_call=scales))
        from functools import partial

        requests.extend(
            plan.request(
                name=f"output:{index}",
                prepare_call=partial(prepare, output=output),
                dependencies=("scales",) if expansion is not None else (),
            )
            for index, (plan, output) in enumerate(zip(plans, outputs, strict=True))
        )
        session.prepare(tuple(requests))
        token, routes = bindings
        assert "routes" in token.views and "routes" not in routes.views
        assert plans[1].scratch_specs()[0].nbytes < plans[0].scratch_specs()[0].nbytes

        def run():
            token.run()
            routes.run()
            _sum[(m, triton.cdiv(h, 256))](
                route_output,
                weights,
                ids,
                actual,
                m,
                h,
                4,
                e,
                256,
                enable_fp_fusion=False,
            )

        run()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for live in (m, min(m, 17), 0, 1, m):
            ids.random_(0, e)
            ids[:, 1].copy_(ids[:, 0])
            ids[live:].fill_(-1)
            ids[-1, -1] = e
            weights.uniform_()
            source.normal_()
            for binding in bindings:
                for view in binding.views.values():
                    view.fill_(0)
                    view.view(torch.uint8).fill_(255)
            route_output.fill_(float("nan"))
            token_output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            valid = (ids >= 0) & (ids < e)
            torch.testing.assert_close(
                route_output.view(torch.int16)[valid],
                token.views["routes"].view_as(route_output).view(torch.int16)[valid],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual, token_output, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("block_m", [16, 32])
def test_csf_fused_task_metadata_matches_task_kernel(ids_dtype, mode, block_m):
    require_b12x()
    from b12x._lib.quant.nvfp4_csf import compile_nvfp4_csf_pair, decode_nvfp4_csf_pair
    from b12x.moe.fused_moe._token_outer import _tasks

    torch.manual_seed(1197)
    e, capacity = 128, 192
    experts, *_ = _weights(e, 128, 128, csf=True)
    decoder = experts._impl.nvfp4_csf
    raw = torch.randint(0, e, (capacity,), dtype=ids_dtype, device="cuda")
    counts = torch.empty(e, dtype=torch.int32, device="cuda")
    unique = torch.arange(e, dtype=ids_dtype, device="cuda")
    metadata = tuple(
        torch.empty(capacity + e, dtype=torch.int32, device="cuda") for _ in range(4)
    )
    actual_expert, actual_begin, expected_expert, expected_begin = metadata
    program = compile_nvfp4_csf_pair(
        decoder.first.geometry,
        decoder.second.geometry,
        mode != 2 and ids_dtype == torch.int64,
        fuse_tasks=True,
    )

    def run():
        decode_nvfp4_csf_pair(
            decoder.first,
            decoder.second,
            counts if mode == 2 else unique if mode in (1, 3) else raw,
            experts._impl.w1_blockscale,
            experts._impl.w2_blockscale,
            mode=mode,
            program=program,
            task_metadata=(counts, actual_expert, actual_begin),
            task_block_m=block_m,
        )

    counts.fill_(1)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for live in (capacity, 33, 0, 1, capacity):
        raw.fill_(-1)
        if live:
            raw[:live].random_(0, e)
            raw[: max(1, live // 2)].fill_(e - 1)
            raw[live - 1] = e
        valid = (raw >= 0) & (raw < e)
        counts.copy_(torch.bincount(raw[valid].long(), minlength=e).int())
        actual_expert.fill_(-1)
        expected_expert.fill_(-1)
        actual_begin.fill_(-333)
        expected_begin.fill_(-777)
        _tasks[(e,)](
            counts,
            expected_expert,
            expected_begin,
            e,
            block_m,
            triton.next_power_of_2(e),
            triton.next_power_of_2(triton.cdiv(capacity, block_m)),
        )
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        torch.testing.assert_close(actual_expert, expected_expert, rtol=0, atol=0)
        active = expected_expert >= 0
        torch.testing.assert_close(
            actual_begin[active], expected_begin[active], rtol=0, atol=0
        )
    graph.reset()


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "m,h,i,e",
    [(1, 128, 128, 4), (17, 384, 128, 4), (48, 6144, 768, 128), (4096, 128, 128, 4)],
)
def test_csf_fused_tasks_match_prepared_chain(m, h, i, e, ids_dtype):
    _check_prepared_moe(
        m, h, i, e, 4, (16, 64, 256, 4), csf=True, ids_dtype=ids_dtype, fuse_tasks=True
    )


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "m,h,i,e",
    [(1, 128, 128, 4), (17, 128, 128, 4), (48, 6144, 768, 128), (4096, 128, 128, 4)],
)
def test_csf_fused_tasks_route_outputs_match_staged_chain(m, h, i, e, ids_dtype):
    test_caller_owned_route_output_matches_staged_chain(
        True, ids_dtype, m, h, i, e, fuse_tasks=True
    )
