"""Independent compact W4A8 FC1 gate/up projection kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x.moe._shared.kernels.routing_ids import local_expert_id

from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_shared_u8,
)
from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_f32x4_to_e4m3x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.mma import e2m1x8_to_qmma_e2m1x8
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x.moe._shared.kernels.w4a8_compact_activation import (
    W4A8CompactMicroActivationKernel,
)


@cute.jit
def _stage_compact_input_k128(
    source: cute.Tensor,
    a_base: Int32,
    sfa_base: Int32,
    tid: Int32,
    token: Int32,
    k128_slice: Int32,
    k: cutlass.Constexpr,
):
    """Quantize a contiguous K128 slice with eight lanes per K32 block."""
    if tid < Int32(32):
        values = cute.make_rmem_tensor((4,), cutlass.Float32)
        offset = (
            Int64(token) * Int64(k)
            + Int64(k128_slice) * Int64(128)
            + Int64(tid * Int32(4))
        )
        for elem in cutlass.range_constexpr(4):
            values[elem] = cutlass.Float32(source[offset + Int64(elem)])
        max_abs = fabs_f32(values[0])
        for elem in cutlass.range_constexpr(1, 4):
            max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
        for shift in cutlass.range_constexpr(3):
            max_abs = fmax_f32(
                max_abs,
                cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
            )
        _, scale_byte = pow2_ceil_ue8m0(
            max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
        )
        inv_scale = ue8m0_to_output_scale(scale_byte)
        payload = cvt_f32x4_to_e4m3x4(
            values[0] * inv_scale,
            values[1] * inv_scale,
            values[2] * inv_scale,
            values[3] * inv_scale,
        )
        st_shared_u32(a_base + tid * Int32(4), payload)
        if tid % Int32(8) == Int32(0):
            st_shared_u8(sfa_base + tid // Int32(8), cutlass.Uint8(scale_byte))


@dsl_user_op
def _mma_fp4_weights_fp8_activation(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    block: int,
    *,
    loc=None,
    ip=None,
):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(value).ir_value(loc=loc, ip=ip)
            for value in (a0, a1, a2, a3, b0, b1, sfa, sfb)
        ],
        f"""
        {{
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0
            {{$0, $1, $2, $3}},
            {{$4, $5, $6, $7}},
            {{$8, $9}},
            {{zero, zero, zero, zero}},
            {{$10}}, {{{int(block)}, 0}},
            {{$11}}, {{{int(block)}, 0}};
        }}
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


class W4A8CompactMicroProjectionKernel:
    """Project compact output tiles with expert weights in the QMMA row operand."""

    tile_m = 16
    tile_n = 128
    tile_k = 128
    stages = 2
    num_warps = 4
    threads_per_cta = 128

    def __init__(
        self,
        k: int,
        n: int,
        num_topk: int,
        *,
        tile_n: int = 128,
        csf_inline: bool = False,
        max_copy_groups: int = 2,
        pipeline_stages: int = 2,
    ):
        if k <= 0 or n < 64 or k % 128 or n % 64:
            raise ValueError(
                "compact W4A8 projection requires K divisible by 128 and "
                "N divisible by 64"
            )
        if num_topk <= 0:
            raise ValueError("num_topk must be positive")
        if type(tile_n) is not int or tile_n not in (64, 128):
            raise ValueError("compact FC1 tile N must be 64 or 128")
        if tile_n == 64 and n % 128 != 64:
            raise ValueError("compact FC1 N64 tiles require the native N64-tail layout")
        if type(max_copy_groups) is not int or max_copy_groups not in (1, 2):
            raise ValueError("compact FC1 permits one or two pending copy groups")
        if type(pipeline_stages) is not int or pipeline_stages not in (1, 2):
            raise ValueError("compact FC1 pipeline stages must be one or two")
        if pipeline_stages == 1 and max_copy_groups != 1:
            raise ValueError("one compact FC1 stage requires one pending copy group")
        self.max_copy_groups = max_copy_groups
        self.stages = pipeline_stages
        self.k = int(k)
        self.n = int(n)
        self.num_topk = int(num_topk)
        self.csf_inline = bool(csf_inline)
        self.n64_tail = self.n % 128 == 64
        self.tile_n = tile_n
        self.num_warps = self.tile_n // 32
        self.threads_per_cta = self.num_warps * 32
        self.n_fragments = self.tile_n // (self.num_warps * 16)
        self.fuse_input_quantization = False
        self.share_projection_input = False
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.b_payload_bytes = self.tile_n * self.tile_k // 2
        self.sfb_bytes = 512 if self.csf_inline else self.tile_n * 4
        self.a_offset = 0
        self.sfa_offset = 128
        self.b_offset = 256
        self.sfb_offset = self.b_offset + self.b_payload_bytes
        self.stage_bytes = self.sfb_offset + self.sfb_bytes
        self.shared_words = self.stages * self.stage_bytes // 4

    @cute.jit
    def __call__(
        self,
        a8: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        a_source = cute.recast_tensor(a8, cutlass.Uint32)
        projected_flat = cute.make_tensor(
            projections.iterator,
            cute.make_layout((projections.shape[0] * projections.shape[1],)),
        )
        grid = num_pairs * Int32(2 * self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(2))
        self.kernel(
            a_source,
            scale_rows,
            w13_rp,
            w13_sfb,
            projected_flat,
            topk_ids,
            alpha,
            input_scale,
            num_pairs,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
    ):
        # Prepared RP is [K32, N32-chunk, lane, n8-in-chunk].  A CTA owns
        # one N128 half, retaining coalesced four-word lane loads.
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_cta):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(512):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(4)
                kb = kc // Int32(4)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global(
                    dst_base + (idx << Int32(4)), get_ptr_as_int64(weights, src_word)
                )

    @cute.jit
    def _stage_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
    ):
        if tid < Int32(32):
            src_word = tile_word_base + Int64(packed_half * Int32(128) + tid * 4)
            cp_async4_shared_global(
                dst_base + (tid << Int32(4)), get_ptr_as_int64(scales, src_word)
            )

    @cute.jit
    def _stage_n64_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_cta):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(4 * (self.tile_n // 32) * 32):
                lane = idx & Int32(31)
                tmp = idx >> Int32(5)
                n32 = tmp % Int32(self.tile_n // 32)
                kb = tmp // Int32(self.tile_n // 32)
                copy_words = Int32(1)
                if cutlass.const_expr(self.tile_n == 128):
                    copy_words = (n32 < n_rows // Int32(32)).to(Int32)
                dst_addr = dst_base + (idx << Int32(4))
                if copy_words != Int32(0):
                    source_word = (
                        Int64(expert_idx) * Int64(self.n * (self.k // 32) * 8)
                        + Int64(projection) * Int64(self.n * (self.k // 32) * 4)
                        + Int64(physical_tile) * Int64(128 * (self.k // 32) * 4)
                        + Int64(k128_slice * Int32(4) + kb)
                        * Int64(n_rows // Int32(32))
                        * Int64(128)
                        + Int64(
                            (packed_half * Int32(self.tile_n // 32) + n32) * Int32(128)
                            + lane * Int32(4)
                        )
                    )
                    cp_async4_shared_global(
                        dst_addr, get_ptr_as_int64(weights, source_word)
                    )
                else:
                    st_shared_u32(dst_addr, Uint32(0))
                    st_shared_u32(dst_addr + Int32(4), Uint32(0))
                    st_shared_u32(dst_addr + Int32(8), Uint32(0))
                    st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_n64_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        if cutlass.const_expr(self.csf_inline):
            if tid < Int32(32):
                target = cute.make_ptr(
                    cutlass.Uint8, dst_base, cute.AddressSpace.smem, assumed_align=16
                )
                group = (
                    projection * Int32(self.csf_reader.plane.storage.groups)
                    + physical_tile * Int32(self.csf_reader.plane.storage.column_groups)
                    + k128_slice
                )
                self.csf_reader.stage_compressed(
                    scales.iterator,
                    expert_idx,
                    group,
                    target,
                    tid,
                    coordinates=(projection, physical_tile, k128_slice),
                )
            return
        if tid < Int32(self.tile_n // 4):
            dst_addr = dst_base + (tid << Int32(4))
            copy_words = Int32(1)
            if cutlass.const_expr(self.tile_n == 128):
                copy_words = (tid < n_rows // Int32(4)).to(Int32)
            if copy_words != Int32(0):
                source_word = (
                    Int64(expert_idx) * Int64(self.n * (self.k // 32) // 2)
                    + Int64(projection) * Int64(self.n * (self.k // 32) // 4)
                    + Int64(physical_tile) * Int64(128 * (self.k // 32) // 4)
                    + Int64(
                        k128_slice * n_rows
                        + packed_half * Int32(self.tile_n)
                        + tid * Int32(4)
                    )
                )
                cp_async4_shared_global(dst_addr, get_ptr_as_int64(scales, source_word))
            else:
                st_shared_u32(dst_addr, Uint32(0))
                st_shared_u32(dst_addr + Int32(4), Uint32(0))
                st_shared_u32(dst_addr + Int32(8), Uint32(0))
                st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_slice(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
        k128_slice: Int32,
    ):
        stage_base = smem_base
        if cutlass.const_expr(self.stages == 2):
            stage_base += (k128_slice & Int32(1)) * Int32(self.stage_bytes)
        token = pair // Int32(self.num_topk)
        if cutlass.const_expr(not self.fuse_input_quantization):
            copy_input = Int32(1)
            if cutlass.const_expr(self.share_projection_input):
                copy_input = (projection_tile < Int32(self.n_tiles)).to(Int32)
            if copy_input != Int32(0):
                if tid < Int32(8):
                    src_word = (
                        Int64(token) * Int64(self.k // 4)
                        + Int64(k128_slice) * Int64(32)
                        + Int64(tid * 4)
                    )
                    cp_async4_shared_global(
                        stage_base + Int32(self.a_offset) + tid * Int32(16),
                        get_ptr_as_int64(a_source, src_word),
                    )
                if tid == Int32(0):
                    sf_src = Int64(token) * Int64(self.k // 32) + Int64(
                        k128_slice
                    ) * Int64(4)
                    cp_async_u32_shared_global(
                        stage_base + Int32(self.sfa_offset),
                        get_ptr_as_int64(scale_words, sf_src),
                    )
        b_base = stage_base + Int32(self.b_offset)
        sfb_base = stage_base + Int32(self.sfb_offset)
        if cutlass.const_expr(self.n64_tail):
            projection = projection_tile // Int32(self.n_tiles)
            output_tile = projection_tile % Int32(self.n_tiles)
            self._stage_n64_b_k128(
                w13_rp,
                b_base,
                expert,
                projection,
                output_tile,
                k128_slice,
                tid,
            )
            self._stage_n64_sfb(
                w13_sfb,
                sfb_base,
                expert,
                projection,
                output_tile,
                k128_slice,
                tid,
            )
        else:
            packed_tile = projection_tile // Int32(2)
            packed_half = projection_tile % Int32(2)
            packed_tile_index = (
                Int64(expert) * Int64(self.n // 128) + Int64(packed_tile)
            ) * Int64(self.k // 128) + Int64(k128_slice)
            tile_word_base = packed_tile_index * Int64(4096)
            self._stage_b_k128(
                w13_rp,
                b_base,
                tile_word_base,
                packed_half,
                tid,
            )
            scale_word_base = packed_tile_index * Int64(256)
            self._stage_sfb(
                w13_sfb,
                sfb_base,
                scale_word_base,
                packed_half,
                tid,
            )

        if cutlass.const_expr(self.fuse_input_quantization):
            if projection_tile < Int32(self.n_tiles):
                # Weight copies are in flight before the shared input is encoded.
                _stage_compact_input_k128(
                    a_source,
                    stage_base + Int32(self.a_offset),
                    stage_base + Int32(self.sfa_offset),
                    tid,
                    token,
                    k128_slice,
                    self.k,
                )

    @cute.jit
    def _run_task(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        input_smem_base = smem_base
        if cutlass.const_expr(self.share_projection_input):
            # Paired projection groups consume group0's encoded input ring.
            input_smem_base -= (projection_tile // Int32(self.n_tiles)) * Int32(
                self.shared_words * 4
            )
        for preload in cutlass.range_constexpr(1):
            if cutlass.const_expr(preload < self.k // 128):
                self._stage_slice(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    smem_base,
                    tid,
                    pair,
                    expert,
                    projection_tile,
                    Int32(preload),
                )
            cute.arch.cp_async_commit_group()

        acc = tuple(
            cute.make_rmem_tensor((4,), cutlass.Float32)
            for _ in range(self.n_fragments)
        )
        for nt in cutlass.range_constexpr(self.n_fragments):
            acc[nt].fill(0.0)
        k128_slice = Int32(0)
        k128_tiles = Int32(self.k // 128)
        b_scale_words = cute.make_rmem_tensor((self.n_fragments,), Uint32)
        b_scale_words.fill(Uint32(0))
        while k128_slice < k128_tiles:
            stage_base = smem_base
            input_stage_base = input_smem_base
            if cutlass.const_expr(self.stages == 2):
                stage_offset = (k128_slice & Int32(1)) * Int32(self.stage_bytes)
                stage_base += stage_offset
                input_stage_base += stage_offset
            b_base = stage_base + Int32(self.b_offset)
            sfb_base = stage_base + Int32(self.sfb_offset)
            if cutlass.const_expr(self.max_copy_groups == 1):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            next_slice = k128_slice + Int32(1)
            if cutlass.const_expr(self.stages == 2):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        smem_base,
                        tid,
                        pair,
                        expert,
                        projection_tile,
                        next_slice,
                    )
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.max_copy_groups == 2):
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
            if cutlass.const_expr(self.csf_inline):
                self.csf_reader.expand_slots(
                    w13_sfb.iterator, sfb_base, tid, self.threads_per_cta, 1
                )
                cute.arch.sync_threads()
                if cutlass.const_expr(self.tile_n == 64):
                    sfb_base += (
                        projection_tile % Int32(self.n_tiles) & Int32(1)
                    ) * Int32(256)
            for nt in cutlass.range_constexpr(self.n_fragments):
                n8 = (
                    warp * Int32(self.n_fragments * 2)
                    + Int32(nt * 2)
                    + (lane & Int32(1))
                )
                b_scale_words[nt] = ld_shared_u32(
                    sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                )
            asc = ld_shared_u32(input_stage_base + Int32(self.sfa_offset))
            for kb in cutlass.range_constexpr(4):
                # One activation is repeated across the eight hardware columns.
                a_src = (
                    input_stage_base
                    + Int32(self.a_offset)
                    + Int32(kb * 32)
                    + c * Int32(8)
                )
                a0, a2 = ld_shared_v2_u32(a_src)
                b_words = cute.make_rmem_tensor((self.n_fragments * 2,), Uint32)
                for chunk in cutlass.range_constexpr(self.n_fragments // 2):
                    source_chunk = (
                        Int32(kb * (self.tile_n // 32))
                        + warp * Int32(self.n_fragments // 2)
                        + Int32(chunk)
                    )
                    bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                        b_base + ((source_chunk * Int32(32) + lane) << Int32(4))
                    )
                    b_words[chunk * 4] = bw0
                    b_words[chunk * 4 + 1] = bw1
                    b_words[chunk * 4 + 2] = bw2
                    b_words[chunk * 4 + 3] = bw3
                for nt in cutlass.range_constexpr(self.n_fragments):
                    w0, w2 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2])
                    w1, w3 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2 + 1])
                    sfb = b_scale_words[nt]
                    fragment = acc[nt]
                    x0, x1, x2, x3 = _mma_fp4_weights_fp8_activation(
                        w0,
                        w1,
                        w2,
                        w3,
                        a0,
                        a2,
                        sfb,
                        asc,
                        block=kb,
                    )
                    fragment[0] = fragment[0] + x0
                    fragment[1] = fragment[1] + x1
                    fragment[2] = fragment[2] + x2
                    fragment[3] = fragment[3] + x3
            if cutlass.const_expr(self.max_copy_groups == 2 or self.stages == 1):
                cute.arch.sync_threads()
            if cutlass.const_expr(self.stages == 1):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        smem_base,
                        tid,
                        pair,
                        expert,
                        projection_tile,
                        next_slice,
                    )
                cute.arch.cp_async_commit_group()
            k128_slice += Int32(1)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scale_expert = expert
        if cutlass.const_expr(input_scale.shape[0] == 1):
            scale_expert = Int32(0)
        value_scale = alpha[expert].to(cutlass.Float32) * input_scale[
            Int64(scale_expert)
        ].to(cutlass.Float32)
        # The prepared RP stores up first.  The seam is canonical gate then up.
        source_tile = projection_tile % Int32(self.n_tiles)
        dst_base = source_tile * Int32(self.tile_n)
        if projection_tile < Int32(self.n_tiles):
            dst_base += Int32(self.n)
        if c == Int32(0):
            col_base = warp * Int32(self.tile_n // self.num_warps) + q
            for nt in cutlass.range_constexpr(self.n_fragments):
                tile_col = col_base + Int32(nt * 16)
                if source_tile * Int32(self.tile_n) + tile_col < Int32(self.n):
                    col = dst_base + tile_col
                    fragment = acc[nt]
                    projections[Int64(pair) * Int64(2 * self.n) + Int64(col)] = (
                        value_scale * fragment[0]
                    ).to(cutlass.BFloat16)
                    projections[
                        Int64(pair) * Int64(2 * self.n) + Int64(col + Int32(8))
                    ] = (value_scale * fragment[2]).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        task = Int32(bidx)
        projection_tiles = Int32(2 * self.n_tiles)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        while task < num_pairs * projection_tiles:
            pair = task % num_pairs
            projection_tile = task // num_pairs
            expert = local_expert_id(topk_ids[Int64(pair)], Int32(alpha.shape[0]))
            if expert >= Int32(0) and expert < Int32(alpha.shape[0]):
                self._run_task(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    projections,
                    alpha,
                    input_scale,
                    shared_ptr_to_u32(storage.words.data_ptr()),
                    tid,
                    cute.arch.make_warp_uniform(cute.arch.warp_idx()),
                    pair,
                    expert,
                    projection_tile,
                )
            task += Int32(gdimx)


class W4A8CompactMicroProjectionActivationKernel(W4A8CompactMicroProjectionKernel):
    """Pair gate/up projection groups and materialize their MXFP8 activation."""

    def __init__(
        self,
        k: int,
        n: int,
        num_topk: int,
        activation: W4A8CompactMicroActivationKernel,
        *,
        csf_inline: bool = False,
        max_copy_groups: int = 2,
        fuse_input_quantization: bool = False,
        pipeline_stages: int = 2,
    ):
        super().__init__(
            k,
            n,
            num_topk,
            csf_inline=csf_inline,
            max_copy_groups=max_copy_groups,
            pipeline_stages=pipeline_stages,
        )
        self.num_warps = 2
        self.threads_per_cta = self.num_warps * 32
        self.n_fragments = self.tile_n // (self.num_warps * 16)
        self.activation = activation
        self.share_projection_input = True
        self.fuse_input_quantization = bool(fuse_input_quantization)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        grid = num_pairs * Int32(self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(2))
        a_source = a
        if cutlass.const_expr(not self.fuse_input_quantization):
            a_source = cute.recast_tensor(a, cutlass.Uint32)
        self.fused_kernel(
            a_source,
            scale_rows,
            w13_rp,
            w13_sfb,
            projections,
            intermediate,
            topk_ids,
            alpha,
            input_scale,
            num_pairs,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta * 2, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.kernel
    def fused_kernel(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        group = tid // Int32(self.threads_per_cta)
        local_tid = tid % Int32(self.threads_per_cta)
        local_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(
            self.num_warps
        )
        task = Int32(bidx)
        projected_flat = cute.make_tensor(
            projections.iterator,
            cute.make_layout((projections.shape[0] * projections.shape[1],)),
        )
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words * 2], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        while task < num_pairs * Int32(self.n_tiles):
            pair = task % num_pairs
            output_tile = task // num_pairs
            expert = local_expert_id(topk_ids[Int64(pair)], Int32(alpha.shape[0]))
            if expert >= Int32(0) and expert < Int32(alpha.shape[0]):
                self._run_task(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    projected_flat,
                    alpha,
                    input_scale,
                    smem_base + group * Int32(self.shared_words * 4),
                    local_tid,
                    local_warp,
                    pair,
                    expert,
                    output_tile + group * Int32(self.n_tiles),
                )
                # Both BF16 projections must be visible before activation reads.
                cute.arch.sync_threads()
                if tid < Int32(32):
                    self.activation.activate_tile(
                        projections, intermediate, pair, output_tile, tid
                    )
                cute.arch.sync_threads()
            task += Int32(gdimx)


__all__ = [
    "W4A8CompactMicroProjectionKernel",
    "W4A8CompactMicroProjectionActivationKernel",
]
