"""Fused paged MXFP4 tile arithmetic, quantization, and input validation.

Selection requires every query head contributing to the score on each rank.
Hierarchical block selection retains its mandatory newest scored block.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..._lib.compiler import KernelCompileSpec, compile as b12x_compile
from ..._lib.program_cache import program_cache
from b12x._lib.intrinsics.conversion import (
    cvt_fp32x2_to_e2m1x2,
    f16x2_to_f32x2,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.atomics import ld_shared_u32
from ..._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

MXFP4_INDEX_PAGE_SIZE = 64
MXFP4_INDEX_PAGE_BYTES = 64 * 68


def index_mxfp4_page_bytes(page_size: int = MXFP4_INDEX_PAGE_SIZE) -> int:
    """Page storage: ``[page_size,64]`` data, then ``[page_size,4]`` UE8M0.

    Adjacent E2M1 values occupy low/high nibbles. Every data row is 16-byte
    aligned; scales are not interleaved into a 68-byte token stride.
    """
    if page_size <= 0 or page_size % 8:
        raise ValueError("MXFP4 page_size must be a positive multiple of eight")
    return page_size * 68


def _ptr(tensor, dtype):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)


@cute.jit
def _flat(ptr: cute.Pointer):
    # All actual accesses are bounded by runtime extents. The large logical
    # extent avoids narrowing pointer arithmetic for high recycled pool pages.
    return cute.make_tensor(ptr, cute.make_layout((Int64(1) << Int64(40),)))


class _Quantize:
    def __init__(self, paged: bool, page_size: int):
        self.paged = paged
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        scales: cute.Pointer,
        slots: cute.Pointer,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(q),
            _flat(scales),
            _flat(slots),
            rows,
            pool_pages,
            page_stride,
        ).launch(grid=((rows * 4 + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        q: cute.Tensor,
        scales: cute.Tensor,
        slots: cute.Tensor,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        group = Int32(bx) * Int32(8) + Int32(tx) // Int32(32)
        row = group // Int32(4)
        lane = Int32(tx) % Int32(32)
        if row < rows:
            v = Float32(x[Int64(group) * Int64(32) + Int64(lane)])
            amax = fabs_f32(v)
            for shift in cutlass.range_constexpr(5):
                amax = fmax_f32(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                )
            amax = fmax_f32(amax, Float32(6.0 * 2.0**-126))
            _, sf = pow2_ceil_ue8m0(amax * Float32(1.0 / 6.0))
            scaled = v * ue8m0_to_output_scale(sf)
            other = cute.arch.shuffle_sync_bfly(scaled, offset=1)
            packed = cvt_fp32x2_to_e2m1x2(scaled, other)
            data_base = Int64(row) * Int64(64)
            scale_base = Int64(row) * Int64(4)
            valid = True
            if cutlass.const_expr(self.paged):
                slot = Int64(slots[row])
                page = slot // Int64(self.page_size)
                token = slot % Int64(self.page_size)
                valid = slot >= Int64(0) and page < pool_pages
                data_base = page * page_stride + token * Int64(64)
                scale_base = (
                    page * page_stride + Int64(self.page_size * 64) + token * Int64(4)
                )
            if valid:
                if lane % Int32(2) == Int32(0):
                    q[
                        data_base
                        + Int64(group % Int32(4)) * Int64(16)
                        + Int64(lane // Int32(2))
                    ] = Uint8(packed)
                if lane == Int32(0):
                    scales[scale_base + Int64(group % Int32(4))] = Uint8(sf)


@dsl_user_op
def _mxfp4_mma(
    d0,
    d1,
    d2,
    d3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    sfa,
    sfb,
    *,
    scale_byte_id=0,
    loc=None,
    ip=None,
):
    """Packed E2M1 m16n8k64 with two UE8M0 group scales per operand row."""
    operands = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
    for scale in (sfa, sfb):
        operands.extend(
            (
                Uint32(scale).ir_value(loc=loc, ip=ip),
                Uint16(scale_byte_id).ir_value(loc=loc, ip=ip),
                Uint16(0).ir_value(loc=loc, ip=ip),
            )
        )
    operands.extend(Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1, d2, d3))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        operands,
        """
        mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
        {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$0,$1,$2,$3},
        {$10}, {$11,$12}, {$13}, {$14,$15};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cute.jit
def _packed_word(data: cute.Tensor, offset: Int64):
    return ld_global_nc_u32(get_ptr_as_int64(data, offset))


@cute.jit
def _scale_pair(data: cute.Tensor, offset: Int64):
    return Uint32(data[offset]) | (Uint32(data[offset + Int64(1)]) << Uint32(8))


@cute.jit
def _fused_qword(data: cute.Tensor, offset: Int64, aligned_q: cutlass.Constexpr[bool]):
    if cutlass.const_expr(aligned_q):
        return _packed_word(data, offset)
    return _scale_pair(data, offset) | (
        _scale_pair(data, offset + Int64(2)) << Uint32(16)
    )


class _TensorCorePagedScore:
    """Packed MXFP4 tile arithmetic with distinct BF16 rounding boundaries."""

    def __init__(self, heads: int, candidates: bool, page_size: int):
        self.heads = heads
        self.candidates = candidates
        self.page_size = page_size

    @cute.jit
    def _load_weight_pairs(self, weights: cute.Tensor, row: Int32):
        tx, _, _ = cute.arch.thread_idx()
        lane = Int32(tx) % Int32(32)
        packed = Uint32(0)
        if lane < Int32((self.heads + 1) // 2):
            w0 = weights[row, lane * Int32(2)]
            w1 = BFloat16(0)
            if cutlass.const_expr(self.heads != 1):
                w1 = weights[row, lane * Int32(2) + Int32(1)]
            packed = _fused_bf16_pack_weights(w0, w1)
        return packed

    @cute.jit
    def _score_tile_fused(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        tile_start: Int32,
        row_index: Int32 | None = None,
        local_output: cutlass.Constexpr[bool] = False,
        output_base: Int32 = 0,
        logical_indices: cute.Tensor | None = None,
        hardware_scale_select: cutlass.Constexpr[bool] = False,
        weight_pairs: Uint32 | None = None,
        aligned_q: cutlass.Constexpr[bool] = False,
        output_width: cutlass.Constexpr[int] = 0,
        carry_query_offsets: cutlass.Constexpr[bool] = False,
        logical_query_index: cutlass.Constexpr[bool] = False,
        unsigned_candidate_pages: cutlass.Constexpr[bool] = False,
    ):
        tx, _, _ = cute.arch.thread_idx()
        _, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(row_index is not None):
            row = row_index
        lane = Int32(tx) % Int32(32)
        store_width = Int32(output_width) if output_width > 0 else width
        col = Int64(tile_start) + Int64(tx // 32) * Int64(16) + Int64(lane // 4)
        if cutlass.const_expr(self.candidates):
            kb = cute.make_rmem_tensor((2,), Int64)
            valid_key = cute.make_rmem_tensor((2,), cutlass.Boolean)
            totals = cute.make_rmem_tensor((2,), Float32)
            totals.fill(0.0)
            column = col + Int64(lane % Int32(2)) * Int64(8)
            # Candidate IDs are signed 32-bit; storage offsets remain 64-bit.
            candidate_pos = Int32(-1)
            valid = column < width
            if cutlass.const_expr(self.candidates):
                valid = valid and column < candidate_lengths[row]
                if valid:
                    candidate_pos = Int32(
                        candidates[Int64(row) * Int64(width) + column]
                    )
            valid = (
                valid
                and candidate_pos >= Int32(0)
                and candidate_pos < lengths[row]
                and candidate_pos < active[0]
            )
            if cutlass.const_expr(unsigned_candidate_pages):
                candidate_page = Uint32(candidate_pos) // Uint32(self.page_size)
                valid = valid and candidate_page < Uint32(page_width)
            else:
                valid = valid and candidate_pos // Int32(self.page_size) < page_width
            page = Int64(-1)
            if valid:
                if cutlass.const_expr(unsigned_candidate_pages):
                    page = Int64(
                        pages[Int64(row) * page_row_stride + Int64(candidate_page)]
                    )
                else:
                    page = Int64(
                        pages[
                            Int64(row) * page_row_stride
                            + Int64(candidate_pos // Int32(self.page_size))
                        ]
                    )
            valid = valid and page >= Int64(0) and page < pool_pages
            if cutlass.const_expr(unsigned_candidate_pages):
                token = Int64(Uint32(candidate_pos) % Uint32(self.page_size))
            else:
                token = Int64(candidate_pos % Int32(self.page_size))
            selected_key_base = page * pool_stride + token * Int64(64)
            selected_scale_base = (
                page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
            )
            selected_scale_valid = valid
            # Selection starts after a CTA barrier, so IDs can precede their scores.
            if cutlass.const_expr(logical_indices is not None):
                if lane % Int32(4) < Int32(2) and column < width:
                    index_offset = Int64(row) * Int64(store_width) + column
                    if cutlass.const_expr(local_output):
                        index_offset = Int64(output_base) + column - Int64(tile_start)
                    logical_indices[index_offset] = (
                        Int32(candidate_pos) if valid else Int32(-1)
                    )
            # Each four-lane group shares both complete 64-bit key bases.
            for part in cutlass.range_constexpr(2):
                address_source_lane = lane // Int32(4) * Int32(4) + Int32(part)
                kb[part] = cute.arch.shuffle_sync(
                    selected_key_base, address_source_lane
                )
                valid_key[part] = cute.arch.shuffle_sync(
                    Int32(valid), address_source_lane
                ) != Int32(0)
        else:
            kb = cute.make_rmem_tensor((2,), Int64)
            sb = cute.make_rmem_tensor((2,), Int64)
            valid_key = cute.make_rmem_tensor((2,), cutlass.Boolean)
            totals = cute.make_rmem_tensor((2,), Float32)
            totals.fill(0.0)
            for part in cutlass.range_constexpr(2):
                column = col + Int32(part * 8)
                pos = column
                valid = column < width
                if cutlass.const_expr(self.candidates):
                    valid = valid and column < candidate_lengths[row]
                    if valid:
                        pos = Int64(
                            candidates[Int64(row) * Int64(width) + Int64(column)]
                        )
                valid = (
                    valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
                )
                valid = valid and pos // Int32(self.page_size) < page_width
                page = Int64(-1)
                if valid:
                    page = Int64(
                        pages[
                            Int64(row) * page_row_stride
                            + Int64(pos // Int32(self.page_size))
                        ]
                    )
                valid = valid and page >= Int64(0) and page < pool_pages
                token = Int64(pos % Int32(self.page_size))
                valid_key[part] = valid
                kb[part] = page * pool_stride + token * Int64(64)
                sb[part] = (
                    page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
                )
                # Selection starts after a CTA barrier, so IDs can precede their scores.
                if cutlass.const_expr(logical_indices is not None):
                    if lane % Int32(4) == Int32(0) and column < width:
                        index_offset = Int64(row) * Int64(store_width) + Int64(column)
                        if cutlass.const_expr(local_output):
                            index_offset = Int64(output_base) + Int64(
                                column - tile_start
                            )
                        logical_indices[index_offset] = (
                            Int32(pos) if valid else Int32(-1)
                        )
            selected_scale_base = sb[0]
            selected_scale_valid = valid_key[0]
            if lane % Int32(2) != Int32(0):
                selected_scale_base = sb[1]
                selected_scale_valid = valid_key[1]
        query_head_index = Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))
        qb = (Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))) * Int64(64)
        qsb = (Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))) * Int64(4)
        for head_tile in cutlass.range((self.heads + 7) // 8, unroll=1):
            head = Int32(head_tile * 8) + lane // Int32(4)
            if cutlass.const_expr(not carry_query_offsets):
                qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
                qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
            elif cutlass.const_expr(logical_query_index):
                qb = query_head_index * Int64(64)
                qsb = query_head_index * Int64(4)
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for kstep in cutlass.range_constexpr(2):
                byte_col = Int64(kstep * 32) + Int64(lane % Int32(4)) * Int64(4)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid_key[0]:
                    a0 = _packed_word(pool, kb[0] + byte_col)
                    a2 = _packed_word(pool, kb[0] + byte_col + Int64(16))
                if valid_key[1]:
                    a1 = _packed_word(pool, kb[1] + byte_col)
                    a3 = _packed_word(pool, kb[1] + byte_col + Int64(16))
                b0, b1 = Uint32(0), Uint32(0)
                scale_default = 0x7F7F7F7F if hardware_scale_select else 0x7F7F
                sfa, sfb = Uint32(scale_default), Uint32(scale_default)
                if selected_scale_valid:
                    sfa = _packed_word(pool, selected_scale_base)
                    if cutlass.const_expr(not hardware_scale_select):
                        sfa = (sfa >> Uint32(kstep * 16)) & Uint32(65535)
                if head < Int32(self.heads):
                    b0 = _fused_qword(q, qb + byte_col, aligned_q)
                    b1 = _fused_qword(q, qb + byte_col + Int64(16), aligned_q)
                    sfb = _fused_qword(qs, qsb, aligned_q)
                    if cutlass.const_expr(not hardware_scale_select):
                        sfb = (sfb >> Uint32(kstep * 16)) & Uint32(65535)
                d0, d1, d2, d3 = _mxfp4_mma(
                    d0,
                    d1,
                    d2,
                    d3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    sfa,
                    sfb,
                    scale_byte_id=kstep * 2 if hardware_scale_select else 0,
                )
            dots = cute.make_rmem_tensor((4,), Float32)
            dots[0], dots[1], dots[2], dots[3] = d0, d1, d2, d3
            packed_products = cute.make_rmem_tensor((2,), Uint32)
            if cutlass.const_expr(weight_pairs is None):
                output_head = Int32(head_tile * 8) + (lane % Int32(4)) * Int32(2)
                w0, w1 = BFloat16(0), BFloat16(0)
                if cutlass.const_expr(self.heads >= 8):
                    w0 = weights[Int64(row) * Int64(self.heads) + Int64(output_head)]
                    w1 = weights[
                        Int64(row) * Int64(self.heads) + Int64(output_head + Int32(1))
                    ]
                else:
                    if output_head < Int32(self.heads):
                        w0 = weights[
                            Int64(row) * Int64(self.heads) + Int64(output_head)
                        ]
                    if output_head + Int32(1) < Int32(self.heads):
                        w1 = weights[
                            Int64(row) * Int64(self.heads)
                            + Int64(output_head + Int32(1))
                        ]
                for pair in cutlass.range_constexpr(2):
                    packed_products[pair] = _fused_bf16_products_packed(
                        dots[pair * 2],
                        dots[pair * 2 + 1],
                        w0,
                        w1,
                        fused_relu=not self.candidates,
                    )
            else:
                weight_word = cute.arch.shuffle_sync(
                    weight_pairs, Int32(head_tile * 4) + lane % Int32(4)
                )
                for pair in cutlass.range_constexpr(2):
                    packed_products[pair] = _fused_bf16_products_word(
                        dots[pair * 2],
                        dots[pair * 2 + 1],
                        weight_word,
                        fused_relu=not self.candidates,
                    )
            for head_pair in cutlass.range_constexpr(min(4, (self.heads + 1) // 2)):
                source_lane = lane // Int32(4) * Int32(4) + Int32(head_pair)
                for pair in cutlass.range_constexpr(2):
                    packed = cute.arch.shuffle_sync(packed_products[pair], source_lane)
                    if cutlass.const_expr(self.heads == 1):
                        totals[pair] += u32_as_f32(packed << Uint32(16))
                    else:
                        totals[pair] = _fused_bf16_pair_accumulate(totals[pair], packed)
            if cutlass.const_expr(carry_query_offsets):
                if cutlass.const_expr(not logical_query_index):
                    qb += Int64(8 * 64)
                    qsb += Int64(8 * 4)
                else:
                    query_head_index += Int64(8)
        if lane % Int32(4) == Int32(0):
            for part in cutlass.range_constexpr(2):
                column = col + Int32(part * 8)
                if column < width:
                    value = BFloat16(-float("inf"))
                    if valid_key[part]:
                        value = BFloat16(totals[part])
                    offset = Int64(row) * Int64(store_width) + Int64(column)
                    if cutlass.const_expr(local_output):
                        offset = Int64(output_base) + Int64(column - tile_start)
                    scores[offset] = value

    @cute.jit
    def _score_tiles_direct_register(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        tile_start: Int32,
        row_index: Int32,
        aligned_q: cutlass.Constexpr[bool] = False,
    ):
        tx, _, _ = cute.arch.thread_idx()
        row = row_index
        lane = Int32(tx) % Int32(32)
        col = Int64(tile_start) + Int64(tx // 32) * Int64(32) + Int64(lane // 4)
        kb = cute.make_rmem_tensor((4,), Int64)
        sb = cute.make_rmem_tensor((4,), Int64)
        valid_key = cute.make_rmem_tensor((4,), cutlass.Boolean)
        logical_ids = cute.make_rmem_tensor((4,), Int32)
        totals = cute.make_rmem_tensor((4,), Float32)
        totals.fill(0.0)
        for part in cutlass.range_constexpr(4):
            column = col + Int32((part % 2) * 8 + (part // 2) * 16)
            pos = column
            valid = column < width
            if cutlass.const_expr(self.candidates):
                valid = valid and column < candidate_lengths[row]
                if valid:
                    pos = Int64(candidates[Int64(row) * Int64(width) + Int64(column)])
            valid = valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
            valid = valid and pos // Int32(self.page_size) < page_width
            page = Int64(-1)
            if valid:
                page = Int64(
                    pages[
                        Int64(row) * page_row_stride
                        + Int64(pos // Int32(self.page_size))
                    ]
                )
            valid = valid and page >= Int64(0) and page < pool_pages
            token = Int64(pos % Int32(self.page_size))
            valid_key[part] = valid
            logical_ids[part] = Int32(pos)
            kb[part] = page * pool_stride + token * Int64(64)
            sb[part] = (
                page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
            )
        for head_tile in cutlass.range((self.heads + 7) // 8):
            head = Int32(head_tile * 8) + lane // Int32(4)
            qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
            qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
            dots = cute.make_rmem_tensor((8,), Float32)
            dots.fill(0.0)
            for kstep in cutlass.range_constexpr(2):
                byte_col = Int64(kstep * 32) + Int64(lane % Int32(4)) * Int64(4)
                b0, b1 = Uint32(0), Uint32(0)
                sfb = Uint32(0x7F7F)
                if head < Int32(self.heads):
                    b0 = _fused_qword(q, qb + byte_col, aligned_q)
                    b1 = _fused_qword(q, qb + byte_col + Int64(16), aligned_q)
                    sfb = (
                        _fused_qword(qs, qsb, aligned_q) >> Uint32(kstep * 16)
                    ) & Uint32(65535)
                for tile in cutlass.range_constexpr(2):
                    pair = tile * 2
                    a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if valid_key[pair]:
                        a0 = _packed_word(pool, kb[pair] + byte_col)
                        a2 = _packed_word(pool, kb[pair] + byte_col + Int64(16))
                    if valid_key[pair + 1]:
                        a1 = _packed_word(pool, kb[pair + 1] + byte_col)
                        a3 = _packed_word(pool, kb[pair + 1] + byte_col + Int64(16))
                    scale_first = lane % Int32(2) == Int32(0)
                    scale_valid = (
                        valid_key[pair] if scale_first else valid_key[pair + 1]
                    )
                    scale_offset = sb[pair] if scale_first else sb[pair + 1]
                    sfa = Uint32(0x7F7F)
                    if scale_valid:
                        sfa = (
                            _packed_word(pool, scale_offset) >> Uint32(kstep * 16)
                        ) & Uint32(65535)
                    dbase = tile * 4
                    dots[dbase], dots[dbase + 1], dots[dbase + 2], dots[dbase + 3] = (
                        _mxfp4_mma(
                            dots[dbase],
                            dots[dbase + 1],
                            dots[dbase + 2],
                            dots[dbase + 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            sfa,
                            sfb,
                        )
                    )
            contributions = cute.make_rmem_tensor((8,), Float32)
            for part in cutlass.range_constexpr(4):
                output_head = (
                    Int32(head_tile * 8)
                    + (lane % Int32(4)) * Int32(2)
                    + Int32(part % 2)
                )
                weight = Float32(0)
                if output_head < Int32(self.heads):
                    weight = Float32(
                        weights[Int64(row) * Int64(self.heads) + Int64(output_head)]
                    )
                for tile in cutlass.range_constexpr(2):
                    dot = fmax_f32(Float32(BFloat16(dots[tile * 4 + part])), Float32(0))
                    contributions[tile * 4 + part] = Float32(BFloat16(dot * weight))
            packed_products = cute.make_rmem_tensor((4,), Uint32)
            for pair in cutlass.range_constexpr(4):
                packed_products[pair] = _fused_pack_bf16_pair(
                    BFloat16(contributions[pair * 2]),
                    BFloat16(contributions[pair * 2 + 1]),
                )
            for head_pair in cutlass.range_constexpr(min(4, (self.heads + 1) // 2)):
                source_lane = lane // Int32(4) * Int32(4) + Int32(head_pair)
                for pair in cutlass.range_constexpr(4):
                    packed = cute.arch.shuffle_sync(packed_products[pair], source_lane)
                    if cutlass.const_expr(self.heads == 1):
                        totals[pair] += u32_as_f32(packed << Uint32(16))
                    else:
                        totals[pair] = _fused_bf16_pair_accumulate(totals[pair], packed)
        value0, value1, value2, value3 = (
            Float32(-float("inf")),
            Float32(-float("inf")),
            Float32(-float("inf")),
            Float32(-float("inf")),
        )
        index0, index1, index2, index3 = Int32(-1), Int32(-1), Int32(-1), Int32(-1)
        if lane % Int32(4) == Int32(0):
            if valid_key[0]:
                value0, index0 = Float32(BFloat16(totals[0])), Int32(logical_ids[0])
            if valid_key[1]:
                value1, index1 = Float32(BFloat16(totals[1])), Int32(logical_ids[1])
            if valid_key[2]:
                value2, index2 = Float32(BFloat16(totals[2])), Int32(logical_ids[2])
            if valid_key[3]:
                value3, index3 = Float32(BFloat16(totals[3])), Int32(logical_ids[3])
        return value0, value1, value2, value3, index0, index1, index2, index3


@dataclass(frozen=True)
class _CompiledLauncher:
    raw: object
    dtypes: tuple

    @property
    def __b12x_dependencies__(self):
        return (self.raw,)

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys(self.raw)


@program_cache
def _compile(kind: str, recipe: tuple, device_index: int):
    if kind == "sort":
        from ._position_sort import _sort_positions

        topk, expand = recipe
        raise_if_kernel_resolution_frozen(
            "triton.compile", target=_sort_positions, cache_key=(kind, recipe)
        )
        with torch.cuda.device(device_index):
            raw = _sort_positions.warmup(
                torch.int32,
                torch.float32,
                torch.int32,
                torch.float32,
                torch.int32,
                torch.int32,
                torch.int32,
                K=topk,
                EXPAND=expand,
                num_warps=8 if expand else 4,
                grid=(1, 1, 1),
            )
        return _CompiledLauncher(raw, ())
    # Pointer-only launch ABIs: all live row/page/width/stride quantities are
    # runtime scalars and do not contribute to compile identity.
    if kind == "quantize":
        obj = _Quantize(*recipe)
        dtypes = (BFloat16, Uint8, Uint8, Int64)
        scalars = (Int32(1), Int64(1), Int64(4352))
    else:
        raise ValueError(f"unknown MXFP4 indexer kernel {kind}")
    key = (kind, recipe)
    raise_if_kernel_resolution_frozen("cute.compile", target=obj, cache_key=key)
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in dtypes
    )
    raw = b12x_compile(
        obj,
        *pointers,
        *scalars,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.indexer.mxfp4", 9, key),
    )
    return _CompiledLauncher(raw, dtypes)


def _launch(kind, recipe, tensors, scalars, *, launcher=None):
    if tensors[0].device.type != "cuda":
        raise ValueError("native MXFP4 indexer requires CUDA tensors")
    with torch.cuda.device(tensors[0].device):
        if launcher is None:
            launcher = _compile(kind, recipe, tensors[0].device.index)
        if kind == "sort":
            launcher.raw[(scalars[0], 1, 1)](*tensors, *recipe)
            return
        launcher.raw(
            *(
                _ptr(t, dtype)
                for t, dtype in zip(tensors, launcher.dtypes, strict=True)
            ),
            *scalars,
            current_cuda_stream(),
        )


class _FusedStateReset:
    def __init__(
        self,
        groups: int,
        row_words: int,
        coarse1: int,
        arrival: int,
        contiguous: bool = False,
    ):
        self.groups = int(groups)
        self.contiguous = bool(contiguous)
        self.words = groups * row_words if self.contiguous else groups * 520
        self.row_words = row_words
        self.coarse1 = coarse1
        self.arrival = arrival

    @cute.jit
    def __call__(self, state: cute.Tensor, live_rows: Int32, stream: cuda.CUstream):
        self.kernel(state, live_rows).launch(
            grid=((self.words + 1023) // 1024, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, state: cute.Tensor, live_rows: Int32):
        cute.arch.griddepcontrol_launch_dependents()
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        word = Int32(bx) * Int32(1024) + Int32(tx) * Int32(4)
        if cutlass.const_expr(self.contiguous):
            for j in cutlass.range_constexpr(4):
                if word + Int32(j) < Int32(self.words):
                    state[Int64(word + Int32(j))] = Int32(0)
        else:
            if (word < Int32(self.words)) & (word // Int32(520) < live_rows):
                group = word // Int32(520)
                slot = word - group * Int32(520)
                physical = slot
                if slot >= Int32(512):
                    physical = slot - Int32(512) + Int32(self.arrival)
                elif slot >= Int32(256):
                    physical = slot - Int32(256) + Int32(self.coarse1)
                offset = Int64(group) * Int64(self.row_words) + Int64(physical)
                st_global_v4_u32(
                    get_ptr_as_int64(state, offset),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
            inactive = Int64(live_rows) * Int64(self.row_words) + Int64(word)
            while inactive < Int64(self.groups * self.row_words):
                st_global_v4_u32(
                    get_ptr_as_int64(state, inactive),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
                inactive += Int64(((self.words + 1023) // 1024) * 1024)


@dataclass(frozen=True)
class _PreparedFusedMXFP4:
    launcher: object
    args: tuple
    scratch: tuple
    owners: tuple
    rows: int
    reset_launcher: object
    reset_args: tuple

    @property
    def __b12x_dependencies__(self):
        return (
            (self.launcher, self.reset_launcher)
            if self.reset_launcher is not None
            else (self.launcher,)
        )

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys(self.__b12x_dependencies__)

    def __call__(self):
        from b12x._lib.compiler import run_compiled

        with torch.cuda.device(self.scratch[-1].device):
            stream = current_cuda_stream()
            if self.reset_launcher is not None:
                run_compiled(
                    self.reset_launcher, (self.reset_args[0], Int32(self.rows), stream)
                )
            args = (*self.args[:13], stream, self.args[14])
            run_compiled(self.launcher, args)


def prepare_fused_mxfp4(
    *,
    q_mxfp4,
    q_scales,
    query_weights,
    index_k_cache,
    page_table,
    cache_lengths,
    active_width,
    output_indices,
    output_scores=None,
    candidate_indices=None,
    candidate_lengths=None,
    page_size=64,
    ctas_per_row=None,
    _vector_output_pairs=None,
):
    """Prepare streaming top-512 selection with complete per-rank head scores.

    Queries and keys use packed E2M1 with four UE8M0 scales per head. Query
    data and scale tensors require four-byte-aligned pointers. Inputs
    remain mutable at fixed addresses. The returned callable resets its fixed
    merge arena, scores and selects without a context-sized score tensor, and
    sorts selected logical positions. Hierarchical block production is separate.
    """
    from .fused_indexer import (
        DSAFusedIndexerKernel,
        _check_mxfp4_cooperative_residency,
        KV_LAYOUT_PAGED,
        _COOP_STATE_WORDS,
        _FUSED_COARSE1,
        _FUSED_STATE_ARRIVAL,
    )
    from .kernel import _to_kernel_tensor

    raise_if_kernel_resolution_frozen(
        "fused MXFP4 preparation", target=prepare_fused_mxfp4
    )
    device = q_mxfp4.device
    if device.type != "cuda" or q_mxfp4.ndim != 3:
        raise ValueError("q_mxfp4 must be a CUDA [rows, heads, 64] tensor")
    rows, heads, packed = map(int, q_mxfp4.shape)
    if rows <= 0 or rows >= 2**31 or heads != 32 or packed != 64:
        raise ValueError(
            "fused MXFP4 requires positive rows and 32 complete index heads"
        )
    page_bytes = index_mxfp4_page_bytes(page_size)

    def checked(tensor, name, shape, dtype, *, contiguous=True, align=1):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            raise ValueError(f"{name} must be {dtype} {tuple(shape)} on {device}")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % align:
            raise ValueError(f"{name} requires {align}-byte alignment")
        return tensor

    checked(q_mxfp4, "q_mxfp4", (rows, heads, 64), torch.uint8, align=4)
    checked(q_scales, "q_scales", (rows, heads, 4), torch.uint8, align=4)
    checked(query_weights, "query_weights", (rows, heads), torch.bfloat16, align=2)
    checked(cache_lengths, "cache_lengths", (rows,), torch.int32, align=4)
    checked(active_width, "active_width", (1,), torch.int32, align=4)
    checked(output_indices, "output_indices", (rows, 512), torch.int32, align=4)
    if output_scores is None:
        output_scores = torch.empty((rows, 512), device=device, dtype=torch.float32)
    checked(output_scores, "output_scores", (rows, 512), torch.float32, align=4)
    if (
        index_k_cache.ndim != 2
        or index_k_cache.shape[1] < page_bytes
        or index_k_cache.stride(1) != 1
        or index_k_cache.stride(0) < page_bytes
        or index_k_cache.stride(0) % 4
    ):
        raise ValueError("index_k_cache must contain aligned packed MXFP4 pages")
    checked(
        index_k_cache,
        "index_k_cache",
        index_k_cache.shape,
        torch.uint8,
        contiguous=False,
        align=4,
    )
    if (
        page_table.ndim != 2
        or page_table.shape[0] not in (1, rows)
        or page_table.stride(1) != 1
        or page_table.stride(0) < 0
    ):
        raise ValueError("page_table must have one shared row or one row per query")
    checked(
        page_table,
        "page_table",
        page_table.shape,
        torch.int32,
        contiguous=False,
        align=4,
    )
    if page_table.shape[0] == 1:
        page_table = page_table.expand(rows, -1)
    max_width = int(page_table.shape[1]) * page_size
    candidates = candidate_indices is not None
    if candidates:
        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != rows:
            raise ValueError("candidate_indices must have one row per query")
        checked(
            candidate_indices,
            "candidate_indices",
            candidate_indices.shape,
            torch.int32,
            align=4,
        )
        checked(candidate_lengths, "candidate_lengths", (rows,), torch.int32, align=4)
        width = int(candidate_indices.shape[1])
    else:
        if candidate_lengths is not None:
            raise ValueError("candidate_lengths requires candidate_indices")
        width = max_width
        candidate_indices = output_indices
        candidate_lengths = cache_lengths
    if not 0 < width < 2**31 or max_width >= 2**31:
        raise ValueError("logical positions must fit int32")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if ctas_per_row is None:
        ctas_per_row = max(1, min((width + 511) // 512, sms // rows))
    ctas_per_row = int(ctas_per_row)
    if ctas_per_row < 1:
        raise ValueError("fused CTAs per row must be positive")
    merge_threshold = width if rows * ctas_per_row > sms else 0
    raw_indices = output_indices
    raw_values = output_scores
    vector_pairs_eligible = (
        max_width <= 65536
        and raw_indices.data_ptr() % 8 == 0
        and raw_values.data_ptr() % 8 == 0
    )
    vector_output_pairs = (
        False if _vector_output_pairs is None else bool(_vector_output_pairs)
    )
    if vector_output_pairs and not vector_pairs_eligible:
        raise ValueError(
            "paired MXFP4 output stores require position capacity <=65536 "
            "and eight-byte-aligned output buffers"
        )
    kernel = DSAFusedIndexerKernel(
        num_heads_static=heads,
        topk=512,
        kv_layout=KV_LAYOUT_PAGED,
        ctas_per_group=ctas_per_row,
        num_sms=sms,
        merge_threshold=merge_threshold,
        max_seq_capacity=width,
        k_quant_page_stride=int(index_k_cache.stride(0)),
        mxfp4=True,
        mxfp4_aligned_q=True,
        candidates=candidates,
        page_size=page_size,
        score_width=width,
        ordered_output=True,
        output_position_capacity=max_width,
        state_reset_pdl=ctas_per_row > 1,
        vector_output_pairs=vector_output_pairs,
    )
    pack_elements, state_elements = kernel.merge_scratch_elements(rows)
    pack_v = torch.empty(pack_elements, dtype=torch.float32, device=device)
    pack_i = torch.empty(pack_elements, dtype=torch.int32, device=device)
    state = torch.zeros(state_elements, dtype=torch.int32, device=device)
    tensors = (
        q_mxfp4,
        query_weights,
        index_k_cache,
        q_scales,
        page_table,
        cache_lengths,
        cache_lengths,
        cache_lengths,
        raw_indices,
        raw_values,
        pack_v,
        pack_i,
        state,
    )
    dtypes = (
        Uint8,
        BFloat16,
        Uint8,
        Uint8,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Float32,
        Float32,
        Int32,
        Int32,
    )
    key = (
        "mxfp4",
        kernel.constructor_config,
        rows,
        heads,
        page_size,
        width,
        ctas_per_row,
        sms,
        candidates,
        max_width,
        vector_output_pairs,
        kernel.merge_threshold,
        kernel.mxfp4_merge_bits,
        kernel.mxfp4_histogram_count_bits,
        kernel.mxfp4_histogram_bitmap,
        kernel.mxfp4_serial_state_words,
        kernel.state_words_per_group,
        kernel.mxfp4_query_offset_mode,
        kernel.mxfp4_unsigned_candidate_pages,
        tuple(
            (tuple(t.shape), tuple(t.stride()), str(t.dtype))
            for t in (*tensors, active_width, candidate_indices)
        ),
    )
    with torch.cuda.device(device):
        args = []
        for t, d in zip(tensors, dtypes, strict=True):
            alignment = 1
            if t is raw_indices or t is raw_values:
                alignment = 8 if vector_output_pairs else 4
            elif t is pack_v or t is pack_i or t is state:
                alignment = 16
            args.append(_to_kernel_tensor(t, d, assumed_align=alignment))
        args = tuple(args)
        operands = tuple(
            _to_kernel_tensor(t, d, assumed_align=1)
            for t, d in (
                (q_scales, Uint8),
                (active_width, Int32),
                (candidate_indices, Int32),
                (candidate_lengths, Int32),
            )
        )
        operands = (*operands, Int32(width))
        args = (*args, current_cuda_stream(), operands)
        launcher = b12x_compile(
            kernel,
            *args,
            compile_spec=KernelCompileSpec.from_key(
                "attention.indexer.fused_indexer", 11, key
            ),
        )
        _check_mxfp4_cooperative_residency(kernel, launcher, num_rows=rows, num_sms=sms)
        reset_launcher = None
        reset_args = ()
        if kernel.merge_in_kernel:
            reset_args = (
                _to_kernel_tensor(state, Int32, assumed_align=16),
                Int32(rows),
            )
            reset_launcher = b12x_compile(
                _FusedStateReset(
                    rows,
                    kernel.state_words_per_group,
                    _FUSED_COARSE1,
                    _FUSED_STATE_ARRIVAL,
                    contiguous=(kernel.state_words_per_group != _COOP_STATE_WORDS),
                ),
                *reset_args,
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.indexer.mxfp4_state_reset",
                    6,
                    (state.numel(), kernel.state_words_per_group),
                ),
            )
    owners = (
        *tensors,
        active_width,
        candidate_indices,
        candidate_lengths,
        output_indices,
        output_scores,
    )
    prepared = _PreparedFusedMXFP4(
        launcher,
        args,
        (raw_indices, raw_values, pack_v, pack_i, state),
        owners,
        rows,
        reset_launcher,
        reset_args,
    )
    prepared()
    return prepared


def _check(tensor, name, shape, dtype, device):
    if tensor is None or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _quantize_q_mxfp4(
    query: torch.Tensor,
    *,
    q_mxfp4: torch.Tensor,
    q_scales: torch.Tensor,
    launcher,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the prepared query-quantizer launcher into caller-owned buffers."""
    if query.ndim < 2 or query.shape[-1] != 128:
        raise ValueError("query must have shape (...,128)")
    _check(query, "query", query.shape, torch.bfloat16, query.device)
    _check(q_mxfp4, "q_mxfp4", (*query.shape[:-1], 64), torch.uint8, query.device)
    _check(q_scales, "q_scales", (*query.shape[:-1], 4), torch.uint8, query.device)
    rows = query.numel() // 128
    if rows:
        _launch(
            "quantize",
            (False, 64),
            (query, q_mxfp4, q_scales, query),
            (rows, 0, 0),
            launcher=launcher,
        )
    return q_mxfp4, q_scales


def _quantize_write_index_k_mxfp4(
    keys: torch.Tensor,
    *,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int = 64,
    launcher,
) -> torch.Tensor:
    """Run the prepared paged-key writer against caller-owned physical slots."""
    page_bytes = index_mxfp4_page_bytes(page_size)
    if keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("keys must have shape (rows,128)")
    _check(keys, "keys", keys.shape, torch.bfloat16, keys.device)
    _check(slot_mapping, "slot_mapping", (keys.shape[0],), torch.int64, keys.device)
    _check_pool(index_k_cache, page_bytes, keys.device)
    if keys.shape[0]:
        _launch(
            "quantize",
            (True, page_size),
            (keys, index_k_cache, index_k_cache, slot_mapping),
            (keys.shape[0], index_k_cache.shape[0], index_k_cache.stride(0)),
            launcher=launcher,
        )
    return index_k_cache


def _check_pool(pool, page_bytes, device):
    if pool.ndim != 2 or pool.shape[1] != page_bytes or pool.dtype != torch.uint8:
        raise ValueError(f"index_k_cache must be uint8 (pages,{page_bytes})")
    if (
        pool.device != device
        or pool.stride(1) != 1
        or pool.stride(0) < page_bytes
        or pool.stride(0) % 16
        or pool.data_ptr() % 16
    ):
        raise ValueError(
            "index_k_cache must have aligned, nonoverlapping page storage on the plan device"
        )


def bind_mxfp4(
    plan,
    *,
    scratch,
    q_mxfp4,
    q_scales,
    query_weights,
    index_k_cache,
    page_table,
    cache_lengths,
    active_width,
    output_indices,
    output_scores,
    candidate_indices,
    candidate_lengths,
    candidate_output,
    candidate_output_lengths,
    score_width=None,
):
    caps = plan.caps
    width_capacity = caps.max_candidates or caps.max_page_table_width * caps.page_size
    if score_width is not None and (
        isinstance(score_width, bool) or not isinstance(score_width, int)
    ):
        raise TypeError("score_width must be an integer")
    score_width = width_capacity if score_width is None else score_width
    if not 0 < score_width <= width_capacity:
        raise ValueError("score_width must be positive and within planned capacity")
    if caps.max_candidates and score_width != caps.max_candidates:
        raise ValueError("candidate scoring retains its planned candidate width")
    if q_mxfp4 is None or q_mxfp4.ndim != 3:
        raise ValueError("MXFP4 recipe requires explicit q_mxfp4 and q_scales")
    rows = q_mxfp4.shape[0]
    if rows <= 0 or rows > caps.max_q_rows:
        raise ValueError("query rows must be positive and within planned capacity")
    _check(q_mxfp4, "q_mxfp4", (rows, caps.num_q_heads, 64), torch.uint8, caps.device)
    _check(q_scales, "q_scales", (rows, caps.num_q_heads, 4), torch.uint8, caps.device)
    if query_weights.ndim == 3 and query_weights.shape[-1] == 1:
        query_weights = query_weights.view(rows, caps.num_q_heads)
    _check(
        query_weights,
        "query_weights",
        (rows, caps.num_q_heads),
        torch.bfloat16,
        caps.device,
    )
    _check_pool(index_k_cache, index_mxfp4_page_bytes(caps.page_size), caps.device)
    if page_table.ndim != 2 or page_table.shape[1] > caps.max_page_table_width:
        raise ValueError("page_table width exceeds planned capacity")
    if (
        page_table.dtype != torch.int32
        or page_table.device != caps.device
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 with unit inner stride on plan device"
        )
    if page_table.shape[0] not in (1, rows):
        raise ValueError("page_table must have one shared row or one row per query")
    _check(cache_lengths, "cache_lengths", (rows,), torch.int32, caps.device)
    _check(active_width, "active_width", (1,), torch.int32, caps.device)
    _check(
        output_indices, "output_indices", (rows, caps.topk), torch.int32, caps.device
    )
    if output_scores is not None:
        _check(
            output_scores,
            "output_scores",
            (rows, caps.topk),
            torch.float32,
            caps.device,
        )
    if caps.max_candidates:
        _check(
            candidate_indices,
            "candidate_indices",
            (rows, caps.max_candidates),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_lengths, "candidate_lengths", (rows,), torch.int32, caps.device
        )
    elif candidate_indices is not None or candidate_lengths is not None:
        raise ValueError("candidate inputs require max_candidates in Caps")
    if caps.candidate_topk_blocks:
        _check(
            candidate_output,
            "candidate_output",
            (rows, caps.candidate_topk_blocks * 8),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_output_lengths,
            "candidate_output_lengths",
            (rows,),
            torch.int32,
            caps.device,
        )
    elif candidate_output is not None or candidate_output_lengths is not None:
        raise ValueError("candidate output requires candidate_topk_blocks in Caps")
    runtime = plan.inner.bind(
        scratch=scratch,
        score_width=score_width,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        candidate_indices=candidate_indices,
        candidate_lengths=candidate_lengths,
        candidate_output=candidate_output,
        candidate_output_lengths=candidate_output_lengths,
    )
    return runtime


@dsl_user_op
def _fused_pack_bf16_pair(lo, hi, *, loc=None, ip=None):
    operands = [
        llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        for v in (lo, hi)
    ]
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "mov.b32 $0, {$1, $2};",
        "=r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_pair_accumulate(total, packed, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.f32(),
        [
            Float32(total).ir_value(loc=loc, ip=ip),
            Uint32(packed).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .b16 lo, hi;
        .reg .f32 partial;
        mov.b32 {lo, hi}, $2;
        add.rn.f32.bf16 partial, lo, $1;
        add.rn.f32.bf16 $0, hi, partial;
        }""",
        "=f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def _fused_bf16_products_packed(d0, d1, w0, w1, *, fused_relu=False, loc=None, ip=None):
    operands = [Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1)]
    operands.extend(
        llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        for v in (w0, w1)
    )
    conversion = (
        "cvt.rn.relu.bf16x2.f32 dots, $2, $1;"
        if fused_relu
        else """cvt.rn.bf16x2.f32 dots, $2, $1;
        mov.b32 zero, 0;
        max.bf16x2 dots, dots, zero;"""
    )
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "{\n.reg .b32 dots, weights, zero;\n"
        + conversion
        + """

        mov.b32 weights, {$3, $4};
        mul.bf16x2 $0, dots, weights;
        }""",
        "=r,f,f,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_products_word(d0, d1, weights, *, fused_relu=False, loc=None, ip=None):
    operands = [Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1)]
    operands.append(Uint32(weights).ir_value(loc=loc, ip=ip))
    conversion = (
        "cvt.rn.relu.bf16x2.f32 dots, $2, $1;"
        if fused_relu
        else """cvt.rn.bf16x2.f32 dots, $2, $1;
        mov.b32 zero, 0;
        max.bf16x2 dots, dots, zero;"""
    )
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "{\n.reg .b32 dots, zero;\n"
        + conversion
        + """

        mul.bf16x2 $0, dots, $3;
        }""",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_pack_weights(w0, w1, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [
            llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            for v in (w0, w1)
        ],
        "mov.b32 $0, {$1, $2};",
        "=r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)
