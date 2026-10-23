"""Bounded fused token selection and hierarchical block candidate publication.

Each partition retains token top-512 and block-max top-2048 independently.
Only those bounded carries cross CTA boundaries; no context-sized scores are
materialized. Block maxima consume the scorer's final BF16-rounded values.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from .fused_indexer import _mxfp4_order_small_output, _mxfp4_radix_select
from .mxfp4 import _flat, _TensorCorePagedScore


def hierarchical_scratch_shapes(rows: int, ctas: int) -> dict:
    """Return dtype/shape pairs for bounded, caller-owned producer scratch."""
    if rows < 1 or ctas < 1:
        raise ValueError("hierarchical rows and CTA count must be positive")
    if rows * ctas > 2**31 - 1:
        raise ValueError("hierarchical rows times CTA count exceeds CUDA grid x limit")
    partial = (rows, ctas, 2560) if ctas > 1 else (1,)
    return {
        "partial_indices": (torch.int32, partial),
        "partial_values": (torch.float32, partial),
        "partial_counts": (torch.int32, (rows, ctas, 2) if ctas > 1 else (1,)),
        "block_indices": (torch.int32, (rows, 2048)),
        "block_values": (torch.float32, (rows, 2048)),
        "selected_values": (torch.float32, (rows, 512)),
        "state": (torch.int32, (2,)),
    }


@cute.jit
def _trim(
    values,
    ids,
    count,
    topk: cutlass.Constexpr[int],
    capacity: cutlass.Constexpr[int],
    tx,
    hist,
    selected,
    scalars,
    priority_id: Int32 | None = None,
):
    if count > Int32(topk):
        _mxfp4_radix_select(
            values,
            ids,
            values,
            ids,
            count,
            Int32(topk),
            tx,
            hist,
            hist,
            selected,
            shared_ptr_to_u32(hist.iterator),
            shared_ptr_to_u32(scalars.iterator),
            shared_ptr_to_u32(scalars.iterator) + Int32(4),
            shared_ptr_to_u32(scalars.iterator) + Int32(8),
            max_items=capacity,
            indices_only=True,
            priority_id=priority_id,
        )
        rv = cute.make_rmem_tensor(((topk + 1023) // 1024,), Float32)
        ri = cute.make_rmem_tensor(((topk + 1023) // 1024,), Int32)
        for slot in cutlass.range_constexpr((topk + 1023) // 1024):
            i = tx + Int32(slot * 1024)
            if i < Int32(topk):
                source = Int32(selected[i])
                rv[slot] = Float32(values[source])
                ri[slot] = Int32(ids[source])
        cute.arch.sync_threads()
        for slot in cutlass.range_constexpr((topk + 1023) // 1024):
            i = tx + Int32(slot * 1024)
            if i < Int32(topk):
                values[i] = rv[slot]
                ids[i] = ri[slot]
        cute.arch.sync_threads()
        count = Int32(topk)
    return count


@cute.jit
def _row_output(ptr):
    return cute.make_tensor(
        ptr,
        cute.make_layout(
            (Int64(1) << Int64(30), Int64(512)), stride=(Int64(512), Int64(1))
        ),
    )


class HierarchicalFusedMXFP4:
    """Pointer-only local/merge programs with runtime rows, widths and strides.

    ``scorer`` may supply the production scorer's eligible byte-load policy.
    The scorer must provide the exact ``_score_tile_fused`` helper and dense
    logical indexing. Compilation and scratch ownership belong to the caller.
    """

    def __init__(self, heads: int, page_size: int, ctas: int = 1, scorer=None):
        if heads < 1 or heads > 64 or 64 % heads:
            raise ValueError("MXFP4 heads must divide 64")
        if page_size < 8 or page_size % 8 or not 1 <= ctas < 2**31:
            raise ValueError("invalid hierarchical page size or CTA count")
        self.ctas = int(ctas)
        self.scorer = (
            scorer
            if scorer is not None
            else _TensorCorePagedScore(heads, False, page_size)
        )

    @cute.jit
    def local(
        self,
        q: cute.Pointer,
        qs: cute.Pointer,
        weights: cute.Pointer,
        pool: cute.Pointer,
        pages: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        block_indices: cute.Pointer,
        block_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        state: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(q),
            _flat(qs),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(lengths),
            _flat(active),
            _row_output(out_indices),
            _row_output(out_values),
            _flat(block_indices),
            _flat(block_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            _flat(state),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
            False,
        ).launch(
            grid=(Int64(rows) * Int64(self.ctas), 1, 1),
            block=(1024, 1, 1),
            stream=stream,
        )

    @cute.jit
    def merge(
        self,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        block_indices: cute.Pointer,
        block_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            None,
            None,
            None,
            None,
            None,
            _flat(lengths),
            _flat(active),
            _row_output(out_indices),
            _row_output(out_values),
            _flat(block_indices),
            _flat(block_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            None,
            rows,
            Int32(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int64(0),
            True,
        ).launch(grid=(rows, 1, 1), block=(1024, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q,
        qs,
        weights,
        pool,
        pages,
        lengths,
        active,
        out_indices,
        out_values,
        block_indices,
        block_values,
        partial_indices,
        partial_values,
        partial_counts,
        state,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        merge: cutlass.Constexpr[bool],
    ):
        tx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        tx = Int32(tx)
        if cutlass.const_expr(merge):
            row, partition = Int32(block), Int32(0)
        else:
            row = Int32(block) // Int32(self.ctas)
            partition = Int32(block) % Int32(self.ctas)
        smem = cutlass.utils.SmemAllocator()
        ov = smem.allocate_tensor(Float32, cute.make_layout((2048,)), byte_alignment=16)
        oi = smem.allocate_tensor(Int32, cute.make_layout((2048,)), byte_alignment=16)
        bv = smem.allocate_tensor(Float32, cute.make_layout((4096,)), byte_alignment=16)
        bi = smem.allocate_tensor(Int32, cute.make_layout((4096,)), byte_alignment=16)
        hist = smem.allocate_tensor(Int32, cute.make_layout((264,)), byte_alignment=16)
        selected = smem.allocate_tensor(
            Int32, cute.make_layout((2048,)), byte_alignment=16
        )
        scalars = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        on, bn = Int32(0), Int32(0)
        visible = cutlass.min(
            cutlass.max(Int32(lengths[row]), Int32(0)),
            cutlass.max(Int32(active[0]), Int32(0)),
        )
        newest = (visible - Int32(1)) // Int32(8) if visible > Int32(0) else Int32(-1)
        if cutlass.const_expr(not merge):
            if (row == Int32(0)) & (partition == Int32(0)) & (tx < Int32(2)):
                state[tx] = Int32(0)
            visible = cutlass.min(
                cutlass.max(Int32(lengths[row]), Int32(0)),
                cutlass.max(Int32(active[0]), Int32(0)),
            )
            extent = cutlass.min(visible, width)
            blocks = (extent + Int32(7)) // Int32(8)
            begin = (Int64(blocks) * Int64(partition) // Int64(self.ctas)) * Int64(8)
            end = cutlass.min(
                Int64(extent),
                (Int64(blocks) * Int64(partition + Int32(1)) // Int64(self.ctas))
                * Int64(8),
            )
            tile = Int32(begin)
            while Int64(tile) < end:
                n = cutlass.min(Int32(512), Int32(end - Int64(tile)))
                nb = (n + Int32(7)) // Int32(8)
                if on + n > Int32(2048):
                    on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
                if bn + nb > Int32(4096):
                    bn = _trim(
                        bv,
                        bi,
                        bn,
                        2048,
                        4096,
                        tx,
                        hist,
                        selected,
                        scalars,
                        priority_id=newest,
                    )
                if tx < n:
                    ov[on + tx] = Float32(-float("inf"))
                    oi[on + tx] = Int32(-1)
                cute.arch.sync_threads()
                self.scorer._score_tile_fused(
                    q,
                    qs,
                    weights,
                    pool,
                    pages,
                    lengths,
                    active,
                    partial_indices,
                    partial_counts,
                    ov,
                    rows,
                    Int32(end),
                    page_width,
                    page_row_stride,
                    pool_stride,
                    pool_pages,
                    tile,
                    row_index=row,
                    local_output=True,
                    output_base=on,
                    logical_indices=oi,
                )
                cute.arch.sync_threads()
                if tx < nb:
                    value = Float32(-float("inf"))
                    for offset in cutlass.range_constexpr(8):
                        index = tx * Int32(8) + Int32(offset)
                        if index < n:
                            value = fmax_f32(value, Float32(ov[on + index]))
                    block = tile // Int32(8) + tx
                    if (visible > Int32(0)) & (
                        block == (visible - Int32(1)) // Int32(8)
                    ):
                        value = Float32(float("inf"))
                    bv[bn + tx] = value
                    bi[bn + tx] = block
                cute.arch.sync_threads()
                on += n
                bn += nb
                tile += n
        else:
            for part in cutlass.range(self.ctas):
                base = (Int64(row) * Int64(self.ctas) + Int64(part)) * Int64(2560)
                counts = (Int64(row) * Int64(self.ctas) + Int64(part)) * Int64(2)
                no = Int32(partial_counts[counts])
                if on + no > Int32(2048):
                    on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
                if tx < no:
                    ov[on + tx] = Float32(partial_values[base + Int64(tx)])
                    oi[on + tx] = Int32(partial_indices[base + Int64(tx)])
                on += no
                block_count = Int32(partial_counts[counts + Int64(1)])
                offset = Int32(0)
                while offset < block_count:
                    n = cutlass.min(Int32(512), block_count - offset)
                    if bn + n > Int32(4096):
                        bn = _trim(
                            bv,
                            bi,
                            bn,
                            2048,
                            4096,
                            tx,
                            hist,
                            selected,
                            scalars,
                            priority_id=newest,
                        )
                    if tx < n:
                        source = base + Int64(512) + Int64(offset) + Int64(tx)
                        bv[bn + tx] = Float32(partial_values[source])
                        bi[bn + tx] = Int32(partial_indices[source])
                    bn += n
                    offset += n
                    cute.arch.sync_threads()
                cute.arch.sync_threads()
        on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
        bn = _trim(
            bv, bi, bn, 2048, 4096, tx, hist, selected, scalars, priority_id=newest
        )
        if cutlass.const_expr(not merge and self.ctas > 1):
            base = (Int64(row) * Int64(self.ctas) + Int64(partition)) * Int64(2560)
            if tx < on:
                partial_values[base + Int64(tx)] = Float32(ov[tx])
                partial_indices[base + Int64(tx)] = Int32(oi[tx])
            for slot in cutlass.range_constexpr(2):
                i = tx + Int32(slot * 1024)
                if i < bn:
                    partial_values[base + Int64(512) + Int64(i)] = Float32(bv[i])
                    partial_indices[base + Int64(512) + Int64(i)] = Int32(bi[i])
            if tx == Int32(0):
                count_base = (Int64(row) * Int64(self.ctas) + Int64(partition)) * Int64(
                    2
                )
                partial_counts[count_base] = on
                partial_counts[count_base + Int64(1)] = bn
        else:
            for slot in cutlass.range_constexpr(2):
                i = tx + Int32(slot * 1024)
                valid = i < bn
                value, index = Float32(-float("inf")), Int32(-1)
                if valid:
                    value, index = Float32(bv[i]), Int32(bi[i])
                base = Int64(row) * Int64(2048) + Int64(i)
                block_values[base] = value
                block_indices[base] = index
            _mxfp4_order_small_output(out_indices, out_values, row, oi, ov, tx, on)
