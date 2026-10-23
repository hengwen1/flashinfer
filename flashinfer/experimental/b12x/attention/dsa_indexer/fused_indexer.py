"""Stream index scores into shared top-k carries and merge cooperating CTAs.

Cache encoding and paged/contiguous addressing are compile-time specializations.
FP8 scoring and MXFP4's BF16 arithmetic boundaries feed the same radix selection
and cross-CTA merge. Fused scoring requires all contributing query heads on the
rank; callers requiring an external score sum use the staged score/select API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import os

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import (
    KernelCompileSpec,
    launch as b12x_launch,
    tensor_compile_fact,
)
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.utils.device import current_cuda_stream

# Kept in lock-step with the source kernels we fuse.
from b12x.attention.dsa_indexer._tuning import (
    FUSED_MERGE_AUTO,
    FUSED_MERGE_CHOICES,
    FUSED_MERGE_COOPERATIVE,
    FUSED_MERGE_SERIAL,
)
from b12x.attention.dsa_indexer.tiled_topk import (
    _RADIX,
    _SCAN_UNROLL,
    _SUPPORTED_TOPK,
    _THREADS_PER_CTA as _RADIX_THREADS,
    _convert_to_uint8,
    _convert_to_uint32,
    _exact_overflow_fallback,
    _smem_ld,
    _smem_red_add,
    _smem_st,
    _smem_xadd,
)
from b12x._lib.intrinsics.atomics import ld_shared_f32
from b12x.attention.dsa_indexer.kernel import (
    _to_kernel_tensor,
    _stream_issue_k_page_cp_async,
    _INDEX_HEAD_DIM,
    _PAGE_SIZE,
    _PAGED_Q_HEAD_TILE,
    _PAGED_THREADS_PER_CTA,
    _PAGED_TOKENS_PER_GROUP,
    _PAGED_WARPS_PER_CTA,
    _WARP_THREADS,
    _compute_mxfp8_tile_partials,
    _compute_mxfp8_tile_partials_qldm,
    _load_index_k_page_scalar,
    _num_q_head_tiles,
    _permuted_offset_128b,
    _reduce_column_pair_sum,
    _repack_k_page_to_permuted,
    _smem_addr_from_b128_offset,
)
from b12x._lib.intrinsics.atomics import (
    atomic_add_global_i32,
    atomic_add_global_u64,
    red_add_global_i32,
    red_add_global_release_i32,
)
from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.intrinsics.packed import f32_to_raw_bits
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_cg_v4_u32,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_v4_u32,
    ldmatrix_m8n8x4_b16,
    ld_shared_v4_u32,
    st_shared_u8,
    st_shared_v4_u32,
    st_global_v4_u32,
    st_global_v2_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.sync import (
    ld_global_acquire_i32,
    spin_wait_global_ge_i32,
    threadfence,
)
from b12x._lib.intrinsics.mma import mxfp8_mma_m16n8k32_f32_e4m3

# Triage kill-switch: B12X_INDEXER_DIRECT_K=0 forces every variant back to
# the staged pipeline (which also restores the v2 cache keys), so serving can
# A/B the direct-L2 score in place without a checkout change.
_DIRECT_K_SCORE_ENABLED = os.environ.get("B12X_INDEXER_DIRECT_K", "1").lower() not in {
    "0",
    "false",
    "no",
    "",
}

_THREADS_PER_CTA = 1024
# Per-CTA K-slice cap; each CTA streams its slice through MMA sub-tiles.
_MAX_CHUNK_ELEMENTS = 8192

# Constexpr K/V-layout specialization (the ONLY contiguous-vs-paged divergence in the
# fused kernel — everything downstream of the SMEM staging tile is shared).
KV_LAYOUT_CONTIGUOUS_MLA = 0
KV_LAYOUT_PAGED = 1

# Route metadata is capture-static; live seqlen is deliberately not a policy input,
# so vLLM graph replay cannot switch backends. The generic policy retains its
# measured small-row limit. C4 uses top-k 512 with either 64 heads (DeepSeek V4)
# or 32 heads (GLM-5.3-Flash), while GLM-5.2 uses 32 heads and top-k 2048.
# L2-flushed DSV4F measurements select fused through B16 for C4 on both
# RTX-class SM120 and GB10/SM121.
# Prefill remains packed-contiguous and never reaches this decode-only resolver.
FUSED_MAX_ROWS = 6
_C4_B12X_FUSED_MAX_ROWS = 16
# GLM re-measured after the two-level tiled fold landed: fused keeps rows 1-16
# (wins/ties at 8K, dominant at 32K-131K, e.g. r1@131K 41us vs 90us tiled) but
# loses rows 32-64 everywhere (r64@131K 1541us vs 1043us tiled, r64@8K 100 vs
# 78) -- the 2-CTAs-per-group merge and one-resident-CTA occupancy starve it.
_GLM32_FUSED_MAX_ROWS = 16
FUSED_MIN_WIDTH = 20480  # (capacity-sizing only; no longer a routing gate)


@cute.jit
def _load_flat_k_tile_scalar(
    k_quant_flat: cute.Tensor,  # [k_rows, 128] uint8
    abs_base: Int32,
    valid_rows: Int32,
    s_k_page_stage: cute.Tensor,
    lane_linear: Int32,
):
    """CONTIGUOUS_MLA K-load: stage a 64-row tile of contiguous K starting at abs_base.

    Rows >= valid_rows are zero-filled (so the MMA contributes 0 for them — masked
    out anyway by the staging-write valid_slots guard).
    """
    linear = lane_linear
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        b = cutlass.Uint8(0)
        if row < valid_rows:
            b = k_quant_flat[abs_base + row, col]
        s_k_page_stage[row, col, Int32(0)] = b
        linear += Int32(_PAGED_THREADS_PER_CTA)


# --- wide (all-thread) page loaders ------------------------------------------
# The per-page K load dominates the fused per-page cost. The kernel.py scalar
# loaders stride by _PAGED_THREADS_PER_CTA (128), so a 8 KB page is 64 sequential
# coalesced rounds. These widen the stride to n_threads (up to 1024) so the page
# loads in ~8 rounds. Same coalesced byte pattern (consecutive tx -> consecutive
# col), just 8x more threads in flight to hide latency.
@cute.jit
def _load_index_k_page_wide(
    k_quant_bytes: cute.Tensor,  # [pages, 64, 128] uint8
    page_id: Int32,
    s_k_page_stage: cute.Tensor,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        s_k_page_stage[row, col, Int32(0)] = k_quant_bytes[page_id, row, col]
        linear += n_threads


@cute.jit
def _load_flat_k_tile_wide(
    k_quant_flat: cute.Tensor,  # [k_rows, 128] uint8
    abs_base: Int32,
    valid_rows: Int32,
    s_k_page_stage: cute.Tensor,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        b = cutlass.Uint8(0)
        if row < valid_rows:
            b = k_quant_flat[abs_base + row, col]
        s_k_page_stage[row, col, Int32(0)] = b
        linear += n_threads


@cute.jit
def _load_permute_k_page_g2s(
    k_quant_bytes: cute.Tensor,  # [pages, 64, 128] uint8 (16B-aligned base)
    page_id: Int32,
    page_stride_bytes: Int64,  # dim-0 byte stride; 8192 for contiguous K, 8448 for
    # the packed paged cache (64*128 quant + 64*4 scales / page)
    k_perm_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    """paged fused global->shared load + permute: read 16-byte K granules straight
    from global into the ldmatrix-permuted SMEM layout. Replaces the scalar
    linear load + separate SMEM->SMEM repack pass (and its barrier) with one
    vectorized pass. Granule addresses are 16B-aligned (page*page_stride_bytes +
    row*128 + vec*16); page_stride_bytes MUST be the real dim-0 stride of
    k_quant_bytes (the packed paged cache interleaves per-page scales, so its page
    stride is 8448, not the contiguous 8192) — using a hardcoded stride reads the
    wrong page for the packed layout. It stays 16B-aligned (8448 = 16*528).
    """
    page_elem_base = Int64(page_id) * page_stride_bytes
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        g_addr = get_ptr_as_int64(
            k_quant_bytes,
            page_elem_base
            + Int64(row) * Int64(_INDEX_HEAD_DIM)
            + Int64(vec_idx) * Int64(16),
        )
        v0, v1, v2, v3 = ld_global_v4_u32(g_addr)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += n_threads


@cute.jit
def _repack_k_page_wide(
    k_linear_base_addr: Int32,
    k_perm_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        src_addr = k_linear_base_addr + Int32(row * _INDEX_HEAD_DIM + vec_idx * 16)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        v0, v1, v2, v3 = ld_shared_v4_u32(src_addr)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += n_threads


@dataclass(frozen=True)
class _FusedLaunchConfig:
    chunk_size: int
    ctas_per_group: int
    num_groups: int
    total_ctas: int


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_fused_launch_config(
    num_rows: int, width: int, device: torch.device
) -> _FusedLaunchConfig:
    """Split query rows across cooperative CTA groups.

    One group owns one query row; ``ctas_per_group`` CTAs split the row's K of
    ``width`` tokens; groups persist grid-stride over rows.
    """
    max_chunk = max(_MAX_CHUNK_ELEMENTS, _THREADS_PER_CTA)
    ctas_per_group = max(1, (width + max_chunk - 1) // max_chunk)
    chunk_size = min(
        max_chunk, _align_up((width + ctas_per_group - 1) // ctas_per_group, 1)
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_groups = min(max(num_rows, 1), max(1, num_sms // ctas_per_group))
    total_ctas = max(1, num_groups * ctas_per_group)
    return _FusedLaunchConfig(
        chunk_size=chunk_size,
        ctas_per_group=ctas_per_group,
        num_groups=num_groups,
        total_ctas=total_ctas,
    )


# --- scratch sizing (per-group combine state + inter-CTA carry slab) ----------
# Combine state per group: arrival + output counters + a small relay slot. The
# inter-CTA carry slab holds each CTA's published running top-k (value f32 +
# index i32) for the next CTA in the relay to fold.
_STATE_WORDS_PER_GROUP = 4  # arrival_counter, output_counter, relay_phase, pad

# Cross-CTA merge auto-switch (ctas_per_group>1): a group whose live row K length
# seq_len <= merge_threshold uses the serial last-CTA reduction (no grid
# barriers); larger rows use the cooperative grid-barrier radix. Both arms are
# exact, so the threshold only trades perf. Planned launches resolve the
# threshold through _resolve_default_merge_threshold; this value is the
# constructor default for direct kernel constructions.
_LAST_CTA_MERGE_MAX = 49152
# Sentinel "never coop" threshold: larger than any live seq_len, so the per-group
# auto-switch always takes the serial last-CTA arm (single-CTA groups, and
# groups that cannot be co-resident).
_FORCE_LAST_CTA = 1 << 30
# K stages of the paged score pipeline (one 64-token page each).
_K_STAGES = 3

# Per-group cross-CTA merge state (int32 words): the cooperative radix
# selects the pivot over three digits of the order-preserving 32-bit key
# (12, 12, and 8 bits), each round accumulating the group's digit histogram
# into its own buffer, followed by the scalars both merge arms share. The
# arrival counter backs the grid barrier, the output counter reserves winner
# slots (and the pack-slab offset on the last-CTA arm), the tie counter
# reserves slots at the pivot, and the total/cleanup words carry the group's
# candidate count and its end-of-launch departure count.
# A 12-bit round publishes both a 256-bin coarse histogram (top 8 bits of
# the digit) and a 4096-bin fine histogram, so the pivot scan reads the
# coarse buffer and then only the 16 fine bins under the coarse pivot.
_FUSED_WIDE_RADIX = 4096
_FUSED_COARSE0 = 0
_FUSED_FINE0 = _FUSED_COARSE0 + _RADIX
_FUSED_COARSE1 = _FUSED_FINE0 + _FUSED_WIDE_RADIX
_FUSED_FINE1 = _FUSED_COARSE1 + _RADIX
_FUSED_COARSE2 = _FUSED_FINE1 + _FUSED_WIDE_RADIX
_FUSED_STATE_ARRIVAL = _FUSED_COARSE2 + _RADIX
_FUSED_STATE_TOTAL = _FUSED_STATE_ARRIVAL + 1
# Output (winner) and tie counters are adjacent and 8-byte aligned so one
# 64-bit fetch-add reserves both slot ranges.
_FUSED_STATE_OUTPUT = _FUSED_STATE_ARRIVAL + 2
_FUSED_STATE_TIES = _FUSED_STATE_ARRIVAL + 3
_FUSED_STATE_CLEANUP = _FUSED_STATE_ARRIVAL + 4
_COOP_STATE_WORDS = _FUSED_STATE_ARRIVAL + 8
assert (_FUSED_STATE_OUTPUT % 2 == 0) and (_COOP_STATE_WORDS % 4 == 0)
assert _FUSED_WIDE_RADIX == 4 * _RADIX_THREADS


@cute.jit
def _fused_state_offset(group_id: Int32, word: Int32) -> Int64:
    return Int64(group_id) * Int64(_COOP_STATE_WORDS) + Int64(word)


@cute.jit
def _fused_state_ptr(state: cute.Tensor, group_id: Int32, word: Int32):
    return get_ptr_as_int64(state, _fused_state_offset(group_id, word))


@cute.jit
def _fused_group_barrier(
    state: cute.Tensor, group_id: Int32, phase: Int32, ctas_per_group: Int32, tx: Int32
) -> Int32:
    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
    # Every warp publishes histogram bins. Thread zero may announce this
    # CTA's arrival only after those writes precede its release operation.
    cute.arch.sync_threads()
    if tx == Int32(0):
        red_add_global_release_i32(arrival_ptr, Int32(1))
        spin_wait_global_ge_i32(arrival_ptr, (phase + Int32(1)) * ctas_per_group)
    cute.arch.sync_threads()
    return phase + Int32(1)


@cute.jit
def _page_id_or_invalid(
    real_page_table: cute.Tensor, q_idx: Int32, col: Int32, page_end: Int32
) -> Int32:
    """Page id of page-table column ``col`` of row ``q_idx``, or -1 past the slice."""
    pid = Int32(-1)
    if col < page_end:
        if col < Int32(real_page_table.shape[1]):
            pid = Int32(real_page_table[q_idx, col])
    return pid


@cute.jit
def _k_stage_addr(stage: Int32, stage0: Int32, stage1: Int32, stage2: Int32) -> Int32:
    """Shared-memory base of K ring stage ``stage`` (0, 1, or 2)."""
    addr = stage0
    if stage == Int32(1):
        addr = stage1
    if stage == Int32(2):
        addr = stage2
    return addr


@cute.jit
def _coop_publish_wide(
    merge_state: cute.Tensor,
    group_id: Int32,
    hist_off: cutlass.Constexpr[int],
    hist_addr: Int32,
    tx: Int32,
    threads: cutlass.Constexpr[int] = 1024,
):
    """Publish and clear the local histogram in four-bin vectors."""
    for hist_pass in cutlass.range_constexpr(1024 // threads):
        hist_tx = tx + Int32(hist_pass * threads)
        c0, c1, c2, c3 = ld_shared_v4_u32(hist_addr + hist_tx * Int32(16))
        base = _fused_state_ptr(
            merge_state, group_id, Int32(hist_off) + hist_tx * Int32(4)
        )
        if c0 != Uint32(0):
            red_add_global_i32(base, Int32(c0))
        if c1 != Uint32(0):
            red_add_global_i32(base + Int64(4), Int32(c1))
        if c2 != Uint32(0):
            red_add_global_i32(base + Int64(8), Int32(c2))
        if c3 != Uint32(0):
            red_add_global_i32(base + Int64(12), Int32(c3))
        st_shared_v4_u32(
            hist_addr + hist_tx * Int32(16), Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        )


@cute.jit
def _coop_wide_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    fine_off: cutlass.Constexpr[int],
    s_c0_values: cute.Tensor,
    carry_count: Int32,
    key_mask: Uint32,
    key_prefix: Uint32,
    digit_shift: cutlass.Constexpr[int],
    wide_hist_addr: Int32,
    coarse_hist_addr: Int32,
    s_hist0: cute.Tensor,
    remaining_k: Int32,
    barrier_phase: Int32,
    ctas_pg: Int32,
    tx: Int32,
    lane: Int32,
    warp_idx: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
    threads: cutlass.Constexpr[int] = 1024,
) -> Int32:
    """One 12-bit cooperative radix round; returns the next barrier phase.

    Every candidate whose key matches ``key_prefix`` under ``key_mask``
    contributes the 12-bit digit at ``digit_shift`` to the CTA's fine
    (4096-bin, in the K ring) and coarse (256-bin, in ``s_hist0``) local
    histograms, both of which must be zero on entry and are left zero. The
    local histograms are published to the group buffers at ``fine_off`` and
    ``coarse_off``, the grid barrier is crossed, and warp 0 resolves the
    pivot into ``s_coop_scalars`` (readable after the trailing CTA barrier).
    """
    i = Int32(tx)
    while i < carry_count:
        key = _convert_to_uint32(Float32(s_c0_values[i]))
        if (key & key_mask) == key_prefix:
            digit = Int32((key >> Uint32(digit_shift)) & Uint32(0xFFF))
            _smem_red_add(wide_hist_addr, digit, Int32(1))
            _smem_red_add(coarse_hist_addr, digit >> Int32(4), Int32(1))
        i += Int32(threads)
    cute.arch.sync_threads()
    _coop_publish_wide(
        merge_state, group_id, fine_off, wide_hist_addr, tx, threads=threads
    )
    if tx < Int32(_RADIX):
        c = Int32(s_hist0[tx])
        if c > Int32(0):
            red_add_global_i32(
                _fused_state_ptr(merge_state, group_id, Int32(coarse_off) + tx), c
            )
        s_hist0[tx] = Int32(0)
    phase = _fused_group_barrier(merge_state, group_id, barrier_phase, ctas_pg, tx)
    if warp_idx == Int32(0):
        _coop_scan_round12(
            merge_state,
            group_id,
            coarse_off,
            fine_off,
            remaining_k,
            lane,
            s_coop_scalars,
            read_total,
        )
    cute.arch.sync_threads()
    return phase


@cute.jit
def _coop_narrow_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    s_c0_values: cute.Tensor,
    carry_count: Int32,
    key_mask: Uint32,
    key_prefix: Uint32,
    coarse_hist_addr: Int32,
    s_hist0: cute.Tensor,
    remaining_k: Int32,
    barrier_phase: Int32,
    ctas_pg: Int32,
    tx: Int32,
    lane: Int32,
    warp_idx: Int32,
    s_coop_scalars: cute.Tensor,
    digit_shift: cutlass.Constexpr[int] = 0,
    read_total: cutlass.Constexpr[bool] = False,
    threads: cutlass.Constexpr[int] = 1024,
) -> Int32:
    """One byte-wide cooperative radix round over the selected key byte.

    Same contract as :func:`_coop_wide_round` with a 256-bin histogram only.
    """
    if tx < Int32(_RADIX):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    i = Int32(tx)
    while i < carry_count:
        key = _convert_to_uint32(Float32(s_c0_values[i]))
        if (key & key_mask) == key_prefix:
            _smem_red_add(
                coarse_hist_addr,
                Int32((key >> Uint32(digit_shift)) & Uint32(0xFF)),
                Int32(1),
            )
        i += Int32(threads)
    cute.arch.sync_threads()
    if tx < Int32(_RADIX):
        c = Int32(s_hist0[tx])
        if c > Int32(0):
            red_add_global_i32(
                _fused_state_ptr(merge_state, group_id, Int32(coarse_off) + tx), c
            )
        s_hist0[tx] = Int32(0)
    phase = _fused_group_barrier(merge_state, group_id, barrier_phase, ctas_pg, tx)
    if warp_idx == Int32(0):
        _coop_scan_round(
            merge_state,
            group_id,
            coarse_off,
            remaining_k,
            lane,
            s_coop_scalars,
            read_total,
        )
    cute.arch.sync_threads()
    return phase


@cute.jit
def _coop_scan_round12(
    merge_state: cute.Tensor,
    group_id: Int32,
    coarse_off: cutlass.Constexpr[int],
    fine_off: cutlass.Constexpr[int],
    remaining_k: Int32,
    lane: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
):
    """Warp 0: resolve a 12-bit pivot digit from the coarse and fine histograms.

    The 256-bin coarse histogram (lane ``l`` owns bins ``8l .. 8l+7``) is
    suffix-scanned for the coarse pivot ``c`` and the count above it; lanes
    0-3 then read the 16 fine bins under ``c`` and refine. Both are read
    through L2 so other CTAs' reductions are visible. ``s_coop_scalars``
    receives the 12-bit pivot bin, the count strictly above it, and the
    bin's own count; with ``read_total`` it also receives the group's
    candidate total.
    """
    base = _fused_state_ptr(merge_state, group_id, Int32(coarse_off) + lane * Int32(8))
    h0, h1, h2, h3 = ld_global_cg_v4_u32(base)
    h4, h5, h6, h7 = ld_global_cg_v4_u32(base + Int64(16))
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [
        s0 + above,
        s1 + above,
        s2 + above,
        s3 + above,
        s4 + above,
        s5 + above,
        s6 + above,
        s7 + above,
        above,
    ]
    coarse_bin = Int32(-1)
    coarse_gt = Int32(-1)
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            coarse_bin = lane * Int32(8) + Int32(j)
            coarse_gt = suffix[j + 1]
    for d in (16, 8, 4, 2, 1):
        coarse_bin = _max_i32(
            coarse_bin, cute.arch.shuffle_sync_bfly(coarse_bin, offset=d)
        )
        coarse_gt = _max_i32(
            coarse_gt, cute.arch.shuffle_sync_bfly(coarse_gt, offset=d)
        )
    f0 = Uint32(0)
    f1 = Uint32(0)
    f2 = Uint32(0)
    f3 = Uint32(0)
    if lane < Int32(4):
        f0, f1, f2, f3 = ld_global_cg_v4_u32(
            _fused_state_ptr(
                merge_state,
                group_id,
                Int32(fine_off) + coarse_bin * Int32(16) + lane * Int32(4),
            )
        )
    t3 = Int32(f3)
    t2 = t3 + Int32(f2)
    t1 = t2 + Int32(f1)
    t0 = t1 + Int32(f0)
    fine_above = t0
    for d in (1, 2):
        other = cute.arch.shuffle_sync_down(fine_above, offset=d)
        if lane + Int32(d) < Int32(4):
            fine_above = fine_above + other
    fine_above = fine_above - t0 + coarse_gt
    fsuffix = [
        t0 + fine_above,
        t1 + fine_above,
        t2 + fine_above,
        t3 + fine_above,
        fine_above,
    ]
    if lane < Int32(4):
        for j in cutlass.range_constexpr(4):
            if (fsuffix[j] >= remaining_k) & (fsuffix[j + 1] < remaining_k):
                s_coop_scalars[0] = Uint32(
                    coarse_bin * Int32(16) + lane * Int32(4) + Int32(j)
                )
                s_coop_scalars[1] = Uint32(fsuffix[j + 1])
                s_coop_scalars[2] = Uint32(fsuffix[j] - fsuffix[j + 1])
    if cutlass.const_expr(read_total):
        if lane == Int32(0):
            s_coop_scalars[3] = Uint32(
                ld_global_acquire_i32(
                    _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_TOTAL))
                )
            )


@cute.jit
def _max_i32(a: Int32, b: Int32) -> Int32:
    r = a
    if b > a:
        r = b
    return r


@cute.jit
def _coop_scan_round(
    merge_state: cute.Tensor,
    group_id: Int32,
    hist_off: cutlass.Constexpr[int],
    remaining_k: Int32,
    lane: Int32,
    s_coop_scalars: cute.Tensor,
    read_total: cutlass.Constexpr[bool],
):
    """Warp 0: suffix-scan a 256-bin group histogram and publish the pivot.

    Lane ``l`` owns bins ``8l .. 8l+7`` of the histogram at word offset
    ``hist_off``, read through L2 so other CTAs' reductions are visible.
    ``s_coop_scalars`` receives the pivot bin (the highest bin whose suffix
    count reaches ``remaining_k``), the count strictly above it, and the
    bin's own count; with ``read_total`` it also receives the group's
    candidate total.
    """
    base = _fused_state_ptr(merge_state, group_id, Int32(hist_off) + lane * Int32(8))
    h0, h1, h2, h3 = ld_global_cg_v4_u32(base)
    h4, h5, h6, h7 = ld_global_cg_v4_u32(base + Int64(16))
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [
        s0 + above,
        s1 + above,
        s2 + above,
        s3 + above,
        s4 + above,
        s5 + above,
        s6 + above,
        s7 + above,
        above,
    ]
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            s_coop_scalars[0] = Uint32(lane * Int32(8) + Int32(j))
            s_coop_scalars[1] = Uint32(suffix[j + 1])
            s_coop_scalars[2] = Uint32(suffix[j] - suffix[j + 1])
    if cutlass.const_expr(read_total):
        if lane == Int32(0):
            s_coop_scalars[3] = Uint32(
                ld_global_acquire_i32(
                    _fused_state_ptr(merge_state, group_id, Int32(_FUSED_STATE_TOTAL))
                )
            )


@cute.jit
def _load_q_bytes_g2s_v4(
    q_bytes: cute.Tensor,
    q_idx: Int32,
    q_row_stride_bytes: Int64,
    real_q_bytes: cutlass.Constexpr[int],
    padded_q_bytes: cutlass.Constexpr[int],
    q_smem_base_addr: Int32,
    tx: Int32,
    n_threads: Int32,
):
    """Vectorized 16-byte query staging for the common contiguous-head layout."""
    linear_vec = tx
    total_vecs = Int32(int(padded_q_bytes) // 16)
    real_vecs = Int32(int(real_q_bytes) // 16)
    row_base = Int64(q_idx) * q_row_stride_bytes
    while linear_vec < total_vecs:
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if linear_vec < real_vecs:
            q_addr = get_ptr_as_int64(q_bytes, row_base + Int64(linear_vec) * Int64(16))
            v0, v1, v2, v3 = ld_global_v4_u32(q_addr)
        st_shared_v4_u32(q_smem_base_addr + linear_vec * Int32(16), v0, v1, v2, v3)
        linear_vec += n_threads


@cute.jit
def _score_tokens_direct_k(
    q_perm_base_addr: Int32,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_quant_bytes: cute.Tensor,
    k_byte_off: Int64,
    lane: Int32,
    num_q_head_tiles: cutlass.Constexpr[int],
):
    """Score 8 tokens against every head tile with K read straight from L2.

    Byte-identical to the smem-staged score core: each lane loads exactly the
    bytes ldmatrix.m8n8x4.b16 would hand it (reg m, lane l = token l//4,
    byte-quad l%4 of K half m), preserving the raw-fragment convention the
    MMA relies on without staging K through shared memory. k_byte_off is the
    lane's base byte offset (page + token-row + quad). After the butterfly
    reduction every lane holds the two column totals for columns
    2*(lane%4) and 2*(lane%4)+1, summed over all head tiles.
    """
    # num_heads is unreferenced: s_w is zero-padded to the padded head count.
    q_row = lane & Int32(15)
    q_half = lane >> Int32(4)
    k0_0 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off))
    k1_0 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(16)))
    k0_1 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(32)))
    k1_1 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(48)))
    k0_2 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(64)))
    k1_2 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(80)))
    k0_3 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(96)))
    k1_3 = ld_global_nc_u32(get_ptr_as_int64(k_quant_bytes, k_byte_off + Int64(112)))
    total0 = Float32(0.0)
    total1 = Float32(0.0)
    qgrp = lane // Int32(4)
    for tile in cutlass.range_constexpr(num_q_head_tiles):
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        q_tile_base = q_perm_base_addr + Int32(
            tile * _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
        )
        for step in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
            q_addr = _smem_addr_from_b128_offset(
                q_tile_base,
                _permuted_offset_128b(
                    q_row, Int32(2 * step) + q_half, Int32(_INDEX_HEAD_DIM // 16)
                ),
            )
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_addr)
            if cutlass.const_expr(step == 0):
                bk0 = k0_0
                bk1 = k1_0
            elif cutlass.const_expr(step == 1):
                bk0 = k0_1
                bk1 = k1_1
            elif cutlass.const_expr(step == 2):
                bk0 = k0_2
                bk1 = k1_2
            else:
                bk0 = k0_3
                bk1 = k1_3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc0,
                acc1,
                acc2,
                acc3,
                a0,
                a1,
                a2,
                a3,
                bk0,
                bk1,
                Uint32(0x7F7F7F7F),
                Uint32(0x7F7F7F7F),
            )
            acc0 = d0
            acc1 = d1
            acc2 = d2
            acc3 = d3
        head0 = Int32(tile * _PAGED_Q_HEAD_TILE) + qgrp
        head1 = head0 + Int32(8)
        w0 = Float32(s_w[head0])
        w1 = Float32(s_w[head1])
        p0 = Float32(fmax_f32(acc0, Float32(0.0)) * w0)
        p0 = Float32(p0 + fmax_f32(acc2, Float32(0.0)) * w1)
        p1 = Float32(fmax_f32(acc1, Float32(0.0)) * w0)
        p1 = Float32(p1 + fmax_f32(acc3, Float32(0.0)) * w1)
        total0 = Float32(total0 + _reduce_column_pair_sum(p0))
        total1 = Float32(total1 + _reduce_column_pair_sum(p1))
    return total0, total1


def _fused_indexer_state_nbytes(launch: _FusedLaunchConfig) -> int:
    return launch.num_groups * _STATE_WORDS_PER_GROUP * 4


def _fused_indexer_carry_nbytes(launch: _FusedLaunchConfig, topk: int) -> int:
    # value f32 + index i32 = 8 bytes per (group, cta, topk) slot.
    return launch.num_groups * launch.ctas_per_group * int(topk) * 8


def fused_indexer_workspace_nbytes(
    num_rows: int,
    width: int,
    topk: int,
    *,
    device: torch.device | str | None = None,
) -> int:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        # CPU planning fallback: assume 1 CTA/group worst case.
        return max(int(num_rows), 1) * (_STATE_WORDS_PER_GROUP * 4 + int(topk) * 8)
    launch = _resolve_fused_launch_config(int(num_rows), int(width), device)
    return _fused_indexer_state_nbytes(launch) + _fused_indexer_carry_nbytes(
        launch, int(topk)
    )


def _fused_indexer_capacity_nbytes(
    max_rows: int, max_width: int, topk: int, *, device: torch.device
) -> int:
    max_rows = max(int(max_rows), 1)
    max_width = max(int(max_width), 1)
    if device.type != "cuda":
        return fused_indexer_workspace_nbytes(max_rows, max_width, topk, device=device)
    # Worst case over the strides that change ctas_per_group / num_groups.
    candidate_widths = {1, max_width}
    if max_width > FUSED_MIN_WIDTH:
        candidate_widths.add(FUSED_MIN_WIDTH + 1)
    return max(
        fused_indexer_workspace_nbytes(max_rows, w, topk, device=device)
        for w in candidate_widths
    )


@dataclass(frozen=True, kw_only=True)
class B12XFusedIndexerScratchCaps:
    device: torch.device | str
    max_rows: int
    max_width: int
    topk: int

    def __post_init__(self) -> None:
        if int(self.topk) not in _SUPPORTED_TOPK:
            raise ValueError(
                f"fused indexer supports topk {_SUPPORTED_TOPK}, got {self.topk}"
            )


# --- dispatch gates -----------------------------------------------------------
def supports_fused_indexer(
    *,
    topk: int,
    num_rows: int,
    width: int,
) -> bool:
    """Whether the fused kernel is applicable (capability gate)."""
    if int(topk) not in _SUPPORTED_TOPK:
        return False
    if int(width) <= 0 or int(num_rows) <= 0:
        return False
    return True


def resolve_fused_indexer_path(
    *,
    topk: int,
    num_rows: int,
    width: int,
    num_heads: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Route to the fused kernel only for small decode batches (rows <= N).

    Row count, head count, top-k, and width here are all capture-time workspace
    metadata; live seqlen is deliberately absent, so the selected route is stable
    across vLLM CUDA-graph replays. The general small-decode gate remains six
    rows. C4's 64-head and 32-head top-k-512 shapes use fused through B16 on
    SM120 and SM121. GLM-5.2's 32-head/top-k-2048 shape uses fused through B16
    and the streamed tiled route beyond. Prefill is selected before this
    decode-only resolver and remains packed-contiguous.
    """
    if not supports_fused_indexer(topk=topk, num_rows=num_rows, width=width):
        return False
    if int(topk) == 512 and num_heads is not None and int(num_heads) in {32, 64}:
        # C4 (DSV4 heads=64, GLM-5.3-Flash heads=32): both RTX-class SM120 and
        # GB10/SM121 use the same fixed-page scorer and top-k specialization.
        return (
            compute_capability in {(12, 0), (12, 1)}
            and int(num_rows) <= _C4_B12X_FUSED_MAX_ROWS
        )
    if int(topk) == 2048 and num_heads is not None and int(num_heads) == 32:
        return int(num_rows) <= _GLM32_FUSED_MAX_ROWS
    return int(num_rows) <= FUSED_MAX_ROWS


def fused_indexer_scratch_max_rows(
    *,
    topk: int,
    num_heads: int,
    compute_capability: tuple[int, int] | None = None,
) -> int:
    """Return the fixed row capacity needed by fused decode scratch."""

    if (
        int(topk) == 512
        and int(num_heads) in {32, 64}
        and compute_capability in {(12, 0), (12, 1)}
    ):
        return _C4_B12X_FUSED_MAX_ROWS
    if int(topk) == 2048 and int(num_heads) == 32:
        return _GLM32_FUSED_MAX_ROWS
    return FUSED_MAX_ROWS


def _resolve_default_ctas_per_group(
    *,
    num_rows: int,
    max_pages: int,
    device: torch.device,
) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Target a single CTA wave (rows * ctas_per_group ~ num_sms): more CTAs
    # only add last-CTA merge work + a second wave. Never split below 1 page.
    return max(1, min(int(max_pages), num_sms // max(1, int(num_rows))))


def _resolve_default_merge_threshold(
    *,
    ctas_per_group: int,
    num_heads: int,
    topk: int,
) -> int:
    """Live-length threshold above which a multi-CTA group merges cooperatively.

    The cooperative arm publishes each radix round's histogram with one grid
    barrier and places its output without another; measured under graph
    replay on sm120 (top-k 512 with 64 heads and top-k 2048 with 32 heads,
    1 to 8 rows, 4k to 64k live tokens) it won or tied the serial last-CTA
    arm at every point, so every co-resident group merges cooperatively at
    any live length. Single-CTA groups have no cross-CTA merge.
    """
    del num_heads, topk
    if int(ctas_per_group) <= 1:
        return _FORCE_LAST_CTA
    return 0


def resolve_fused_merge_threshold(
    fused_merge: str,
    *,
    ctas_per_group: int,
    num_heads: int,
    topk: int,
) -> int:
    """Live-length threshold implementing the policy's ``fused_merge`` choice.

    A group whose live row length exceeds the threshold merges cooperatively;
    otherwise it takes the serial last-CTA arm. Single-CTA groups have no
    cross-CTA merge. ``auto`` applies the planner default
    (:func:`_resolve_default_merge_threshold`).
    """
    if fused_merge not in FUSED_MERGE_CHOICES:
        raise ValueError(
            f"unsupported fused_merge {fused_merge!r}; expected one of "
            f"{FUSED_MERGE_CHOICES}"
        )
    if int(ctas_per_group) <= 1 or fused_merge == FUSED_MERGE_SERIAL:
        return _FORCE_LAST_CTA
    if fused_merge == FUSED_MERGE_COOPERATIVE:
        return 0
    return _resolve_default_merge_threshold(
        ctas_per_group=int(ctas_per_group),
        num_heads=int(num_heads),
        topk=int(topk),
    )


def fused_indexer_decode_warmup_rows(
    *,
    topk: int,
    num_heads: int,
    max_pages: int,
    device: torch.device,
    fused_merge: str = FUSED_MERGE_AUTO,
) -> tuple[int, ...]:
    """Return one safe launch row count per compiled decode policy.

    The default fused planner specializes ``ctas_per_group`` and its merge
    threshold by row count. Each returned row count selects a distinct policy
    while preserving the planner's one-machine-wave launch bound. Callers can
    launch these shapes before graph capture to populate every runtime variant.
    ``fused_merge`` must match the policy choice the serving plan applies.
    """
    props = torch.cuda.get_device_properties(device)
    compute_capability = (
        (int(props.major), int(props.minor))
        if hasattr(props, "major") and hasattr(props, "minor")
        else None
    )
    max_rows = fused_indexer_scratch_max_rows(
        topk=int(topk),
        num_heads=int(num_heads),
        compute_capability=compute_capability,
    )

    width = int(max_pages) * _PAGE_SIZE
    rows_by_policy: list[int] = []
    seen_policies: set[tuple[int, int]] = set()
    for rows in range(1, max_rows + 1):
        if not resolve_fused_indexer_path(
            topk=int(topk),
            num_rows=rows,
            width=width,
            num_heads=int(num_heads),
            compute_capability=compute_capability,
        ):
            continue
        ctas_per_group = _resolve_default_ctas_per_group(
            num_rows=rows,
            max_pages=int(max_pages),
            device=device,
        )
        merge_threshold = resolve_fused_merge_threshold(
            fused_merge,
            ctas_per_group=ctas_per_group,
            num_heads=int(num_heads),
            topk=int(topk),
        )
        policy = (ctas_per_group, merge_threshold)
        if policy not in seen_policies:
            seen_policies.add(policy)
            rows_by_policy.append(rows)
    return tuple(rows_by_policy)


# --- scratch plan trio (layout-agnostic: sizes only) --------------------------
@dataclass(frozen=True)
class B12XFusedIndexerScratchPlan:
    caps: B12XFusedIndexerScratchCaps
    workspace_nbytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def workspace_view(
        self,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Materialize the int32 workspace view from a bound arena slice."""
        arena = scratch_tensor(scratch, self._scratch_specs, owner="fused indexer")
        return arena[: self.workspace_nbytes].view(torch.int32)


def plan_fused_indexer_scratch(
    caps: B12XFusedIndexerScratchCaps,
) -> B12XFusedIndexerScratchPlan:
    device = torch.device(caps.device)
    workspace_nbytes = _fused_indexer_capacity_nbytes(
        caps.max_rows, caps.max_width, caps.topk, device=device
    )
    return B12XFusedIndexerScratchPlan(
        caps=caps,
        workspace_nbytes=workspace_nbytes,
        _scratch_specs=(
            scratch_buffer_spec(
                "fused_indexer.state",
                nbytes=workspace_nbytes,
                device=device,
            ),
        ),
    )


# --- fused kernel SharedStorage -----------------------------------------------
# 1024-thread CTA. Score phase: warps 0-3 (tx < _PAGED_THREADS_PER_CTA) run the
# scorer MMA over a K-sub-tile into the logit staging (s_stage_*); all 1024
# threads then run the tiled_topk radix folding {staging ∪ carry} -> carry.
# Layout = scorer fields (q/weights/k_page/perm/scales/partial_logits) + radix
# scratch (hist0/1, out_idx, counters, cand0/1) + the SMEM running-topk carry
# (double-buffered value+gindex) + the per-sub-tile logit staging (value+gindex).
# Batching: score directly into an over-sized carry (topk + slack); run the radix
# only to "trim" carry back to topk when it would overflow -> one radix per
# ~slack/page pages instead of one per page (the dominant cost). Slack is a
# perf/SMEM knob.
# The fused scorer is synchronization/instruction limited rather than HBM
# limited on SM120 (the 64-head DSV4 B32/128K production launch reaches only
# ~45% DRAM throughput).  Spend the otherwise-idle one-block-per-SM shared-memory
# headroom on a larger DSV4 carry, batching fifty-two 64-token pages between
# exact radix trims.  DSV4 serving graphs always have a large static context
# capacity, so do not create capacity-dependent variants that are unreachable
# in production.  Other contracts retain the smaller footprint, notably GLM's
# top-k-2048 specialization, which cannot fit the larger carry.
# 448 (not 512): the deferred-reduce pipeline's third K stage + partials/scales
# rings cost ~10KB of SMEM; topk-2048 (GLM) needs 64 fewer carry entries to fit
# the 101376B SM120 carveout. One trim per ~7 pages instead of 8.
_BATCH_SLACK = 448
_DSV4_BATCH_SLACK = 2816


def _resolve_fused_batch_slack(
    *,
    kv_layout: int,
    num_heads: int,
    topk: int,
) -> int:
    """Select the capture-static carry batching size for one fused variant."""

    if int(kv_layout) == KV_LAYOUT_PAGED and int(num_heads) == 64 and int(topk) == 512:
        return _DSV4_BATCH_SLACK
    return _BATCH_SLACK


@lru_cache(maxsize=64)
def _fused_indexer_shared_storage_cls(
    padded_q_heads: int,
    tokens_per_work: int,
    num_q_head_tiles: int,
    topk: int,
    carry_cap: int,
    cands: int,
    selection_groups: int = 1,
    score_batch_tiles: int = 1,
    compact: bool = False,
    preserve_histogram: bool = False,
):
    class SharedStorage:
        pass

    def _u8_page(n_bytes=_PAGE_SIZE * _INDEX_HEAD_DIM, *, histogram=False):
        full_page = not compact or (histogram and preserve_histogram)
        return cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(n_bytes) if full_page else 1],
            1024 if full_page else 16,
        ]

    def _i32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Int32, int(n)], 128]

    def _f32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Float32, int(n)], 128]

    annotations = {
        # --- scorer fields (mirror _paged_indexer_shared_storage_cls) ---
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8, 1 if compact else int(padded_q_heads) * _INDEX_HEAD_DIM
            ],
            16,
        ],
        "weights": _f32(1 if compact else int(padded_q_heads)),
        "k_page": _u8_page(histogram=True),
        "k_page_perm": _u8_page(histogram=True),
        "k_page3": _u8_page(4096)
        if selection_groups == 4 and score_batch_tiles == 2
        else _u8_page(),
        "scales": _f32(1 if compact else 4 * _PAGE_SIZE),
        # Second per-page scale slot: the paged branch ping-pongs K between
        # k_page_perm (stage 0) and k_page (stage 1, otherwise unused there),
        # so the scale staging needs a matching second buffer.
        "partial_logits": _f32(
            1 if compact else int(tokens_per_work) * int(num_q_head_tiles)
        ),
        "partial_logits2": _f32(
            1 if compact else int(tokens_per_work) * int(num_q_head_tiles)
        ),
        # --- radix scratch (mirror tiled_topk SharedStorage) ---
        "hist0": _i32(384 * selection_groups),
        "hist1": _i32(384 * selection_groups),
        "out_idx": _i32(int(topk) * selection_groups),
        "counter": _i32(selection_groups),
        "thr_id": _i32(selection_groups),
        "ni0": _i32(1),
        "ni1": _i32(1),
        "last_rem": _i32(selection_groups),
        # relay[0]=pack offset, relay[1]=arrival-old (last-CTA broadcast scalars)
        "relay": _i32(max(2, selection_groups)),
        # cooperative-merge SMEM: suffix-scan of the global histogram + per-CTA
        # scalars (prefix/remaining_k/pivot_bin/rem) + counters (gt-count/base).
        "coop_suffix": _i32(int(_RADIX)),
        "coop_scalars": cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 8], 128],
        "coop_counters": _i32(2),
        "cand0": _i32(cands),
        "cand1": _i32(cands),
        # --- running accumulator: scored tokens append here; over-sized to topk+slack ---
        "carry0_values": _f32(int(carry_cap)),
        "carry0_gindex": _i32(int(carry_cap)),
        # --- trim output (exactly topk) ---
        "carry1_values": _f32(int(topk) * selection_groups),
        "carry1_gindex": _i32(int(topk) * selection_groups),
    }
    SharedStorage.__annotations__ = annotations
    return cute.struct(SharedStorage)


@dsl_user_op
def _mxfp4_publish_cutoff(address: Int32, key: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip), Uint32(key).ir_value(loc=loc, ip=ip)],
        "red.shared.max.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _mxfp4_selection_barrier(threads: cutlass.Constexpr[int], barrier_id: Int32):
    if cutlass.const_expr(threads == 1024):
        cute.arch.sync_threads()
    else:
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)


@cute.jit
def _mxfp4_private_radix_select(
    vin_v: cute.Tensor,
    vin_g: cute.Tensor,
    vout_v: cute.Tensor,
    vout_g: cute.Tensor,
    n_total: Int32,
    topk_static: Int32,
    tx: Int32,
    s_hist0: cute.Tensor,
    s_hist1: cute.Tensor,
    s_out: cute.Tensor,
    h0: Int32,
    ctr: Int32,
    thr: Int32,
    lr: Int32,
    selection_threads: cutlass.Constexpr[int] = 1024,
    barrier_id: Int32 = 0,
    maintain_histogram: cutlass.Constexpr[bool] = False,
    cached_histogram_valid: cutlass.Boolean = False,
):
    """Select exact top-k from BF16-rounded scores with two ordered-key bytes.

    A valid hist1[128:384] cache counts vin_v[:n_total]. Maintained calls
    replace it with counts for their top-k output.
    """
    prefix = Uint32(0)
    prefix_mask = Uint32(0)
    remaining = topk_static
    if tx == Int32(0):
        _smem_st(ctr, Int32(0), Int32(0))
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    if cutlass.const_expr(maintain_histogram):
        if cached_histogram_valid:
            if tx < Int32(256):
                s_hist0[tx] = Int32(s_hist1[Int32(128) + tx])
            _mxfp4_selection_barrier(selection_threads, barrier_id)
        else:
            if tx < Int32(257):
                s_hist0[tx] = Int32(0)
            _mxfp4_selection_barrier(selection_threads, barrier_id)
            i = Int32(tx)
            while i < n_total:
                key = _convert_to_uint32(Float32(vin_v[i])) >> Uint32(16)
                if (key & prefix_mask) == prefix:
                    digit = (key >> Uint32(8)) & Uint32(255)
                    _smem_red_add(h0, Int32(digit), Int32(1))
                i += Int32(selection_threads)
            _mxfp4_selection_barrier(selection_threads, barrier_id)
    else:
        if tx < Int32(257):
            s_hist0[tx] = Int32(0)
        _mxfp4_selection_barrier(selection_threads, barrier_id)
        i = Int32(tx)
        while i < n_total:
            key = _convert_to_uint32(Float32(vin_v[i])) >> Uint32(16)
            if (key & prefix_mask) == prefix:
                digit = (key >> Uint32(8)) & Uint32(255)
                _smem_red_add(h0, Int32(digit), Int32(1))
            i += Int32(selection_threads)
        _mxfp4_selection_barrier(selection_threads, barrier_id)
    value = Int32(0)
    if tx < Int32(256):
        value = Int32(s_hist0[tx])
    for stage in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_down(value, 1 << stage)
        if tx % Int32(32) + Int32(1 << stage) < Int32(32):
            value += other
    if tx < Int32(256) and tx % Int32(32) == Int32(0):
        s_hist1[tx // Int32(32)] = value
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    if tx < Int32(256):
        for w in cutlass.range_constexpr(8):
            if Int32(w) > tx // Int32(32):
                value += Int32(s_hist1[w])
        s_hist0[tx] = value
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    if tx < Int32(256):
        suffix = Int32(s_hist0[tx])
        suffix_next = Int32(0)
        if tx + Int32(1) < Int32(256):
            suffix_next = Int32(s_hist0[tx + Int32(1)])
        if suffix >= remaining and suffix_next < remaining:
            _smem_st(thr, Int32(0), tx)
            _smem_st(lr, Int32(0), remaining - suffix_next)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    pivot = _smem_ld(thr, Int32(0))
    remaining = _smem_ld(lr, Int32(0))
    if cutlass.const_expr(maintain_histogram):
        # BF16 scan prefixes use hist1[0:8]; its tail holds carried counts.
        if tx < Int32(256):
            retained = Int32(0)
            if tx > pivot:
                retained = Int32(s_hist0[tx])
                if tx < Int32(255):
                    retained -= Int32(s_hist0[tx + Int32(1)])
            elif tx == pivot:
                retained = remaining
            s_hist1[Int32(128) + tx] = retained
    i = Int32(tx)
    while i < n_total:
        key = _convert_to_uint32(Float32(vin_v[i])) >> Uint32(16)
        if (key & prefix_mask) == prefix:
            digit = Int32((key >> Uint32(8)) & Uint32(255))
            if digit > pivot:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                if pos < topk_static:
                    s_out[pos] = i

        i += Int32(selection_threads)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    prefix |= Uint32(pivot) << Uint32(8)
    prefix_mask |= Uint32(255) << Uint32(8)
    if tx < Int32(257):
        s_hist0[tx] = Int32(0)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    i = Int32(tx)
    while i < n_total:
        key = _convert_to_uint32(Float32(vin_v[i])) >> Uint32(16)
        if (key & prefix_mask) == prefix:
            digit = (key >> Uint32(0)) & Uint32(255)
            _smem_red_add(h0, Int32(digit), Int32(1))
        i += Int32(selection_threads)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    value = Int32(0)
    if tx < Int32(256):
        value = Int32(s_hist0[tx])
    for stage in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_down(value, 1 << stage)
        if tx % Int32(32) + Int32(1 << stage) < Int32(32):
            value += other
    if tx < Int32(256) and tx % Int32(32) == Int32(0):
        s_hist1[tx // Int32(32)] = value
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    if tx < Int32(256):
        for w in cutlass.range_constexpr(8):
            if Int32(w) > tx // Int32(32):
                value += Int32(s_hist1[w])
        s_hist0[tx] = value
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    if tx < Int32(256):
        suffix = Int32(s_hist0[tx])
        suffix_next = Int32(0)
        if tx + Int32(1) < Int32(256):
            suffix_next = Int32(s_hist0[tx + Int32(1)])
        if suffix >= remaining and suffix_next < remaining:
            _smem_st(thr, Int32(0), tx)
            _smem_st(lr, Int32(0), remaining - suffix_next)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    pivot = _smem_ld(thr, Int32(0))
    remaining = _smem_ld(lr, Int32(0))
    i = Int32(tx)
    while i < n_total:
        key = _convert_to_uint32(Float32(vin_v[i])) >> Uint32(16)
        if (key & prefix_mask) == prefix:
            digit = Int32((key >> Uint32(0)) & Uint32(255))
            if digit > pivot:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                if pos < topk_static:
                    s_out[pos] = i
            if digit == pivot:
                old_rem = _smem_xadd(lr, Int32(0), Int32(-1))
                if old_rem > Int32(0):
                    s_out[topk_static - old_rem] = i
        i += Int32(selection_threads)
    _mxfp4_selection_barrier(selection_threads, barrier_id)
    prefix |= Uint32(pivot) << Uint32(0)
    prefix_mask |= Uint32(255) << Uint32(0)
    i = Int32(tx)
    while i < topk_static:
        sel = Int32(s_out[i])
        vout_v[i] = Float32(vin_v[sel])
        vout_g[i] = Int32(vin_g[sel])
        i += Int32(selection_threads)


@cute.jit
def _mxfp4_scan_global8(
    base_addr: Int64, remaining_k: Int32, lane: Int32, scalars: cute.Tensor
):
    """Resolve a 256-bin global histogram with coherent L2 reads."""
    base = base_addr + Int64(lane) * Int64(32)
    h0, h1, h2, h3 = ld_global_cg_v4_u32(base)
    h4, h5, h6, h7 = ld_global_cg_v4_u32(base + Int64(16))
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [
        s0 + above,
        s1 + above,
        s2 + above,
        s3 + above,
        s4 + above,
        s5 + above,
        s6 + above,
        s7 + above,
        above,
    ]
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            scalars[0] = Uint32(lane * Int32(8) + Int32(j))
            scalars[1] = Uint32(suffix[j + 1])
            scalars[2] = Uint32(suffix[j] - suffix[j + 1])


@cute.jit
def _mxfp4_scan_packed_global8(
    base_addr: Int64, remaining_k: Int32, lane: Int32, scalars: cute.Tensor
):
    """Resolve 256 logical bins packed into 128 global words."""
    base = base_addr + Int64(lane) * Int64(16)
    p0, p1, p2, p3 = ld_global_cg_v4_u32(base)
    h0 = p0 & Uint32(65535)
    h1 = p0 >> Uint32(16)
    h2 = p1 & Uint32(65535)
    h3 = p1 >> Uint32(16)
    h4 = p2 & Uint32(65535)
    h5 = p2 >> Uint32(16)
    h6 = p3 & Uint32(65535)
    h7 = p3 >> Uint32(16)
    s7 = Int32(h7)
    s6 = s7 + Int32(h6)
    s5 = s6 + Int32(h5)
    s4 = s5 + Int32(h4)
    s3 = s4 + Int32(h3)
    s2 = s3 + Int32(h2)
    s1 = s2 + Int32(h1)
    s0 = s1 + Int32(h0)
    above = s0
    for d in (1, 2, 4, 8, 16):
        other = cute.arch.shuffle_sync_down(above, offset=d)
        if lane + Int32(d) < Int32(_WARP_THREADS):
            above = above + other
    above = above - s0
    suffix = [
        s0 + above,
        s1 + above,
        s2 + above,
        s3 + above,
        s4 + above,
        s5 + above,
        s6 + above,
        s7 + above,
        above,
    ]
    for j in cutlass.range_constexpr(8):
        if (suffix[j] >= remaining_k) & (suffix[j + 1] < remaining_k):
            scalars[0] = Uint32(lane * Int32(8) + Int32(j))
            scalars[1] = Uint32(suffix[j + 1])
            scalars[2] = Uint32(suffix[j] - suffix[j + 1])


@dsl_user_op
def _mxfp4_mark_active(addr: Int64, bits: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(bits).ir_value(loc=loc, ip=ip)],
        "red.global.or.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _mxfp4_onepass_merge(
    state: cute.Tensor,
    group: Int32,
    cta: Int32,
    ctas: Int32,
    tx: Int32,
    lane: Int32,
    warp: Int32,
    carry_values: cute.Tensor,
    carry_indices: cute.Tensor,
    count: Int32,
    out_values: cute.Tensor,
    out_indices: cute.Tensor,
    hist: cute.Tensor,
    hist_addr: Int32,
    scalars: cute.Tensor,
    counters: cute.Tensor,
    counter_addr: Int32,
    fine_words: cutlass.Constexpr[int],
    active_cap: cutlass.Constexpr[int],
    count_bits: cutlass.Constexpr[int],
    use_bitmap: cutlass.Constexpr[bool],
    cleanup_work_addr: Int32,
    thread_count: cutlass.Constexpr[int],
):
    # Bitmap compaction reuses 8 KiB of the dead K staging ring.
    bitmap_words = fine_words // 32
    state_words = fine_words + _RADIX + 8 + (bitmap_words if use_bitmap else active_cap)
    fine = Int64(group) * Int64(state_words)
    coarse = fine + Int64(fine_words)
    meta = coarse + Int64(256)
    active = meta + Int64(8)
    i = Int32(tx)
    key = Uint32(0)
    slot = Int32(0)
    previous = Int32(0)
    fine_key = Uint32(0)
    increment = Int32(0)
    bitmap_word = Int32(0)
    bitmap_bits = Uint32(0)
    work_index = Int32(0)
    total = Int32(0)
    high = Uint32(0)
    threshold = Uint32(0)
    key_mask = Uint32(0)
    greater = Int32(0)
    pos = Int32(0)
    packed = Int64(0)
    old = Int64(0)
    if tx < Int32(256):
        hist[tx] = Int32(0)
    cute.arch.sync_threads()
    while i < count:
        key = _convert_to_uint32(Float32(carry_values[i])) >> Uint32(16)
        if cutlass.const_expr(count_bits == 16):
            fine_key = key >> Uint32(1)
            increment = Int32(1) << Int32((key & Uint32(1)) * Uint32(16))
        else:
            fine_key = key
            increment = Int32(1)
        previous = atomic_add_global_i32(
            get_ptr_as_int64(state, fine + Int64(fine_key)), increment
        )
        if previous == Int32(0):
            if cutlass.const_expr(use_bitmap):
                _mxfp4_mark_active(
                    get_ptr_as_int64(state, active + Int64(fine_key >> Uint32(5))),
                    Uint32(1) << (fine_key & Uint32(31)),
                )
            else:
                slot = atomic_add_global_i32(get_ptr_as_int64(state, meta), Int32(1))
                state[active + Int64(slot)] = Int32(fine_key)
        _smem_red_add(hist_addr, Int32(key >> Uint32(8)), Int32(1))
        i += Int32(thread_count)
    cute.arch.sync_threads()
    if tx < Int32(256):
        if Int32(hist[tx]) != Int32(0):
            red_add_global_i32(
                get_ptr_as_int64(state, coarse + Int64(tx)), Int32(hist[tx])
            )
    if tx == Int32(0):
        red_add_global_i32(get_ptr_as_int64(state, meta + Int64(2)), count)
    cute.arch.sync_threads()
    if tx == Int32(0):
        red_add_global_release_i32(get_ptr_as_int64(state, meta + Int64(1)), Int32(1))
        spin_wait_global_ge_i32(get_ptr_as_int64(state, meta + Int64(1)), ctas)
        scalars[3] = Uint32(
            ld_global_acquire_i32(get_ptr_as_int64(state, meta + Int64(2)))
        )
    cute.arch.sync_threads()
    total = Int32(scalars[3])
    if total <= Int32(512):
        if tx == Int32(0):
            old = atomic_add_global_u64(
                get_ptr_as_int64(state, meta + Int64(4)), Int64(count)
            )
            scalars[4] = Uint32(old)
        cute.arch.sync_threads()
        i = Int32(tx)
        while i < count:
            pos = Int32(scalars[4]) + i
            out_values[group, pos] = Float32(carry_values[i])
            out_indices[group, pos] = Int32(carry_indices[i])
            i += Int32(thread_count)
        if cta == Int32(0):
            i = total + Int32(tx)
            while i < Int32(512):
                out_values[group, i] = Float32(float("-inf"))
                out_indices[group, i] = Int32(-1)
                i += Int32(thread_count)
    else:
        if warp == Int32(0):
            _mxfp4_scan_global8(
                get_ptr_as_int64(state, coarse), Int32(512), lane, scalars
            )
        cute.arch.sync_threads()
        high = Uint32(scalars[0])
        greater = Int32(scalars[1])
        threshold = high << Uint32(8)
        key_mask = Uint32(0xFF00)
        if Int32(scalars[2]) != Int32(512) - greater:
            if warp == Int32(0):
                if cutlass.const_expr(count_bits == 16):
                    _mxfp4_scan_packed_global8(
                        get_ptr_as_int64(state, fine + Int64(high << Uint32(7))),
                        Int32(512) - greater,
                        lane,
                        scalars,
                    )
                else:
                    _mxfp4_scan_global8(
                        get_ptr_as_int64(state, fine + Int64(high << Uint32(8))),
                        Int32(512) - greater,
                        lane,
                        scalars,
                    )
            cute.arch.sync_threads()
            threshold = threshold | Uint32(scalars[0])
            greater = greater + Int32(scalars[1])
            key_mask = Uint32(0xFFFF)
        if tx < Int32(2):
            counters[tx] = Int32(0)
        cute.arch.sync_threads()
        i = Int32(tx)
        while i < count:
            key = (
                _convert_to_uint32(Float32(carry_values[i])) >> Uint32(16)
            ) & key_mask
            if key > threshold:
                _smem_red_add(counter_addr, Int32(0), Int32(1))
            elif key == threshold:
                _smem_red_add(counter_addr, Int32(1), Int32(1))
            i += Int32(thread_count)
        cute.arch.sync_threads()
        if tx == Int32(0):
            packed = Int64(Int32(counters[0])) | (
                Int64(Int32(counters[1])) << Int64(32)
            )
            old = atomic_add_global_u64(
                get_ptr_as_int64(state, meta + Int64(4)), packed
            )
            scalars[4] = Uint32(old)
            scalars[5] = Uint32(old >> Int64(32))
            counters[0] = Int32(0)
            counters[1] = Int32(0)
        cute.arch.sync_threads()
        i = Int32(tx)
        while i < count:
            key = (
                _convert_to_uint32(Float32(carry_values[i])) >> Uint32(16)
            ) & key_mask
            pos = Int32(-1)
            if key > threshold:
                pos = Int32(scalars[4]) + _smem_xadd(counter_addr, Int32(0), Int32(1))
            elif key == threshold:
                pos = (
                    greater
                    + Int32(scalars[5])
                    + _smem_xadd(counter_addr, Int32(1), Int32(1))
                )
            if (pos >= Int32(0)) & (pos < Int32(512)):
                out_values[group, pos] = Float32(carry_values[i])
                out_indices[group, pos] = Int32(carry_indices[i])
            i += Int32(thread_count)
    cute.arch.sync_threads()
    if tx == Int32(0):
        threadfence()
        scalars[7] = Uint32(
            atomic_add_global_i32(get_ptr_as_int64(state, meta + Int64(6)), Int32(1))
        )
    cute.arch.sync_threads()
    if Int32(scalars[7]) == ctas - Int32(1):
        threadfence()
        if cutlass.const_expr(use_bitmap):
            cute.arch.cp_async_wait_group(0)
            if tx == Int32(0):
                counters[0] = Int32(0)
            cute.arch.sync_threads()
            i = Int32(tx)
            while i < Int32(bitmap_words):
                bitmap_bits = Uint32(
                    ld_global_acquire_i32(get_ptr_as_int64(state, active + Int64(i)))
                )
                if bitmap_bits != Uint32(0):
                    slot = _smem_xadd(counter_addr, Int32(0), Int32(1))
                    _smem_st(cleanup_work_addr, slot, i)
                    _smem_st(
                        cleanup_work_addr,
                        slot + Int32(bitmap_words),
                        Int32(bitmap_bits),
                    )
                i += Int32(thread_count)
            cute.arch.sync_threads()
            work_index = warp
            while work_index < Int32(counters[0]):
                bitmap_word = _smem_ld(cleanup_work_addr, work_index)
                bitmap_bits = Uint32(
                    _smem_ld(cleanup_work_addr, work_index + Int32(bitmap_words))
                )
                if (bitmap_bits & (Uint32(1) << Uint32(lane))) != Uint32(0):
                    state[fine + Int64(bitmap_word) * Int64(32) + Int64(lane)] = Int32(
                        0
                    )
                if lane == Int32(0):
                    state[active + Int64(bitmap_word)] = Int32(0)
                work_index += Int32(thread_count // 32)
        else:
            if tx == Int32(0):
                scalars[6] = Uint32(
                    ld_global_acquire_i32(get_ptr_as_int64(state, meta))
                )
            cute.arch.sync_threads()
            i = Int32(tx)
            while i < Int32(scalars[6]):
                fine_key = Uint32(
                    ld_global_acquire_i32(get_ptr_as_int64(state, active + Int64(i)))
                )
                state[fine + Int64(fine_key)] = Int32(0)
                state[active + Int64(i)] = Int32(0)
                i += Int32(thread_count)
        if tx < Int32(256):
            state[coarse + Int64(tx)] = Int32(0)
        cute.arch.sync_threads()
        if tx < Int32(8):
            state[meta + Int64(tx)] = Int32(0)

    return Int32(scalars[7]) == ctas - Int32(1)


@cute.jit
def _mxfp4_radix_select(
    vin_v: cute.Tensor,
    vin_g: cute.Tensor,
    vout_v: cute.Tensor,
    vout_g: cute.Tensor,
    n_total: Int32,
    topk_static: Int32,
    tx: Int32,
    s_hist0: cute.Tensor,
    s_hist1: cute.Tensor,
    s_out: cute.Tensor,
    h0: Int32,
    ctr: Int32,
    thr: Int32,
    lr: Int32,
    max_items: cutlass.Constexpr[int] = 2048,
    indices_only: cutlass.Constexpr[bool] = False,
    priority_id: Int32 | None = None,
    threads: cutlass.Constexpr[int] = 1024,
):
    """Select exact top-k from BF16-rounded scores with two ordered-key bytes."""
    keys = cute.make_rmem_tensor(((max_items + threads - 1) // threads,), Uint32)
    keys.fill(0)
    for slot in cutlass.range_constexpr((max_items + threads - 1) // threads):
        idx = tx + Int32(slot * threads)
        if idx < n_total:
            keys[slot] = _convert_to_uint32(Float32(vin_v[idx])) >> Uint32(16)
            if cutlass.const_expr(priority_id is not None):
                if Int32(vin_g[idx]) == priority_id:
                    keys[slot] = Uint32(65535)
    prefix = Uint32(0)
    prefix_mask = Uint32(0)
    remaining = topk_static
    if tx == Int32(0):
        _smem_st(ctr, Int32(0), Int32(0))
    cute.arch.sync_threads()
    if tx < Int32(257):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    for slot in cutlass.range_constexpr((max_items + threads - 1) // threads):
        i = tx + Int32(slot * threads)
        if i < n_total:
            key = Uint32(keys[slot])
            if (key & prefix_mask) == prefix:
                digit = (key >> Uint32(8)) & Uint32(255)
                _smem_red_add(h0, Int32(digit), Int32(1))

    cute.arch.sync_threads()
    if tx < Int32(32):
        base = h0 + tx * Int32(32)
        c0, c1, c2, c3 = ld_shared_v4_u32(base)
        c4, c5, c6, c7 = ld_shared_v4_u32(base + Int32(16))
        s7 = Int32(c7)
        s6 = s7 + Int32(c6)
        s5 = s6 + Int32(c5)
        s4 = s5 + Int32(c4)
        s3 = s4 + Int32(c3)
        s2 = s3 + Int32(c2)
        s1 = s2 + Int32(c1)
        s0 = s1 + Int32(c0)
        above = s0
        for distance in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_down(above, 1 << distance)
            if tx + Int32(1 << distance) < Int32(32):
                above += other
        above -= s0
        suffix = [
            s0 + above,
            s1 + above,
            s2 + above,
            s3 + above,
            s4 + above,
            s5 + above,
            s6 + above,
            s7 + above,
            above,
        ]
        for digit in cutlass.range_constexpr(8):
            if (suffix[digit] >= remaining) & (suffix[digit + 1] < remaining):
                _smem_st(thr, Int32(0), tx * Int32(8) + Int32(digit))
                _smem_st(lr, Int32(0), remaining - suffix[digit + 1])
    cute.arch.sync_threads()
    pivot = _smem_ld(thr, Int32(0))
    remaining = _smem_ld(lr, Int32(0))
    for slot in cutlass.range_constexpr((max_items + threads - 1) // threads):
        i = tx + Int32(slot * threads)
        if i < n_total:
            key = Uint32(keys[slot])
            if (key & prefix_mask) == prefix:
                digit = Int32((key >> Uint32(8)) & Uint32(255))
                if digit > pivot:
                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                    if pos < topk_static:
                        s_out[pos] = i

    cute.arch.sync_threads()
    prefix |= Uint32(pivot) << Uint32(8)
    prefix_mask |= Uint32(255) << Uint32(8)
    if tx < Int32(257):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    for slot in cutlass.range_constexpr((max_items + threads - 1) // threads):
        i = tx + Int32(slot * threads)
        if i < n_total:
            key = Uint32(keys[slot])
            if (key & prefix_mask) == prefix:
                digit = (key >> Uint32(0)) & Uint32(255)
                _smem_red_add(h0, Int32(digit), Int32(1))

    cute.arch.sync_threads()
    if tx < Int32(32):
        base = h0 + tx * Int32(32)
        c0, c1, c2, c3 = ld_shared_v4_u32(base)
        c4, c5, c6, c7 = ld_shared_v4_u32(base + Int32(16))
        s7 = Int32(c7)
        s6 = s7 + Int32(c6)
        s5 = s6 + Int32(c5)
        s4 = s5 + Int32(c4)
        s3 = s4 + Int32(c3)
        s2 = s3 + Int32(c2)
        s1 = s2 + Int32(c1)
        s0 = s1 + Int32(c0)
        above = s0
        for distance in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_down(above, 1 << distance)
            if tx + Int32(1 << distance) < Int32(32):
                above += other
        above -= s0
        suffix = [
            s0 + above,
            s1 + above,
            s2 + above,
            s3 + above,
            s4 + above,
            s5 + above,
            s6 + above,
            s7 + above,
            above,
        ]
        for digit in cutlass.range_constexpr(8):
            if (suffix[digit] >= remaining) & (suffix[digit + 1] < remaining):
                _smem_st(thr, Int32(0), tx * Int32(8) + Int32(digit))
                _smem_st(lr, Int32(0), remaining - suffix[digit + 1])
    cute.arch.sync_threads()
    pivot = _smem_ld(thr, Int32(0))
    remaining = _smem_ld(lr, Int32(0))
    for slot in cutlass.range_constexpr((max_items + threads - 1) // threads):
        i = tx + Int32(slot * threads)
        if i < n_total:
            key = Uint32(keys[slot])
            if (key & prefix_mask) == prefix:
                digit = Int32((key >> Uint32(0)) & Uint32(255))
                if digit > pivot:
                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                    if pos < topk_static:
                        s_out[pos] = i
                if digit == pivot:
                    old_rem = _smem_xadd(lr, Int32(0), Int32(-1))
                    if old_rem > Int32(0):
                        s_out[topk_static - old_rem] = i

    cute.arch.sync_threads()
    prefix |= Uint32(pivot) << Uint32(0)
    prefix_mask |= Uint32(255) << Uint32(0)
    if cutlass.const_expr(not indices_only):
        i = Int32(tx)
        while i < topk_static:
            sel = Int32(s_out[i])
            vout_v[i] = Float32(vin_v[sel])
            vout_g[i] = Int32(vin_g[sel])
            i += Int32(threads)


@cute.jit
def _fused_radix_select(
    vin_v: cute.Tensor,
    vin_g: cute.Tensor,
    vout_v: cute.Tensor,
    vout_g: cute.Tensor,
    n_total: Int32,
    topk_static: Int32,
    cands: cutlass.Constexpr[int],
    tx: Int32,
    s_hist0: cute.Tensor,
    s_hist1: cute.Tensor,
    s_out: cute.Tensor,
    s_cand0: cute.Tensor,
    s_cand1: cute.Tensor,
    h0: Int32,
    ctr: Int32,
    thr: Int32,
    ni0: Int32,
    ni1: Int32,
    lr: Int32,
    threads: cutlass.Constexpr[int] = 1024,
):
    """Exact radix top-k of vin[0:n_total] -> vout[0:topk_static] (single source).

    Assumes n_total > topk_static (callers only trim when over capacity). Reuses the
    tiled_topk coarse-8-bit-histogram + byte-refine. All CTA threads must call this.
    """
    topk = topk_static
    if tx < Int32(257):
        s_hist0[tx] = Int32(0)
    cute.arch.sync_threads()
    # Single contiguous source: read vin_v[idx] directly (no virtual-index branch)
    # and unroll the full-length scan by _SCAN_UNROLL for memory-level parallelism,
    # mirroring tiled_topk's select (this is the rows=1 last-CTA merge hot loop).
    idx_base = Int32(tx)
    full_scan_limit = n_total - Int32((_SCAN_UNROLL - 1) * threads)
    while idx_base < full_scan_limit:
        for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
            sidx = idx_base + Int32(scan_u * threads)
            bin8 = _convert_to_uint8(Float32(vin_v[sidx]))
            _smem_red_add(h0, Int32(bin8), Int32(1))
        idx_base += Int32(threads * _SCAN_UNROLL)
    while idx_base < n_total:
        bin8 = _convert_to_uint8(Float32(vin_v[idx_base]))
        _smem_red_add(h0, Int32(bin8), Int32(1))
        idx_base += Int32(threads)
    cute.arch.sync_threads()
    for stage in cutlass.range_constexpr(8):
        j = Int32(1 << stage)
        if tx < Int32(256):
            if (stage & 1) == 0:
                value = Int32(s_hist0[tx])
                if tx < Int32(256) - j:
                    value = value + Int32(s_hist0[tx + j])
                s_hist1[tx] = value
            else:
                value = Int32(s_hist1[tx])
                if tx < Int32(256) - j:
                    value = value + Int32(s_hist1[tx + j])
                s_hist0[tx] = value
        cute.arch.sync_threads()
    if tx < Int32(256):
        val_tx = Int32(s_hist0[tx])
        val_tx1 = Int32(s_hist0[tx + Int32(1)])
        if val_tx > topk:
            if val_tx1 <= topk:
                _smem_st(thr, Int32(0), Int32(tx))
                _smem_st(ni0, Int32(0), Int32(0))
                _smem_st(ctr, Int32(0), Int32(0))
    cute.arch.sync_threads()
    threshold_bin = _smem_ld(thr, Int32(0))
    topk = topk - Int32(s_hist0[threshold_bin + Int32(1)])
    # Captured below (refine-setup) = true #candidates in the coarse threshold bin; used
    # to detect candidate-buffer overflow and fall back to an exact re-scan radix.
    bin_count = Int32(0)

    if topk == Int32(0):
        idx_base = Int32(tx)
        while idx_base < n_total:
            bin8 = _convert_to_uint8(Float32(vin_v[idx_base]))
            if Int32(bin8) > threshold_bin:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                s_out[pos] = idx_base
            idx_base += Int32(threads)

    if topk != Int32(0):
        cute.arch.sync_threads()
        if tx < Int32(257):
            s_hist0[tx] = Int32(0)
        cute.arch.sync_threads()
        idx_base = Int32(tx)
        full_scan_limit = n_total - Int32((_SCAN_UNROLL - 1) * threads)
        while idx_base < full_scan_limit:
            for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                sidx = idx_base + Int32(scan_u * threads)
                raw_input = Float32(vin_v[sidx])
                bin8 = _convert_to_uint8(raw_input)
                if Int32(bin8) > threshold_bin:
                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                    s_out[pos] = sidx
                else:
                    if Int32(bin8) == threshold_bin:
                        cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                        if cand_pos < Int32(cands):
                            s_cand0[cand_pos] = sidx
                            key32 = _convert_to_uint32(raw_input)
                            sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                            _smem_red_add(h0, Int32(sub_bin), Int32(1))
            idx_base += Int32(threads * _SCAN_UNROLL)
        while idx_base < n_total:
            raw_input = Float32(vin_v[idx_base])
            bin8 = _convert_to_uint8(raw_input)
            if Int32(bin8) > threshold_bin:
                pos = _smem_xadd(ctr, Int32(0), Int32(1))
                s_out[pos] = idx_base
            else:
                if Int32(bin8) == threshold_bin:
                    cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                    if cand_pos < Int32(cands):
                        s_cand0[cand_pos] = idx_base
                        key32 = _convert_to_uint32(raw_input)
                        sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                        _smem_red_add(h0, Int32(sub_bin), Int32(1))
            idx_base += Int32(threads)
        cute.arch.sync_threads()
        # ni0 now holds the FULL threshold-bin candidate count: the xadd ran for every
        # bin-matching key, even past `cands`. Capture it to detect refine overflow.
        bin_count = Int32(_smem_ld(ni0, Int32(0)))

        for round_idx in cutlass.range_constexpr(4):
            if topk != Int32(-1) and bin_count <= Int32(cands):
                r_idx_is_0 = (round_idx % 2) == 0
                r_idx_next_is_0 = not r_idx_is_0
                raw_num_input = (
                    _smem_ld(ni0, Int32(0))
                    if cutlass.const_expr(r_idx_is_0)
                    else _smem_ld(ni1, Int32(0))
                )
                num_input = (
                    raw_num_input if raw_num_input < Int32(cands) else Int32(cands)
                )
                for stage in cutlass.range_constexpr(8):
                    j = Int32(1 << stage)
                    if tx < Int32(256):
                        if (stage & 1) == 0:
                            value = Int32(s_hist0[tx])
                            if tx < Int32(256) - j:
                                value = value + Int32(s_hist0[tx + j])
                            s_hist1[tx] = value
                        else:
                            value = Int32(s_hist1[tx])
                            if tx < Int32(256) - j:
                                value = value + Int32(s_hist1[tx + j])
                            s_hist0[tx] = value
                    cute.arch.sync_threads()
                if tx < Int32(256):
                    val_tx = Int32(s_hist0[tx])
                    val_tx1 = Int32(s_hist0[tx + Int32(1)])
                    if val_tx > topk:
                        if val_tx1 <= topk:
                            _smem_st(thr, Int32(0), Int32(tx))
                            if cutlass.const_expr(r_idx_next_is_0):
                                _smem_st(ni0, Int32(0), Int32(0))
                            else:
                                _smem_st(ni1, Int32(0), Int32(0))
                            _smem_st(lr, Int32(0), topk - val_tx1)
                cute.arch.sync_threads()
                sub_threshold = _smem_ld(thr, Int32(0))
                topk = topk - Int32(s_hist0[sub_threshold + Int32(1)])

                if topk == Int32(0):
                    i = Int32(tx)
                    while i < num_input:
                        c_idx = (
                            Int32(s_cand0[i])
                            if cutlass.const_expr(r_idx_is_0)
                            else Int32(s_cand1[i])
                        )
                        offset = Int32(24 - round_idx * 8)
                        raw_val = Float32(vin_v[c_idx])
                        key32 = _convert_to_uint32(raw_val)
                        bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                        if Int32(bin) > sub_threshold:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = c_idx
                        i += Int32(threads)
                    topk = Int32(-1)

                if topk != Int32(-1):
                    cute.arch.sync_threads()
                    if tx < Int32(257):
                        s_hist0[tx] = Int32(0)
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < num_input:
                        c_idx = (
                            Int32(s_cand0[i])
                            if cutlass.const_expr(r_idx_is_0)
                            else Int32(s_cand1[i])
                        )
                        raw_val = Float32(vin_v[c_idx])
                        offset = Int32(24 - round_idx * 8)
                        key32 = _convert_to_uint32(raw_val)
                        bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                        if Int32(bin) > sub_threshold:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = c_idx
                        else:
                            if Int32(bin) == sub_threshold:
                                if cutlass.const_expr(round_idx == 3):
                                    old_rem = _smem_xadd(lr, Int32(0), Int32(-1))
                                    if old_rem > Int32(0):
                                        s_out[topk_static - old_rem] = c_idx
                                else:
                                    cand_pos = (
                                        _smem_xadd(ni0, Int32(0), Int32(1))
                                        if cutlass.const_expr(r_idx_next_is_0)
                                        else _smem_xadd(ni1, Int32(0), Int32(1))
                                    )
                                    if cand_pos < Int32(cands):
                                        if cutlass.const_expr(r_idx_next_is_0):
                                            s_cand0[cand_pos] = c_idx
                                        else:
                                            s_cand1[cand_pos] = c_idx
                                        sub_bin = (
                                            key32 >> Uint32(24 - (round_idx + 1) * 8)
                                        ) & Uint32(0xFF)
                                        _smem_red_add(h0, Int32(sub_bin), Int32(1))
                        i += Int32(threads)
                    cute.arch.sync_threads()

    # ---- exact overflow fallback ----
    # The buffered coarse+refine above drops winners when more than `cands`
    # candidates share the coarse threshold bucket (clustered scores; the
    # cross-CTA merge can pack up to ctas_per_group*topk candidates). Redo the
    # selection exactly by re-scanning vin -- see _exact_overflow_fallback. The
    # single contiguous vin_v source uses the flat=True load path.
    if bin_count > Int32(cands):
        _exact_overflow_fallback(
            tx,
            n_total,
            topk_static,
            s_hist0,
            s_hist1,
            s_out,
            h0,
            ctr,
            thr,
            ni0,
            ni1,
            lr,
            flat=True,
            flat_values=vin_v,
            input_tensor=vin_v,
            carry_values=vin_v,
            row_base=Int32(0),
            row_start=Int32(0),
            carry_base=Int32(0),
            chunk_len=Int32(0),
            block_q=1,
            block_k=1,
            is_tiled=False,
            is_first=True,
            threads=threads,
        )
    cute.arch.sync_threads()
    i = Int32(tx)
    while i < topk_static:
        sel = Int32(s_out[i])
        vout_v[i] = Float32(vin_v[sel])
        vout_g[i] = Int32(vin_g[sel])
        i += Int32(threads)


@cute.jit
def _mxfp4_order_output_packed_pairs(
    out_indices,
    out_values,
    group_id,
    ids,
    vals,
    tx,
    count,
    shared_input: cutlass.Constexpr,
    cleanup_state: cute.Tensor,
    cleanup_words: cutlass.Constexpr,
    threads: cutlass.Constexpr[int] = 1024,
):
    cute.arch.sync_threads()
    if cutlass.const_expr(cleanup_words > 0):
        if tx >= Int32(256):
            word = (tx - Int32(256)) * Int32(4)
            while word < Int32(cleanup_words):
                st_global_v4_u32(
                    get_ptr_as_int64(
                        cleanup_state, _fused_state_offset(group_id, word)
                    ),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
                word += Int32((threads - 256) * 4)
    if tx < Int32(256):
        keys = cute.make_rmem_tensor((2,), Uint32)
        for item in cutlass.range_constexpr(2):
            index = tx + Int32(item * 256)
            key = Uint32(0xFFFFFFFF)
            if index < count:
                if cutlass.const_expr(shared_input):
                    pos = Int32(ids[index])
                    value = Float32(vals[index])
                else:
                    pos = Int32(out_indices[group_id, index])
                    value = Float32(out_values[group_id, index])
                if (pos >= Int32(0)) & (value > Float32(float("-inf"))):
                    key = (Uint32(pos) << Uint32(16)) | (
                        f32_to_raw_bits(value) >> Uint32(16)
                    )
            keys[item] = key
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        for level in cutlass.range_constexpr(1, 10):
            for stage in cutlass.range_constexpr(level):
                stride = 1 << (level - 1 - stage)
                others = cute.make_rmem_tensor((2,), Uint32)
                if cutlass.const_expr(stride == 256):
                    others[0] = keys[1]
                    others[1] = keys[0]
                elif cutlass.const_expr(stride < 32):
                    for item in cutlass.range_constexpr(2):
                        others[item] = cute.arch.shuffle_sync_bfly(
                            keys[item], offset=stride
                        )
                else:
                    for item in cutlass.range_constexpr(2):
                        ids[tx + Int32(item * 256)] = Int32(keys[item])
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    for item in cutlass.range_constexpr(2):
                        others[item] = Uint32(
                            ids[(tx ^ Int32(stride)) + Int32(item * 256)]
                        )
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                for item in cutlass.range_constexpr(2):
                    index = tx + Int32(item * 256)
                    ascending = (index & Int32(1 << level)) == Int32(0)
                    lower = (index & Int32(stride)) == Int32(0)
                    take_min = ascending == lower
                    key = keys[item]
                    other = others[item]
                    if ((key > other) & take_min) | ((key < other) & (not take_min)):
                        key = other
                    keys[item] = key
        for item in cutlass.range_constexpr(2):
            index = tx + Int32(item * 256)
            key = keys[item]
            valid = key != Uint32(0xFFFFFFFF)
            out_indices[group_id, index] = (
                Int32(key >> Uint32(16)) if valid else Int32(-1)
            )
            out_values[group_id, index] = (
                u32_as_f32(key << Uint32(16)) if valid else Float32(float("-inf"))
            )


@cute.jit
def _mxfp4_order_output_packed_vectors(
    out_indices,
    out_values,
    group_id,
    ids,
    vals,
    tx,
    count,
    shared_input: cutlass.Constexpr,
    cleanup_state: cute.Tensor,
    cleanup_words: cutlass.Constexpr,
    threads: cutlass.Constexpr[int] = 1024,
):
    cute.arch.sync_threads()
    if cutlass.const_expr(cleanup_words > 0):
        if tx >= Int32(256):
            word = (tx - Int32(256)) * Int32(4)
            while word < Int32(cleanup_words):
                st_global_v4_u32(
                    get_ptr_as_int64(
                        cleanup_state, _fused_state_offset(group_id, word)
                    ),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
                word += Int32((threads - 256) * 4)
    if tx < Int32(256):
        shared_pairs = cute.recast_tensor(ids, Uint64)
        keys = cute.make_rmem_tensor((2,), Uint32)
        for item in cutlass.range_constexpr(2):
            index = tx * Int32(2) + Int32(item)
            key = Uint32(0xFFFFFFFF)
            if index < count:
                if cutlass.const_expr(shared_input):
                    pos = Int32(ids[index])
                    value = Float32(vals[index])
                else:
                    pos = Int32(out_indices[group_id, index])
                    value = Float32(out_values[group_id, index])
                if (pos >= Int32(0)) & (value > Float32(float("-inf"))):
                    key = (Uint32(pos) << Uint32(16)) | (
                        f32_to_raw_bits(value) >> Uint32(16)
                    )
            keys[item] = key
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        for level in cutlass.range_constexpr(1, 10):
            for stage in cutlass.range_constexpr(level):
                stride = 1 << (level - 1 - stage)
                others = cute.make_rmem_tensor((2,), Uint32)
                if cutlass.const_expr(stride == 1):
                    others[0] = keys[1]
                    others[1] = keys[0]
                elif cutlass.const_expr(stride < 64):
                    for item in cutlass.range_constexpr(2):
                        others[item] = cute.arch.shuffle_sync_bfly(
                            keys[item], offset=stride // 2
                        )
                else:
                    shared_pairs[tx] = Uint64(keys[0]) | (Uint64(keys[1]) << Uint64(32))
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    pair = Uint64(shared_pairs[tx ^ Int32(stride // 2)])
                    others[0] = Uint32(pair)
                    others[1] = Uint32(pair >> Uint64(32))
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                for item in cutlass.range_constexpr(2):
                    index = tx * Int32(2) + Int32(item)
                    ascending = (index & Int32(1 << level)) == Int32(0)
                    lower = (index & Int32(stride)) == Int32(0)
                    take_min = ascending == lower
                    key = keys[item]
                    other = others[item]
                    if ((key > other) & take_min) | ((key < other) & (not take_min)):
                        key = other
                    keys[item] = key
        key0 = keys[0]
        key1 = keys[1]
        valid0 = key0 != Uint32(0xFFFFFFFF)
        valid1 = key1 != Uint32(0xFFFFFFFF)
        position0 = key0 >> Uint32(16) if valid0 else Uint32(0xFFFFFFFF)
        position1 = key1 >> Uint32(16) if valid1 else Uint32(0xFFFFFFFF)
        value0 = key0 << Uint32(16) if valid0 else Uint32(0xFF800000)
        value1 = key1 << Uint32(16) if valid1 else Uint32(0xFF800000)
        offset = Int64(group_id) * Int64(512) + Int64(tx) * Int64(2)
        st_global_v2_u32(get_ptr_as_int64(out_indices, offset), position0, position1)
        st_global_v2_u32(get_ptr_as_int64(out_values, offset), value0, value1)


@cute.jit
def _mxfp4_order_output(
    out_indices,
    out_values,
    group_id,
    ids,
    vals,
    tx,
    count,
    shared_input: cutlass.Constexpr,
    cleanup_state: cute.Tensor,
    cleanup_words: cutlass.Constexpr,
    position_capacity: cutlass.Constexpr,
    vector_output_pairs: cutlass.Constexpr,
    threads: cutlass.Constexpr[int] = 1024,
):
    if cutlass.const_expr(position_capacity <= 65536):
        if cutlass.const_expr(vector_output_pairs):
            _mxfp4_order_output_packed_vectors(
                out_indices,
                out_values,
                group_id,
                ids,
                vals,
                tx,
                count,
                shared_input,
                cleanup_state,
                cleanup_words,
                threads=threads,
            )
        else:
            _mxfp4_order_output_packed_pairs(
                out_indices,
                out_values,
                group_id,
                ids,
                vals,
                tx,
                count,
                shared_input,
                cleanup_state,
                cleanup_words,
                threads=threads,
            )
    else:
        _mxfp4_order_output_full(
            out_indices,
            out_values,
            group_id,
            ids,
            vals,
            tx,
            count,
            shared_input,
            cleanup_state,
            cleanup_words,
            threads=threads,
        )


@cute.jit
def _mxfp4_order_output_full(
    out_indices,
    out_values,
    group_id,
    ids,
    vals,
    tx,
    count,
    shared_input: cutlass.Constexpr,
    cleanup_state: cute.Tensor,
    cleanup_words: cutlass.Constexpr,
    threads: cutlass.Constexpr[int] = 1024,
):
    cute.arch.sync_threads()
    if cutlass.const_expr(threads > 512):
        if cutlass.const_expr(cleanup_words > 0):
            if tx >= Int32(512):
                word = (tx - Int32(512)) * Int32(4)
                while word < Int32(cleanup_words):
                    st_global_v4_u32(
                        get_ptr_as_int64(
                            cleanup_state, _fused_state_offset(group_id, word)
                        ),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
                    )
                    word += Int32((threads - 512) * 4)
    if tx < Int32(512):
        pos = Int32(2147483647)
        value = Float32(float("-inf"))
        if tx < count:
            if cutlass.const_expr(shared_input):
                raw = Int32(ids[tx])
                value = Float32(vals[tx])
            else:
                raw = Int32(out_indices[group_id, tx])
                value = Float32(out_values[group_id, tx])
            if (raw >= Int32(0)) & (value > Float32(float("-inf"))):
                pos = raw
        cute.arch.barrier(barrier_id=1, number_of_threads=512)
        for level in cutlass.range_constexpr(1, 10):
            for stage in cutlass.range_constexpr(level):
                stride = 1 << (level - 1 - stage)
                other_pos = Int32(2147483647)
                other_value = Float32(float("-inf"))
                if cutlass.const_expr(stride < 32):
                    other_pos = cute.arch.shuffle_sync_bfly(pos, offset=stride)
                    other_value = cute.arch.shuffle_sync_bfly(value, offset=stride)
                else:
                    if tx < Int32(512):
                        ids[tx] = pos
                        vals[tx] = value
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                    if tx < Int32(512):
                        other_pos = Int32(ids[tx ^ Int32(stride)])
                        other_value = Float32(vals[tx ^ Int32(stride)])
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                ascending = (tx & Int32(1 << level)) == Int32(0)
                lower = (tx & Int32(stride)) == Int32(0)
                take_min = ascending == lower
                if ((pos > other_pos) & take_min) | (
                    (pos < other_pos) & (not take_min)
                ):
                    pos = other_pos
                    value = other_value
        if tx < Int32(512):
            valid = pos != Int32(2147483647)
            out_indices[group_id, tx] = pos if valid else Int32(-1)
            out_values[group_id, tx] = value if valid else Float32(float("-inf"))
    if cutlass.const_expr(threads == 512 and cleanup_words > 0):
        cute.arch.sync_threads()
        word = tx * Int32(4)
        while word < Int32(cleanup_words):
            st_global_v4_u32(
                get_ptr_as_int64(cleanup_state, _fused_state_offset(group_id, word)),
                Uint32(0),
                Uint32(0),
                Uint32(0),
                Uint32(0),
            )
            word += Int32(threads * 4)


def _check_mxfp4_cooperative_residency(
    kernel, launcher, *, num_rows: int, num_sms: int
) -> int:
    """Validate the resolved current-device launcher before warmup or capture."""
    from b12x._lib.compile_plan import DeferredCuTeKernel
    from b12x.preparation.types import CandidateResourceIneligibleError

    if not kernel.mxfp4 or num_rows < 1 or num_sms < 1:
        raise ValueError("MXFP4 residency requires positive row and SM counts")
    if isinstance(launcher, DeferredCuTeKernel):
        raise RuntimeError("MXFP4 residency requires a resolved executable")
    executable = getattr(launcher, "_executable", None)
    if executable is None or not executable.kernel_info:
        raise RuntimeError("MXFP4 compiled kernel metadata is unavailable")
    shared_bytes = kernel._get_shared_storage_cls().__sizeof__()
    launch_threads = kernel.radix_threads
    if kernel.mxfp4_small_output:
        launch_threads = 512 if kernel.score_width == kernel.topk else 1024
        shared_bytes = 512 * 8
    library = cuda.CUlibrary(int(executable.library))
    capacity = None
    for name in executable.kernel_info:
        status, compiled_kernel = cuda.cuLibraryGetKernel(library, name.encode())
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("cannot resolve MXFP4 kernel resources")
        status, function = cuda.cuKernelGetFunction(compiled_kernel)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("cannot query MXFP4 kernel resources")
        status, resident = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, launch_threads, shared_bytes
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("cannot query MXFP4 kernel occupancy")
        capacity = resident if capacity is None else min(capacity, resident)
    if capacity < 1:
        raise CandidateResourceIneligibleError(
            "MXFP4 kernel has no resident CUDA blocks"
        )
    cooperative = (
        kernel.merge_in_kernel and kernel.max_seq_capacity > kernel.merge_threshold
    )
    if cooperative:
        budget = min(capacity, kernel.coop_resident_blocks)
        if num_rows * kernel.ctas_per_group > num_sms * budget:
            raise CandidateResourceIneligibleError(
                "cooperative MXFP4 grid exceeds compiled kernel residency"
            )
    return capacity


@cute.jit
def _mxfp4_order_small_output(
    out_indices,
    out_values,
    group_id,
    ids,
    vals,
    tx,
    count,
    register_input: tuple | None = None,
):
    if cutlass.const_expr(register_input is None):
        cute.arch.sync_threads()
    if tx < Int32(512):
        pos = Int32(2147483647)
        value = Float32(float("-inf"))
        raw = Int32(-1)
        if cutlass.const_expr(register_input is None):
            if tx < count:
                raw = Int32(ids[tx])
                value = Float32(vals[tx])
        else:
            raw = Int32(register_input[0])
            value = Float32(register_input[1])
        if (raw >= Int32(0)) & (value > Float32(float("-inf"))):
            pos = raw
        if cutlass.const_expr(register_input is None):
            cute.arch.barrier(barrier_id=1, number_of_threads=512)
        for level in cutlass.range_constexpr(1, 10):
            for stage in cutlass.range_constexpr(level):
                stride = 1 << (level - 1 - stage)
                other_pos = Int32(2147483647)
                other_value = Float32(float("-inf"))
                if cutlass.const_expr(stride < 32):
                    other_pos = cute.arch.shuffle_sync_bfly(pos, offset=stride)
                    other_value = cute.arch.shuffle_sync_bfly(value, offset=stride)
                else:
                    if tx < Int32(512):
                        ids[tx] = pos
                        vals[tx] = value
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                    if tx < Int32(512):
                        other_pos = Int32(ids[tx ^ Int32(stride)])
                        other_value = Float32(vals[tx ^ Int32(stride)])
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                ascending = (tx & Int32(1 << level)) == Int32(0)
                lower = (tx & Int32(stride)) == Int32(0)
                take_min = ascending == lower
                if ((pos > other_pos) & take_min) | (
                    (pos < other_pos) & (not take_min)
                ):
                    pos = other_pos
                    value = other_value
        if tx < Int32(512):
            valid = pos != Int32(2147483647)
            out_indices[group_id, tx] = pos if valid else Int32(-1)
            out_values[group_id, tx] = value if valid else Float32(float("-inf"))


class DSAFusedIndexerKernel:
    """Fused DSA indexer: score a query row's K slice and select its top-k.

    One CTA per (row, slice). ``ctas_per_group`` CTAs split a
    row's pages; each scores its slice and the group merges across CTAs.

    Score phase (paged): 64-token pages stream through a 3-stage cp.async
    ring one page ahead of the MMA; the score warps run the mxfp8 MMA
    against the ldmatrix-permuted Q staged in shared memory and a page's
    reduce and carry append are deferred behind the next page's barrier.
    The 4-head-tile paged contract at 3 rows or more instead scores 8
    tokens per warp straight from L2 (``direct_k_score``). Scored tokens
    append to an over-sized carry accumulator that a radix select trims
    back to ``topk`` when it would overflow, so the trim runs once per
    batch of pages rather than per page.

    Merge phase (``ctas_per_group > 1``): groups above ``merge_threshold``
    select cooperatively; smaller groups pack their carries into a global slab
    for last-CTA selection. FP8 uses 12/12/8-bit cooperative refinement; MXFP4
    defaults to two 8-bit rounds. Explicit MXFP4 16-bit refinement publishes a
    full BF16 histogram in one barrier, with full-word or bounded halfword
    counts and list or bitmap cleanup. Its prepared layout applies to every
    live length. Statically serial MXFP4 groups may use two-counter state.
    Every merge arm resets its state before the next launch or graph replay.
    """

    def __init__(
        self,
        *,
        num_heads_static: int,
        topk: int,
        kv_layout: int = KV_LAYOUT_PAGED,
        paged_output: bool = False,
        ctas_per_group: int = 1,
        num_sms: int = 0,
        merge_threshold: int = _LAST_CTA_MERGE_MAX,
        k_quant_page_stride: int = _PAGE_SIZE * _INDEX_HEAD_DIM,
        k_scales_row_stride: int = _PAGE_SIZE,
        max_seq_capacity: int = 1 << 30,
        vectorized_q_load: bool = False,
        q_row_stride_bytes: int = 0,
        mxfp4: bool = False,
        mxfp4_small_output: bool = False,
        mxfp4_warp_weights: bool = False,
        mxfp4_aligned_q: bool = False,
        candidates: bool = False,
        page_size: int = 64,
        score_width: int = 0,
        ordered_output: bool = False,
        output_position_capacity: int = 1 << 31,
        state_reset_pdl: bool = False,
        vector_output_pairs: bool = False,
        tile_schedule: str = "contiguous",
        grid_layout: str = "flat",
        mxfp4_query_offset_mode: int | None = None,
        mxfp4_unsigned_candidate_pages: bool | None = None,
        mxfp4_merge_bits: int = 8,
        mxfp4_histogram_count_bits: int = 16,
        mxfp4_histogram_bitmap: bool = False,
        mxfp4_serial_state_words: int = _COOP_STATE_WORDS,
        mxfp4_partition_count: int = 1,
        mxfp4_score_batch_tiles: int | None = None,
        mxfp4_cutoff_strategy: str | None = None,
        mxfp4_threads: int = 1024,
        mxfp4_score_passes: int = 1,
        mxfp4_coop_residency: int = 1,
        mxfp4_compact_shared: bool = False,
    ):
        self.constructor_config = tuple(
            (name, value) for name, value in locals().items() if name != "self"
        )
        if tile_schedule not in ("contiguous", "cyclic"):
            raise ValueError("tile_schedule must be contiguous or cyclic")
        if grid_layout not in ("flat", "row_major", "slice_major"):
            raise ValueError("grid_layout must be flat, row_major, or slice_major")
        self.tile_schedule = tile_schedule
        self.grid_layout = grid_layout
        self.num_heads_static = int(num_heads_static)
        self.topk = int(topk)
        self.kv_layout = int(kv_layout)
        self.ctas_per_group = max(1, int(ctas_per_group))
        self.merge_threshold = int(merge_threshold)
        self.max_seq_capacity = max(0, int(max_seq_capacity))
        self.vectorized_q_load = bool(vectorized_q_load)
        self.q_row_stride_bytes = int(q_row_stride_bytes)
        self.mxfp4 = bool(mxfp4)
        self.mxfp4_small_output = bool(mxfp4_small_output)
        if self.mxfp4_small_output and (
            not self.mxfp4 or int(score_width) > self.topk or self.ctas_per_group != 1
        ):
            raise ValueError(
                "small MXFP4 output requires one CTA and capacity at most topk"
            )
        self.mxfp4_warp_weights = bool(mxfp4 and mxfp4_warp_weights)
        self.mxfp4_aligned_q = bool(mxfp4 and mxfp4_aligned_q)
        self.radix_threads = int(mxfp4_threads) if self.mxfp4 else _RADIX_THREADS
        self.mxfp4_score_passes = int(mxfp4_score_passes) if self.mxfp4 else 1
        self.coop_resident_blocks = int(mxfp4_coop_residency) if self.mxfp4 else 1
        self.mxfp4_compact_shared = self.mxfp4 and bool(mxfp4_compact_shared)
        if self.mxfp4:
            if self.radix_threads not in (512, 1024):
                raise ValueError("MXFP4 block size must be 512 or 1024 threads")
            if self.mxfp4_score_passes < 1:
                raise ValueError("MXFP4 score passes must be positive")
            if self.coop_resident_blocks not in (1, 2):
                raise ValueError("MXFP4 residency must be one or two blocks")
            if self.coop_resident_blocks == 2 and self.radix_threads != 512:
                raise ValueError("two-block MXFP4 residency requires 512 threads")
        self.mxfp4_chunk = self.radix_threads // 2
        self.mxfp4_score_group = self.mxfp4_chunk * self.mxfp4_score_passes
        self.candidates = bool(candidates)
        self.page_size = int(page_size)
        self.score_width = int(score_width)
        self.ordered_output = bool(ordered_output)
        self.output_position_capacity = int(output_position_capacity)
        self.state_reset_pdl = bool(state_reset_pdl)
        self.vector_output_pairs = bool(vector_output_pairs)
        self.mxfp4_merge_bits = int(mxfp4_merge_bits)
        self.mxfp4_histogram_count_bits = int(mxfp4_histogram_count_bits)
        self.mxfp4_histogram_bitmap = bool(mxfp4_histogram_bitmap)
        self.mxfp4_serial_state_words = int(mxfp4_serial_state_words)
        self.mxfp4_query_offset_mode = 0
        self.mxfp4_unsigned_candidate_pages = False
        if self.mxfp4:
            self.mxfp4_query_offset_mode = (
                (1 if self.candidates and self.score_width > self.topk else 0)
                if mxfp4_query_offset_mode is None
                else int(mxfp4_query_offset_mode)
            )
            if self.mxfp4_query_offset_mode not in (0, 1, 2):
                raise ValueError("MXFP4 query offset mode must be 0, 1, or 2")
            self.mxfp4_unsigned_candidate_pages = (
                False
                if mxfp4_unsigned_candidate_pages is None
                else bool(mxfp4_unsigned_candidate_pages)
            )
            from .mxfp4 import _TensorCorePagedScore

            if (
                self.kv_layout != KV_LAYOUT_PAGED
                or bool(paged_output)
                or self.topk != 512
            ):
                raise ValueError("fused MXFP4 requires paged logical top-512 output")
            self.vectorized_q_load = False
            self.mxfp4_scorer = _TensorCorePagedScore(
                self.num_heads_static, self.candidates, self.page_size
            )
        # dim-0 byte stride of k_quant_bytes for the PAGED wide load. Defaults to
        # the contiguous 8192 (64*128); the packed paged cache interleaves per-page
        # scales so its real page stride is 8448 -- run_fused_paged_indexer passes
        # k_quant_bytes.stride(0) so the load reads the correct page.
        self.k_quant_page_stride = int(k_quant_page_stride)
        # dim-0 ELEMENT stride of the (pages, 64) f32 scales view: the packed
        # cache interleaves per-page scales, while standalone scale tensors are
        # normally contiguous.  Keep this independent from the quant-page byte
        # stride so both layouts obey their actual tensor contracts.
        self.k_scales_row_stride = int(k_scales_row_stride)
        if self.kv_layout not in (KV_LAYOUT_CONTIGUOUS_MLA, KV_LAYOUT_PAGED):
            raise ValueError(f"bad kv_layout {self.kv_layout}")
        self.paged_output = bool(paged_output)
        if self.paged_output and self.kv_layout != KV_LAYOUT_PAGED:
            raise ValueError("physical-slot output requires the paged K/V layout")
        if self.topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"fused indexer supports topk {_SUPPORTED_TOPK}, got {self.topk}"
            )
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"fused indexer supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        # All-warp scoring: spread the page's 64 tokens across as many warps as the
        # head-tiling leaves free, scoring the whole page in ONE pass instead of the
        # 4-warp / page_splits-deep sequential loop. token_groups covers the page
        # (8 tokens/group => 8 groups) bounded by warps available after head-tiling.
        # heads=16 (tiles=1): 8 warps; heads=32 (tiles=2): 16; heads=64 (tiles=4): 32.
        total_warps = self.radix_threads // _WARP_THREADS
        self.token_groups = min(
            _PAGE_SIZE // _PAGED_TOKENS_PER_GROUP,  # 8 => one pass over the page
            total_warps // self.num_q_head_tiles,  # warps free after head-tiling
        )
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        # Warp-owns-tokens direct-L2 score: no K staging, no page barriers.
        # Gated to the 4-head-tile paged decode contract (DSV4 64-head) at
        # rows >= 3 (ctas_per_group <= 16): those rows re-read an L2-resident
        # K where 16B fragment loads are free. rows 1-2 stream mostly-cold K
        # from DRAM, where the staged pipeline's contiguous 128B page loads
        # win (measured +25% at rows=1, +3.5% at rows=2); they keep the
        # historical path.
        self.direct_k_score = (
            _DIRECT_K_SCORE_ENABLED
            and self.kv_layout == KV_LAYOUT_PAGED
            and self.num_q_head_tiles == 4
            and self.ctas_per_group <= 16
        )
        if self.mxfp4:
            self.direct_k_score = True
        self.score_warps = self.token_groups * self.num_q_head_tiles
        self.score_threads = self.score_warps * _WARP_THREADS
        # Over-sized accumulator: scored tokens append until topk+slack, then
        # trim.  Keep the large measured slack specific to the paged DSV4
        # contract so top-k-2048 and contiguous contracts retain their existing
        # shared-memory residency.
        batch_slack = _resolve_fused_batch_slack(
            kv_layout=self.kv_layout,
            num_heads=self.num_heads_static,
            topk=self.topk,
        )
        if self.mxfp4:
            batch_slack = 1536
        self.carry_cap = int(self.topk) + int(batch_slack)
        self.mxfp4_carry_slots = (
            self.carry_cap + self.radix_threads - 1
        ) // self.radix_threads
        if self.mxfp4 and self.mxfp4_score_group > self.carry_cap - self.topk:
            raise ValueError("MXFP4 score group exceeds carry append capacity")
        # MXFP4 trims rescan the carry rather than buffer pivot candidates.
        # Other encodings retain enough candidate slots for a full carry.
        self.cands = 512 if self.mxfp4 else self.carry_cap
        self.selection_groups = int(mxfp4_partition_count)
        if self.selection_groups != 1 and self.tile_schedule != "contiguous":
            raise ValueError("private MXFP4 carries require contiguous tile scheduling")
        if self.selection_groups != 1 and self.mxfp4_compact_shared:
            raise ValueError("private MXFP4 carries require their full shared layout")
        if self.selection_groups != 1 and self.mxfp4_small_output:
            raise ValueError("small MXFP4 output requires one carry partition")
        if self.selection_groups != 1 and self.mxfp4_score_passes != 1:
            raise ValueError("private MXFP4 carries use score batch tiles")
        if self.selection_groups not in (1, 2, 4):
            raise ValueError("MXFP4 partition count must be 1, 2, or 4")
        if self.selection_groups != 1 and (
            not self.mxfp4 or self.num_heads_static != 32 or self.radix_threads != 1024
        ):
            raise ValueError("private MXFP4 carries require 32 heads and 1024 threads")
        self.mxfp4_score_batch_tiles = (
            (1 if self.selection_groups == 1 else 2)
            if mxfp4_score_batch_tiles is None
            else int(mxfp4_score_batch_tiles)
        )
        if self.mxfp4_score_batch_tiles not in (1, 2) or (
            self.selection_groups == 1 and self.mxfp4_score_batch_tiles != 1
        ):
            raise ValueError("paired score tiles require private MXFP4 carries")
        max_cta_span = (
            (self.score_width + _PAGE_SIZE * self.ctas_per_group - 1)
            // (_PAGE_SIZE * self.ctas_per_group)
        ) * _PAGE_SIZE
        self.mxfp4_cutoff_strategy = mxfp4_cutoff_strategy
        if self.mxfp4_cutoff_strategy is None:
            self.mxfp4_cutoff_strategy = (
                "shared_compact"
                if self.selection_groups == 4 and max_cta_span >= 32768
                else "local"
            )
        if self.mxfp4_cutoff_strategy not in ("local", "shared", "shared_compact") or (
            self.selection_groups != 4 and self.mxfp4_cutoff_strategy != "local"
        ):
            raise ValueError("shared cutoff strategies require four MXFP4 partitions")
        self.mxfp4_share_cutoff = self.mxfp4_cutoff_strategy != "local"
        if self.selection_groups == 2:
            self.carry_cap = 2 * (1024 if self.mxfp4_score_batch_tiles == 1 else 1536)
            self.cands = 2048
        elif self.selection_groups == 4:
            self.carry_cap = 4 * (896 if self.mxfp4_score_batch_tiles == 1 else 1024)
            self.cands = 1
        if (
            self.selection_groups != 1
            and self._get_shared_storage_cls().size_in_bytes() > 101376
        ):
            raise ValueError(
                "private MXFP4 carries exceed the SM120 shared-memory limit"
            )
        # In-kernel cross-CTA merge (ctas_per_group > 1): each CTA atomically
        # packs its REAL local candidates into a per-group global slab of
        # capacity ctas_per_group * topk; the last-arriving CTA radix-selects the
        # final top-k over the packed reals (no host merge launch).
        self.merge_in_kernel = self.ctas_per_group > 1
        # Cooperative groups need a proven resident grid. Preparation checks
        # the resolved launcher's occupancy before using this budget.
        self.num_sms = int(num_sms)
        self.coop_co_resident = (
            self.num_sms <= 0
            or self.ctas_per_group <= self.num_sms * self.coop_resident_blocks
        )
        self.coop_merge_possible = (
            self.merge_in_kernel
            and self.max_seq_capacity > self.merge_threshold
            and self.coop_co_resident
        )
        self.pack_cap = self.ctas_per_group * int(self.topk)
        # Compact counters require a statically serial group. One-pass histograms
        # use their own arena for every live length, including empty groups.
        if self.mxfp4:
            if self.mxfp4_merge_bits not in (8, 16):
                raise ValueError("MXFP4 cooperative radix width must be 8 or 16")
            if self.mxfp4_serial_state_words not in (2, _COOP_STATE_WORDS):
                raise ValueError(
                    "MXFP4 serial state requires 2 or 8968 words per group"
                )
            if self.coop_merge_possible and self.mxfp4_merge_bits == 16:
                if self.mxfp4_histogram_count_bits not in (16, 32):
                    raise ValueError("BF16 histogram counts must use 16 or 32 bits")
                if (
                    self.mxfp4_histogram_bitmap
                    and self.mxfp4_histogram_count_bits != 16
                ):
                    raise ValueError(
                        "bitmap cleanup is defined for packed 16-bit counts"
                    )
                if self.mxfp4_histogram_count_bits == 16 and self.pack_cap > 8192:
                    raise ValueError(
                        "packed BF16 histograms require at most 8192 published candidates"
                    )
                if self.pack_cap >= 2**31:
                    raise ValueError("BF16 histogram candidate totals must fit int32")
        self.mxfp4_histogram_fine_words = (
            65536 if self.mxfp4_histogram_count_bits == 32 else 32768
        )
        self.mxfp4_histogram_active_cap = min(
            self.mxfp4_histogram_fine_words, self.pack_cap
        )
        self.mxfp4_histogram_state_words = (
            self.mxfp4_histogram_fine_words
            + _RADIX
            + 8
            + (
                self.mxfp4_histogram_fine_words // 32
                if self.mxfp4_histogram_bitmap
                else self.mxfp4_histogram_active_cap
            )
        )
        self.state_words_per_group = (
            2
            if not self.merge_in_kernel
            else self.mxfp4_serial_state_words
            if self.mxfp4 and not self.coop_merge_possible
            else self.mxfp4_histogram_state_words
            if self.mxfp4 and self.mxfp4_merge_bits == 16
            else _COOP_STATE_WORDS
        )

    def merge_scratch_elements(self, num_groups: int) -> tuple[int, int]:
        """Return pack-slab and int32-state element counts for caller-owned storage."""
        if int(num_groups) <= 0:
            raise ValueError("merge groups must be positive")
        if not self.merge_in_kernel:
            return 1, 2
        return int(num_groups) * self.pack_cap, int(
            num_groups
        ) * self.state_words_per_group

    def _get_shared_storage_cls(self):
        return _fused_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
            self.topk,
            self.carry_cap,
            self.cands,
            self.selection_groups,
            self.mxfp4_score_batch_tiles,
            compact=self.mxfp4_compact_shared,
            preserve_histogram=self.mxfp4_compact_shared and self.coop_merge_possible,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        pack_values: cute.Tensor,
        pack_indices: cute.Tensor,
        merge_state: cute.Tensor,
        stream: cuda.CUstream,
        mxfp4_operands: tuple = (),
    ):
        if cutlass.const_expr(self.mxfp4_small_output):
            if cutlass.const_expr(self.score_width == self.topk):
                self.register_all_keys_kernel(
                    q_bytes,
                    weights,
                    k_quant_bytes,
                    real_page_table,
                    seqlens_per_query,
                    out_indices,
                    out_values,
                    merge_state,
                    mxfp4_operands,
                ).launch(
                    grid=(q_bytes.shape[0], 1, 1), block=(512, 1, 1), stream=stream
                )
            else:
                self.ordered_all_keys_kernel(
                    q_bytes,
                    weights,
                    k_quant_bytes,
                    real_page_table,
                    seqlens_per_query,
                    out_indices,
                    out_values,
                    merge_state,
                    mxfp4_operands,
                ).launch(
                    grid=(q_bytes.shape[0], 1, 1), block=(1024, 1, 1), stream=stream
                )
        else:
            grid = (q_bytes.shape[0] * self.ctas_per_group, 1, 1)
            if cutlass.const_expr(self.mxfp4 and self.grid_layout == "row_major"):
                grid = (self.ctas_per_group, q_bytes.shape[0], 1)
            elif cutlass.const_expr(self.mxfp4 and self.grid_layout == "slice_major"):
                grid = (q_bytes.shape[0], self.ctas_per_group, 1)
            self.kernel(
                q_bytes,
                weights,
                k_quant_bytes,
                k_scales,
                real_page_table,
                seqlens_per_query,
                k_start,
                k_end,
                out_indices,
                out_values,
                pack_values,
                pack_indices,
                merge_state,
                mxfp4_operands,
            ).launch(
                grid=grid,
                block=[self.radix_threads, 1, 1],
                # The 1024-thread radix CTA is architecturally limited to one
                # resident block on SM120 (1536 threads/SM).  The cooperative
                # planner also caps the default launch to one machine-wide wave.
                min_blocks_per_mp=self.coop_resident_blocks,
                stream=stream,
                use_pdl=self.state_reset_pdl,
            )

    @cute.kernel
    def ordered_all_keys_kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        merge_state: cute.Tensor,
        mxfp4_operands: tuple,
    ):
        """Score and order keys within a single prepared top-k CTA."""
        tx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if (row == Int32(0)) & (tx == Int32(0)):
            merge_state[Int64(0)] = Int32(0)
            merge_state[Int64(1)] = Int32(0)
        smem = cutlass.utils.SmemAllocator()
        ids = smem.allocate_tensor(Int32, cute.make_layout((512,)), byte_alignment=16)
        vals = smem.allocate_tensor(
            Float32, cute.make_layout((512,)), byte_alignment=16
        )
        page_row_stride = Int64(real_page_table.stride[0])
        if real_page_table.shape[0] == Int32(1):
            page_row_stride = Int64(0)
        q_flat = cute.make_tensor(
            q_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        qs_flat = cute.make_tensor(
            mxfp4_operands[0].iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        weights_flat = cute.make_tensor(
            weights.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        pool_flat = cute.make_tensor(
            k_quant_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        pages_flat = cute.make_tensor(
            real_page_table.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        candidates_flat = cute.make_tensor(
            mxfp4_operands[2].iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        if tx < Int32(512):
            ids[tx] = Int32(-1)
            vals[tx] = Float32(-float("inf"))
        cute.arch.sync_threads()
        tile_start = Int32(0)
        while tile_start < Int32(self.topk):
            if tile_start < Int32(mxfp4_operands[4]):
                self.mxfp4_scorer._score_tile_fused(
                    q_flat,
                    qs_flat,
                    weights_flat,
                    pool_flat,
                    pages_flat,
                    seqlens_per_query,
                    mxfp4_operands[1],
                    candidates_flat,
                    mxfp4_operands[3],
                    vals,
                    Int32(q_bytes.shape[0]),
                    Int32(mxfp4_operands[4]),
                    Int32(real_page_table.shape[1]),
                    page_row_stride,
                    Int64(k_quant_bytes.stride[0]),
                    Int64(k_quant_bytes.shape[0]),
                    tile_start,
                    row_index=Int32(row),
                    logical_indices=ids,
                    local_output=True,
                    output_base=tile_start,
                    output_width=self.topk,
                    aligned_q=self.mxfp4_aligned_q,
                )
            elif tx < Int32(512):
                column = Int64(tile_start) + Int64(tx)
                ids[column] = Int32(-1)
                vals[column] = Float32(-float("inf"))
            tile_start += Int32(512)

        _mxfp4_order_small_output(
            out_indices, out_values, Int32(row), ids, vals, Int32(tx), Int32(512)
        )

    @cute.kernel
    def register_all_keys_kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        merge_state: cute.Tensor,
        mxfp4_operands: tuple,
    ):
        """Feed warp-remapped score registers directly into logical ordering."""
        tx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if (row == Int32(0)) & (tx == Int32(0)):
            merge_state[Int64(0)] = Int32(0)
            merge_state[Int64(1)] = Int32(0)
        smem = cutlass.utils.SmemAllocator()
        ids = smem.allocate_tensor(Int32, cute.make_layout((512,)), byte_alignment=16)
        vals = smem.allocate_tensor(
            Float32, cute.make_layout((512,)), byte_alignment=16
        )
        page_row_stride = Int64(real_page_table.stride[0])
        if real_page_table.shape[0] == Int32(1):
            page_row_stride = Int64(0)
        q_flat = cute.make_tensor(
            q_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        qs_flat = cute.make_tensor(
            mxfp4_operands[0].iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        weights_flat = cute.make_tensor(
            weights.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        pool_flat = cute.make_tensor(
            k_quant_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        pages_flat = cute.make_tensor(
            real_page_table.iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        candidates_flat = cute.make_tensor(
            mxfp4_operands[2].iterator, cute.make_layout((Int64(1) << Int64(40),))
        )
        a0, a1, b0, b1, ag0, ag1, bg0, bg1 = (
            self.mxfp4_scorer._score_tiles_direct_register(
                q_flat,
                qs_flat,
                weights_flat,
                pool_flat,
                pages_flat,
                seqlens_per_query,
                mxfp4_operands[1],
                candidates_flat,
                mxfp4_operands[3],
                Int32(mxfp4_operands[4]),
                Int32(real_page_table.shape[1]),
                page_row_stride,
                Int64(k_quant_bytes.stride[0]),
                Int64(k_quant_bytes.shape[0]),
                Int32(0),
                Int32(row),
                aligned_q=self.mxfp4_aligned_q,
            )
        )
        lane = Int32(tx) % Int32(32)
        source_lane = (lane % Int32(8)) * Int32(4)
        a0 = cute.arch.shuffle_sync(a0, source_lane)
        a1 = cute.arch.shuffle_sync(a1, source_lane)
        ag0 = cute.arch.shuffle_sync(ag0, source_lane)
        ag1 = cute.arch.shuffle_sync(ag1, source_lane)
        b0 = cute.arch.shuffle_sync(b0, source_lane)
        b1 = cute.arch.shuffle_sync(b1, source_lane)
        bg0 = cute.arch.shuffle_sync(bg0, source_lane)
        bg1 = cute.arch.shuffle_sync(bg1, source_lane)
        first_part = lane % Int32(16) < Int32(8)
        a_value = a0 if first_part else a1
        b_value = b0 if first_part else b1
        a_index = ag0 if first_part else ag1
        b_index = bg0 if first_part else bg1
        value = a_value if lane < Int32(16) else b_value
        logical_id = a_index if lane < Int32(16) else b_index
        _mxfp4_order_small_output(
            out_indices,
            out_values,
            Int32(row),
            ids,
            vals,
            Int32(tx),
            Int32(512),
            register_input=(logical_id, value),
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        out_indices: cute.Tensor,
        out_values: cute.Tensor,
        pack_values: cute.Tensor,
        pack_indices: cute.Tensor,
        merge_state: cute.Tensor,
        mxfp4_operands: tuple = (),
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, bid_y, _ = cute.arch.block_idx()
        bid = Int32(bid)
        # Relay: ctas_per_group CTAs cooperate on one row (group). Each owns a
        # page-slice and emits a LOCAL top-k into candidate[cta_in_group, group, :];
        # the host merges the per-CTA partials. ctas_per_group==1 => single-CTA.
        ctas_pg = Int32(self.ctas_per_group)
        if cutlass.const_expr(self.mxfp4 and self.grid_layout == "slice_major"):
            group_id = bid
            cta_in_group = Int32(bid_y)
        elif cutlass.const_expr(self.mxfp4 and self.grid_layout == "row_major"):
            group_id = Int32(bid_y)
            cta_in_group = bid
        else:
            group_id = bid // ctas_pg
            cta_in_group = bid - group_id * ctas_pg
        q_idx = group_id
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        topk_static = Int32(self.topk)
        num_heads = Int32(self.num_heads_static)
        # All-warp scoring: warps [0, score_warps) run the MMA + reduce (the page
        # is covered in one pass). K-load / repack stay at _PAGED_THREADS_PER_CTA.
        score_threads = Int32(self.score_threads)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())

        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        q_smem_base_addr = shared_ptr_to_u32(storage.q_bytes.data_ptr())
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        k_page3_base_addr = shared_ptr_to_u32(storage.k_page3.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(
            cute.make_layout((_PAGE_SIZE,), stride=(1,))
        )
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_partial_logits2 = storage.partial_logits2.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_c0_values = storage.carry0_values.get_tensor(
            cute.make_layout((self.carry_cap,), stride=(1,))
        )
        s_c0_gindex = storage.carry0_gindex.get_tensor(
            cute.make_layout((self.carry_cap,), stride=(1,))
        )
        s_c1_values = storage.carry1_values.get_tensor(
            cute.make_layout((self.topk * self.selection_groups,), stride=(1,))
        )
        s_c1_gindex = storage.carry1_gindex.get_tensor(
            cute.make_layout((self.topk * self.selection_groups,), stride=(1,))
        )
        s_hist0 = storage.hist0.get_tensor(
            cute.make_layout((384 * self.selection_groups,), stride=(1,))
        )
        s_hist1 = storage.hist1.get_tensor(
            cute.make_layout((384 * self.selection_groups,), stride=(1,))
        )
        s_out = storage.out_idx.get_tensor(
            cute.make_layout((self.topk * self.selection_groups,), stride=(1,))
        )
        s_cand0 = storage.cand0.get_tensor(cute.make_layout((self.cands,), stride=(1,)))
        s_cand1 = storage.cand1.get_tensor(cute.make_layout((self.cands,), stride=(1,)))
        s_relay = storage.relay.get_tensor(
            cute.make_layout((max(2, self.selection_groups),), stride=(1,))
        )
        s_coop_suffix = storage.coop_suffix.get_tensor(
            cute.make_layout((_RADIX,), stride=(1,))
        )
        s_coop_scalars = storage.coop_scalars.get_tensor(
            cute.make_layout((8,), stride=(1,))
        )
        s_coop_counters = storage.coop_counters.get_tensor(
            cute.make_layout((2,), stride=(1,))
        )
        # coop local histogram reuses hist0 (>=256); its u32 base address for atomics:
        coop_hist_addr = shared_ptr_to_u32(storage.hist0.data_ptr())
        coop_ctr_addr = shared_ptr_to_u32(storage.coop_counters.data_ptr())
        h0 = shared_ptr_to_u32(storage.hist0.data_ptr())
        ctr = shared_ptr_to_u32(storage.counter.data_ptr())
        thr = shared_ptr_to_u32(storage.thr_id.data_ptr())
        ni0 = shared_ptr_to_u32(storage.ni0.data_ptr())
        ni1 = shared_ptr_to_u32(storage.ni1.data_ptr())
        lr = shared_ptr_to_u32(storage.last_rem.data_ptr())

        # Q vectors and head weights do not depend on the row length, so their
        # loads are issued before the seq_len -> page-table -> page-fetch
        # chain and complete while it resolves; they are stored to shared
        # memory once the first pages are in flight. Thread ``t`` owns Q
        # vector ``t`` (16 bytes of one head) in the ldmatrix-permuted
        # per-head-tile layout and weight ``t``.
        q_v0 = Uint32(0)
        q_v1 = Uint32(0)
        q_v2 = Uint32(0)
        q_v3 = Uint32(0)
        q_dst_addr = Int32(0)
        q_owner = Int32(0)
        if cutlass.const_expr(self.vectorized_q_load):
            if tx < Int32(self.padded_q_heads * (_INDEX_HEAD_DIM // 16)):
                q_owner = Int32(1)
                q_head = tx // Int32(_INDEX_HEAD_DIM // 16)
                q_vec = tx - q_head * Int32(_INDEX_HEAD_DIM // 16)
                if q_head < num_heads:
                    q_v0, q_v1, q_v2, q_v3 = ld_global_v4_u32(
                        get_ptr_as_int64(
                            q_bytes,
                            Int64(q_idx) * Int64(self.q_row_stride_bytes)
                            + Int64(q_head) * Int64(_INDEX_HEAD_DIM)
                            + Int64(q_vec) * Int64(16),
                        )
                    )
                q_tile = q_head // Int32(_PAGED_Q_HEAD_TILE)
                q_dst_addr = _smem_addr_from_b128_offset(
                    q_smem_base_addr
                    + q_tile * Int32(_PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM),
                    _permuted_offset_128b(
                        q_head - q_tile * Int32(_PAGED_Q_HEAD_TILE),
                        q_vec,
                        Int32(_INDEX_HEAD_DIM // 16),
                    ),
                )
        if cutlass.const_expr(not self.mxfp4_compact_shared):
            w_val = Float32(0.0)
            if tx < Int32(self.padded_q_heads):
                if tx < num_heads:
                    w_val = Float32(weights[q_idx, tx])

        # PAGED: valid k = [0, seqlen) via page table. CONTIGUOUS_MLA: contiguous k =
        # [k_start, k_end) with abs_start = k_start (no page table).
        abs_start = Int32(0)
        if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
            seq_len = Int32(seqlens_per_query[q_idx])
        else:
            abs_start = Int32(k_start[q_idx])
            seq_len = Int32(k_end[q_idx]) - abs_start
        mxfp4_score_width = Int32(0)
        if cutlass.const_expr(self.mxfp4):
            mxfp4_score_width = Int32(mxfp4_operands[4])
            seq_len = cutlass.min(seq_len, mxfp4_operands[1][0])
            if cutlass.const_expr(self.candidates):
                seq_len = cutlass.min(mxfp4_score_width, mxfp4_operands[3][q_idx])
                if seqlens_per_query[q_idx] <= Int32(0) or mxfp4_operands[1][
                    0
                ] <= Int32(0):
                    seq_len = Int32(0)
            else:
                seq_len = cutlass.min(seq_len, mxfp4_score_width)
        if seq_len < Int32(0):
            seq_len = Int32(0)
        total_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
        if cutlass.const_expr(self.mxfp4):
            total_pages = Int32(
                (Int64(seq_len) + Int64(_PAGE_SIZE - 1)) // Int64(_PAGE_SIZE)
            )
        # This CTA's page-slice of the row (balanced split across ctas_pg CTAs).
        pages_per_cta = (total_pages + ctas_pg - Int32(1)) // ctas_pg
        page_start = cta_in_group * pages_per_cta
        page_end = page_start + pages_per_cta
        if page_end > total_pages:
            page_end = total_pages
        if page_start > total_pages:
            page_start = total_pages

        # Page ids of the slice are read two pages ahead of their use (pid_cur
        # and pid_p1 hold pages p and p+1 at the top of iteration p) so the
        # page-table load never sits between a landed page and the next fetch
        # issue. The first page is issued before Q staging so its fetch
        # overlaps the Q and weight staging.
        scales_base_addr = shared_ptr_to_u32(storage.scales.data_ptr())
        pid_cur = Int32(-1)
        pid_p1 = Int32(-1)
        pid_p2 = Int32(-1)
        if cutlass.const_expr(
            self.kv_layout == KV_LAYOUT_PAGED and not self.direct_k_score
        ):
            pid_cur = _page_id_or_invalid(real_page_table, q_idx, page_start, page_end)
            pid_p1 = _page_id_or_invalid(
                real_page_table, q_idx, page_start + Int32(1), page_end
            )
            if pid_cur >= Int32(0):
                _stream_issue_k_page_cp_async(
                    k_quant_bytes,
                    k_scales,
                    pid_cur,
                    k_page_perm_base_addr,
                    scales_base_addr,
                    Int32(0),
                    tx,
                    k_quant_page_stride=int(self.k_quant_page_stride),
                    k_scales_row_stride=int(self.k_scales_row_stride),
                    issue_threads=self.radix_threads,
                )
            cute.arch.cp_async_commit_group()

        # Stage q + weights. Q lands straight in the ldmatrix-permuted
        # per-head-tile layout so the score core consumes raw fragments on
        # BOTH operands (no byte packs, no 16b->8b fragment swizzles); heads
        # past num_heads stage zeros.
        if cutlass.const_expr(self.vectorized_q_load):
            if q_owner != Int32(0):
                st_shared_v4_u32(q_dst_addr, q_v0, q_v1, q_v2, q_v3)
        elif cutlass.const_expr(not self.mxfp4):
            # Preserve arbitrary tensor head/row strides.  Each thread moves
            # individual bytes directly into the same XOR-permuted granules as
            # the vectorized loader; this is the correctness fallback for views
            # whose 128-byte head rows are not 16-byte vector-load compatible.
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                vec_idx = col_idx // Int32(16)
                byte_idx = col_idx - vec_idx * Int32(16)
                tile_idx = head_idx // Int32(_PAGED_Q_HEAD_TILE)
                row_in_tile = head_idx - tile_idx * Int32(_PAGED_Q_HEAD_TILE)
                tile_base = q_smem_base_addr + tile_idx * Int32(
                    _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
                )
                dst_addr = (
                    _smem_addr_from_b128_offset(
                        tile_base,
                        _permuted_offset_128b(
                            row_in_tile, vec_idx, Int32(_INDEX_HEAD_DIM // 16)
                        ),
                    )
                    + byte_idx
                )
                q_byte = cutlass.Uint8(0)
                if head_idx < num_heads:
                    q_byte = q_bytes[q_idx, head_idx, col_idx]
                st_shared_u8(dst_addr, q_byte)
                q_linear += Int32(self.radix_threads)
        if cutlass.const_expr(not self.mxfp4_compact_shared):
            if tx < Int32(self.padded_q_heads):
                s_w[tx] = w_val
            cute.arch.sync_threads()

        head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
        token_group = warp_idx // Int32(self.num_q_head_tiles)

        if cutlass.const_expr(self.mxfp4 and self.selection_groups == 4):
            if cutlass.const_expr(self.mxfp4_share_cutoff):
                if tx == Int32(0):
                    _smem_st(ni0, Int32(0), Int32(0))
                cute.arch.sync_threads()
            filter_key = Uint32(0)
            partition = tx // Int32(256)
            partition_thread = tx % Int32(256)
            partition_warp = partition_thread // Int32(32)
            partition_barrier = partition + Int32(1)
            carry_count = Int32(0)
            cutoff = Float32(-float("inf"))
            span_first = Int32(
                cutlass.min(Int64(page_start) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            span_end = Int32(
                cutlass.min(Int64(page_end) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            tile_start = span_first
            q_flat = cute.make_tensor(
                q_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            qs_flat = cute.make_tensor(
                mxfp4_operands[0].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            weights_flat = cute.make_tensor(
                weights.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            if cutlass.const_expr(self.mxfp4_warp_weights):
                weight_pairs = self.mxfp4_scorer._load_weight_pairs(weights, q_idx)
            else:
                weight_pairs = None
            mxfp4_page_row_stride = Int64(real_page_table.stride[0])
            if real_page_table.shape[0] == Int32(1):
                mxfp4_page_row_stride = Int64(0)
            pool_flat = cute.make_tensor(
                k_quant_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            pages_flat = cute.make_tensor(
                real_page_table.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            candidates_flat = cute.make_tensor(
                mxfp4_operands[2].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            p_values = cute.make_tensor(
                s_c0_values.iterator + partition * Int32(self.carry_cap // 4),
                cute.make_layout((self.carry_cap // 4,), stride=(1,)),
            )
            p_indices = cute.make_tensor(
                s_c0_gindex.iterator + partition * Int32(self.carry_cap // 4),
                cute.make_layout((self.carry_cap // 4,), stride=(1,)),
            )
            p_selected_values = cute.make_tensor(
                s_c1_values.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_selected_indices = cute.make_tensor(
                s_c1_gindex.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_hist0 = cute.make_tensor(
                s_hist0.iterator + partition * Int32(384),
                cute.make_layout((384,), stride=(1,)),
            )
            p_hist1 = cute.make_tensor(
                s_hist1.iterator + partition * Int32(384),
                cute.make_layout((384,), stride=(1,)),
            )
            p_out = cute.make_tensor(
                s_out.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_hist_address = h0 + partition * Int32(1536)
            p_counter = ctr + partition * Int32(4)
            p_threshold = thr + partition * Int32(4)
            p_remaining = lr + partition * Int32(4)
            p_cached_hist_address = (
                shared_ptr_to_u32(storage.hist1.data_ptr())
                + partition * Int32(1536)
                + Int32(512)
            )
            while tile_start < span_end:
                count = cutlass.min(
                    Int32(128),
                    cutlass.max(
                        span_end - tile_start - partition * Int32(128), Int32(0)
                    ),
                )
                if cutlass.const_expr(self.mxfp4_score_batch_tiles == 2):
                    count += cutlass.min(
                        Int32(128),
                        cutlass.max(
                            span_end - tile_start - Int32(512) - partition * Int32(128),
                            Int32(0),
                        ),
                    )
                if carry_count + Int32(128 * self.mxfp4_score_batch_tiles) > Int32(
                    self.carry_cap // 4
                ):
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    _mxfp4_private_radix_select(
                        p_values,
                        p_indices,
                        p_selected_values,
                        p_selected_indices,
                        carry_count,
                        topk_static,
                        partition_thread,
                        p_hist0,
                        p_hist1,
                        p_out,
                        p_hist_address,
                        p_counter,
                        p_threshold,
                        p_remaining,
                        selection_threads=256,
                        barrier_id=partition_barrier,
                        maintain_histogram=True,
                        cached_histogram_valid=(cutoff != Float32(-float("inf"))),
                    )
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    i = Int32(partition_thread)
                    while i < topk_static:
                        p_values[i] = Float32(p_selected_values[i])
                        p_indices[i] = Int32(p_selected_indices[i])
                        i += Int32(256)
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    carry_count = topk_static
                    value = Float32(float("inf"))
                    i = Int32(partition_thread)
                    while i < topk_static:
                        value = cutlass.min(value, Float32(p_values[i]))
                        i += Int32(256)
                    for delta in (16, 8, 4, 2, 1):
                        value = cutlass.min(
                            value, cute.arch.shuffle_sync_bfly(value, offset=delta)
                        )
                    if lane == Int32(0):
                        s_scale[partition * Int32(16) + partition_warp] = value
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    if partition_warp == Int32(0):
                        value = Float32(float("inf"))
                        if lane < Int32(8):
                            value = Float32(s_scale[partition * Int32(16) + lane])
                        for delta in (16, 8, 4, 2, 1):
                            value = cutlass.min(
                                value, cute.arch.shuffle_sync_bfly(value, offset=delta)
                            )
                        if lane == Int32(0):
                            s_scale[partition * Int32(16) + Int32(8)] = value
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    cutoff = Float32(s_scale[partition * Int32(16) + Int32(8)])
                    if cutlass.const_expr(self.mxfp4_share_cutoff):
                        if partition_thread == Int32(0):
                            key = _convert_to_uint32(cutoff)
                            if cutoff == Float32(0):
                                key = Uint32(0x7FFFFFFF)
                            _mxfp4_publish_cutoff(ni0, key)
                            _smem_st(
                                p_threshold,
                                Int32(0),
                                _smem_xadd(ni0, Int32(0), Int32(0)),
                            )
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    if cutlass.const_expr(self.mxfp4_share_cutoff):
                        filter_key = Uint32(_smem_ld(p_threshold, Int32(0)))
                        own_key = _convert_to_uint32(cutoff)
                        if cutoff == Float32(0):
                            own_key = Uint32(0x7FFFFFFF)
                        if cutlass.const_expr(
                            self.mxfp4_cutoff_strategy == "shared_compact"
                        ):
                            if filter_key > own_key:
                                if partition_thread == Int32(0):
                                    _smem_st(p_counter, Int32(0), Int32(0))
                                cute.arch.barrier(
                                    barrier_id=partition_barrier, number_of_threads=256
                                )
                                for retained_batch in cutlass.range_constexpr(2):
                                    retained_index = partition_thread + Int32(
                                        retained_batch * 256
                                    )
                                    retained_value = Float32(
                                        p_selected_values[retained_index]
                                    )
                                    retained_logical = Int32(
                                        p_selected_indices[retained_index]
                                    )
                                    retained_key = _convert_to_uint32(retained_value)
                                    accept_retained = retained_key >= filter_key
                                    mask = cute.arch.vote_ballot_sync(accept_retained)
                                    rank = Int32(
                                        cute.arch.popc(
                                            mask
                                            & ((Uint32(1) << Uint32(lane)) - Uint32(1))
                                        )
                                    )
                                    size = Int32(cute.arch.popc(mask))
                                    base = Int32(0)
                                    if lane == Int32(0) and size > Int32(0):
                                        base = _smem_xadd(p_counter, Int32(0), size)
                                    base = cute.arch.shuffle_sync(base, Int32(0))
                                    if accept_retained:
                                        p_values[base + rank] = retained_value
                                        p_indices[base + rank] = retained_logical
                                    else:
                                        _smem_red_add(
                                            p_cached_hist_address,
                                            Int32(retained_key >> Uint32(24)),
                                            Int32(-1),
                                        )
                                cute.arch.barrier(
                                    barrier_id=partition_barrier, number_of_threads=256
                                )
                                carry_count = _smem_ld(p_counter, Int32(0))
                                cute.arch.barrier(
                                    barrier_id=partition_barrier, number_of_threads=256
                                )
                for score_batch in cutlass.range_constexpr(
                    self.mxfp4_score_batch_tiles
                ):
                    self.mxfp4_scorer._score_tile_fused(
                        q_flat,
                        qs_flat,
                        weights_flat,
                        pool_flat,
                        pages_flat,
                        seqlens_per_query,
                        mxfp4_operands[1],
                        candidates_flat,
                        mxfp4_operands[3],
                        s_c0_values,
                        Int32(q_bytes.shape[0]),
                        mxfp4_score_width,
                        Int32(real_page_table.shape[1]),
                        mxfp4_page_row_stride,
                        Int64(k_quant_bytes.stride[0]),
                        Int64(k_quant_bytes.shape[0]),
                        tile_start + Int32(score_batch) * Int32(512),
                        row_index=q_idx,
                        local_output=True,
                        output_base=partition * Int32(self.carry_cap // 4 - 128)
                        + carry_count
                        + Int32(score_batch) * Int32(128),
                        logical_indices=s_c0_gindex,
                        weight_pairs=weight_pairs,
                        aligned_q=self.mxfp4_aligned_q,
                        carry_query_offsets=self.mxfp4_query_offset_mode != 0,
                        logical_query_index=self.mxfp4_query_offset_mode == 2,
                        unsigned_candidate_pages=self.mxfp4_unsigned_candidate_pages,
                    )
                filter_live = cutoff != Float32(-float("inf"))
                if cutlass.const_expr(self.mxfp4_share_cutoff):
                    filter_live = filter_key > Uint32(0x007FFFFF)
                if filter_live:
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    value = Float32(-float("inf"))
                    logical = Int32(-1)
                    accept = False
                    if partition_thread < count:
                        value = Float32(p_values[carry_count + partition_thread])
                        logical = Int32(p_indices[carry_count + partition_thread])
                        if cutlass.const_expr(self.mxfp4_share_cutoff):
                            accept = _convert_to_uint32(value) >= filter_key
                        else:
                            accept = value >= cutoff
                    if partition_thread == Int32(0):
                        _smem_st(p_counter, Int32(0), Int32(0))
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    if accept:
                        slot = _smem_xadd(p_counter, Int32(0), Int32(1))
                        p_values[carry_count + slot] = value
                        p_indices[carry_count + slot] = logical
                        digit = Int32(_convert_to_uint32(value) >> Uint32(24))
                        _smem_red_add(p_cached_hist_address, digit, Int32(1))
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                    carry_count += _smem_ld(p_counter, Int32(0))
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=256
                    )
                else:
                    carry_count += count
                tile_start += Int32(512 * self.mxfp4_score_batch_tiles)
            if carry_count > topk_static:
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=256)
                _mxfp4_private_radix_select(
                    p_values,
                    p_indices,
                    p_selected_values,
                    p_selected_indices,
                    carry_count,
                    topk_static,
                    partition_thread,
                    p_hist0,
                    p_hist1,
                    p_out,
                    p_hist_address,
                    p_counter,
                    p_threshold,
                    p_remaining,
                    selection_threads=256,
                    barrier_id=partition_barrier,
                    maintain_histogram=True,
                    cached_histogram_valid=(cutoff != Float32(-float("inf"))),
                )
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=256)
                i = Int32(partition_thread)
                while i < topk_static:
                    p_values[i] = Float32(p_selected_values[i])
                    p_indices[i] = Int32(p_selected_indices[i])
                    i += Int32(256)
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=256)
                carry_count = topk_static
            if partition_thread == Int32(0):
                s_relay[partition] = carry_count
            cute.arch.sync_threads()
            # Local top-k carries contain an exact global top-k.
            gathered_values = cute.make_rmem_tensor((2,), Float32)
            gathered_indices = cute.make_rmem_tensor((2,), Int32)
            gathered_targets = cute.make_rmem_tensor((2,), Int32)
            gathered_targets.fill(-1)
            for slot in cutlass.range_constexpr(2):
                item = tx + Int32(slot * 1024)
                source_partition = item // Int32(512)
                local_index = item % Int32(512)
                prefix = Int32(0)
                source_count = Int32(0)
                for group in cutlass.range_constexpr(4):
                    value = Int32(s_relay[group])
                    if Int32(group) < source_partition:
                        prefix += value
                    if Int32(group) == source_partition:
                        source_count = value
                if local_index < source_count:
                    source = source_partition * Int32(self.carry_cap // 4) + local_index
                    gathered_values[slot] = Float32(s_c0_values[source])
                    gathered_indices[slot] = Int32(s_c0_gindex[source])
                    gathered_targets[slot] = prefix + local_index
            cute.arch.sync_threads()
            for slot in cutlass.range_constexpr(2):
                target = Int32(gathered_targets[slot])
                if target >= Int32(0):
                    s_c0_values[target] = gathered_values[slot]
                    s_c0_gindex[target] = gathered_indices[slot]
            cute.arch.sync_threads()
            carry_count = Int32(0)
            for group in cutlass.range_constexpr(4):
                carry_count += Int32(s_relay[group])
            if carry_count > topk_static:
                _mxfp4_private_radix_select(
                    s_c0_values,
                    s_c0_gindex,
                    s_c1_values,
                    s_c1_gindex,
                    carry_count,
                    topk_static,
                    tx,
                    s_hist0,
                    s_hist1,
                    s_out,
                    h0,
                    ctr,
                    thr,
                    lr,
                )
                cute.arch.sync_threads()
                if tx < topk_static:
                    s_c0_values[tx] = Float32(s_c1_values[tx])
                    s_c0_gindex[tx] = Int32(s_c1_gindex[tx])
                cute.arch.sync_threads()
                carry_count = topk_static
        elif cutlass.const_expr(self.mxfp4 and self.selection_groups == 2):
            partition = tx // Int32(512)
            partition_thread = tx % Int32(512)
            partition_warp = partition_thread // Int32(32)
            partition_barrier = partition + Int32(1)
            carry_count = Int32(0)
            cutoff = Float32(-float("inf"))
            span_first = Int32(
                cutlass.min(Int64(page_start) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            span_end = Int32(
                cutlass.min(Int64(page_end) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            tile_start = span_first
            q_flat = cute.make_tensor(
                q_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            qs_flat = cute.make_tensor(
                mxfp4_operands[0].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            weights_flat = cute.make_tensor(
                weights.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            if cutlass.const_expr(self.mxfp4_warp_weights):
                weight_pairs = self.mxfp4_scorer._load_weight_pairs(weights, q_idx)
            else:
                weight_pairs = None
            mxfp4_page_row_stride = Int64(real_page_table.stride[0])
            if real_page_table.shape[0] == Int32(1):
                mxfp4_page_row_stride = Int64(0)
            pool_flat = cute.make_tensor(
                k_quant_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            pages_flat = cute.make_tensor(
                real_page_table.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            candidates_flat = cute.make_tensor(
                mxfp4_operands[2].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            p_values = cute.make_tensor(
                s_c0_values.iterator + partition * Int32(self.carry_cap // 2),
                cute.make_layout((self.carry_cap // 2,), stride=(1,)),
            )
            p_indices = cute.make_tensor(
                s_c0_gindex.iterator + partition * Int32(self.carry_cap // 2),
                cute.make_layout((self.carry_cap // 2,), stride=(1,)),
            )
            p_selected_values = cute.make_tensor(
                s_c1_values.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_selected_indices = cute.make_tensor(
                s_c1_gindex.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_hist0 = cute.make_tensor(
                s_hist0.iterator + partition * Int32(384),
                cute.make_layout((384,), stride=(1,)),
            )
            p_hist1 = cute.make_tensor(
                s_hist1.iterator + partition * Int32(384),
                cute.make_layout((384,), stride=(1,)),
            )
            p_out = cute.make_tensor(
                s_out.iterator + partition * Int32(512),
                cute.make_layout((512,), stride=(1,)),
            )
            p_hist_address = h0 + partition * Int32(1536)
            p_counter = ctr + partition * Int32(4)
            p_threshold = thr + partition * Int32(4)
            p_remaining = lr + partition * Int32(4)
            while tile_start < span_end:
                count = cutlass.min(
                    Int32(256),
                    cutlass.max(
                        span_end - tile_start - partition * Int32(256), Int32(0)
                    ),
                )
                if cutlass.const_expr(self.mxfp4_score_batch_tiles == 2):
                    count += cutlass.min(
                        Int32(256),
                        cutlass.max(
                            span_end - tile_start - Int32(512) - partition * Int32(256),
                            Int32(0),
                        ),
                    )
                if carry_count + Int32(256 * self.mxfp4_score_batch_tiles) > Int32(
                    self.carry_cap // 2
                ):
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    if partition_thread < Int32(256):
                        _mxfp4_private_radix_select(
                            p_values,
                            p_indices,
                            p_selected_values,
                            p_selected_indices,
                            carry_count,
                            topk_static,
                            partition_thread,
                            p_hist0,
                            p_hist1,
                            p_out,
                            p_hist_address,
                            p_counter,
                            p_threshold,
                            p_remaining,
                            selection_threads=256,
                            barrier_id=partition + Int32(3),
                        )
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    p_values[partition_thread] = Float32(
                        p_selected_values[partition_thread]
                    )
                    p_indices[partition_thread] = Int32(
                        p_selected_indices[partition_thread]
                    )
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    carry_count = topk_static
                    value = Float32(p_values[partition_thread])
                    for delta in (16, 8, 4, 2, 1):
                        value = cutlass.min(
                            value, cute.arch.shuffle_sync_bfly(value, offset=delta)
                        )
                    if lane == Int32(0):
                        s_scale[partition * Int32(32) + partition_warp] = value
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    if partition_warp == Int32(0):
                        value = Float32(float("inf"))
                        if lane < Int32(16):
                            value = Float32(s_scale[partition * Int32(32) + lane])
                        for delta in (16, 8, 4, 2, 1):
                            value = cutlass.min(
                                value, cute.arch.shuffle_sync_bfly(value, offset=delta)
                            )
                        if lane == Int32(0):
                            s_scale[partition * Int32(32) + Int32(16)] = value
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    cutoff = Float32(s_scale[partition * Int32(32) + Int32(16)])
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                for score_batch in cutlass.range(self.mxfp4_score_batch_tiles):
                    self.mxfp4_scorer._score_tile_fused(
                        q_flat,
                        qs_flat,
                        weights_flat,
                        pool_flat,
                        pages_flat,
                        seqlens_per_query,
                        mxfp4_operands[1],
                        candidates_flat,
                        mxfp4_operands[3],
                        s_c0_values,
                        Int32(q_bytes.shape[0]),
                        mxfp4_score_width,
                        Int32(real_page_table.shape[1]),
                        mxfp4_page_row_stride,
                        Int64(k_quant_bytes.stride[0]),
                        Int64(k_quant_bytes.shape[0]),
                        tile_start + Int32(score_batch) * Int32(512),
                        row_index=q_idx,
                        local_output=True,
                        output_base=partition * Int32(self.carry_cap // 2 - 256)
                        + carry_count
                        + Int32(score_batch) * Int32(256),
                        logical_indices=s_c0_gindex,
                        weight_pairs=weight_pairs,
                        aligned_q=self.mxfp4_aligned_q,
                        carry_query_offsets=self.mxfp4_query_offset_mode != 0,
                        logical_query_index=self.mxfp4_query_offset_mode == 2,
                        unsigned_candidate_pages=self.mxfp4_unsigned_candidate_pages,
                    )
                if cutoff != Float32(-float("inf")):
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    value = Float32(-float("inf"))
                    logical = Int32(-1)
                    accept = False
                    if partition_thread < count:
                        value = Float32(p_values[carry_count + partition_thread])
                        logical = Int32(p_indices[carry_count + partition_thread])
                        accept = value >= cutoff
                    if partition_thread == Int32(0):
                        _smem_st(p_counter, Int32(0), Int32(0))
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    if accept:
                        slot = _smem_xadd(p_counter, Int32(0), Int32(1))
                        p_values[carry_count + slot] = value
                        p_indices[carry_count + slot] = logical
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                    carry_count += _smem_ld(p_counter, Int32(0))
                    cute.arch.barrier(
                        barrier_id=partition_barrier, number_of_threads=512
                    )
                else:
                    carry_count += count
                tile_start += Int32(512 * self.mxfp4_score_batch_tiles)
            if carry_count > topk_static:
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=512)
                if partition_thread < Int32(256):
                    _mxfp4_private_radix_select(
                        p_values,
                        p_indices,
                        p_selected_values,
                        p_selected_indices,
                        carry_count,
                        topk_static,
                        partition_thread,
                        p_hist0,
                        p_hist1,
                        p_out,
                        p_hist_address,
                        p_counter,
                        p_threshold,
                        p_remaining,
                        selection_threads=256,
                        barrier_id=partition + Int32(3),
                    )
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=512)
                p_values[partition_thread] = Float32(
                    p_selected_values[partition_thread]
                )
                p_indices[partition_thread] = Int32(
                    p_selected_indices[partition_thread]
                )
                cute.arch.barrier(barrier_id=partition_barrier, number_of_threads=512)
                carry_count = topk_static
            if partition_thread == Int32(0):
                s_relay[partition] = carry_count
            cute.arch.sync_threads()
            # Local top-k carries contain an exact global top-k.
            left_count = Int32(s_relay[0])
            right_count = Int32(s_relay[1])
            if tx < right_count:
                s_c0_values[left_count + tx] = Float32(
                    s_c0_values[Int32(self.carry_cap // 2) + tx]
                )
                s_c0_gindex[left_count + tx] = Int32(
                    s_c0_gindex[Int32(self.carry_cap // 2) + tx]
                )
            cute.arch.sync_threads()
            carry_count = left_count + right_count
            if carry_count > topk_static:
                _mxfp4_private_radix_select(
                    s_c0_values,
                    s_c0_gindex,
                    s_c1_values,
                    s_c1_gindex,
                    carry_count,
                    topk_static,
                    tx,
                    s_hist0,
                    s_hist1,
                    s_out,
                    h0,
                    ctr,
                    thr,
                    lr,
                )
                cute.arch.sync_threads()
                if tx < topk_static:
                    s_c0_values[tx] = Float32(s_c1_values[tx])
                    s_c0_gindex[tx] = Int32(s_c1_gindex[tx])
                cute.arch.sync_threads()
                carry_count = topk_static
        elif cutlass.const_expr(self.mxfp4):
            carry_count = Int32(0)
            cutoff = Float32(-float("inf"))
            span_first = Int32(
                cutlass.min(Int64(page_start) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            span_end = Int32(
                cutlass.min(Int64(page_end) * Int64(_PAGE_SIZE), Int64(seq_len))
            )
            tile_start = span_first
            if cutlass.const_expr(self.tile_schedule == "cyclic"):
                span_end = seq_len
                tile_start = Int32(
                    cutlass.min(
                        Int64(cta_in_group) * Int64(self.mxfp4_score_group),
                        Int64(seq_len),
                    )
                )
            q_flat = cute.make_tensor(
                q_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            qs_flat = cute.make_tensor(
                mxfp4_operands[0].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            weights_flat = cute.make_tensor(
                weights.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            if cutlass.const_expr(self.mxfp4_warp_weights):
                weight_pairs = self.mxfp4_scorer._load_weight_pairs(weights, q_idx)
            else:
                weight_pairs = None
            pool_flat = cute.make_tensor(
                k_quant_bytes.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            pages_flat = cute.make_tensor(
                real_page_table.iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            candidates_flat = cute.make_tensor(
                mxfp4_operands[2].iterator, cute.make_layout((Int64(1) << Int64(40),))
            )
            mxfp4_page_row_stride = Int64(real_page_table.stride[0])
            if real_page_table.shape[0] == Int32(1):
                mxfp4_page_row_stride = Int64(0)
            while tile_start < span_end:
                count = cutlass.min(
                    Int32(self.mxfp4_score_group), span_end - tile_start
                )
                if carry_count + Int32(self.mxfp4_score_group) > Int32(
                    self.carry_cap
                ) and cutoff > Float32(-float("inf")):
                    values = cute.make_rmem_tensor((self.mxfp4_carry_slots,), Float32)
                    indices = cute.make_rmem_tensor((self.mxfp4_carry_slots,), Int32)
                    cute.arch.sync_threads()
                    for slot in cutlass.range_constexpr(self.mxfp4_carry_slots):
                        idx = tx + Int32(slot * self.radix_threads)
                        values[slot] = Float32(-float("inf"))
                        indices[slot] = Int32(-1)
                        if idx < carry_count:
                            values[slot] = Float32(s_c0_values[idx])
                            indices[slot] = Int32(s_c0_gindex[idx])
                    if tx == Int32(0):
                        _smem_st(ni0, Int32(0), Int32(0))
                    cute.arch.sync_threads()
                    for slot in cutlass.range_constexpr(self.mxfp4_carry_slots):
                        idx = tx + Int32(slot * self.radix_threads)
                        if idx < carry_count and values[slot] >= cutoff:
                            pos = _smem_xadd(ni0, Int32(0), Int32(1))
                            s_c0_values[pos] = values[slot]
                            s_c0_gindex[pos] = indices[slot]
                    cute.arch.sync_threads()
                    carry_count = _smem_ld(ni0, Int32(0))
                if carry_count + Int32(self.mxfp4_score_group) > Int32(self.carry_cap):
                    cute.arch.sync_threads()
                    _mxfp4_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        h0,
                        ctr,
                        thr,
                        lr,
                        threads=self.radix_threads,
                    )
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < topk_static:
                        s_c0_values[i] = Float32(s_c1_values[i])
                        s_c0_gindex[i] = Int32(s_c1_gindex[i])
                        i += Int32(self.radix_threads)
                    cute.arch.sync_threads()
                    carry_count = topk_static
                    value = Float32(float("inf"))
                    if tx < topk_static:
                        value = Float32(s_c0_values[tx])
                    for shift in cutlass.range_constexpr(5):
                        value = cutlass.min(
                            value, cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
                        )
                    if tx % Int32(32) == Int32(0):
                        s_c1_values[tx // Int32(32)] = value
                    cute.arch.sync_threads()
                    value = Float32(float("inf"))
                    if tx % Int32(32) < Int32(self.radix_threads // 32):
                        value = Float32(s_c1_values[tx % Int32(32)])
                    for shift in cutlass.range_constexpr(5):
                        value = cutlass.min(
                            value, cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
                        )
                    cutoff = value
                    cute.arch.sync_threads()
                for score_pass in cutlass.range_constexpr(self.mxfp4_score_passes):
                    self.mxfp4_scorer._score_tile_fused(
                        q_flat,
                        qs_flat,
                        weights_flat,
                        pool_flat,
                        pages_flat,
                        seqlens_per_query,
                        mxfp4_operands[1],
                        candidates_flat,
                        mxfp4_operands[3],
                        s_c0_values,
                        Int32(q_bytes.shape[0]),
                        mxfp4_score_width,
                        Int32(real_page_table.shape[1]),
                        mxfp4_page_row_stride,
                        Int64(k_quant_bytes.stride[0]),
                        Int64(k_quant_bytes.shape[0]),
                        tile_start + Int32(score_pass * self.mxfp4_chunk),
                        row_index=q_idx,
                        local_output=True,
                        output_base=carry_count + Int32(score_pass * self.mxfp4_chunk),
                        logical_indices=s_c0_gindex,
                        hardware_scale_select=self.candidates
                        and self.score_width <= 512,
                        weight_pairs=weight_pairs,
                        aligned_q=self.mxfp4_aligned_q,
                        carry_query_offsets=self.mxfp4_query_offset_mode != 0,
                        logical_query_index=self.mxfp4_query_offset_mode == 2,
                        unsigned_candidate_pages=self.mxfp4_unsigned_candidate_pages,
                    )
                carry_count += count
                if cutlass.const_expr(self.tile_schedule == "cyclic"):
                    tile_start += Int32(
                        cutlass.min(
                            Int64(self.ctas_per_group) * Int64(self.mxfp4_score_group),
                            Int64(span_end) - Int64(tile_start),
                        )
                    )
                else:
                    tile_start += count
            cute.arch.sync_threads()
            if carry_count > topk_static:
                _mxfp4_radix_select(
                    s_c0_values,
                    s_c0_gindex,
                    s_c1_values,
                    s_c1_gindex,
                    carry_count,
                    topk_static,
                    tx,
                    s_hist0,
                    s_hist1,
                    s_out,
                    h0,
                    ctr,
                    thr,
                    lr,
                    threads=self.radix_threads,
                )
                cute.arch.sync_threads()
                i = Int32(tx)
                while i < topk_static:
                    s_c0_values[i] = Float32(s_c1_values[i])
                    s_c0_gindex[i] = Int32(s_c1_gindex[i])
                    i += Int32(self.radix_threads)
                cute.arch.sync_threads()
                carry_count = topk_static
        elif cutlass.const_expr(self.direct_k_score):
            # Warp-owns-tokens score: every warp streams 8 consecutive tokens
            # per round with K fragments read straight from L2 (no staging,
            # no producer pipeline, no per-page barrier). Slot bookkeeping is
            # uniform register arithmetic, so different rounds write disjoint
            # carry regions and the only CTA rendezvous are trims.
            carry_count = Int32(0)
            span_first = page_start * Int32(_PAGE_SIZE)
            span_end_tok = page_end * Int32(_PAGE_SIZE)
            if span_end_tok > seq_len:
                span_end_tok = seq_len
            span = span_end_tok - span_first
            if span < Int32(0):
                span = Int32(0)
            lane4 = lane % Int32(4)
            round_base = Int32(0)
            while round_base < span:
                round_valid = span - round_base
                if round_valid > Int32(256):
                    round_valid = Int32(256)
                if carry_count + Int32(256) > Int32(self.carry_cap):
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                        threads=self.radix_threads,
                    )
                    cute.arch.sync_threads()
                    ti = Int32(tx)
                    while ti < topk_static:
                        s_c0_values[ti] = Float32(s_c1_values[ti])
                        s_c0_gindex[ti] = Int32(s_c1_gindex[ti])
                        ti += Int32(self.radix_threads)
                    cute.arch.sync_threads()
                    carry_count = topk_static
                my_rel = round_base + warp_idx * Int32(8)
                pid = Int32(-1)
                tip0 = Int32(0)
                if my_rel < span:
                    my_page = my_rel // Int32(_PAGE_SIZE)
                    tip0 = my_rel - my_page * Int32(_PAGE_SIZE)
                    pcol = page_start + my_page
                    if pcol < Int32(real_page_table.shape[1]):
                        pid = Int32(real_page_table[q_idx, pcol])
                t0 = Float32(0.0)
                t1 = Float32(0.0)
                if pid >= Int32(0):
                    k_off = Int64(pid) * Int64(self.k_quant_page_stride) + Int64(
                        (tip0 + lane // Int32(4)) * Int32(_INDEX_HEAD_DIM)
                        + lane4 * Int32(4)
                    )
                    t0, t1 = _score_tokens_direct_k(
                        q_smem_base_addr,
                        s_w,
                        num_heads,
                        k_quant_bytes,
                        k_off,
                        lane,
                        self.num_q_head_tiles,
                    )
                if lane < Int32(4):
                    for cc in cutlass.range_constexpr(2):
                        col = Int32(2) * lane4 + Int32(cc)
                        tok_rel = my_rel + col
                        if tok_rel < span:
                            val = Float32(-3.4028235e38)
                            gidx = Int32(-1)
                            if pid >= Int32(0):
                                if cutlass.const_expr(cc == 0):
                                    tsel = t0
                                else:
                                    tsel = t1
                                scale_addr = get_ptr_as_int64(
                                    k_scales,
                                    Int64(pid) * Int64(self.k_scales_row_stride)
                                    + Int64(tip0 + col),
                                )
                                val = Float32(tsel * ld_global_nc_f32(scale_addr))
                                if cutlass.const_expr(self.paged_output):
                                    gidx = pid * Int32(_PAGE_SIZE) + tip0 + col
                                else:
                                    gidx = span_first + tok_rel
                            s_c0_values[carry_count + tok_rel - round_base] = val
                            s_c0_gindex[carry_count + tok_rel - round_base] = gidx
                carry_count = carry_count + round_valid
                round_base += Int32(256)
            cute.arch.sync_threads()
            if carry_count > topk_static:
                _fused_radix_select(
                    s_c0_values,
                    s_c0_gindex,
                    s_c1_values,
                    s_c1_gindex,
                    carry_count,
                    topk_static,
                    self.cands,
                    tx,
                    s_hist0,
                    s_hist1,
                    s_out,
                    s_cand0,
                    s_cand1,
                    h0,
                    ctr,
                    thr,
                    ni0,
                    ni1,
                    lr,
                    threads=self.radix_threads,
                )
                cute.arch.sync_threads()
                ti = Int32(tx)
                while ti < topk_static:
                    s_c0_values[ti] = Float32(s_c1_values[ti])
                    s_c0_gindex[ti] = Int32(s_c1_gindex[ti])
                    ti += Int32(self.radix_threads)
                cute.arch.sync_threads()
                carry_count = topk_static
        else:
            carry_count = Int32(0)
            # Paged K pipeline: page 0 of the slice was issued before Q
            # staging; iteration p issues page p+1 into the K stage page p-2
            # used (the 3-stage ring holds p-1, p, and p+1), then waits for
            # page p. Deferred-reduce flow (page_splits == 1): page p's
            # reduce and carry append run at iteration p+1 behind the
            # pipeline barrier, so one barrier per page orders both the K
            # stage and the partials handoff, and page p-1's scales stay live
            # for it (4-slot scales ring).
            k_stage = Int32(0)
            sc_slot = Int32(0)
            cur_scale_addr = scales_base_addr
            prev_do = Int32(0)
            prev_valid = Int32(0)
            prev_out_base = Int32(0)
            prev_scale_addr = scales_base_addr
            prev_pl2 = Int32(0)
            cur_pl2 = Int32(0)
            page_col = page_start
            while page_col < page_end:
                page_base = page_col * Int32(_PAGE_SIZE)
                valid_slots = seq_len - page_base
                if valid_slots > Int32(_PAGE_SIZE):
                    valid_slots = Int32(_PAGE_SIZE)
                n_new = Int32(0)
                do_page = Int32(0)
                page_id = Int32(0)
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    page_id = pid_cur
                    if (page_id >= Int32(0)) & (valid_slots > Int32(0)):
                        do_page = Int32(1)
                    # Read the id of page p+2 now; it is issued at iteration p+1.
                    pid_p2 = _page_id_or_invalid(
                        real_page_table, q_idx, page_col + Int32(2), page_end
                    )
                else:
                    if valid_slots > Int32(0):
                        do_page = Int32(1)

                cur_k_base = _k_stage_addr(
                    k_stage, k_page_perm_base_addr, k_page_base_addr, k_page3_base_addr
                )
                cur_scale_base = scales_base_addr
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    nxt_valid = Int32(0)
                    if (page_col + Int32(1) < page_end) & (pid_p1 >= Int32(0)):
                        nxt_valid = Int32(1)
                    next_stage = k_stage + Int32(1)
                    if next_stage == Int32(_K_STAGES):
                        next_stage = Int32(0)
                    nsc = sc_slot + Int32(1)
                    if nsc == Int32(4):
                        nsc = Int32(0)
                    if nxt_valid != Int32(0):
                        _stream_issue_k_page_cp_async(
                            k_quant_bytes,
                            k_scales,
                            pid_p1,
                            _k_stage_addr(
                                next_stage,
                                k_page_perm_base_addr,
                                k_page_base_addr,
                                k_page3_base_addr,
                            ),
                            scales_base_addr + nsc * Int32(_PAGE_SIZE * 4),
                            Int32(0),
                            tx,
                            k_quant_page_stride=int(self.k_quant_page_stride),
                            k_scales_row_stride=int(self.k_scales_row_stride),
                            issue_threads=self.radix_threads,
                        )
                    cute.arch.cp_async_commit_group()
                    # Page p has landed once every group but the newest has
                    # completed; the barrier then orders page p-1's partials
                    # for the deferred reduce.
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                    cur_scale_base = cur_scale_addr
                    if cutlass.const_expr(self.page_splits == 1):
                        # Reduce + append the PREVIOUS page: its partials (all
                        # head tiles) and its scales are ordered by the pipeline
                        # barrier above; the score of the current page then runs
                        # into the other partials buffer with no trailing barrier.
                        if prev_do != Int32(0):
                            if tx < score_threads:
                                if (head_tile_slot == Int32(0)) & (
                                    lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                ):
                                    slot_idx = (
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                        + lane
                                    )
                                    if slot_idx < prev_valid:
                                        logit = Float32(0.0)
                                        h_i = Int32(0)
                                        if prev_pl2 == Int32(0):
                                            while h_i < Int32(self.num_q_head_tiles):
                                                logit = Float32(
                                                    logit
                                                    + s_partial_logits[slot_idx, h_i]
                                                )
                                                h_i += Int32(1)
                                        else:
                                            while h_i < Int32(self.num_q_head_tiles):
                                                logit = Float32(
                                                    logit
                                                    + s_partial_logits2[slot_idx, h_i]
                                                )
                                                h_i += Int32(1)
                                        s_c0_values[carry_count + slot_idx] = Float32(
                                            logit
                                            * ld_shared_f32(
                                                prev_scale_addr + slot_idx * Int32(4)
                                            )
                                        )
                                        s_c0_gindex[carry_count + slot_idx] = (
                                            prev_out_base + slot_idx
                                        )
                            carry_count = carry_count + prev_valid
                            prev_do = Int32(0)
                if do_page != Int32(0):
                    n_new = valid_slots
                    if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                        pass
                    else:
                        # CONTIGUOUS_MLA: masked wide scalar load into linear staging, then repack.
                        _load_flat_k_tile_wide(
                            k_quant_bytes,
                            abs_start + page_base,
                            valid_slots,
                            s_k_page_stage,
                            tx,
                            Int32(self.radix_threads),
                        )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            sv = Float32(0.0)
                            if scale_idx < valid_slots:
                                sv = Float32(
                                    k_scales[abs_start + page_base + scale_idx]
                                )
                            s_scale[scale_idx] = sv
                            scale_idx += Int32(self.radix_threads)
                        cute.arch.sync_threads()
                        _repack_k_page_wide(
                            k_page_base_addr,
                            k_page_perm_base_addr,
                            tx,
                            Int32(self.radix_threads),
                        )
                        cute.arch.sync_threads()

                    if cutlass.const_expr(
                        self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
                    ):
                        tb_defer = token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                        if tx < score_threads:
                            if tb_defer < valid_slots:
                                htb_defer = head_tile_slot * Int32(_PAGED_Q_HEAD_TILE)
                                if cur_pl2 == Int32(0):
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        tb_defer,
                                        htb_defer,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                                else:
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        tb_defer,
                                        htb_defer,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits2,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                        prev_do = Int32(1)
                        prev_valid = valid_slots
                        prev_scale_addr = cur_scale_addr
                        prev_pl2 = cur_pl2
                        cur_pl2 = cur_pl2 ^ Int32(1)
                        prev_out_base = abs_start + page_base
                        if cutlass.const_expr(self.paged_output):
                            prev_out_base = page_id * Int32(_PAGE_SIZE)
                    else:
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            # No partial-logits pre-zero: every group with token_base <
                            # valid_slots runs the MMA (writing all its token columns for
                            # all head tiles), and the reduce only reads slot_idx <
                            # valid_slots -> it never consumes a stale (unwritten) partial.
                            token_base = split_idx * Int32(
                                self.tokens_per_work
                            ) + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            if tx < score_threads:
                                if token_base < valid_slots:
                                    head_tile_base = head_tile_slot * Int32(
                                        _PAGED_Q_HEAD_TILE
                                    )
                                    _compute_mxfp8_tile_partials_qldm(
                                        q_smem_base_addr,
                                        s_w,
                                        num_heads,
                                        cur_k_base,
                                        token_base,
                                        head_tile_base,
                                        head_tile_slot,
                                        lane,
                                        s_partial_logits,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                    )
                            cute.arch.sync_threads()
                            if tx < score_threads:
                                if (head_tile_slot == Int32(0)) & (
                                    lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                ):
                                    slot_idx = token_base + lane
                                    if slot_idx < valid_slots:
                                        logit = Float32(0.0)
                                        partial_row = (
                                            token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                            + lane
                                        )
                                        h_i = Int32(0)
                                        while h_i < Int32(self.num_q_head_tiles):
                                            logit = Float32(
                                                logit
                                                + s_partial_logits[partial_row, h_i]
                                            )
                                            h_i += Int32(1)
                                        s_c0_values[carry_count + slot_idx] = Float32(
                                            logit
                                            * ld_shared_f32(
                                                cur_scale_base + slot_idx * Int32(4)
                                            )
                                        )
                                        output_idx = abs_start + page_base + slot_idx
                                        if cutlass.const_expr(
                                            self.kv_layout == KV_LAYOUT_PAGED
                                            and self.paged_output
                                        ):
                                            output_idx = (
                                                page_id * Int32(_PAGE_SIZE) + slot_idx
                                            )
                                        s_c0_gindex[carry_count + slot_idx] = output_idx
                            # page_splits==1 (always for 1/2/4 head tiles): no post-reduce
                            # barrier — the next page's load barrier (or the trim's leading
                            # sync, or the pre-output sync) orders this reduce before any
                            # later writer of s_partial_logits / reader of the carry. Multi-
                            # split would need the barrier between splits, so keep it then.
                            if cutlass.const_expr(self.page_splits > 1):
                                cute.arch.sync_threads()
                            split_idx += Int32(1)
                # ---- accumulate + conditional trim ----
                is_last = page_col == (page_end - Int32(1))
                need_trim = Int32(0) != Int32(0)
                if cutlass.const_expr(
                    self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
                ):
                    # Deferred flow: the count already advanced at the deferred
                    # append; the current page appends next iteration (or in the
                    # post-loop tail), so only capacity pressure trims here -- the
                    # final trim runs after the loop.
                    need_trim = carry_count + Int32(_PAGE_SIZE) > Int32(self.carry_cap)
                else:
                    # This page's n_new scored tokens were written into carry0[carry_count:].
                    if do_page != Int32(0):
                        carry_count = carry_count + n_new
                    # Trim carry0 back to topk only when the next page would overflow the
                    # over-sized accumulator, or at the very end. This runs the radix once
                    # per ~(_BATCH_SLACK/_PAGE_SIZE) pages instead of every page.
                    need_trim = (
                        carry_count + Int32(_PAGE_SIZE) > Int32(self.carry_cap)
                    ) | (is_last & (carry_count > topk_static))
                if cutlass.const_expr(self.kv_layout == KV_LAYOUT_PAGED):
                    k_stage = k_stage + Int32(1)
                    if k_stage == Int32(_K_STAGES):
                        k_stage = Int32(0)
                    pid_cur = pid_p1
                    pid_p1 = pid_p2
                    sc_slot = sc_slot + Int32(1)
                    if sc_slot == Int32(4):
                        sc_slot = Int32(0)
                    cur_scale_addr = scales_base_addr + sc_slot * Int32(_PAGE_SIZE * 4)
                if need_trim:
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                        threads=self.radix_threads,
                    )
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < topk_static:
                        s_c0_values[i] = Float32(s_c1_values[i])
                        s_c0_gindex[i] = Int32(s_c1_gindex[i])
                        i += Int32(self.radix_threads)
                    cute.arch.sync_threads()
                    carry_count = topk_static
                page_col += Int32(1)

            cute.arch.sync_threads()
            if cutlass.const_expr(
                self.kv_layout == KV_LAYOUT_PAGED and self.page_splits == 1
            ):
                # Drain the deferred pipeline: the last page's partials are ordered
                # by the barrier above; append it, then run the final trim the
                # in-loop path no longer performs.
                if prev_do != Int32(0):
                    if tx < score_threads:
                        if (head_tile_slot == Int32(0)) & (
                            lane < Int32(_PAGED_TOKENS_PER_GROUP)
                        ):
                            slot_idx = (
                                token_group * Int32(_PAGED_TOKENS_PER_GROUP) + lane
                            )
                            if slot_idx < prev_valid:
                                logit = Float32(0.0)
                                h_i = Int32(0)
                                if prev_pl2 == Int32(0):
                                    while h_i < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits[slot_idx, h_i]
                                        )
                                        h_i += Int32(1)
                                else:
                                    while h_i < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits2[slot_idx, h_i]
                                        )
                                        h_i += Int32(1)
                                s_c0_values[carry_count + slot_idx] = Float32(
                                    logit
                                    * ld_shared_f32(
                                        prev_scale_addr + slot_idx * Int32(4)
                                    )
                                )
                                s_c0_gindex[carry_count + slot_idx] = (
                                    prev_out_base + slot_idx
                                )
                    carry_count = carry_count + prev_valid
                if carry_count > topk_static:
                    cute.arch.sync_threads()
                    _fused_radix_select(
                        s_c0_values,
                        s_c0_gindex,
                        s_c1_values,
                        s_c1_gindex,
                        carry_count,
                        topk_static,
                        self.cands,
                        tx,
                        s_hist0,
                        s_hist1,
                        s_out,
                        s_cand0,
                        s_cand1,
                        h0,
                        ctr,
                        thr,
                        ni0,
                        ni1,
                        lr,
                        threads=self.radix_threads,
                    )
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < topk_static:
                        s_c0_values[i] = Float32(s_c1_values[i])
                        s_c0_gindex[i] = Int32(s_c1_gindex[i])
                        i += Int32(self.radix_threads)
                    cute.arch.sync_threads()
                    carry_count = topk_static
        if cutlass.const_expr(not self.merge_in_kernel):
            if cutlass.const_expr(self.mxfp4 and self.ordered_output):
                _mxfp4_order_output(
                    out_indices,
                    out_values,
                    group_id,
                    s_c0_gindex,
                    s_c0_values,
                    tx,
                    carry_count,
                    True,
                    merge_state,
                    0,
                    self.output_position_capacity,
                    self.vector_output_pairs,
                    threads=self.radix_threads,
                )
                if (group_id == Int32(0)) & (tx == Int32(0)):
                    merge_state[Int64(0)] = Int32(0)
                    merge_state[Int64(1)] = Int32(0)
            else:
                i = Int32(tx)
                while i < topk_static:
                    is_valid = i < carry_count
                    out_values[group_id, i] = (
                        Float32(s_c0_values[i]) if is_valid else Float32(float("-inf"))
                    )
                    out_indices[group_id, i] = (
                        Int32(s_c0_gindex[i]) if is_valid else Int32(-1)
                    )
                    i += Int32(self.radix_threads)
        else:
            if cutlass.const_expr(self.state_reset_pdl):
                cute.arch.griddepcontrol_wait()
            # ---- in-kernel cross-CTA merge: runtime-selected strategy ----
            # All CTAs in a group share the row's live K length (seq_len, from
            # metadata), so they pick the SAME arm with no extra barrier:
            #   seq_len >  merge_threshold -> cooperative grid-barrier radix over
            #     the SMEM carries (parallel; amortizes the barriers at long K);
            #   seq_len <= merge_threshold -> serial last-CTA reduction over a
            #     packed slab (no grid barriers; cheaper for few-row/short-K).
            # Both arms index the per-group _COOP_STATE_WORDS state block via
            # _fused_state_offset, so groups taking different arms never collide.
            ctas_pg = Int32(self.ctas_per_group)
            barrier_phase = Int32(0)
            cap = Int32(self.pack_cap)
            slab_base = Int64(group_id) * Int64(cap)
            pack_v_row = cute.make_tensor(
                pack_values.iterator + slab_base,
                cute.make_layout((self.pack_cap,), stride=(1,)),
            )
            pack_i_row = cute.make_tensor(
                pack_indices.iterator + slab_base,
                cute.make_layout((self.pack_cap,), stride=(1,)),
            )
            # Pre-declare every variable assigned inside the dynamic arms below;
            # cute forbids a None->typed transition inside a dynamic `if`.
            i = Int32(0)
            base = Int32(0)
            off = Int32(0)
            total = Int32(0)
            key = Uint32(0)
            mask = Uint32(0)
            prefix = Uint32(0)
            shift = Uint32(0)
            pivot_bin = Int32(0)
            count_gt = Int32(0)
            digit = Int32(0)
            bin_count = Int32(0)
            winners = Int32(0)
            remaining_k = Int32(0)
            done = Int32(0)
            local_gt = Int32(0)
            local_eq = Int32(0)
            gt_base = Int32(0)
            eq_base = Int32(0)
            pos = Int32(0)
            c = Int32(0)
            coop_possible = Int32(1 if self.coop_merge_possible else 0)
            if cutlass.const_expr(
                self.mxfp4 and self.mxfp4_merge_bits == 16 and self.coop_merge_possible
            ):
                last_departure = _mxfp4_onepass_merge(
                    merge_state,
                    group_id,
                    cta_in_group,
                    ctas_pg,
                    tx,
                    lane,
                    warp_idx,
                    s_c0_values,
                    s_c0_gindex,
                    carry_count,
                    out_values,
                    out_indices,
                    s_hist0,
                    coop_hist_addr,
                    s_coop_scalars,
                    s_coop_counters,
                    coop_ctr_addr,
                    self.mxfp4_histogram_fine_words,
                    self.mxfp4_histogram_active_cap,
                    self.mxfp4_histogram_count_bits,
                    self.mxfp4_histogram_bitmap,
                    k_page_base_addr,
                    self.radix_threads,
                )
                if cutlass.const_expr(self.ordered_output):
                    if last_departure:
                        _mxfp4_order_output(
                            out_indices,
                            out_values,
                            group_id,
                            s_c0_gindex,
                            s_c0_values,
                            tx,
                            Int32(512),
                            False,
                            merge_state,
                            0,
                            self.output_position_capacity,
                            self.vector_output_pairs,
                            threads=self.radix_threads,
                        )
            elif (seq_len > Int32(self.merge_threshold)) & (coop_possible != Int32(0)):
                # ---- cooperative grid-barrier radix merge ----
                # MXFP4 refines the high sixteen BF16 key bits in two byte rounds.
                # FP8 round 0 histograms the top 12 key bits of every candidate,
                # round 1 the next 12 bits of the candidates matching the
                # round-0 pivot, and round 2 the low byte of those matching
                # the round-1 pivot. Each round accumulates the group's
                # histogram into its own global buffer, so one grid barrier
                # per round publishes it, and every CTA then reads the buffer
                # back and derives the same pivot digit. The round-0 barrier
                # also publishes the group's candidate total. Winners above
                # the final pivot prefix and the tie fill at the prefix are
                # placed through two group counters whose bases follow from
                # the histogram, so no barrier separates them, and the
                # refinement stops once the pivot bin holds exactly the
                # remaining slots. The last CTA to leave zeroes the state
                # block. The 4096-bin local histogram lives in the K ring,
                # which is idle once scoring is done.
                if tx == Int32(0):
                    red_add_global_i32(
                        _fused_state_ptr(
                            merge_state, group_id, Int32(_FUSED_STATE_TOTAL)
                        ),
                        carry_count,
                    )
                cute.arch.cp_async_wait_group(0)
                wide_hist_addr = k_page_base_addr
                for hist_pass in cutlass.range_constexpr(1024 // self.radix_threads):
                    st_shared_v4_u32(
                        wide_hist_addr
                        + (tx + Int32(hist_pass * self.radix_threads)) * Int32(16),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
                    )
                if tx < Int32(_RADIX):
                    s_hist0[tx] = Int32(0)
                cute.arch.sync_threads()
                if cutlass.const_expr(self.mxfp4):
                    # BF16 scores are determined by the high sixteen ordered-key bits.
                    barrier_phase = _coop_narrow_round(
                        merge_state,
                        group_id,
                        _FUSED_COARSE0,
                        s_c0_values,
                        carry_count,
                        Uint32(0),
                        Uint32(0),
                        coop_hist_addr,
                        s_hist0,
                        topk_static,
                        barrier_phase,
                        ctas_pg,
                        tx,
                        lane,
                        warp_idx,
                        s_coop_scalars,
                        24,
                        True,
                        threads=self.radix_threads,
                    )
                else:
                    barrier_phase = _coop_wide_round(
                        merge_state,
                        group_id,
                        _FUSED_COARSE0,
                        _FUSED_FINE0,
                        s_c0_values,
                        carry_count,
                        Uint32(0),
                        Uint32(0),
                        20,
                        wide_hist_addr,
                        coop_hist_addr,
                        s_hist0,
                        topk_static,
                        barrier_phase,
                        ctas_pg,
                        tx,
                        lane,
                        warp_idx,
                        s_coop_scalars,
                        True,
                        threads=self.radix_threads,
                    )
                total = Int32(s_coop_scalars[3])
                if total <= topk_static:
                    # Every candidate survives: pack at an atomic base; CTA 0 pads.
                    if tx == Int32(0):
                        s_coop_counters[1] = atomic_add_global_i32(
                            _fused_state_ptr(
                                merge_state, group_id, Int32(_FUSED_STATE_OUTPUT)
                            ),
                            carry_count,
                        )
                    cute.arch.sync_threads()
                    base = Int32(s_coop_counters[1])
                    i = Int32(tx)
                    while i < carry_count:
                        out_values[group_id, base + i] = Float32(s_c0_values[i])
                        out_indices[group_id, base + i] = Int32(s_c0_gindex[i])
                        i += Int32(self.radix_threads)
                    if cta_in_group == Int32(0):
                        i = total + Int32(tx)
                        while i < topk_static:
                            out_values[group_id, i] = Float32(float("-inf"))
                            out_indices[group_id, i] = Int32(-1)
                            i += Int32(self.radix_threads)
                else:
                    pivot_bin = Int32(s_coop_scalars[0])
                    count_gt = Int32(s_coop_scalars[1])
                    bin_count = Int32(s_coop_scalars[2])
                    if cutlass.const_expr(self.mxfp4):
                        prefix = Uint32(pivot_bin) << Uint32(24)
                        mask = Uint32(0xFF000000)
                    else:
                        prefix = Uint32(pivot_bin) << Uint32(20)
                        mask = Uint32(0xFFF00000)
                    winners = count_gt
                    remaining_k = topk_static - count_gt
                    done = Int32(0)
                    if bin_count == remaining_k:
                        done = Int32(1)
                    if done == Int32(0):
                        if cutlass.const_expr(self.mxfp4):
                            barrier_phase = _coop_narrow_round(
                                merge_state,
                                group_id,
                                _FUSED_COARSE1,
                                s_c0_values,
                                carry_count,
                                mask,
                                prefix,
                                coop_hist_addr,
                                s_hist0,
                                remaining_k,
                                barrier_phase,
                                ctas_pg,
                                tx,
                                lane,
                                warp_idx,
                                s_coop_scalars,
                                16,
                                threads=self.radix_threads,
                            )
                        else:
                            barrier_phase = _coop_wide_round(
                                merge_state,
                                group_id,
                                _FUSED_COARSE1,
                                _FUSED_FINE1,
                                s_c0_values,
                                carry_count,
                                mask,
                                prefix,
                                8,
                                wide_hist_addr,
                                coop_hist_addr,
                                s_hist0,
                                remaining_k,
                                barrier_phase,
                                ctas_pg,
                                tx,
                                lane,
                                warp_idx,
                                s_coop_scalars,
                                False,
                                threads=self.radix_threads,
                            )
                        pivot_bin = Int32(s_coop_scalars[0])
                        count_gt = Int32(s_coop_scalars[1])
                        bin_count = Int32(s_coop_scalars[2])
                        if cutlass.const_expr(self.mxfp4):
                            prefix = prefix | (Uint32(pivot_bin) << Uint32(16))
                            mask = Uint32(0xFFFF0000)
                            done = Int32(1)
                        else:
                            prefix = prefix | (Uint32(pivot_bin) << Uint32(8))
                            mask = Uint32(0xFFFFFF00)
                        winners = winners + count_gt
                        remaining_k = remaining_k - count_gt
                        if bin_count == remaining_k:
                            done = Int32(1)
                    if done == Int32(0):
                        barrier_phase = _coop_narrow_round(
                            merge_state,
                            group_id,
                            _FUSED_COARSE2,
                            s_c0_values,
                            carry_count,
                            mask,
                            prefix,
                            coop_hist_addr,
                            s_hist0,
                            remaining_k,
                            barrier_phase,
                            ctas_pg,
                            tx,
                            lane,
                            warp_idx,
                            s_coop_scalars,
                            threads=self.radix_threads,
                        )
                        pivot_bin = Int32(s_coop_scalars[0])
                        count_gt = Int32(s_coop_scalars[1])
                        bin_count = Int32(s_coop_scalars[2])
                        prefix = prefix | Uint32(pivot_bin)
                        mask = Uint32(0xFFFFFFFF)
                        winners = winners + count_gt
                        remaining_k = remaining_k - count_gt
                    # Count this CTA's winners and pivot ties, then reserve
                    # their group slots: winners fill [0, winners), ties fill
                    # [winners, topk) in arrival order.
                    if tx == Int32(0):
                        s_coop_counters[0] = Int32(0)
                        s_coop_counters[1] = Int32(0)
                    cute.arch.sync_threads()
                    i = Int32(tx)
                    while i < carry_count:
                        key = _convert_to_uint32(Float32(s_c0_values[i])) & mask
                        if key > prefix:
                            _smem_xadd(coop_ctr_addr, Int32(0), Int32(1))
                        else:
                            if key == prefix:
                                _smem_xadd(coop_ctr_addr, Int32(1), Int32(1))
                        i += Int32(self.radix_threads)
                    cute.arch.sync_threads()
                    if tx == Int32(0):
                        local_gt = Int32(s_coop_counters[0])
                        local_eq = Int32(s_coop_counters[1])
                        # One 64-bit fetch-add reserves the winner slots (low
                        # word) and the tie slots (high word).
                        reserved = atomic_add_global_u64(
                            _fused_state_ptr(
                                merge_state, group_id, Int32(_FUSED_STATE_OUTPUT)
                            ),
                            Int64(local_gt) + (Int64(local_eq) << Int64(32)),
                        )
                        s_coop_scalars[0] = Uint32(
                            (reserved & Int64(0xFFFFFFFF)).to(Int32)
                        )
                        s_coop_scalars[1] = Uint32(
                            winners + (reserved >> Int64(32)).to(Int32)
                        )
                        s_coop_counters[0] = Int32(0)
                        s_coop_counters[1] = Int32(0)
                    cute.arch.sync_threads()
                    gt_base = Int32(s_coop_scalars[0])
                    eq_base = Int32(s_coop_scalars[1])
                    i = Int32(tx)
                    while i < carry_count:
                        key = _convert_to_uint32(Float32(s_c0_values[i])) & mask
                        if key > prefix:
                            pos = gt_base + _smem_xadd(
                                coop_ctr_addr, Int32(0), Int32(1)
                            )
                            out_values[group_id, pos] = Float32(s_c0_values[i])
                            out_indices[group_id, pos] = Int32(s_c0_gindex[i])
                        else:
                            if key == prefix:
                                pos = eq_base + _smem_xadd(
                                    coop_ctr_addr, Int32(1), Int32(1)
                                )
                                if pos < topk_static:
                                    out_values[group_id, pos] = Float32(s_c0_values[i])
                                    out_indices[group_id, pos] = Int32(s_c0_gindex[i])
                        i += Int32(self.radix_threads)
                # Departure: once its output is placed, every CTA publishes
                # that it is done with the group state, and the last one
                # zeroes the whole block (round histograms and counters) so
                # the next launch or graph replay starts clean. The fence
                # orders this CTA's reservation and histogram traffic before
                # its departure becomes visible, so the zeroing can never
                # overtake a pending reservation.
                cute.arch.sync_threads()
                if tx == Int32(0):
                    threadfence()
                    s_relay[0] = atomic_add_global_i32(
                        _fused_state_ptr(
                            merge_state, group_id, Int32(_FUSED_STATE_CLEANUP)
                        ),
                        Int32(1),
                    )
                cute.arch.sync_threads()
                if Int32(s_relay[0]) == (ctas_pg - Int32(1)):
                    if cutlass.const_expr(self.mxfp4 and self.ordered_output):
                        threadfence()
                        _mxfp4_order_output(
                            out_indices,
                            out_values,
                            group_id,
                            s_c0_gindex,
                            s_c0_values,
                            tx,
                            Int32(512),
                            False,
                            merge_state,
                            _COOP_STATE_WORDS,
                            self.output_position_capacity,
                            self.vector_output_pairs,
                            threads=self.radix_threads,
                        )
                    else:
                        i = Int32(tx)
                        while i < Int32(_COOP_STATE_WORDS):
                            merge_state[_fused_state_offset(group_id, i)] = Int32(0)
                            i += Int32(self.radix_threads)
            else:
                # ---- in-kernel cross-CTA merge (relay) ----
                # Each CTA atomically packs its carry_count REAL candidates into the
                # per-group global slab; the last-arriving CTA radix-selects the final
                # top-k over the packed reals and writes the row (no host merge launch).
                # 1. reserve a contiguous slab slot for this CTA's real candidates.
                #    woff/arrival live in this group's coop-state block (via
                #    _FUSED_STATE_OUTPUT / _FUSED_STATE_ARRIVAL), so a group on
                #    this arm never collides with a sibling group on the coop arm.
                if cutlass.const_expr(
                    self.mxfp4
                    and not self.coop_merge_possible
                    and self.mxfp4_serial_state_words == 2
                ):
                    output_offset = Int64(group_id) * Int64(2)
                    arrival_offset = output_offset + Int64(1)
                else:
                    output_offset = _fused_state_offset(
                        group_id, Int32(_FUSED_STATE_OUTPUT)
                    )
                    arrival_offset = _fused_state_offset(
                        group_id, Int32(_FUSED_STATE_ARRIVAL)
                    )
                if tx == Int32(0):
                    woff_ptr = get_ptr_as_int64(merge_state, output_offset)
                    s_relay[0] = atomic_add_global_i32(woff_ptr, carry_count)
                cute.arch.sync_threads()
                off = Int32(s_relay[0])
                # 2. publish carry0[0:carry_count] into slab[slab_base+off : +count]
                i = Int32(tx)
                while i < carry_count:
                    pack_values[slab_base + off + i] = Float32(s_c0_values[i])
                    pack_indices[slab_base + off + i] = Int32(s_c0_gindex[i])
                    i += Int32(self.radix_threads)
                cute.arch.sync_threads()
                # 3. release the writes, then bump arrival; old==ctas-1 => I am last
                if tx == Int32(0):
                    threadfence()
                    arr_ptr = get_ptr_as_int64(merge_state, arrival_offset)
                    s_relay[1] = atomic_add_global_i32(arr_ptr, Int32(1))
                cute.arch.sync_threads()
                if Int32(s_relay[1]) == (ctas_pg - Int32(1)):
                    threadfence()  # acquire: observe every producer's packed writes
                    total = Int32(merge_state[output_offset])
                    if total > topk_static:
                        _fused_radix_select(
                            pack_v_row,
                            pack_i_row,
                            s_c1_values,
                            s_c1_gindex,
                            total,
                            topk_static,
                            self.cands,
                            tx,
                            s_hist0,
                            s_hist1,
                            s_out,
                            s_cand0,
                            s_cand1,
                            h0,
                            ctr,
                            thr,
                            ni0,
                            ni1,
                            lr,
                            threads=self.radix_threads,
                        )
                        cute.arch.sync_threads()
                        i = Int32(tx)
                        while i < topk_static:
                            out_values[group_id, i] = Float32(s_c1_values[i])
                            out_indices[group_id, i] = Int32(s_c1_gindex[i])
                            i += Int32(self.radix_threads)
                    else:
                        i = Int32(tx)
                        while i < topk_static:
                            if i < total:
                                out_values[group_id, i] = Float32(pack_v_row[i])
                                out_indices[group_id, i] = Int32(pack_i_row[i])
                            else:
                                out_values[group_id, i] = Float32(float("-inf"))
                                out_indices[group_id, i] = Int32(-1)
                            i += Int32(self.radix_threads)
                    if cutlass.const_expr(self.mxfp4 and self.ordered_output):
                        cute.arch.sync_threads()
                        _mxfp4_order_output(
                            out_indices,
                            out_values,
                            group_id,
                            s_c0_gindex,
                            s_c0_values,
                            tx,
                            Int32(512),
                            False,
                            merge_state,
                            0
                            if self.mxfp4_serial_state_words == 2
                            and not self.coop_merge_possible
                            else _COOP_STATE_WORDS,
                            self.output_position_capacity,
                            self.vector_output_pairs,
                            threads=self.radix_threads,
                        )
                    # reset the group's counters so the next launch / graph replay starts clean
                    cute.arch.sync_threads()
                    if tx == Int32(0):
                        merge_state[output_offset] = Int32(0)
                        merge_state[arrival_offset] = Int32(0)


@lru_cache(maxsize=64)
def _build_fused_indexer_kernel(
    kv_layout: int,
    num_heads_static: int,
    topk: int,
    paged_output: bool,
    ctas_per_group: int,
    num_sms: int = 0,
    merge_threshold: int = _LAST_CTA_MERGE_MAX,
    k_quant_page_stride: int = _PAGE_SIZE * _INDEX_HEAD_DIM,
    k_scales_row_stride: int = _PAGE_SIZE,
    max_seq_capacity: int = 1 << 30,
    vectorized_q_load: bool = False,
    q_row_stride_bytes: int = 0,
):
    return DSAFusedIndexerKernel(
        num_heads_static=num_heads_static,
        topk=topk,
        kv_layout=kv_layout,
        paged_output=paged_output,
        ctas_per_group=ctas_per_group,
        num_sms=num_sms,
        merge_threshold=merge_threshold,
        k_quant_page_stride=k_quant_page_stride,
        k_scales_row_stride=k_scales_row_stride,
        max_seq_capacity=max_seq_capacity,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )


def _arg_sig(*tensors: torch.Tensor):
    return tuple((tuple(t.shape), str(t.dtype)) for t in tensors)


def _fused_indexer_tensor_key(name: str, tensor: torch.Tensor) -> tuple[object, ...]:
    dynamic_row_names = {
        "q",
        "w",
        "pt",
        "sl",
        "kstart",
        "kend",
        "oi",
        "ov",
        "pv",
        "pi",
        "st",
    }
    dynamic_dims = (0,) if name in dynamic_row_names and int(tensor.ndim) >= 1 else ()
    return tensor_compile_fact(name, tensor, dynamic_dims=dynamic_dims)


def _launch_fused(kernel, cute_args, key_tensors, policy, *, launcher=None):
    """Graph-safe launch via b12x_launch (compile-once + replayable run_compiled).

    Unlike a bare cute.compile()+call, the b12x_launch path is CUDA-graph-capturable
    (the existing indexer kernels rely on it). Row-bearing key tensors use dynamic
    row dimensions; policy distinguishes the constexpr variant.  A prepared owner
    supplies the already-resolved launcher; this branch must never consult the
    compiler cache during serving replay.
    """
    # The variant string names the traced kernel body: the direct-K score
    # path and the cross-CTA merge arm (cooperative grid barrier, or the
    # serial last-CTA arm for a group that oversubscribes the SMs, see
    # coop_co_resident) are distinct traces. Its version suffix is bumped
    # whenever the traced body changes, so a cached cubin is never reused
    # for a different kernel.
    if kernel.direct_k_score:
        # direct-K needs ctas_per_group <= 16, co-resident on every real SM12x part,
        # so this keeps its historical key. The suffix keeps the key honest (no trace
        # collision) if a <16-SM device ever oversubscribes a direct-K build.
        variant = (
            "fused_indexer_v10_directk64"
            if kernel.coop_co_resident
            else "fused_indexer_v10_directk64_oversub_serial"
        )
    elif kernel.coop_co_resident:
        variant = "fused_indexer_v8_coop"
    else:
        variant = "fused_indexer_v8_oversub_serial"
    cache_key = tuple(_fused_indexer_tensor_key(name, t) for name, t in key_tensors) + (
        (variant,) + tuple(policy),
    )
    labels = tuple(name for name, _ in key_tensors) + ("policy",)
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.fused_indexer", 1, cache_key, labels=labels
    )
    if launcher is not None:
        from b12x._lib.compiler import run_compiled

        return run_compiled(launcher, cute_args)
    return b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=cute_args,
        runtime_args=cute_args,
    )


def _alloc_merge_scratch(rows: int, topk: int, ctas_per_group: int, dev):
    """Per-group merge state + pack slab for the in-kernel cross-CTA merge.

    ctas_per_group == 1 takes no merge path -> minimal dummies. Otherwise the
    runtime auto-switch can take either arm per group, so allocate BOTH: the
    per-group state at _COOP_STATE_WORDS int32 (cooperative histogram/barrier/
    output/total counters, which the last-CTA arm reuses for its woff/arrival
    words) and the global pack slab (rows * ctas_per_group * topk) the last-CTA
    arm packs into. State is zeroed each call; under CUDA-graph capture the memset
    is captured and replays clean (the per-launch reset).
    """
    if ctas_per_group <= 1:
        pack_v = torch.empty((1,), dtype=torch.float32, device=dev)
        pack_i = torch.empty((1,), dtype=torch.int32, device=dev)
        state = torch.zeros((2,), dtype=torch.int32, device=dev)
    else:
        cap = rows * ctas_per_group * int(topk)
        pack_v = torch.empty((cap,), dtype=torch.float32, device=dev)
        pack_i = torch.empty((cap,), dtype=torch.int32, device=dev)
        state = torch.zeros((rows * _COOP_STATE_WORDS,), dtype=torch.int32, device=dev)
    return pack_v, pack_i, state


def fused_indexer_scratch_capacity(
    max_rows: int, topk: int, num_sms: int
) -> tuple[int, int]:
    """Frugal FIXED capacity for the fused cross-CTA merge scratch -- for a workspace to
    allocate ONCE and reuse, never grow.

    The merge packs each cooperating CTA's local top-k (already trimmed to `topk`) into a
    per-group slab slot. With ctas_per_group = num_sms // rows, the total packed
    candidates = rows * ctas_per_group * topk <= num_sms * topk for ANY rows in
    [1, num_sms]. So the pack slab is bounded by num_sms * topk -- INDEPENDENT of batch
    size AND sequence length (the candidate count is capped by per-CTA top-k trimming,
    not seq_len). This is the whole point: unlike a per-seq gather buffer, the fused merge
    scratch never scales with context/max_model_len, so the frugal constant cap is exact.

    Returns (pack_elems, state_words): pack_values and pack_indices each need pack_elems
    (f32 / i32); merge_state needs state_words int32.
    """
    pack_elems = max(1, int(num_sms)) * int(topk)
    state_words = max(1, int(max_rows)) * _COOP_STATE_WORDS
    return pack_elems, state_words


def _can_vectorize_q_load(q_bytes: torch.Tensor) -> bool:
    if hasattr(q_bytes, "fake_mode") or q_bytes.is_meta:
        # Preparation represents the declared alignment as a storage offset.
        address = q_bytes.storage_offset() * q_bytes.element_size()
    else:
        address = int(q_bytes.data_ptr())
    return (
        address % 16 == 0
        and int(q_bytes.stride(2)) == 1
        and int(q_bytes.stride(1)) == _INDEX_HEAD_DIM
        and int(q_bytes.stride(0)) % 16 == 0
    )


def run_fused_paged_indexer(
    *,
    q_bytes: torch.Tensor,  # uint8 view of fp8 q, [rows, heads, 128]
    weights: torch.Tensor,  # f32, [rows, heads]
    k_quant_bytes: torch.Tensor,  # uint8, [pages, 64, 128]
    k_scales: torch.Tensor,  # f32, [pages, 64]
    real_page_table: torch.Tensor,  # i32, [rows, max_pages]
    seqlens: torch.Tensor,  # i32, [rows]
    num_heads: int,
    topk: int,
    out_indices: torch.Tensor | None = None,
    out_values: torch.Tensor | None = None,
    ctas_per_group: int | None = None,
    merge_threshold: int | None = None,
    pack_values: torch.Tensor | None = None,
    pack_indices: torch.Tensor | None = None,
    merge_state: torch.Tensor | None = None,
    merge_state_preinitialized: bool = False,
    output_physical_slots: bool = False,
    launcher=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged fused indexer. ctas_per_group>1 splits the row's K across cooperating CTAs.
    Returns (indices, values).

    pack_values/pack_indices/merge_state: optional caller-owned (workspace) scratch for
    serving. Size them with fused_indexer_scratch_capacity(max_rows, topk, num_sms) ONCE
    (fixed capacity, never grows -- the merge scratch is seq-independent). When omitted,
    scratch is allocated per call (benchmarks/tests). A caller-owned merge_state must
    either be zero-initialized once and pass merge_state_preinitialized=True, or it is
    zeroed here for backwards compatibility. Both merge arms self-reset every scalar
    they carry across launches, so a preinitialized state needs no replay-time memset.

    merge_threshold drives the per-group runtime cross-CTA merge auto-switch: a row
    whose live seq_len <= merge_threshold uses the serial last-CTA reduction (no grid
    barriers; faster for few-row/short-K); larger rows use the cooperative
    grid-barrier radix. 0 forces cooperative; a very large value forces last-CTA.
    A group that oversubscribes the SMs (ctas_per_group > num_sms) always takes the
    serial last-CTA arm regardless of merge_threshold: the cooperative grid barrier
    needs every CTA of the group co-resident (1 CTA/SM), impossible past num_sms."""
    rows = int(q_bytes.shape[0])
    dev = q_bytes.device
    max_pages = int(real_page_table.shape[1])
    if ctas_per_group is None:
        ctas_per_group = _resolve_default_ctas_per_group(
            num_rows=rows,
            max_pages=max_pages,
            device=dev,
        )
    ctas_per_group = max(1, int(ctas_per_group))
    if merge_threshold is None:
        merge_threshold = _resolve_default_merge_threshold(
            ctas_per_group=ctas_per_group,
            num_heads=int(num_heads),
            topk=int(topk),
        )

    out_i = (
        torch.empty((rows, topk), dtype=torch.int32, device=dev)
        if out_indices is None
        else out_indices
    )
    out_v = (
        torch.empty((rows, topk), dtype=torch.float32, device=dev)
        if out_values is None
        else out_values
    )
    if pack_values is not None and pack_indices is not None and merge_state is not None:
        # Workspace-owned fixed-capacity scratch. Capacity is validated against
        # fused_indexer_scratch_capacity by the workspace at allocation time.
        pack_need = rows * ctas_per_group * int(topk)
        state_need = rows * _COOP_STATE_WORDS
        if pack_values.numel() < pack_need or pack_indices.numel() < pack_need:
            raise ValueError(
                f"fused indexer pack scratch too small: need {pack_need}, have "
                f"{min(pack_values.numel(), pack_indices.numel())} "
                f"(size via fused_indexer_scratch_capacity)"
            )
        if merge_state.numel() < state_need:
            raise ValueError(
                f"fused indexer merge_state too small: need {state_need}, have "
                f"{merge_state.numel()} (size via fused_indexer_scratch_capacity)"
            )
        pack_v = pack_values.narrow(0, 0, pack_need)
        pack_i = pack_indices.narrow(0, 0, pack_need)
        state = merge_state.narrow(0, 0, state_need)
        if not bool(merge_state_preinitialized):
            state.zero_()
    else:
        pack_v, pack_i, state = _alloc_merge_scratch(rows, topk, ctas_per_group, dev)
    # Real dim-0 byte stride of the K-quant tensor: the packed paged cache interleaves
    # per-page scales (stride 8448), a plain [pages,64,128] view is contiguous (8192).
    # The wide g2s load must use this, not a hardcoded stride, to read the right page.
    k_quant_page_stride = int(k_quant_bytes.stride(0))
    k_scales_row_stride = int(k_scales.stride(0))
    vectorized_q_load = _can_vectorize_q_load(q_bytes)
    q_row_stride_bytes = int(q_bytes.stride(0)) if vectorized_q_load else 0
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    kernel = _build_fused_indexer_kernel(
        KV_LAYOUT_PAGED,
        int(num_heads),
        int(topk),
        bool(output_physical_slots),
        ctas_per_group,
        num_sms=int(num_sms),
        merge_threshold=int(merge_threshold),
        k_quant_page_stride=k_quant_page_stride,
        k_scales_row_stride=k_scales_row_stride,
        max_seq_capacity=max_pages * _PAGE_SIZE,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )
    # k_start/k_end are constexpr-dead in the PAGED kernel. Reuse the existing
    # int32 row metadata instead of capturing a pointless zero-fill kernel.
    unused_k_bounds = seqlens
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(unused_k_bounds, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(unused_k_bounds, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(state, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    key_tensors = [
        ("q", q_bytes),
        ("w", weights),
        ("kq", k_quant_bytes),
        ("ks", k_scales),
        ("pt", real_page_table),
        ("sl", seqlens),
        ("kstart", unused_k_bounds),
        ("kend", unused_k_bounds),
        ("oi", out_i),
        ("ov", out_v),
        ("pv", pack_v),
        ("pi", pack_i),
        ("st", state),
    ]
    _launch_fused(
        kernel,
        args,
        key_tensors,
        (
            KV_LAYOUT_PAGED,
            int(num_heads),
            int(topk),
            bool(output_physical_slots),
            int(ctas_per_group),
            int(merge_threshold),
            k_quant_page_stride,
            max_pages * _PAGE_SIZE,
            k_scales_row_stride,
            bool(vectorized_q_load),
            q_row_stride_bytes,
        ),
        launcher=launcher,
    )
    return out_i, out_v


def run_fused_indexer_mla(
    *,
    q_bytes: torch.Tensor,  # uint8 view of fp8 q, [rows, heads, 128]
    weights: torch.Tensor,  # f32, [rows, heads]
    k_quant_bytes: torch.Tensor,  # uint8, [k_rows, 128] (contiguous MLA K)
    k_scales: torch.Tensor,  # f32, [k_rows]
    k_start: torch.Tensor,  # i32, [rows]
    k_end: torch.Tensor,  # i32, [rows]
    num_heads: int,
    topk: int,
    out_indices: torch.Tensor | None = None,
    out_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v1 FLAT/MLA fused indexer over contiguous K with per-row [k_start, k_end). ctas=1."""
    rows = int(q_bytes.shape[0])
    dev = q_bytes.device
    out_i = (
        torch.empty((rows, topk), dtype=torch.int32, device=dev)
        if out_indices is None
        else out_indices
    )
    out_v = (
        torch.empty((rows, topk), dtype=torch.float32, device=dev)
        if out_values is None
        else out_values
    )
    pack_v, pack_i, state = _alloc_merge_scratch(rows, topk, 1, dev)
    vectorized_q_load = _can_vectorize_q_load(q_bytes)
    q_row_stride_bytes = int(q_bytes.stride(0)) if vectorized_q_load else 0
    kernel = _build_fused_indexer_kernel(
        KV_LAYOUT_CONTIGUOUS_MLA,
        int(num_heads),
        int(topk),
        False,
        1,
        vectorized_q_load=vectorized_q_load,
        q_row_stride_bytes=q_row_stride_bytes,
    )
    # page table / seqlens unused for FLAT; pass minimal dummies.
    dummy_pt = torch.zeros((rows, 1), dtype=torch.int32, device=dev)
    dummy_sl = torch.zeros((rows,), dtype=torch.int32, device=dev)
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(dummy_pt, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(dummy_sl, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_start, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_end, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_v, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(pack_i, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(state, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    key_tensors = [
        ("q", q_bytes),
        ("w", weights),
        ("kq", k_quant_bytes),
        ("ks", k_scales),
        ("pt", dummy_pt),
        ("sl", dummy_sl),
        ("kstart", k_start),
        ("kend", k_end),
        ("oi", out_i),
        ("ov", out_v),
        ("pv", pack_v),
        ("pi", pack_i),
        ("st", state),
    ]
    _launch_fused(
        kernel,
        args,
        key_tensors,
        (
            KV_LAYOUT_CONTIGUOUS_MLA,
            int(num_heads),
            int(topk),
            1,
            bool(vectorized_q_load),
            q_row_stride_bytes,
        ),
    )
    return out_i, out_v
