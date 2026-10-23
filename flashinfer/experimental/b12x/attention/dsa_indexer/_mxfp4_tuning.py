"""Concrete constructor choices for fused MXFP4 DSA indexing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class FusedMXFP4Config:
    ctas_per_row: int
    tile_schedule: str = "contiguous"
    grid_layout: str = "flat"
    ordered_output: bool = True
    state_reset_pdl: bool = False
    vector_output_pairs: bool = False
    mxfp4_warp_weights: bool = False
    mxfp4_aligned_q: bool = False
    mxfp4_query_offset_mode: int = 0
    mxfp4_unsigned_candidate_pages: bool = False
    mxfp4_merge_bits: int = 8
    mxfp4_histogram_count_bits: int = 16
    mxfp4_histogram_bitmap: bool = False
    mxfp4_serial_state_words: int = 8968
    mxfp4_small_output: bool = False
    mxfp4_partition_count: int = 1
    mxfp4_score_batch_tiles: int = 1
    mxfp4_cutoff_strategy: str = "local"
    mxfp4_threads: int = 1024
    mxfp4_score_passes: int = 1
    mxfp4_coop_residency: int = 1
    mxfp4_compact_shared: bool = False

    def __post_init__(self):
        if type(self.ctas_per_row) is not int or self.ctas_per_row <= 0:
            raise ValueError("ctas_per_row must be a positive integer")
        if self.tile_schedule not in ("contiguous", "cyclic"):
            raise ValueError("tile_schedule must be contiguous or cyclic")
        if self.grid_layout not in ("flat", "row_major", "slice_major"):
            raise ValueError("grid_layout must be flat, row_major, or slice_major")
        for name in (
            "mxfp4_small_output",
            "mxfp4_compact_shared",
            "ordered_output",
            "state_reset_pdl",
            "vector_output_pairs",
            "mxfp4_warp_weights",
            "mxfp4_aligned_q",
            "mxfp4_unsigned_candidate_pages",
            "mxfp4_histogram_bitmap",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name, values in (
            ("mxfp4_threads", (512, 1024)),
            ("mxfp4_coop_residency", (1, 2)),
            ("mxfp4_partition_count", (1, 2, 4)),
            ("mxfp4_score_batch_tiles", (1, 2)),
            ("mxfp4_query_offset_mode", (0, 1, 2)),
            ("mxfp4_merge_bits", (8, 16)),
            ("mxfp4_histogram_count_bits", (16, 32)),
            ("mxfp4_serial_state_words", (2, 8968)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in values
            ):
                raise ValueError(f"unsupported {name}")
        if (
            type(self.mxfp4_score_passes) is not int
            or not 1 <= self.mxfp4_score_passes <= 3072 // self.mxfp4_threads
        ):
            raise ValueError("MXFP4 score passes exceed carry append capacity")
        if self.mxfp4_cutoff_strategy not in ("local", "shared", "shared_compact"):
            raise ValueError("unsupported MXFP4 cutoff strategy")
        if self.mxfp4_coop_residency == 2 and self.mxfp4_threads != 512:
            raise ValueError("two-block MXFP4 residency requires 512 threads")
        if self.mxfp4_partition_count != 1 and (
            self.mxfp4_threads != 1024
            or self.mxfp4_compact_shared
            or self.mxfp4_score_passes != 1
            or self.tile_schedule != "contiguous"
            or self.mxfp4_small_output
        ):
            raise ValueError(
                "private MXFP4 partitions require contiguous full-shared 1024-thread scoring"
            )
        if self.mxfp4_partition_count == 1 and self.mxfp4_score_batch_tiles != 1:
            raise ValueError("paired score tiles require private MXFP4 partitions")
        if self.mxfp4_partition_count != 4 and self.mxfp4_cutoff_strategy != "local":
            raise ValueError("shared cutoff requires four MXFP4 partitions")
        if self.vector_output_pairs and not self.ordered_output:
            raise ValueError("paired stores require ordered output")
        if self.mxfp4_histogram_bitmap and self.mxfp4_histogram_count_bits != 16:
            raise ValueError("bitmap histograms require 16-bit counts")

    @classmethod
    def from_config(cls, payload):
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("MXFP4 fused configuration fields differ from schema")
        return cls(**dict(payload))

    def to_dict(self):
        return asdict(self)

    def constructor_options(self):
        return {
            "ctas_per_group": self.ctas_per_row,
            "mxfp4_small_output": self.mxfp4_small_output,
            "mxfp4_partition_count": self.mxfp4_partition_count,
            "mxfp4_score_batch_tiles": self.mxfp4_score_batch_tiles,
            "mxfp4_cutoff_strategy": self.mxfp4_cutoff_strategy,
            "mxfp4_threads": self.mxfp4_threads,
            "mxfp4_score_passes": self.mxfp4_score_passes,
            "mxfp4_coop_residency": self.mxfp4_coop_residency,
            "mxfp4_compact_shared": self.mxfp4_compact_shared,
            "tile_schedule": self.tile_schedule,
            "grid_layout": self.grid_layout,
            "ordered_output": self.ordered_output,
            "state_reset_pdl": self.state_reset_pdl,
            "vector_output_pairs": self.vector_output_pairs,
            "mxfp4_warp_weights": self.mxfp4_warp_weights,
            "mxfp4_aligned_q": self.mxfp4_aligned_q,
            "mxfp4_query_offset_mode": self.mxfp4_query_offset_mode,
            "mxfp4_unsigned_candidate_pages": self.mxfp4_unsigned_candidate_pages,
            "mxfp4_merge_bits": self.mxfp4_merge_bits,
            "mxfp4_histogram_count_bits": self.mxfp4_histogram_count_bits,
            "mxfp4_histogram_bitmap": self.mxfp4_histogram_bitmap,
            "mxfp4_serial_state_words": self.mxfp4_serial_state_words,
        }


def default_ctas(rows: int, width: int, sms: int) -> int:
    """Balance one-resident CTA waves against four/eight score-tile carries."""
    tiles = (width + 511) // 512
    ctas = max(1, min(tiles, sms // rows))
    if rows * ctas * 4 < sms * 3 and tiles >= 8:
        ctas = min(tiles, ctas * 4)
    else:
        depth = (tiles + ctas - 1) // ctas
        ctas = (tiles + depth - 1) // depth
    if ctas > 1 and rows * ctas > sms and tiles <= sms:
        carry = max(ctas, (tiles + 3) // 4)
        compact = max(ctas, (tiles + 7) // 8)
        carry_steps = ((rows * carry + sms - 1) // sms) * ((tiles + carry - 1) // carry)
        compact_steps = ((rows * compact + sms - 1) // sms) * (
            (tiles + compact - 1) // compact
        )
        ctas = compact if compact_steps <= carry_steps else carry
    return ctas


def configuration_values(query, device):
    width = query.max_candidates or query.max_page_table_width * query.page_size
    sms = device.sm_count
    tiles = (width + 511) // 512
    return tuple(
        sorted(
            {
                1,
                max(1, min(tiles, sms // query.max_q_rows)),
                max(1, min(tiles, 2 * sms // query.max_q_rows)),
                (tiles + 3) // 4,
                (tiles + 7) // 8,
                default_ctas(query.max_q_rows, width, sms),
            }
        )
    )


def default_fused_config(query, device, *, ctas_per_row=None):
    """Resolve a resource-based backup, including explicitly pinned CTA counts."""
    width = query.max_candidates or query.max_page_table_width * query.page_size
    sms = device.sm_count
    ctas = (
        default_ctas(query.max_q_rows, width, sms)
        if ctas_per_row is None
        else ctas_per_row
    )
    if getattr(query, "candidate_topk_blocks", 0):
        return FusedMXFP4Config(ctas_per_row=ctas)
    serial = query.max_q_rows * ctas > sms
    operands = query.operands
    aligned_q = all(
        operands.get(name) is not None and operands[name]["alignment"] >= 4
        for name in ("q_mxfp4", "q_scales")
    )
    return FusedMXFP4Config(
        ctas_per_row=ctas,
        tile_schedule="cyclic" if not serial and ctas > 1 else "contiguous",
        grid_layout="row_major"
        if query.max_q_rows <= 65535
        else "slice_major"
        if ctas <= 65535
        else "flat",
        state_reset_pdl=ctas > 1,
        mxfp4_aligned_q=aligned_q,
        mxfp4_serial_state_words=2 if serial else 8968,
        mxfp4_histogram_count_bits=16,
    )


def small_output_config(
    width: int, *, aligned_q: bool = False, ordered_output: bool = True
):
    """Record fixed small-output geometry and its executed load/store composition."""
    return FusedMXFP4Config(
        ctas_per_row=1,
        mxfp4_small_output=True,
        mxfp4_threads=512 if width == 512 else 1024,
        mxfp4_aligned_q=aligned_q,
        ordered_output=ordered_output,
    )
