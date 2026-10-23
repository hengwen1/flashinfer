"""Preparation tuning contract for native DSA indexer execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._efficiency import (
    capture_autotune_effort,
    capture_exhaustive_search,
)
from b12x.preparation.tuning import (
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

from ._mxfp4_tuning import (
    FusedMXFP4Config,
    configuration_values,
    default_fused_config,
    small_output_config,
)

FUSED_MERGE_AUTO = "auto"
FUSED_MERGE_COOPERATIVE = "cooperative"
FUSED_MERGE_SERIAL = "serial"
FUSED_MERGE_CHOICES = (FUSED_MERGE_AUTO, FUSED_MERGE_COOPERATIVE, FUSED_MERGE_SERIAL)
_BACKEND = "native"


@dataclass(frozen=True, kw_only=True)
class DsaIndexerQuery:
    source_layout: str
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    num_idx_heads: int
    max_q_rows: int
    max_k_rows: int
    top_k: int
    page_size: int
    score_mode: str
    shared_page_table: bool
    max_page_table_width: int
    route: str
    output_physical_slots: bool
    supertile_k: int
    prefill_block_k: int
    reserve_paged_logits: bool
    paged_logits_k_rows: int
    operands: FrozenMapping
    cache_format: str = "fp8"
    max_candidates: int = 0
    candidate_topk_blocks: int = 0
    exhaustive: bool = field(default_factory=capture_exhaustive_search)
    autotune_effort: str = field(default_factory=capture_autotune_effort)


@dataclass(frozen=True, kw_only=True)
class DsaIndexerConfig:
    backend: str
    fused_merge: str = FUSED_MERGE_AUTO
    mxfp4: FusedMXFP4Config | None = None

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "DsaIndexerConfig":
        if set(payload) != {"backend", "fused_merge", "mxfp4"}:
            raise ValueError(
                "DSA indexer config requires backend, fused_merge and mxfp4"
            )
        fused = payload["mxfp4"]
        return cls(
            backend=payload["backend"],
            fused_merge=payload["fused_merge"],
            mxfp4=None if fused is None else FusedMXFP4Config.from_config(fused),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "fused_merge": self.fused_merge,
            "mxfp4": None if self.mxfp4 is None else self.mxfp4.to_dict(),
        }


def _encode_query(query: DsaIndexerQuery) -> dict[str, object]:
    if not isinstance(query, DsaIndexerQuery):
        raise TypeError("query must be DsaIndexerQuery")
    return {name: getattr(query, name) for name in DsaIndexerQuery.__dataclass_fields__}


def _encode_config(config: DsaIndexerConfig) -> dict[str, object]:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    return config.to_dict()


def _validate_query(query: DsaIndexerQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, DsaIndexerQuery):
        raise TypeError("query must be DsaIndexerQuery")
    if query.source_layout not in ("paged", "contiguous"):
        raise ValueError("unsupported DSA source layout")
    if query.mode not in ("decode", "prefill"):
        raise ValueError("unsupported DSA mode")
    if query.route not in ("auto", "paged_fused", "paged_tiled", "packed_contiguous"):
        raise ValueError("unsupported DSA route")
    if query.page_size <= 0 or query.max_q_rows <= 0 or query.top_k <= 0:
        raise ValueError("DSA capacities must be positive")
    if not isinstance(query.operands, FrozenMapping):
        raise TypeError("DSA declarations require immutable operand metadata")
    if query.cache_format not in ("fp8", "mxfp4"):
        raise ValueError("unsupported DSA cache format")
    if query.cache_format == "mxfp4":
        if query.route not in ("auto", "paged_fused"):
            raise ValueError("MXFP4 route must be auto or paged_fused")
        if (
            query.top_k != 512
            or query.page_size % 8
            or not 0 < query.max_k_rows < 2**31
        ):
            raise ValueError(
                "MXFP4 requires logical top-k=512 and a positive page size"
            )
        if query.num_q_heads <= 0 or query.num_q_heads > 64 or 64 % query.num_q_heads:
            raise ValueError("MXFP4 index heads must divide 64")
        if (
            not 0 <= query.max_candidates <= 16384
            or query.candidate_topk_blocks not in (0, 2048)
        ):
            raise ValueError("unsupported MXFP4 candidate capacity")
        if query.max_candidates and query.candidate_topk_blocks:
            raise ValueError("MXFP4 source and reindex candidates are exclusive")


def _validate_config(query, config: DsaIndexerConfig, _device) -> None:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    if config.backend != _BACKEND:
        raise ValueError(
            f"unsupported attention.dsa_indexer backend {config.backend!r}"
        )
    if config.fused_merge not in FUSED_MERGE_CHOICES:
        raise ValueError(f"unsupported fused_merge {config.fused_merge!r}")

    if query.cache_format == "mxfp4":
        if not isinstance(config.mxfp4, FusedMXFP4Config):
            raise ValueError("MXFP4 requires a fused configuration")
        fused = config.mxfp4
        if query.candidate_topk_blocks and (
            config.fused_merge != FUSED_MERGE_AUTO
            or fused != FusedMXFP4Config(ctas_per_row=fused.ctas_per_row)
        ):
            raise ValueError("hierarchical MXFP4 supports only CTA configuration")
        width = query.max_candidates or query.max_page_table_width * query.page_size
        if fused.mxfp4_small_output and (
            config.fused_merge != FUSED_MERGE_AUTO
            or fused
            != small_output_config(
                width,
                aligned_q=fused.mxfp4_aligned_q,
                ordered_output=fused.ordered_output,
            )
        ):
            raise ValueError(
                "small-output MXFP4 requires canonical fixed geometry and neutral streaming options"
            )
        serial = (
            config.fused_merge == "serial"
            or query.max_q_rows * fused.ctas_per_row
            > _device.sm_count * fused.mxfp4_coop_residency
        )
        if config.fused_merge == "cooperative" and serial:
            raise ValueError("cooperative MXFP4 grid exceeds resident capacity")
        if fused.grid_layout == "row_major" and (
            query.max_q_rows > 65535 or fused.ctas_per_row > 2**31 - 1
        ):
            raise ValueError("row-major MXFP4 grid exceeds CUDA dimension limits")
        if fused.grid_layout == "slice_major" and (
            fused.ctas_per_row > 65535 or query.max_q_rows > 2**31 - 1
        ):
            raise ValueError("slice-major MXFP4 grid exceeds CUDA dimension limits")
        if (
            fused.grid_layout == "flat"
            and query.max_q_rows * fused.ctas_per_row > 2**31 - 1
        ):
            raise ValueError("flat MXFP4 grid exceeds CUDA dimension limits")
        if fused.mxfp4_merge_bits == 16 and (serial or fused.ctas_per_row == 1):
            raise ValueError("16-bit MXFP4 merge requires a cooperative multi-CTA grid")
        if fused.mxfp4_merge_bits == 8 and (
            fused.mxfp4_histogram_count_bits != 16 or fused.mxfp4_histogram_bitmap
        ):
            raise ValueError(
                "cooperative histogram policies require 16-bit MXFP4 merge"
            )
        if fused.mxfp4_serial_state_words == 2 and not serial:
            raise ValueError("compact MXFP4 state requires a statically serial grid")
        if fused.mxfp4_aligned_q and any(
            query.operands.get(name) is None or query.operands[name]["alignment"] < 4
            for name in ("q_mxfp4", "q_scales")
        ):
            raise ValueError(
                "aligned MXFP4 loads require declared four-byte Q and scale alignment"
            )
        if fused.vector_output_pairs and (
            query.max_k_rows > 65536
            or any(
                query.operands.get(name) is None
                or query.operands[name]["alignment"] < 8
                for name in ("output_indices", "output_scores")
            )
        ):
            raise ValueError(
                "paired output stores require eight-byte outputs and bounded positions"
            )
        if fused.mxfp4_partition_count != 1 and query.num_q_heads != 32:
            raise ValueError("private MXFP4 partitions require 32 query heads")
        if fused.mxfp4_small_output and (width > 512 or fused.ctas_per_row != 1):
            raise ValueError(
                "small MXFP4 output requires one CTA and capacity at most512"
            )
        if fused.ctas_per_row > max(1, (width + 511) // 512):
            raise ValueError("MXFP4 CTA count exceeds score tiles")
        if not query.candidate_topk_blocks:
            from types import SimpleNamespace
            from ._mxfp4_preparation import _kernel

            _kernel(query, config, SimpleNamespace(identity=_device))
    elif config.mxfp4 is not None:
        raise ValueError("MXFP4 fused configuration requires the MXFP4 cache recipe")


def _default(query, device) -> DsaIndexerConfig:
    return DsaIndexerConfig(
        backend=_BACKEND,
        mxfp4=default_fused_config(query, device)
        if query.cache_format == "mxfp4"
        else None,
    )


def _tuning_ctas(query: DsaIndexerQuery, device: DeviceIdentity | None) -> int:
    from .fused_indexer import resolve_fused_indexer_path
    from .kernel import _num_q_head_tiles

    if query.source_layout != "paged" or query.route in (
        "paged_tiled",
        "packed_contiguous",
    ):
        return 0
    if query.shared_page_table or query.mode == "prefill":
        if query.route == "paged_fused":
            raise ValueError("fused paged DSA is decode-only")
        return 0
    if device is None:
        raise ValueError("paged DSA tuning requires a device identity")
    supported = (
        os.getenv("B12X_FUSED_INDEXER", "1") != "0"
        and resolve_fused_indexer_path(
            topk=query.top_k,
            num_rows=query.max_q_rows,
            width=query.max_page_table_width * query.page_size,
            num_heads=query.num_q_heads,
            compute_capability=device.compute_capability,
        )
        and _num_q_head_tiles(query.num_q_heads) in (1, 2, 4)
    )
    if not supported:
        if query.route == "paged_fused":
            raise ValueError("fused paged DSA is unavailable for this query")
        return 0
    return max(
        1, min(query.max_page_table_width, device.sm_count // max(1, query.max_q_rows))
    )


def _equivalence(query, device, config):
    if query.cache_format == "mxfp4":
        serial = (
            config.fused_merge == "serial"
            or query.max_q_rows * config.mxfp4.ctas_per_row
            > device.sm_count * config.mxfp4.mxfp4_coop_residency
        )
        if query.candidate_topk_blocks:
            return {"hierarchical_ctas": config.mxfp4.ctas_per_row}
        return FrozenMapping(
            {
                "merge": "serial" if serial else "cooperative",
                "mxfp4": config.mxfp4.to_dict(),
            }
        )
    from .fused_indexer import resolve_fused_merge_threshold

    return resolve_fused_merge_threshold(
        config.fused_merge,
        ctas_per_group=_tuning_ctas(query, device),
        num_heads=query.num_q_heads,
        topk=query.top_k,
    )


def _parameters(query, device):
    choices = (None,)
    if query.cache_format == "mxfp4":
        configs = []
        for ctas in configuration_values(query, device):
            base = default_fused_config(query, device, ctas_per_row=ctas)
            if query.candidate_topk_blocks:
                configs.append(FrozenMapping(base.to_dict()))
                continue
            for schedule in ("contiguous", "cyclic") if ctas > 1 else ("contiguous",):
                grids = (
                    ("flat", "row_major", "slice_major")
                    if query.max_q_rows <= 65535
                    and ctas <= 65535
                    and query.max_q_rows * ctas <= 2**31 - 1
                    else (base.grid_layout,)
                )
                for grid in grids:
                    configs.append(
                        FrozenMapping(
                            replace(
                                base, tile_schedule=schedule, grid_layout=grid
                            ).to_dict()
                        )
                    )
        if not query.candidate_topk_blocks:
            base = default_fused_config(query, device)
            variants = [
                replace(base, mxfp4_warp_weights=not base.mxfp4_warp_weights),
                replace(base, mxfp4_query_offset_mode=1),
                replace(base, mxfp4_query_offset_mode=2),
                replace(base, ordered_output=False, vector_output_pairs=False),
            ]
            if query.max_candidates:
                variants.append(replace(base, mxfp4_unsigned_candidate_pages=True))
            if base.ctas_per_row > 1:
                variants.append(replace(base, state_reset_pdl=False))
                if query.max_q_rows * base.ctas_per_row <= device.sm_count:
                    variants.append(
                        replace(
                            base, mxfp4_merge_bits=16, mxfp4_histogram_count_bits=32
                        )
                    )
                    if base.ctas_per_row * 512 <= 8192:
                        variants.extend(
                            (
                                replace(base, mxfp4_merge_bits=16),
                                replace(
                                    base,
                                    mxfp4_merge_bits=16,
                                    mxfp4_histogram_bitmap=True,
                                ),
                            )
                        )
            if base.mxfp4_serial_state_words == 2:
                variants.append(replace(base, mxfp4_serial_state_words=8968))
            if base.mxfp4_aligned_q:
                variants.append(replace(base, mxfp4_aligned_q=False))
            if query.max_k_rows <= 65536 and all(
                query.operands.get(name) is not None
                and query.operands[name]["alignment"] >= 8
                for name in ("output_indices", "output_scores")
            ):
                variants.append(replace(base, vector_output_pairs=True))
            if query.num_q_heads == 32:
                for partitions in (2, 4):
                    for batch in (1, 2):
                        for cutoff in (
                            ("local",)
                            if partitions == 2
                            else ("local", "shared", "shared_compact")
                        ):
                            variants.append(
                                replace(
                                    base,
                                    tile_schedule="contiguous",
                                    mxfp4_partition_count=partitions,
                                    mxfp4_score_batch_tiles=batch,
                                    mxfp4_cutoff_strategy=cutoff,
                                )
                            )
            if (
                base.ctas_per_row == 1
                and (query.max_candidates or query.max_k_rows) <= 512
            ):
                variants.append(
                    small_output_config(
                        query.max_candidates or query.max_k_rows,
                        aligned_q=base.mxfp4_aligned_q,
                    )
                )
            for passes in (1, 2, 4):
                variants.append(
                    replace(base, mxfp4_threads=512, mxfp4_score_passes=passes)
                )
                variants.append(
                    replace(
                        base,
                        mxfp4_threads=512,
                        mxfp4_score_passes=passes,
                        mxfp4_compact_shared=True,
                    )
                )
            for ctas in configuration_values(query, device):
                if device.sm_count < query.max_q_rows * ctas <= device.sm_count * 2:
                    variants.append(
                        replace(
                            default_fused_config(query, device, ctas_per_row=ctas),
                            mxfp4_threads=512,
                            mxfp4_compact_shared=True,
                            mxfp4_coop_residency=2,
                            mxfp4_score_passes=2,
                            mxfp4_serial_state_words=8968,
                        )
                    )
            configs.extend(FrozenMapping(config.to_dict()) for config in variants)
        choices = tuple(dict.fromkeys(configs))
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "backend": (_BACKEND,),
            "fused_merge": (FUSED_MERGE_AUTO,),
            "mxfp4": choices,
        },
        exhaustive=query.exhaustive,
        autotune_effort=query.autotune_effort,
    )


TUNING = TuningContract(
    component_id="attention.dsa_indexer",
    query_schema_version=24,
    config_schema_version=24,
    query_fields=frozenset(DsaIndexerQuery.__dataclass_fields__),
    config_fields=frozenset(DsaIndexerConfig.__dataclass_fields__),
    encode_query=_encode_query,
    encode_config=_encode_config,
    decode_config=DsaIndexerConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default,
    knobs=(
        Knob(name="backend", values=(_BACKEND,), binding=ParameterBinding.COMPILE),
        Knob(name="fused_merge", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="mxfp4", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=8,
    equivalence_key=_equivalence,
    parameters=_parameters,
    materialize=lambda query, device, choice: DsaIndexerConfig.from_config(choice),
)

__all__ = [
    "DsaIndexerConfig",
    "DsaIndexerQuery",
    "TUNING",
    "FUSED_MERGE_AUTO",
    "FUSED_MERGE_CHOICES",
    "FUSED_MERGE_COOPERATIVE",
    "FUSED_MERGE_SERIAL",
]
