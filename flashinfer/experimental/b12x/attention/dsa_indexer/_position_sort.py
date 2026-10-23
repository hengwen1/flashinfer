"""Sort selected logical positions and expand hierarchical candidate blocks."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize_on_alignment=[
        "indices",
        "values",
        "output",
        "output_values",
        "lengths",
        "active_width",
        "output_lengths",
    ]
)
def _sort_positions(
    indices,
    values,
    output,
    output_values,
    lengths,
    active_width,
    output_lengths,
    K: tl.constexpr,
    EXPAND: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slots = tl.arange(0, K)
    positions = tl.load(indices + row * K + slots)
    scores = tl.load(values + row * K + slots)
    visible = tl.minimum(
        tl.maximum(tl.load(lengths + row), 0),
        tl.maximum(tl.load(active_width), 0),
    )
    valid = (positions >= 0) & (scores > -float("inf"))
    if EXPAND:
        valid = valid & (positions * 8 < visible)
    else:
        valid = valid & (positions < visible)
    positions = tl.where(valid, positions, 2147483647)
    if EXPAND:
        positions = tl.sort(positions, descending=False)
        tokens = positions[:, None].to(tl.int64) * 8 + tl.arange(0, 8)[None, :]
        live = (positions[:, None] != 2147483647) & (tokens < visible)
        offsets = (row * K + slots[:, None]) * 8 + tl.arange(0, 8)[None, :]
        tl.store(output + offsets, tl.where(live, tokens, -1))
        counts = tl.where(
            positions == 2147483647,
            0,
            tl.minimum(8, tl.maximum(visible - positions.to(tl.int64) * 8, 0)),
        )
        tl.store(output_lengths + row, tl.sum(counts, 0))
    else:
        # Retain each score's source slot while sorting its logical position.
        keys = (positions.to(tl.int64) << 32) | slots.to(tl.int64)
        keys = tl.sort(keys, descending=False)
        scores = tl.gather(scores, (keys & 0xFFFFFFFF).to(tl.int32), 0)
        positions = (keys >> 32).to(tl.int32)
        live = positions != 2147483647
        tl.store(output + row * K + slots, tl.where(live, positions, -1))
        tl.store(output_values + row * K + slots, tl.where(live, scores, -float("inf")))
