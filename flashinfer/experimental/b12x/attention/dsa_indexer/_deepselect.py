"""Prepared DeepSelect BF16 selection with logical-position output ordering."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["width"],
    do_not_specialize_on_alignment=[
        "lengths",
        "active",
        "candidate_lengths",
        "output",
    ],
)
def _selection_lengths(
    lengths, active, candidate_lengths, output, width, CANDIDATES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    extent = tl.minimum(
        tl.maximum(tl.load(lengths + row), 0), tl.maximum(tl.load(active), 0)
    )
    if CANDIDATES:
        extent = tl.maximum(tl.load(candidate_lengths + row), 0)
    tl.store(output + row, tl.minimum(extent, width))


@triton.jit(
    do_not_specialize=["width", "candidate_width"],
    do_not_specialize_on_alignment=[
        "scores",
        "indices",
        "candidates",
        "lengths",
        "active",
        "output",
        "output_values",
    ],
)
def _finish_selection(
    scores,
    indices,
    candidates,
    lengths,
    active,
    output,
    output_values,
    width,
    candidate_width,
    CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slots = tl.arange(0, 512)
    selected = tl.load(indices + row * 512 + slots)
    values = tl.load(
        scores + row * width + selected,
        (selected >= 0) & (selected < width),
        other=-float("inf"),
    ).to(tl.float32)
    positions = selected
    if CANDIDATES:
        positions = tl.load(
            candidates + row * candidate_width + selected,
            (selected >= 0) & (selected < candidate_width),
            other=-1,
        )
    visible = tl.minimum(
        tl.maximum(tl.load(lengths + row), 0), tl.maximum(tl.load(active), 0)
    )
    valid = (positions >= 0) & (positions < visible) & (values > -float("inf"))
    positions = tl.where(valid, positions, 2147483647)
    keys = (positions.to(tl.int64) << 32) | slots.to(tl.int64)
    keys = tl.sort(keys, descending=False)
    values = tl.gather(values, (keys & 0xFFFFFFFF).to(tl.int32), 0)
    positions = (keys >> 32).to(tl.int32)
    live = positions != 2147483647
    tl.store(output + row * 512 + slots, tl.where(live, positions, -1))
    tl.store(output_values + row * 512 + slots, tl.where(live, values, -float("inf")))


@dataclass(frozen=True)
class PreparedDeepSelect:
    module: object
    lengths: object
    finish: object
    candidates: bool

    @property
    def __b12x_dependencies__(self):
        return self.lengths, self.finish

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys((self.lengths, self.finish))

    def validate_device(self, device_index):
        with torch.cuda.device(device_index):
            if not self.module.sm120_topk512_available():
                raise RuntimeError(
                    "DeepSelect requires its SM120 BF16 top-512 CUDA kernel; "
                    "build deep-select==1.0.0+sm120.top512 for sm_120a"
                )

    def __call__(
        self,
        scores,
        indices,
        candidates,
        candidate_lengths,
        lengths,
        active,
        select_lengths,
        output,
        output_values,
    ):
        rows, width = scores.shape
        self.lengths[(rows, 1, 1)](
            lengths, active, candidate_lengths, select_lengths, width, self.candidates
        )
        self.module.topk(
            scores,
            512,
            None,
            select_lengths,
            False,
            False,
            None,
            indices,
            None,
            -1,
            -float("inf"),
            False,
            True,
        )
        self.finish[(rows, 1, 1)](
            scores,
            indices,
            candidates,
            lengths,
            active,
            output,
            output_values,
            width,
            candidates.shape[1],
            self.candidates,
        )


def prepare(candidates, device_index):
    from deep_select import deep_select_cuda

    with torch.cuda.device(device_index):
        lengths = _selection_lengths.warmup(
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            512,
            CANDIDATES=candidates,
            grid=(1, 1, 1),
            num_warps=4,
        )
        finish = _finish_selection.warmup(
            torch.bfloat16,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.float32,
            512,
            512,
            CANDIDATES=candidates,
            grid=(1, 1, 1),
            num_warps=4,
        )
    return PreparedDeepSelect(deep_select_cuda, lengths, finish, candidates)
