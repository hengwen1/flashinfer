"""Packed MXFP4 fused selection under mutable CUDA graph inputs."""

from dataclasses import dataclass

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.dsa_indexer import mxfp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="MXFP4 fused selection requires SM120",
)


def _decode(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    scale = (scales.int() << 23).view(torch.float32).repeat_interleave(32, -1)
    return (
        lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1, 1) * scale
    ).bfloat16()


@dataclass
class _Case:
    args: dict
    page_size: int
    width: int
    physical_base: int
    host_keys: torch.Tensor
    host_key_scales: torch.Tensor

    def mutate(self, step, visible, weight_kind):
        a = self.args
        rows, heads, _ = a["q_mxfp4"].shape
        generator = torch.Generator().manual_seed(41009 + step)
        packed = torch.randint(
            0, 256, (rows, heads, 64), dtype=torch.uint8, generator=generator
        )
        scales = (
            torch.tensor([124, 125, 126, 127], dtype=torch.uint8)
            .expand(rows, heads, 4)
            .clone()
        )
        scales = scales.roll(step % 4, -1)
        weights = torch.randn((rows, heads), generator=generator).bfloat16()
        weights *= torch.tensor([2**-8, 0.125, 8, 64], dtype=torch.bfloat16).repeat(
            heads // 4
        )
        if weight_kind == "negative":
            weights = -weights.abs()
        elif weight_kind == "zero":
            weights.zero_()
        a["q_mxfp4"].copy_(packed)
        a["q_scales"].copy_(scales)
        a["query_weights"].copy_(weights)
        lengths = torch.tensor(
            [max(0, visible - r * 11) for r in range(rows)], dtype=torch.int32
        )
        if rows > 2 and step % 2:
            lengths[-1] = -7
        a["cache_lengths"].copy_(lengths)
        a["active_width"].fill_(visible)
        page_rows, page_columns = a["page_table"].shape
        tables = torch.stack(
            [
                torch.randperm(page_columns, generator=generator).int()
                + self.physical_base
                for _ in range(page_rows)
            ]
        )
        if step % 2:
            tables[:, 0] = -1
            tables[:, 2] = a["index_k_cache"].shape[0]
            tables[:, -1] = 2147483647
        a["page_table"].copy_(tables)
        if a.get("candidate_indices") is not None:
            count = a["candidate_indices"].shape[1]
            candidates = torch.stack(
                [
                    torch.randperm(self.width, generator=generator)[:count]
                    for _ in range(rows)
                ]
            ).int()
            candidates[:, :4] = torch.tensor(
                [-1, self.width, 2147483647, -2147483648], dtype=torch.int32
            )
            a["candidate_indices"].copy_(candidates)
            counts = torch.tensor(
                [count - r * 13 for r in range(rows)], dtype=torch.int32
            )
            if visible <= 0:
                counts.zero_()
            a["candidate_lengths"].copy_(counts)
        return self.reference(packed, scales, weights, tables, lengths, visible)

    def reference(self, packed, scales, weights, tables, lengths, visible):
        query = _decode(packed, scales).float()
        keys = _decode(self.host_keys, self.host_key_scales).float()
        rows = query.shape[0]
        logical = torch.arange(self.width)
        candidates = self.args.get("candidate_indices")
        if candidates is not None:
            logical = candidates.cpu().long()
            candidate_lengths = self.args["candidate_lengths"].cpu()
        else:
            logical = logical.expand(rows, self.width)
        result = torch.full(logical.shape, -torch.inf)
        for row in range(rows):
            positions = logical[row]
            valid = (
                (positions >= 0) & (positions < lengths[row]) & (positions < visible)
            )
            valid &= positions // self.page_size < tables.shape[1]
            if candidates is not None:
                valid &= torch.arange(positions.numel()) < candidate_lengths[row]
            safe = positions.clamp(0, self.width - 1)
            pages = tables[
                0 if tables.shape[0] == 1 else row, safe // self.page_size
            ].long()
            valid &= (pages >= self.physical_base) & (
                pages < self.args["index_k_cache"].shape[0]
            )
            key_rows = (
                (pages - self.physical_base) * self.page_size + safe % self.page_size
            ).clamp(0, keys.shape[0] - 1)
            dots = (query[row] @ keys[key_rows].T).bfloat16().relu()
            products = (dots * weights[row, :, None]).bfloat16().float()
            total = torch.zeros(positions.numel())
            for head in range(query.shape[1]):
                total += products[head]
            result[row] = total.bfloat16().float().masked_fill(~valid, -torch.inf)
        return result, logical


def _case(
    *, page_size=64, gathered=False, shared=False, high=False, offset=False, width=2048
):
    device = torch.device("cuda", 0)
    rows, heads = 4, 32
    page_bytes = page_size * 68
    page_stride = page_bytes + 32
    base = 2**31 // page_stride + 19 if high else 0
    pages = width // page_size
    pool_backing = torch.empty(
        (base + pages, page_stride), dtype=torch.uint8, device=device
    )
    pool = pool_backing[:, :page_bytes]
    generator = torch.Generator().manual_seed(73031)
    keys = torch.randint(0, 256, (width, 64), dtype=torch.uint8, generator=generator)
    scales = (
        torch.tensor([127, 128, 129, 130], dtype=torch.uint8).expand(width, 4).clone()
    )
    pool[base:, : page_size * 64].copy_(keys.reshape(pages, page_size * 64))
    pool[base:, page_size * 64 :].copy_(scales.reshape(pages, page_size * 4))

    def empty(shape, dtype):
        count = 1
        for value in shape:
            count *= value
        storage = torch.empty(count + (4 if offset else 0), dtype=dtype, device=device)
        return storage[4 if offset else 0 :].view(shape)

    table_rows = 1 if shared else rows
    table_backing = torch.empty(
        (table_rows, pages + 3), dtype=torch.int32, device=device
    )
    args = dict(
        q_mxfp4=empty((rows, heads, 64), torch.uint8),
        q_scales=empty((rows, heads, 4), torch.uint8),
        query_weights=empty((rows, heads), torch.bfloat16),
        index_k_cache=pool,
        page_table=table_backing[:, :pages],
        cache_lengths=empty((rows,), torch.int32),
        active_width=empty((1,), torch.int32),
        output_indices=empty((rows, 512), torch.int32),
        output_scores=empty((rows, 512), torch.float32),
        page_size=page_size,
        ctas_per_row=2,
    )
    if gathered:
        args.update(
            candidate_indices=empty((rows, 1024), torch.int32),
            candidate_lengths=empty((rows,), torch.int32),
        )
    return _Case(args, page_size, width, base, keys, scales)


def _baseline_scores(case):
    a = case.args
    gathered = a.get("candidate_indices") is not None
    width = a["candidate_indices"].shape[1] if gathered else case.width
    output = torch.empty((4, width), dtype=torch.bfloat16, device="cuda:0")
    recipe = (32, gathered, case.page_size)
    launcher = mxfp4._compile("score_tensorcore", recipe, 0)
    mxfp4._launch(
        "score_tensorcore",
        recipe,
        (
            a["q_mxfp4"],
            a["q_scales"],
            a["query_weights"],
            a["index_k_cache"],
            a["page_table"],
            a["cache_lengths"],
            a["active_width"],
            a.get("candidate_indices", a["cache_lengths"]),
            a.get("candidate_lengths", a["cache_lengths"]),
            output,
        ),
        (
            4,
            width,
            a["page_table"].shape[1],
            0 if a["page_table"].shape[0] == 1 else a["page_table"].stride(0),
            a["index_k_cache"].stride(0),
            a["index_k_cache"].shape[0],
        ),
        launcher=launcher,
    )
    return output


def _assert_selected(case, expected, logical):
    indices = case.args["output_indices"].cpu().long()
    values = case.args["output_scores"].cpu()
    for row in range(indices.shape[0]):
        count = min(512, int(torch.isfinite(expected[row]).sum()))
        chosen = indices[row, :count]
        assert torch.all(chosen >= 0)
        assert torch.all(chosen[1:] >= chosen[:-1])
        assert torch.all(indices[row, count:] == -1)
        assert torch.all(values[row, count:] == -torch.inf)
        oracle = expected[row].topk(count).values.sort().values
        torch.testing.assert_close(
            values[row, :count].sort().values, oracle, rtol=0, atol=0
        )
        for slot, position in enumerate(chosen):
            matches = logical[row] == position
            assert bool(matches.any())
            associated = expected[row, matches]
            assert bool((associated == values[row, slot]).any())
        assert chosen.unique().numel() == chosen.numel()


@pytest.mark.parametrize(
    "page_size,gathered,shared", [(64, False, False), (128, True, True)]
)
def test_packed_score_oracle_matches_unchanged_native(page_size, gathered, shared):
    case = _case(page_size=page_size, gathered=gathered, shared=shared)
    for step, visible, kind in (
        (0, 2048, "mixed"),
        (1, 1771, "negative"),
        (2, 0, "zero"),
    ):
        expected, _ = case.mutate(step, visible, kind)
        actual = _baseline_scores(case).cpu().float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "page_size,gathered,shared,width",
    [
        (64, False, False, 2048),
        (128, False, True, 2048),
        (64, True, True, 2048),
        (128, True, False, 2048),
        (64, False, False, 32768),
    ],
)
def test_fused_mxfp4_mutable_capture(page_size, gathered, shared, width):
    case = _case(
        page_size=page_size, gathered=gathered, shared=shared, offset=True, width=width
    )
    case.mutate(0, case.width, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, width, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, width - 5, "zero"),
            (4, width, "negative"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in call.scratch:
                scratch.fill_(17)
            case.args["output_indices"].fill_(-17)
            case.args["output_scores"].fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[key] == after[key], key
            assert not bool(call.scratch[-1].count_nonzero())
            _assert_selected(case, expected, logical)
    graph.reset()


def test_fused_mxfp4_high_physical_byte_offsets():
    case = _case(high=True, gathered=True)
    expected, logical = case.mutate(1, case.width, "negative")
    assert case.physical_base * case.args["index_k_cache"].stride(0) > 2**31
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    call()
    torch.cuda.synchronize()
    _assert_selected(case, expected, logical)


@pytest.mark.parametrize(
    "name", ["q_mxfp4", "q_scales", "query_weights", "output_indices", "output_scores"]
)
def test_fused_mxfp4_noncontiguous_inputs_are_explicit(name):
    case = _case()
    expected, logical = case.mutate(0, case.width, "mixed")
    original = case.args[name]
    storage = torch.empty(
        (*original.shape[:-1], original.shape[-1] * 2),
        dtype=original.dtype,
        device=original.device,
    )
    strided = storage[..., ::2]
    strided.copy_(original)
    case.args[name] = strided
    try:
        call = mxfp4.prepare_fused_mxfp4(**case.args)
    except (TypeError, ValueError) as error:
        assert any(
            word in str(error).lower() for word in ("contiguous", "stride", "layout")
        )
    else:
        call()
        torch.cuda.synchronize()
        _assert_selected(case, expected, logical)


@pytest.mark.parametrize("name", ["q_mxfp4", "q_scales"])
def test_fused_mxfp4_rejects_misaligned_packed_loads(name):
    case = _case()
    case.mutate(0, case.width, "mixed")
    original = case.args[name]
    backing = torch.empty(
        original.numel() + 1, dtype=original.dtype, device=original.device
    )
    offset = backing[1:].view_as(original)
    offset.copy_(original)
    case.args[name] = offset
    with pytest.raises((TypeError, ValueError), match="align|offset|pointer"):
        mxfp4.prepare_fused_mxfp4(**case.args)


def test_fused_mxfp4_rejects_hierarchical_producer():
    case = _case()
    case.mutate(0, case.width, "mixed")
    with pytest.raises((TypeError, ValueError), match="candidate|producer|unsupported"):
        mxfp4.prepare_fused_mxfp4(
            **case.args,
            candidate_output=torch.empty(
                (4, 16384), dtype=torch.int32, device="cuda:0"
            ),
            candidate_output_lengths=torch.empty(
                (4,), dtype=torch.int32, device="cuda:0"
            ),
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two assigned CUDA devices"
)
def test_fused_mxfp4_restores_callers_cuda_device():
    case = _case(shared=True)
    expected, logical = case.mutate(0, case.width, "mixed")
    with torch.cuda.device(1):
        call = mxfp4.prepare_fused_mxfp4(**case.args)
        assert torch.cuda.current_device() == 1
        call()
        assert torch.cuda.current_device() == 1
        torch.cuda.synchronize(0)
        _assert_selected(case, expected, logical)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.device(0):
            stream.wait_stream(torch.cuda.current_stream())
            with kernel_resolution_guard(), torch.cuda.graph(graph, stream=stream):
                call()
        assert torch.cuda.current_device() == 1
        expected, logical = case.mutate(4, case.width - 7, "negative")
        torch.cuda.synchronize(0)
        with kernel_resolution_guard():
            call()
            assert torch.cuda.current_device() == 1
            graph.replay()
            assert torch.cuda.current_device() == 1
        torch.cuda.synchronize(0)
        _assert_selected(case, expected, logical)
        graph.reset()
        assert torch.cuda.current_device() == 1
