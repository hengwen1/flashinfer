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


@pytest.mark.parametrize("page_size,gathered", [(64, False), (128, True)])
@pytest.mark.parametrize("ctas_per_row", [1, 2])
@pytest.mark.parametrize(
    "output_offsets,output_form",
    [
        ((1, 1), None),
        ((1, 2), None),
        ((2, 1), None),
        ((2, 2), None),
        ((2, 2), False),
        ((2, 2), True),
    ],
)
def test_fused_mxfp4_mutable_capture_four_byte_output_views(
    page_size, gathered, ctas_per_row, output_offsets, output_form
):
    case = _case(page_size=page_size, gathered=gathered)
    case.args["ctas_per_row"] = ctas_per_row
    case.args["_vector_output_pairs"] = output_form
    for name, offset in zip(
        ("output_indices", "output_scores"), output_offsets, strict=True
    ):
        original = case.args[name]
        backing = torch.empty(
            original.numel() + offset, dtype=original.dtype, device=original.device
        )
        output = backing[offset:].view_as(original)
        assert output.data_ptr() % 16 == offset * 4
        case.args[name] = output
    case.mutate(0, case.width, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, case.width, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, case.width - 5, "zero"),
            (4, case.width, "negative"),
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


@pytest.mark.parametrize("output_form", [False, True])
def test_fused_mxfp4_capture_after_query_producer(output_form):
    case = _case(page_size=128, width=32768)
    case.args["_vector_output_pairs"] = output_form
    case.mutate(0, case.width, "mixed")
    a = case.args
    query = torch.zeros((4, 32, 128), dtype=torch.bfloat16, device=a["q_mxfp4"].device)
    weights = torch.empty_like(a["query_weights"])
    weights.copy_(a["query_weights"])
    quantizer = mxfp4._compile("quantize", (False, 64), query.device.index)

    def quantize():
        mxfp4._quantize_q_mxfp4(
            query, q_mxfp4=a["q_mxfp4"], q_scales=a["q_scales"], launcher=quantizer
        )

    quantize()
    call = mxfp4.prepare_fused_mxfp4(**a)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            a["query_weights"].copy_(weights)
            quantize()
            call()
        for step, visible, kind in (
            (0, case.width, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, case.width - 5, "mixed"),
        ):
            case.mutate(step, visible, kind)
            weights.copy_(a["query_weights"])
            generator = torch.Generator().manual_seed(7749 + step)
            query.copy_((torch.randn(query.shape, generator=generator) / 4).bfloat16())
            quantize()
            expected_packed = a["q_mxfp4"].cpu()
            expected_scales = a["q_scales"].cpu()
            expected_weights = weights.cpu()
            expected, logical = case.reference(
                expected_packed,
                expected_scales,
                expected_weights,
                a["page_table"].cpu(),
                a["cache_lengths"].cpu(),
                visible,
            )
            for scratch in call.scratch:
                scratch.fill_(17)
            a["q_mxfp4"].fill_(0xA5)
            a["q_scales"].fill_(0xA5)
            a["query_weights"].fill_(float("nan"))
            a["output_indices"].fill_(-17)
            a["output_scores"].fill_(float("nan"))
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
            torch.testing.assert_close(
                a["q_mxfp4"].cpu(), expected_packed, rtol=0, atol=0
            )
            torch.testing.assert_close(
                a["q_scales"].cpu(), expected_scales, rtol=0, atol=0
            )
            torch.testing.assert_close(
                a["query_weights"].cpu(), expected_weights, rtol=0, atol=0
            )
            assert not bool(call.scratch[-1].count_nonzero())
            _assert_selected(case, expected, logical)
    graph.reset()


@pytest.mark.parametrize(
    "width,output_form", [(65536, False), (65536, True), (65664, False)]
)
def test_fused_mxfp4_capture_high_logical_positions(width, output_form):
    case = _case(page_size=128, gathered=True, width=width)
    case.args["_vector_output_pairs"] = output_form
    case.mutate(0, width, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    call()
    torch.cuda.synchronize()
    high = torch.arange(width - 16, width, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, width, "mixed"),
            (1, 0, "zero"),
            (2, width, "negative"),
        ):
            case.mutate(step, visible, kind)
            a = case.args
            a["candidate_indices"][:, :16].copy_(high.expand(4, -1))
            a["candidate_lengths"].fill_(16)
            expected, logical = case.reference(
                a["q_mxfp4"].cpu(),
                a["q_scales"].cpu(),
                a["query_weights"].cpu(),
                a["page_table"].cpu(),
                a["cache_lengths"].cpu(),
                visible,
            )
            for scratch in call.scratch:
                scratch.fill_(17)
            a["output_indices"].fill_(-17)
            a["output_scores"].fill_(float("nan"))
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
            if visible == width:
                assert torch.equal(a["output_indices"][0, :16].cpu(), high)
    graph.reset()


@pytest.mark.parametrize("width,offset", [(2048, 1), (65664, 0)])
def test_fused_mxfp4_vector_output_form_requires_eligible_storage(width, offset):
    case = _case(page_size=128, gathered=True, width=width)
    case.args["_vector_output_pairs"] = True
    if offset:
        original = case.args["output_indices"]
        backing = torch.empty(
            original.numel() + offset, dtype=original.dtype, device=original.device
        )
        case.args["output_indices"] = backing[offset:].view_as(original)
    with pytest.raises(ValueError, match="paired MXFP4 output stores require"):
        mxfp4.prepare_fused_mxfp4(**case.args)


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


@pytest.mark.parametrize(
    "page_size,gathered,ctas,bits,count_bits,bitmap,serial_words,paired,one_row",
    [
        (64, False, 2, 8, 16, False, 2, False, False),
        (128, True, 2, 8, 16, False, 8968, True, False),
        (64, False, 2, 8, 16, False, 2, True, True),
        (128, False, 17, 16, 32, False, 8968, False, False),
        (64, True, 13, 16, 16, False, 8968, True, False),
        (128, False, 16, 16, 16, True, 8968, False, False),
        (64, True, 16, 16, 16, True, 8968, True, False),
    ],
)
@pytest.mark.parametrize("threads", [1024, 512])
def test_fused_mxfp4_explicit_merge_state_capture(
    monkeypatch,
    threads,
    page_size,
    gathered,
    ctas,
    bits,
    count_bits,
    bitmap,
    serial_words,
    paired,
    one_row,
):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel
    observed = []

    def configured_kernel(**kwargs):
        if bits == 8:
            kwargs["merge_threshold"] = kwargs["score_width"]
        kwargs.update(
            mxfp4_threads=threads,
            mxfp4_merge_bits=bits,
            mxfp4_histogram_count_bits=count_bits,
            mxfp4_histogram_bitmap=bitmap,
            mxfp4_serial_state_words=serial_words,
        )
        kernel = constructor(**kwargs)
        observed.append(kernel)
        return kernel

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)
    case = _case(page_size=page_size, gathered=gathered, shared=gathered)
    a = case.args
    a["ctas_per_row"] = ctas
    a["_vector_output_pairs"] = paired
    if one_row:
        for key in (
            "q_mxfp4",
            "q_scales",
            "query_weights",
            "cache_lengths",
            "output_indices",
            "output_scores",
            "page_table",
        ):
            a[key] = a[key][:1]
    case.mutate(0, case.width, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**a)
    pack, state = observed[-1].merge_scratch_elements(a["q_mxfp4"].shape[0])
    assert call.scratch[2].numel() == pack
    assert call.scratch[-1].numel() == state
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, case.width, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, case.width - 5, "zero"),
            (4, case.width, "negative"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in call.scratch:
                scratch.fill_(17)
            a["output_indices"].fill_(-17)
            a["output_scores"].fill_(float("nan"))
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


@pytest.mark.parametrize("count_bits,bitmap", [(32, False), (16, False), (16, True)])
@pytest.mark.parametrize("threads", [1024, 512])
def test_fused_mxfp4_histogram_counter_edges(monkeypatch, count_bits, bitmap, threads):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(
            mxfp4_threads=threads,
            mxfp4_merge_bits=16,
            mxfp4_histogram_count_bits=count_bits,
            mxfp4_histogram_bitmap=bitmap,
        )
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)

    for page in (64, 128):
        width = 8192
        case = _case(
            page_size=page, gathered=False, shared=False, offset=True, width=width
        )
        a = case.args
        a["ctas_per_row"] = 16
        rows, heads, _ = a["q_mxfp4"].shape
        pages = width // page
        q = torch.full((rows, heads, 64), 0x11, dtype=torch.uint8)
        sf = torch.full((rows, heads, 4), 127, dtype=torch.uint8)
        tables = torch.arange(pages, dtype=torch.int32).expand(rows, -1).clone()
        lengths = torch.full((rows,), width, dtype=torch.int32)
        case.host_key_scales.fill_(127)

        def mutate(odd_count, sign):
            case.host_keys.fill_(0x11)
            case.host_keys[:odd_count, 0] = 0x12
            pool = a["index_k_cache"][case.physical_base :]
            pool[:, : page * 64].copy_(case.host_keys.reshape(pages, page * 64))
            pool[:, page * 64 :].copy_(case.host_key_scales.reshape(pages, page * 4))
            w = torch.zeros((rows, heads), dtype=torch.bfloat16)
            w[:, 0] = sign / 32
            for name, src in [
                ("q_mxfp4", q),
                ("q_scales", sf),
                ("query_weights", w),
                ("page_table", tables),
                ("cache_lengths", lengths),
            ]:
                a[name].copy_(src)
            a["active_width"].fill_(width)
            expected, logical = case.reference(q, sf, w, tables, lengths, width)
            if sign:
                assert bool(expected.eq(sign * 1.0).sum(-1).eq(width - odd_count).all())
                assert bool(expected.eq(sign * 1.0078125).sum(-1).eq(odd_count).all())
            else:
                assert bool(expected.eq(0).all())
            return expected, logical

        mutate(0, 1)
        call = mxfp4.prepare_fused_mxfp4(**a)
        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard():
            with torch.cuda.graph(graph):
                call()
            for sign in (1, -1):
                for odd in (0, 8192, 4096, 1, 8191):
                    expected, logical = mutate(odd, sign)
                    for t in call.scratch:
                        t.fill_(17)
                    a["output_indices"].fill_(-17)
                    a["output_scores"].fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in [
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ]:
                        assert before[key] == after[key], key
                    assert int(call.scratch[-1].count_nonzero()) == 0, (
                        page,
                        sign,
                        odd,
                        "dirty state",
                    )
                    _assert_selected(case, expected, logical)
            expected, logical = mutate(0, 0)
            for t in call.scratch:
                t.fill_(17)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert int(call.scratch[-1].count_nonzero()) == 0
            _assert_selected(case, expected, logical)
        graph.reset()


@pytest.mark.parametrize("threads", [1024, 512])
def test_fused_mxfp4_histogram_bitmap_cleanup_waves(monkeypatch, threads):
    count_bits, bitmap = 16, True
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(
            mxfp4_threads=threads,
            mxfp4_merge_bits=16,
            mxfp4_histogram_count_bits=count_bits,
            mxfp4_histogram_bitmap=bitmap,
        )
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)

    width = 8192
    page = 64
    case = _case(page_size=page, gathered=False, shared=False, offset=True, width=width)
    a = case.args
    a["ctas_per_row"] = 16
    rows, heads, _ = a["q_mxfp4"].shape
    pages = width // page
    q = torch.zeros((rows, heads, 64), dtype=torch.uint8)
    q[:, 0, :32] = 0x11
    q[:, 1, 32:] = 0x11
    sf = torch.full((rows, heads, 4), 127, dtype=torch.uint8)
    w = torch.zeros((rows, heads), dtype=torch.bfloat16)
    w[:, 0] = 1 / 16
    w[:, 1] = -1 / 16
    tables = torch.arange(pages, dtype=torch.int32).expand(rows, -1).clone()
    lengths = torch.full((rows,), width, dtype=torch.int32)

    def mutate(span):
        n = 4 * (2 * span + 1)
        code = torch.arange(width) % n
        negative = code // (2 * (2 * span + 1)) != 0
        local = code % (2 * (2 * span + 1))
        mantissa = local % 2
        exponent = local // 2 - span
        case.host_keys.zero_()
        for neg in (False, True):
            for frac in (0, 1):
                mask = (negative == neg) & (mantissa == frac)
                case.host_keys[mask, 32 if neg else 0 : 64 if neg else 32] = (
                    0x33 if frac else 0x11
                )
        case.host_key_scales.copy_(
            (exponent + 127 - mantissa).to(torch.uint8).unsqueeze(-1).expand(-1, 4)
        )
        pool = a["index_k_cache"][case.physical_base :]
        pool[:, : page * 64].copy_(case.host_keys.reshape(pages, page * 64))
        pool[:, page * 64 :].copy_(case.host_key_scales.reshape(pages, page * 4))
        for name, src in [
            ("q_mxfp4", q),
            ("q_scales", sf),
            ("query_weights", w),
            ("page_table", tables),
            ("cache_lengths", lengths),
        ]:
            a[name].copy_(src)
        a["active_width"].fill_(width)
        expected, logical = case.reference(q, sf, w, tables, lengths, width)
        direct = (
            torch.exp2(exponent.float())
            * (1 + mantissa.float() / 2)
            * torch.where(negative, -1.0, 1.0)
        )
        assert bool(expected.eq(direct).all())
        bits = expected[0].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        ordered = torch.where(
            (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
        )
        unique = (ordered >> 22).unique().numel()
        assert unique == n and unique > 32, (span, unique, n)
        return expected, logical, unique

    mutate(40)
    call = mxfp4.prepare_fused_mxfp4(**a)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for span in (40, 37, 20):
            expected, logical, n = mutate(span)
            for t in call.scratch:
                t.fill_(17)
            a["output_indices"].fill_(-17)
            a["output_scores"].fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in [
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ]:
                assert before[key] == after[key], key
            assert int(call.scratch[-1].count_nonzero()) == 0, (span, "dirty state")
            _assert_selected(case, expected, logical)
    graph.reset()


@pytest.mark.parametrize(
    "ctas,count_bits,bitmap,message",
    [(17, 16, False, "8192"), (16, 32, True, "bitmap"), (16, 8, False, "16 or 32")],
)
def test_fused_mxfp4_histogram_eligibility(ctas, count_bits, bitmap, message):
    from b12x.attention.dsa_indexer.fused_indexer import DSAFusedIndexerKernel

    with pytest.raises(ValueError, match=message):
        DSAFusedIndexerKernel(
            num_heads_static=32,
            topk=512,
            ctas_per_group=ctas,
            num_sms=188,
            merge_threshold=0,
            max_seq_capacity=8192,
            mxfp4=True,
            mxfp4_merge_bits=16,
            mxfp4_histogram_count_bits=count_bits,
            mxfp4_histogram_bitmap=bitmap,
        )


@pytest.mark.parametrize(
    "partitions,batches,cutoff,page_size,gathered,ctas,wide_positions",
    [
        (2, 1, "local", 64, False, 1, False),
        (2, 2, "local", 128, True, 2, False),
        (4, 1, "local", 128, True, 1, False),
        (4, 2, "local", 64, False, 2, False),
        (4, 2, "shared", 128, False, 1, False),
        (4, 2, "shared_compact", 128, True, 1, False),
        (4, 2, "shared_compact", 64, False, 1, False),
        (4, 2, "shared_compact", 64, True, 2, True),
    ],
)
def test_private_carry_capture(
    monkeypatch, partitions, batches, cutoff, page_size, gathered, ctas, wide_positions
):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def private_kernel(**kwargs):
        return constructor(
            **kwargs,
            mxfp4_partition_count=partitions,
            mxfp4_score_batch_tiles=batches,
            mxfp4_cutoff_strategy=cutoff,
        )

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", private_kernel)
    width = 131072 if wide_positions else 32768
    case = _case(page_size=page_size, gathered=gathered, width=width)
    if gathered:
        case.args["candidate_indices"] = torch.empty(
            (4, min(width, 32768)), dtype=torch.int32, device="cuda"
        )
    case.mutate(0, width, "mixed")
    case.args["ctas_per_row"] = ctas
    call = mxfp4.prepare_fused_mxfp4(
        **case.args, _vector_output_pairs=not wide_positions
    )
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, width, "mixed"),
            (1, 1025, "negative"),
            (2, 0, "zero"),
            (3, width - 5, "zero"),
            (4, 513, "mixed"),
            (5, 767, "negative"),
            (6, 1537, "mixed"),
            (7, width, "negative"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in call.scratch:
                scratch.fill_(17)
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


def test_private_carry_constructor_eligibility():
    from b12x.attention.dsa_indexer.fused_indexer import DSAFusedIndexerKernel

    for kwargs in (
        dict(mxfp4=False, mxfp4_partition_count=2),
        dict(mxfp4=True, mxfp4_partition_count=3),
        dict(mxfp4=True, mxfp4_partition_count=2, mxfp4_cutoff_strategy="shared"),
        dict(mxfp4=True, mxfp4_score_batch_tiles=2),
    ):
        with pytest.raises(ValueError):
            DSAFusedIndexerKernel(num_heads_static=32, topk=512, **kwargs)
    with pytest.raises(ValueError):
        DSAFusedIndexerKernel(
            num_heads_static=16, topk=512, mxfp4=True, mxfp4_partition_count=4
        )


@pytest.mark.parametrize("partitions", [2, 4])
def test_private_carry_isolated_partition_replay(monkeypatch, partitions):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def private_kernel(**kwargs):
        return constructor(
            **kwargs,
            mxfp4_partition_count=partitions,
            mxfp4_score_batch_tiles=2,
            mxfp4_cutoff_strategy="shared_compact" if partitions == 4 else "local",
        )

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", private_kernel)
    case = _case(page_size=128, gathered=True, shared=True, width=32768)
    a = case.args
    a["ctas_per_row"] = 1
    a["candidate_indices"] = torch.empty(
        (4, case.width), dtype=torch.int32, device="cuda"
    )
    columns = torch.arange(case.width, device="cuda")
    owner = (columns % 512) // (512 // partitions)

    def mutate(step, partition, kind):
        case.mutate(step, case.width, kind)
        a["candidate_indices"][:, owner != partition] = -1
        return case.reference(
            a["q_mxfp4"].cpu(),
            a["q_scales"].cpu(),
            a["query_weights"].cpu(),
            a["page_table"].cpu(),
            a["cache_lengths"].cpu(),
            case.width,
        )

    mutate(0, 0, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**a)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for partition in range(partitions):
            for kind in ("mixed", "negative", "zero"):
                expected, logical = mutate(2 * partition, partition, kind)
                for scratch in call.scratch:
                    scratch.fill_(17)
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                _assert_selected(case, expected, logical)
    graph.reset()


@pytest.mark.parametrize("gathered", [False, True])
@pytest.mark.parametrize("width", [256, 512])
def test_fused_mxfp4_small_output_capture(monkeypatch, gathered, width):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(mxfp4_small_output=True, ctas_per_group=1)
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)
    if not gathered:
        test_fused_mxfp4_mutable_capture(64, False, True, width)
        return
    case = _case(page_size=64, gathered=True, shared=True, width=2048)
    case.args["candidate_indices"] = case.args["candidate_indices"][:, :width].clone()
    case.args["ctas_per_row"] = 1
    case.mutate(0, 2048, "mixed")
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, 2048, "mixed"),
            (1, 73, "negative"),
            (2, 0, "zero"),
            (3, 2048, "negative"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in call.scratch:
                scratch.fill_(17)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert not bool(call.scratch[-1].count_nonzero())
            _assert_selected(case, expected, logical)
    graph.reset()


@pytest.mark.parametrize("tile_schedule", ["contiguous", "cyclic"])
@pytest.mark.parametrize("grid_layout", ["flat", "row_major", "slice_major"])
def test_fused_mxfp4_geometry_capture(monkeypatch, tile_schedule, grid_layout):
    from b12x.attention.dsa_indexer import fused_indexer

    constructor = fused_indexer.DSAFusedIndexerKernel

    def configured_kernel(**kwargs):
        kwargs.update(tile_schedule=tile_schedule, grid_layout=grid_layout)
        return constructor(**kwargs)

    monkeypatch.setattr(fused_indexer, "DSAFusedIndexerKernel", configured_kernel)
    test_fused_mxfp4_mutable_capture(64, False, False, 2048)


@pytest.mark.parametrize("page_size,width", [(24, 2064), (64, 4096)])
def test_fused_mxfp4_serial_merge_overflow_capture(page_size, width):
    case = _case(page_size=page_size, shared=True, width=width)
    case.args["ctas_per_row"] = 64
    case.mutate(0, width, "zero")
    call = mxfp4.prepare_fused_mxfp4(**case.args)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard():
        with torch.cuda.graph(graph):
            call()
        for step, visible, kind in (
            (0, width, "zero"),
            (1, width, "negative"),
            (2, 0, "zero"),
            (3, width, "mixed"),
        ):
            expected, logical = case.mutate(step, visible, kind)
            for scratch in call.scratch:
                scratch.fill_(17)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert not bool(call.scratch[-1].count_nonzero())
            _assert_selected(case, expected, logical)
    graph.reset()
