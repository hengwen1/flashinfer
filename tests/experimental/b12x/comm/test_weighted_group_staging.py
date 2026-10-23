"""Exact ordered route reduction before collective transport staging."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cuda.bindings import driver as cuda
import pytest
import torch

from b12x._lib.intrinsics.memory import st_global_v4_u32
from b12x._lib.utils.pointers import make_ptr
from b12x.comm.pcie._oneshot_cute import _FusedOneshotLaunch
from b12x.testing.reference.helpers import require_b12x


class _StagingProbe(_FusedOneshotLaunch):
    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weights: cute.Pointer,
        ids: cute.Pointer,
        addend: cute.Pointer,
        output: cute.Pointer,
        packs: Int32,
        count: Int32,
        stream: cuda.CUstream,
    ):
        self.probe(source, weights, ids, addend, output, packs, count).launch(
            grid=((count + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def probe(
        self,
        source: cute.Pointer,
        weights: cute.Pointer,
        ids: cute.Pointer,
        addend: cute.Pointer,
        output: cute.Pointer,
        packs: Int32,
        count: Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        index = Int64(block * 128 + thread)
        if index < Int64(count):
            words = self._input_words(
                Int64(source.toint()),
                Int64(addend.toint()),
                index,
                Int64(weights.toint()),
                Int64(ids.toint()),
                packs,
            )
            st_global_v4_u32(Int64(output.toint()) + index * Int64(16), *words)


def _ordered(values, weights, ids, limit):
    result = torch.zeros_like(values[:, 0], dtype=torch.float32)
    for group in range(values.shape[1]):
        valid = (ids[:, group] >= 0) & (ids[:, group] < limit)
        term = torch.where(valid[:, None], values[:, group].float(), 0.0)
        product = term * weights[:, group, None]
        result = result + product
    return result.to(values.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("groups", [3, 4])
def test_weighted_group_words_preserve_order_and_rounding(dtype, ids_dtype, groups):
    require_b12x()
    torch.manual_seed(3407)
    name = str(dtype).split(".")[-1]
    launch = _StagingProbe(
        name,
        4,
        0,
        "stage_pull",
        False,
        False,
        True,
        0,
        128,
        input_addend=True,
        group_size=groups,
        group_id_limit=128,
        group_ids_dtype=str(ids_dtype).split(".")[-1],
    )
    null = make_ptr(cutlass.Uint32, 0, cute.AddressSpace.gmem, assumed_align=4)
    raw = cute.compile(
        launch, null, null, null, null, null, Int32(1), Int32(1), cuda.CUstream(0)
    )
    for rows in (1, 17, 48, 4096):
        hidden = 6144 if rows == 48 else 128
        values = torch.randn(rows, groups, hidden, device="cuda", dtype=dtype)
        weights = torch.randn(rows, groups, device="cuda")
        ids = torch.randint(-1, 130, (rows, groups), device="cuda", dtype=ids_dtype)
        addend = torch.randn(rows, hidden, device="cuda", dtype=dtype) / 16
        output = torch.empty_like(addend)

        def run():
            pointers = tuple(
                make_ptr(
                    cutlass.Uint32,
                    x.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for x in (values, weights, ids, addend, output)
            )
            raw(
                *pointers,
                hidden * dtype.itemsize // 16,
                output.numel() * dtype.itemsize // 16,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for iteration in range(4):
            ids.add_(3).remainder_(131).sub_(1)
            if iteration == 1:
                ids.fill_(-1)
                values.fill_(float("nan"))
            elif iteration == 2:
                values.normal_()
                ids.fill_(0)
                weights[0, 0] = float("nan")
            else:
                weights.normal_()
            expected = (
                _ordered(values, weights, ids, 128).float() + addend.float()
            ).to(dtype)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
            finite = torch.isfinite(expected)
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            torch.testing.assert_close(
                output.view(bits)[finite], expected.view(bits)[finite], rtol=0, atol=0
            )
        graph.reset()
