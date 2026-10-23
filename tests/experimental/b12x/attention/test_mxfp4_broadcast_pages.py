"""Prepared MXFP4 indexing preserves broadcast page-table layouts through replay."""

import pytest
import torch

from b12x.attention import dsa_indexer as api
from b12x.preparation import PreparationSession, PreparedCall


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="MXFP4 fused selection requires SM120",
)


@pytest.mark.parametrize("page_layout", ["broadcast", "shared", "contiguous"])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("candidates", [False, True])
def test_prepared_page_table_layout_preserves_every_short_row(
    page_layout, page_size, candidates
):
    rows, heads, width, device = 8, 32, 1024, torch.device("cuda:0")
    pages = width // page_size
    physical = torch.arange(1, pages + 1, dtype=torch.int32, device=device)[None]
    table = physical
    if page_layout == "broadcast":
        table = physical.expand(rows, -1)
    elif page_layout == "contiguous":
        table = physical.repeat(rows, 1)
    lengths = torch.tensor(
        [0, 1, 2, 17, 63, 64, 127, 511], device=device, dtype=torch.int32
    )
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    arguments = dict(
        q_mxfp4=torch.zeros((rows, heads, 64), dtype=torch.uint8, device=device),
        q_scales=torch.full((rows, heads, 4), 127, dtype=torch.uint8, device=device),
        query_weights=torch.ones((rows, heads), dtype=torch.bfloat16, device=device),
        index_k_cache=torch.zeros(
            (pages + 1, page_size * 68), dtype=torch.uint8, device=device
        ),
        page_table=table,
        cache_lengths=lengths,
        active_width=torch.tensor([511], dtype=torch.int32, device=device),
        output_indices=output,
    )
    if candidates:
        arguments.update(
            candidate_indices=torch.arange(
                width, dtype=torch.int32, device=device
            ).repeat(rows, 1),
            candidate_lengths=torch.full(
                (rows,), width, dtype=torch.int32, device=device
            ),
        )
    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=pages,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        max_candidates=width if candidates else 0,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        plan = api.plan(caps, invocation=api.invocation_from_tensors(caps, **arguments))
        retained = []

        def prepare(state):
            (spec,) = state.layout.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, **arguments)
            retained.append((binding, scratch))
            return PreparedCall(
                run=lambda: state.run(binding), owners=(binding, scratch)
            )

        session.prepare((plan.request(name="broadcast-pages", prepare_call=prepare),))
        binding, scratch = retained[0]
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.state.run(binding)
        try:
            for invalid in (False, True, False):
                physical[:, 0] = -1 if invalid else 1
                if page_layout == "contiguous":
                    table.copy_(physical.expand_as(table))
                output.fill_(-42)
                scratch.fill_(0xA5)
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                expected = torch.full_like(output, -1)
                for row, length in enumerate(lengths.tolist()):
                    first = min(length, page_size) if invalid else 0
                    expected[row, : length - first] = torch.arange(
                        first, length, device=device
                    )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
        finally:
            graph.reset()
