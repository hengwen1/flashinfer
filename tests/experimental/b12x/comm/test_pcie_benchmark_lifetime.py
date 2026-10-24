"""Collective benchmark operands retire without requiring cyclic collection."""

from contextlib import nullcontext
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from b12x.comm.pcie._benchmark import benchmark_call
from b12x.preparation.session import _CallGuard, _Trial


def state(surface):
    runtime = SimpleNamespace(device=torch.device("cpu"), capture=nullcontext)
    if surface == "DcpAllToAll.all_to_all":
        call = dict(
            shape=(8, 4, 64),
            stride=(256, 64, 1),
            out_shape=(8, 8, 64),
            out_stride=(512, 64, 1),
            dtype="bfloat16",
        )
    else:
        call = dict(dtype="bfloat16")
    query = SimpleNamespace(
        surface=surface, call=call, setup=dict(shape=(8, 128)), world_size=2, rank=0
    )
    return SimpleNamespace(query=query, runtime=runtime, bind_input=lambda tensor: None)


@pytest.mark.parametrize(
    "surface,inplace",
    [
        ("OneshotAllReduce.all_reduce", False),
        ("OneshotAllReduce.all_reduce", True),
        ("DcpAllToAll.all_to_all", False),
    ],
)
def test_benchmark_operands_release_with_gc_disabled(surface, inplace):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        call = benchmark_call(state(surface), inplace=inplace)
        inputs = call.owners[0]
        references = [
            weakref.ref(inputs),
            weakref.ref(inputs.source),
            weakref.ref(inputs.out),
        ]
        references.extend(weakref.ref(seed) for seed in inputs.seeds)
        for index, producer in enumerate(call.benchmark_producers):
            inputs.source.fill_(123)
            producer()
            assert inputs.step == index
            assert torch.equal(inputs.source, inputs.seeds[index])
        assert call.produce is call.benchmark_producers[0]
        trial = _Trial(0, None, None, call, _CallGuard(call), None, inputs.state, 0)
        del producer, inputs, call
        trial.close()
        # No gc.collect is allowed between dropping the call and observing liveness.
        assert not any(reference() is not None for reference in references)
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()
