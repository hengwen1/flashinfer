"""Compare complete prepared PCIe all-reduces with changing CUDA graph inputs.

Run with torch.distributed.run and an explicit CUDA_VISIBLE_DEVICES assignment.
Transport overrides affect benchmark declarations only; auto uses normal routing.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from b12x.comm.pcie import _oneshot_preparation as oneshot
from b12x.comm.pcie import _twoshot_preparation as twoshot
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.preparation import CollectiveRequirement, FrozenMapping, PreparationSession


def prepare(session, manager, inp, out, arm, threads, blocks):
    if arm == "auto":
        plan = manager.plan(inp, out=out)
        target = manager
    elif arm == "twoshot":
        target = manager._bf16_twoshot
        if target is None or not target.accepts(inp):
            raise ValueError("two-shot does not support this prepared shape")
        query = twoshot.query_from_runtime(
            target, surface="PCIeTwoShotBF16.all_reduce", call={"inp": inp, "out": out}
        )
        plan = twoshot.plan(query, runtime=target)
    else:
        target = manager._runtime._prepared_channel_for_stream(None, None)
        query = oneshot.query_from_runtime(
            target, surface="OneshotAllReduce.all_reduce", call={"inp": inp, "out": out}
        )
        transport = {
            "pull": "pull",
            "oneshot": "tp4_push_sentinel",
            "rsag": "tp4_push_rsag",
        }[arm]
        if arm != "pull" and not query.call["remote_push_region_packs"]:
            raise ValueError("payload readiness requires prepared incoming storage")
        call = {**query.call, "transport": transport}
        if arm != "pull":
            call.update(threads=threads, blocks=blocks)
        query = replace(query, call=FrozenMapping(call))
        plan = oneshot.plan(query, runtime=target)

    def prime(state):
        if state.query.surface.startswith("PCIeTwoShotBF16."):
            return twoshot.prepared_call(state, payload=inp, out=out)
        return oneshot._prepare_plain_call(state, inp=inp, out=out)

    name = f"{arm}-{inp.numel()}"
    session.prepare(
        (
            plan.request(
                name=name,
                collective=CollectiveRequirement(
                    key=name, ranks=tuple(range(manager.world_size))
                ),
                prepare_call=prime,
            ),
        ),
        coordinator=lambda progress: progress.ready_collectives[0].key
        if progress.ready_collectives
        else None,
    )
    return plan, lambda: target.all_reduce(inp, plan=plan, out=out)


def reference(operands, rank, owner_reduction):
    world, elements = operands.shape
    if not owner_reduction:
        total = operands[rank].float()
        for peer in range(1, world):
            total += operands[(rank + peer) % world].float()
        return total.to(operands.dtype)
    packs = elements // 8
    part = packs // world
    result = torch.empty(elements, dtype=operands.dtype)
    for owner in range(world):
        begin = owner * part * 8
        end = elements if owner == world - 1 else (owner + 1) * part * 8
        total = operands[owner, begin:end].float()
        for peer in range(1, world):
            total += operands[(owner + peer) % world, begin:end].float()
        result[begin:end] = total.to(operands.dtype)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bytes",
        default="10240,20480,40960,65536,81920,163840,327680,655360,1048576,2097152",
    )
    parser.add_argument("--arms", default="pull,oneshot,rsag,twoshot")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--chain", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=77231)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.bytes.split(",")]
    arms = args.arms.split(",")
    if any(size <= 0 or size % 16 for size in sizes):
        parser.error("message sizes must be positive multiples of 16 bytes")
    if any(arm not in ("auto", "pull", "oneshot", "rsag", "twoshot") for arm in arms):
        parser.error("unknown comparison arm")
    if min(args.samples, args.replays, args.chain, args.warmup) <= 0:
        parser.error("measurement counts must be positive")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    if world != 4 and any(arm in ("oneshot", "rsag", "twoshot") for arm in arms):
        raise ValueError("explicit payload-readiness arms currently require TP4")
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=max(sizes),
        eager_buffer_bytes=max(sizes),
    )
    manager.prepare_channels(("benchmark:allreduce",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    records = []
    graphs = {}
    try:
        for size in sizes:
            inp = torch.empty((size // 2,), dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            inp.fill_(rank + 1)
            calls, plans, graphs = {}, {}, {}
            for arm in arms:
                if rank == 0:
                    print(f"preparing bytes={size} arm={arm}", flush=True)
                plans[arm], calls[arm] = prepare(
                    session, manager, inp, out, arm, args.threads, args.blocks
                )
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with session.capture(), manager.capture(), torch.cuda.graph(graph):
                    for _ in range(args.chain):
                        calls[arm]()
                graphs[arm] = graph
            for step in range(5):
                generator = torch.Generator().manual_seed(args.seed + step)
                operands = torch.randn(
                    (world, inp.numel()), generator=generator
                ).bfloat16()
                if step == 1:
                    operands.zero_()
                elif step == 2:
                    operands[1] = -operands[0]
                elif step == 3:
                    operands.mul_(2**-100)
                elif step == 4:
                    operands.mul_(128)
                inp.copy_(operands[rank])
                for arm, graph in graphs.items():
                    out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for counter in (
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ):
                        assert before[counter] == after[counter], (arm, counter)
                    transport = plans[arm].query.call.get("transport", "twoshot")
                    expected = reference(
                        operands, rank, transport in ("tp4_push_rsag", "twoshot")
                    )
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        inp.cpu(), operands[rank], rtol=0, atol=0
                    )
            for graph in graphs.values():
                for _ in range(args.warmup):
                    graph.replay()
            torch.cuda.synchronize()
            samples = {arm: [] for arm in arms}
            for sample in range(args.samples):
                for arm in arms if sample % 2 == 0 else reversed(arms):
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.replays):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(
                        start.elapsed_time(end) * 1000 / (args.replays * args.chain)
                    )
            rank_samples = [None] * world
            dist.all_gather_object(rank_samples, samples)
            record = {
                "bytes": size,
                "world_size": world,
                "surfaces": {
                    arm: dict(
                        surface=plans[arm].query.surface,
                        call=dict(plans[arm].query.call),
                    )
                    for arm in arms
                },
                "rank_samples_us": rank_samples,
                "max_rank_median_us": {
                    arm: max(median(values[arm]) for values in rank_samples)
                    for arm in arms
                },
            }
            records.append(record)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "arguments": {**vars(args), "output": str(args.output)},
                            "results": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps({"bytes": size, **record["max_rank_median_us"]}),
                    flush=True,
                )
            for graph in graphs.values():
                graph.reset()
            graphs = {}
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
