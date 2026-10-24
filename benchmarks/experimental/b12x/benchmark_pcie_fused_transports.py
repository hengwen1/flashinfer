"""Compare complete BF16 PCIe all-reduce + residual + RMSNorm CUDA graphs.

Launch with torch.distributed.run and an explicit CUDA_VISIBLE_DEVICES assignment.
The residual and reduction accumulate in FP32; normalized output is BF16.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from b12x.comm.pcie import _oneshot_preparation as preparation
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.preparation import CollectiveRequirement, FrozenMapping, PreparationSession

from benchmark_pcie_allreduce_transports import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--arms", default="auto,pull,push,sentinel")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--chain", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9137)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    arms = args.arms.split(",")
    modes = {
        "pull": "stage_pull",
        "push": "stage_remote_push",
        "sentinel": "stage_bf16_sentinel",
    }
    if any(arm != "auto" and arm not in modes for arm in arms):
        parser.error("unknown comparison arm")
    if (
        min(rows + [args.hidden, args.samples, args.replays, args.chain, args.warmup])
        <= 0
        or args.hidden % 8
    ):
        parser.error(
            "positive counts and a hidden size divisible by eight are required"
        )
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    if world != 4:
        raise ValueError("the compared transport family requires four ranks")
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=max(rows) * args.hidden * 2,
        eager_buffer_bytes=max(rows) * args.hidden * 2,
    )
    manager.prepare_channels(("benchmark:fused",))
    target = manager._runtime._prepared_channel_for_stream(None, None)
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    records, graphs = [], {}
    try:
        for count in rows:
            inp = torch.ones((count, args.hidden), dtype=torch.bfloat16, device=device)
            residual = torch.zeros_like(inp, dtype=torch.float32)
            residual_out = torch.empty_like(residual)
            weight = torch.ones(args.hidden, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            plans, graphs = {}, {}
            for arm in arms:
                if rank == 0:
                    print(f"preparing rows={count} arm={arm}", flush=True)
                query = preparation.query_from_runtime(
                    target,
                    surface="OneshotAllReduce.all_reduce_fused_add_rms_norm",
                    call={
                        "inp": inp,
                        "residual": residual,
                        "weight": weight,
                        "out": out,
                    },
                )
                if arm != "auto":
                    query = replace(
                        query, call=FrozenMapping({**query.call, "mode": modes[arm]})
                    )
                plan = preparation.plan(query, runtime=target)
                plans[arm] = plan
                name = f"{arm}-{count}"
                session.prepare(
                    (
                        plan.request(
                            name=name,
                            collective=CollectiveRequirement(
                                key=name, ranks=tuple(range(world))
                            ),
                            prepare_call=lambda state: preparation._prepare_fused_call(
                                state,
                                inp=inp,
                                residual=residual,
                                weight=weight,
                                out=out,
                                residual_out=residual_out,
                                epsilon=1e-6,
                            ),
                        ),
                    ),
                    coordinator=lambda progress: progress.ready_collectives[0].key
                    if progress.ready_collectives
                    else None,
                )
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with session.capture(), manager.capture(), torch.cuda.graph(graph):
                    for _ in range(args.chain):
                        target.all_reduce_fused_add_rms_norm(
                            inp,
                            residual,
                            weight,
                            1e-6,
                            plan=plan,
                            out=out,
                            residual_out=residual_out,
                        )
                graphs[arm] = graph
            for step in range(5):
                generator = torch.Generator().manual_seed(args.seed + step)
                operands = torch.randn(
                    (world, inp.numel()), generator=generator
                ).bfloat16()
                initial_residual = torch.randn(inp.shape, generator=generator)
                scale = torch.randn(args.hidden, generator=generator).bfloat16()
                if step == 1:
                    operands.zero_()
                elif step == 2:
                    operands[1] = -operands[0]
                elif step == 3:
                    operands.mul_(2**-100)
                inp.copy_(operands[rank].view_as(inp))
                residual.copy_(initial_residual)
                weight.copy_(scale)
                expected_residual = (
                    reference(operands.float(), rank, False).view_as(residual)
                    + initial_residual
                )
                expected = (
                    expected_residual
                    * torch.rsqrt(
                        expected_residual.square().mean(-1, keepdim=True) + 1e-6
                    )
                    * scale.float()
                ).bfloat16()
                for arm, graph in graphs.items():
                    out.fill_(float("nan"))
                    residual_out.fill_(float("nan"))
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
                    torch.testing.assert_close(
                        residual_out.cpu(), expected_residual, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        out.cpu(), expected, rtol=8e-3, atol=8e-3
                    )
                    torch.testing.assert_close(
                        inp.cpu(), operands[rank].view_as(inp), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        residual.cpu(), initial_residual, rtol=0, atol=0
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
                "rows": count,
                "hidden": args.hidden,
                "modes": {arm: plans[arm].query.call["mode"] for arm in arms},
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
                    json.dumps({"rows": count, **record["max_rank_median_us"]}),
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
    main()
