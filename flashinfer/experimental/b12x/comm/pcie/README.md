# PCIe collective preparation

`AllReduce.plan` declares the eligible transports for an established channel.
`PreparationSession` races them jointly across the communicator and retains the
selected native execution state before graph capture. Each callsite submits its
actual shape, dtype, layout, arithmetic options, and aliasing contract. Equivalent
declarations coalesce; request names and layer numbers do not create separate
tuning keys. Every caller still binds its own serving operands.

## Candidate operations

| Declaration | Candidates |
| --- | --- |
| Staged `AllReduce.all_reduce`, TP2–8 | Native pull; qualified TP2 peer push; qualified TP4 BF16 payload-ready one-shot and fused reduce-scatter/all-gather; provisioned TP4 BF16 two-shot with disjoint output |
| Staged fused residual addition and RMSNorm | Native staged transports compatible with declared input, residual, normalization, grouping, and output arithmetic |
| DCP `all_gather_heads` and `lse_reduce_scatter`, TP2/4/8 | Thread count and block limit; equivalent effective grids are measured once |

TP4 payload-ready transports require their reserved incoming slots and qualified
SM120 hardware. Two-shot retains its native capacity and alignment constraints.
Unsupported alternatives are absent, rather than failing over during execution.
Low effort compares base launch configurations; medium/high/max also explore
launch geometry. Tuning off retains the static dispatch policy. Explicit
`CollectiveConfig(implementation=..., threads=..., blocks=...)` pins a legal
configuration. Native low-level plans still use `PcieConfig`.

The TP9–16 hierarchy, DMA rings, FP8 sequence-parallel transports, paired gathers,
and owner/top-k exchanges retain their native fixed declarations. Transport
codecs with different arithmetic are not interchangeable tuning candidates.

## Benchmark and synchronization contract

The `_benchmark.py` fixtures allocate private operands, generate and reset inputs,
capture candidate calls, and check arithmetic before timing. They preserve
declared tensor layouts and residual aliasing. A caller can supply an explicit
`benchmark_call` for a different representative execution context.

All ranks compile and bind a candidate before any rank primes it. Each rank
queues the same captured calls behind a stream gate; the coordinator releases
the sample block after every rank reports readiness. An untimed collective
aligns GPU entry. Corresponding event samples reduce with **maximum across
ranks** before repeat sizing, elimination, and winner selection. The tournament
retains a champion and one challenger, bounding candidate operand residency to
two fixtures plus graph/event and cache-eviction storage.

`PreparationProgress.ready_collective_tuning` carries
`CollectiveTuningRequirement(key, ranks, step, payload)`. An embedding coordinator
exchanges these contributions in communicator rank order and resumes
`PreparationJob.advance(collective_tuning=contributions)`. All members must report
the same key, ranks, and step. Query signatures reject inconsistent shapes or
arithmetic before candidate entry. Cancellation is agreed at drained group
boundaries. Coordinators must propagate preparation exceptions to every member.

Collective candidates execute on the whole group; they are not divided among
the individual-kernel tuner's candidate shards. Independent TP replicas tune
their own communication group. The selection key includes the ordered GPU UUIDs,
rank-local query/program identities, and measurement policy. A valid cached
record from any member is adopted by the group. Cache-only mode fails if none
has a record. Interrupted collective races restart; completed selections reuse
the ordinary selection cache.

## Validation

Run the standalone group test with two or four visible peer-accessible SM120
GPUs:

```sh
B12X_RUN_COLLECTIVE_TUNING_TEST=1 python -m pytest -q \
  tests/experimental/b12x/comm/test_collective_tuning_gpu.py
```

The test exercises duplicate declarations, cache reuse with a missing member
cache, one-rank cancellation, inconsistent shapes, cache-only misses, and graph
replay. `benchmarks/experimental/b12x/benchmark_pcie_allreduce_transports.py`
compares explicit transports and the static dispatch policy with CUDA graphs.
