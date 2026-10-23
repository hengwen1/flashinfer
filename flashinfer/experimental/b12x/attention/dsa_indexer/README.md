# Fused MXFP4 sparse indexing

The prepared MXFP4 indexer computes exact scores and selects up to 512 logical
key positions on SM12x GPUs. Dense indexing scans paged keys; candidate consumers
score only the supplied logical positions. A hierarchical source also publishes
up to 2048 eight-token blocks for a consumer. Scores flow through bounded token
and block carries. Execution does not allocate or expose a context-sized score
matrix.

## Preparation and execution

Use `b12x.attention.dsa_indexer` with `Caps(cache_format="mxfp4", ...)`.
The lifecycle is `plan(Caps(...), invocation=...)`,
`PreparationSession.prepare`, `bind(Plan, ...)`, then `run(binding)`.
`invocation_from_tensors` describes actual operand shapes, strides, and storage
alignment for preparation. The preparation callback binds trial storage and
returns a `PreparedCall` whose `run` performs the complete operation being tuned.
For a query-to-selection measurement, that callback includes query quantization
and indexing. Cache writes prepare stored keys separately.

Preparation selects concrete kernel configurations, compiles and loads the
programs, plans scratch, and warms execution before capture. After preparation,
allocate caller-owned scratch from `Plan.scratch_specs()` and bind the input,
output, and scratch tensors. Keep the prepared session, bindings, tensor owners,
and compiled programs alive throughout capture and replay. `run` computes
selection and orders its output; `quantize_q_mxfp4` and
`quantize_write_index_k_mxfp4` prepare packed queries and paged keys respectively.
MXFP4 does not provide a separate score/select boundary for an external score
reduction. Each rank must supply every contributing index head.

Supported index-head counts are 1, 2, 4, 8, 16, 32, and 64, with head dimension
128. Page sizes are positive multiples of eight tokens. Each physical page uses
`index_mxfp4_page_bytes(page_size)` bytes: 64 packed E2M1 bytes per token followed
by four UE8M0 scale bytes per token. The page table maps logical pages to
physical pages. Physical page strides may exceed the page payload. Packed query
and scale tensors accept contiguous byte-offset views; preparation selects loads
compatible with their actual alignment. Query weights use BF16. Outputs contain
512 int32 logical positions and, when requested, associated FP32 values.

`max_q_rows`, `max_page_table_width`, and `max_candidates` reserve capacities.
Bindings may use fewer rows and a bounded `score_width`; device lengths and
`active_width` control visibility during replay. `score_width` limits the fused
scan without allocating scores. For candidate consumers it bounds candidate
slots, while returned positions remain logical key positions. Negative candidate
IDs, unavailable pages, inactive rows, and positions outside the live bounds are
omitted. Unused output positions are `-1`, with values `-inf`.

Capture only prepared configurations with fixed-address input, output, and
scratch storage. Mutable query data, weights, page tables, lengths, candidate
lists, and visibility bounds may change between replays. Scratch needs no
persistent initialization: each invocation initializes its required state.
Replay performs no compilation, host kernel selection, or GPU allocation.

## Arithmetic and hierarchical candidates

Quantization uses per-group scales for four groups of 32 elements and
round-to-nearest-even E2M1 encoding. Query and key values dequantize to BF16.
Each dot accumulates in FP32 and rounds to BF16 before ReLU; multiplication by
its head weight rounds to BF16. Head contributions accumulate in increasing
head order in FP32, followed by a final BF16 score round. Selected scores widen
exactly to FP32. Top-k is exact; equal-score cutoff ties may choose different
logical IDs. Outputs are sorted by logical position and preserve each selected
position's associated score.

Set `candidate_topk_blocks=2048` for a hierarchical source. Its block scores are
maxima of the final BF16 token scores over eight-token blocks. When the newest
visible block intersects the scored extent, selection retains it even when its
score would otherwise exclude it.
Selected blocks are ordered and expanded into at most 16384 logical token IDs,
with partial blocks clipped to live visibility. Expansion can include positions
beyond `score_width` within a selected block. `candidate_output_lengths`
records the number of published IDs; padding is `-1`. Token top-512 output is
selected independently of block publication.

A consumer declares `max_candidates` and binds `candidate_indices` and
`candidate_lengths`. It evaluates those keys with its own query and weights;
source scores are not reused. Source publication and consumer execution have
separate preparation and scratch ownership. Global intermediates hold bounded
partial selections rather than full token or block score arrays.

## Configuration and measurement

The preparation tuner measures eligible concrete configurations under the
session's cache policy; the benchmark requests warm-cache tuning. Choices include
work partitioning, scoring schedule, merge
organization, output stores, and load alignment. Semantic, alignment, shared
memory, and residency constraints determine eligibility. Selection completes
before capture, and the selected configuration is retained by the prepared Plan.
A work- and resource-based heuristic supplies a concrete configuration when
measurement is disabled. A pinned configuration must resolve dependent geometry,
state, and ordering choices together.

The benchmark uses this production Plan and `run` path:

```bash
python benchmarks/experimental/b12x/benchmark_mxfp4_indexer.py \
  --rows 1,8 --heads 32 --widths 4096 --capacity 4096 \
  --page-size 128 --page-stride 8704 --modes dense,candidates \
  --candidate-capacity 16384 --preparation autotune \
  --samples 3 --replays 5 --warmup 5 --output /tmp/mxfp4-indexer.json
```

Install the checkout's `b12x` extra in the preparation environment. MXFP4
selection uses the fused CuTe implementation and requires no DeepSelect package.
The benchmark has one arm, `fused`; `--arms fused` is accepted explicitly.
`--preparation autotune` measures configurations; `--preparation heuristic`
uses the heuristic configuration. `--fused-ctas-per-row` pins the CTA count
through configuration resolution before capture. The JSON includes the selected
configuration, preparation cache policy, per-case medians, and raw samples.

Timed graphs include query quantization, required state initialization, fused
scoring and selection, and logical-position ordering. Candidate publication by
an independent source query occurs outside the timed consumer graph. Timings
exclude learned projections, cache writes, tensor-parallel collectives, and
whole-model serving. They measure the declared component workload rather than
end-to-end model latency.

`--rows` specifies physical query rows; `--widths` specifies indexed keys, and
`--capacity` reserves indexed-key capacity. `--full-score-width` keeps the dense
scan extent at that capacity while visibility changes; it does not create a
score matrix. `--profile-form c2-dense --modes dense` describes page-64 indexing
with logical context twice the indexed-key count.
`--profile-form c1-reindex --modes candidates --candidate-capacity 16384`
describes the page-128 hierarchical consumer. `--model-path` reads a local model
configuration. Plain page-128 top-512 and 512-entry consumers need no profile
label. Consumer capacities supported by the benchmark are 512 and 16384.

Warm graph replay is the default. `--cold-l2` requires `--replays 1` and evicts
L2 before each measured replay. Compare matching row lists, capacities, seeds,
and cache policies; repeat with independent seeds. `--checks` controls mutable
replay checks covering full, tail, short, and zero visibility. Correctness checks
use independent quantization and score references with exact BF16 boundaries,
selected-score multisets, logical ordering, candidate association, poisoned
scratch, frozen kernel resolution, and unchanged allocation/free counters.

Production correctness and capture coverage lives in
`tests/experimental/b12x/attention/test_dsa_indexer_mxfp4.py`,
`test_dsa_indexer_mxfp4_boundaries.py`, and
`tests/experimental/b12x/test_mxfp4_position_sort.py`. These exercise Plan/run,
mutable captured visibility, hierarchical publication and consumption, unaligned
storage, high physical page addresses, and candidate output offsets beyond
signed 32-bit element indexing. Focused packed arithmetic and fused replay tests
are in `tests/experimental/b12x/attention/test_fused_mxfp4.py` and
`test_fused_mxfp4_head_sum.py`.
