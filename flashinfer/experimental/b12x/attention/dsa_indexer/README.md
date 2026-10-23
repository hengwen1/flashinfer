# MXFP4 sparse selection

The MXFP4 indexer publishes BF16 scores and selects 512 logical positions.
SM120 decode bindings with 16-byte-aligned score storage, 32-byte-aligned
internal index storage, and score widths divisible by 512 and below 2²³ use
DeepSelect's BF16 kernel. The selected values are widened exactly to FP32,
candidate slots are mapped to logical positions, and outputs are ordered by
logical position. Other score widths, storage alignments, prefill, and device
architectures use the existing FP32 radix selector. Hierarchical candidate
selection retains the radix selector for 2048 blocks and the mandatory
newest-block priority.

DeepSelect-eligible bindings require NaN-free visible scores. Positive infinity
participates in selection; negative infinity marks omitted positions and padding.
DeepSelect traps on visible NaNs, whereas the radix path ranks their bit
patterns and removes their positions from the output. Equal-score cutoff ties
may select different logical IDs.

Preparation requires `deep-select==1.0.0+sm120.top512`. The
[dependency patch](../../../../../3rdparty_patches/deepselect/0001-sm120-top512.patch)
adds the SM120 specialization, CUDA dispatch and capability query, build source
entry, and exact package version to the `deepinfer-sm120` branch at commit
`ef20ee1a5c77a9365137573e2327402296a2a0e5`. Unmodified upstream source does not
provide this package version or the required specialization.

Run these commands from the FlashInfer checkout root, using the Python
environment that will prepare models. It must already contain PyTorch,
setuptools, wheel, and ninja, with a CUDA toolkit that supports `sm_120a`:

```bash
flashinfer_checkout="$(pwd)"
git clone --branch deepinfer-sm120 https://github.com/lucifer1004/DeepSelect.git ../DeepSelect-sm120-top512
git -C ../DeepSelect-sm120-top512 checkout --detach ef20ee1a5c77a9365137573e2327402296a2a0e5
git -C ../DeepSelect-sm120-top512 submodule update --init --recursive
git -C ../DeepSelect-sm120-top512 apply "$flashinfer_checkout/3rdparty_patches/deepselect/0001-sm120-top512.patch"
cd ../DeepSelect-sm120-top512
DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA DEEP_SELECT_CUDA_ARCHS=120a \
  NVCC_THREADS=1 MAX_JOBS=6 \
  python setup.py build_ext --inplace build_py bdist_wheel
python -m pip install --no-deps dist/deep_select-1.0.0+sm120.top512-*.whl
cd "$flashinfer_checkout"
python -m pip install --no-build-isolation -e '.[b12x]'
```

The wheel is specific to the Python ABI and platform used to build it. SM120
preparation checks that the installed CUDA extension contains a loadable
specialization before warming or capture. Native serving uses the cubins in
the exported model package and does not install this Python dependency.

DeepSelect is MIT licensed, copyright 2025 DeepSeek. The patch directory
includes its [license](../../../../../3rdparty_patches/deepselect/LICENSE), and
the patched checkout and wheel retain the upstream license. The prepared state
retains the loaded extension and compiled length/position kernels throughout
graph capture and export.

## Fused MXFP4 benchmark

`prepare_fused_mxfp4` specializes the existing fused DSA kernel for packed
MXFP4 queries and paged keys. It prepares a fixed callable for 32 replicated
query heads, page sizes 64 or 128, and top-512 selection, including gathered
candidate consumers. The callable retains its inputs, compiled programs, and
scratch. Its execution resets merge state, computes scores and selection, and
orders the selected logical positions. Hierarchical 2048-block candidate
publication is unsupported. The factory is used explicitly by benchmarks;
production preparation and dispatch do not select it.

From the checkout root, with the dependency above installed:

```bash
python benchmarks/experimental/b12x/benchmark_mxfp4_indexer.py \
  --rows 1,8 --widths 4096 --capacity 4096 \
  --page-size 128 --page-stride 8704 --modes dense,candidates \
  --candidate-capacity 512 --samples 3 --replays 5 --warmup 5 \
  --stage-timings --output /tmp/mxfp4-indexer.json
```

The three arms are `native_staged`, `deepselect_staged`, and `fused`.
Each timed graph includes query quantization and ordered top-512 output.
Native staged selection retains the optimized logical-position sorter.
Consumers use candidates published by an independent source query; source
publication is outside their timed boundary. `--stage-timings` reports separate
quantization, score, and selector diagnostics. These times exclude learned
projections, cache writes, tensor-parallel communication, and model serving.

Use `--rows` for physical query rows, including packed decode rows up to 128,
and `--widths` for indexed keys, not uncompressed context tokens. `--capacity`
declares the fixed indexed-key capacity. `--full-score-width` retains that
capacity in staged score matrices when live widths are smaller. The production
geometry helper is available through `--profile-form c2-dense --modes dense`
(page 64, logical context twice the indexed-key count) or
`--profile-form c1-reindex --modes candidates --candidate-capacity 16384`
(page 128). Supply `--model-path` to load a local V4.1 configuration. Plain
page-128 top-512 and 512-entry consumer cases have no production profile label;
the C1 source also requires hierarchical publication.

Representative decode rows are `1,2,4,7,8,14,16,28,32,64,128`. Useful independent
length cases are 4096 and 32768 indexed keys, up to 524288 for C2 at a 1M logical
context, and up to 1048576 for plain page-128 selection. Consumer capacities
are 512 and 16384. Select cases explicitly instead of taking their full cross
product. `--arms` selects comparison arms and `--fused-ctas-per-row` fixes fused
geometry for experiments; omitting it uses the factory's initial heuristic.

Samples alternate arm order. Warm graph replay is the default; `--cold-l2`
requires `--replays 1` and evicts L2 before every measured replay. Repeat with
independent `--seed` values for evaluation. Raw samples and configuration are
written to JSON. `--checks` controls changing-input replay cases; its default
four cases include full, tail, short, and zero visibility. Every arm checks
exact MXFP4 packing, the selected score multiset, logical ordering and candidate
mapping, inactive rows, poisoned scratch, frozen kernel resolution, and unchanged
allocation/free counters. The independent score oracle rounds dot and weighted
product stages to BF16, accumulates heads in increasing order in FP32, and rounds
the published score to BF16. Equal-score cutoff ties may choose different IDs.

Focused graph and compatibility checks are in
`tests/experimental/b12x/attention/test_fused_mxfp4.py`. Component timing alone
does not establish a production dispatch improvement.
