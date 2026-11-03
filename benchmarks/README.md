# Operator experiments

Default A/B is **index baseline → current paired range/index kernel**. There is
no intermediate range implementation. A negative `time_change_pct` means less
GPU time. No speedup is assumed for an arbitrary synthetic case.

## Correctness

```bash
python -m unittest tests.test_planning
CUDA_VISIBLE_DEVICES=0 python -m tests.check_gpu
CUDA_VISIBLE_DEVICES=0 python -m tests.check_boundaries
```

The CPU test covers every two-range split and exclusive payload sizing. The
additional boundary check exercises all 63 splits on GPU plus gapped and
head-major outer strides. The
Hopper test covers S=128/256/512, continuous/two-range/permuted/mixed maps, and
causal/full/sparse/fully masked attention. Both backward implementations are
checked against FP32 math, against each other, and after CUDA Graph replay.
The supported tensors are BF16 with batch 1, D128 and GQA4. Row maps must be
permutations; duplicate rows would violate gradient write ownership.

## A/B

On one otherwise idle Hopper GPU, run each repeat as a new OS process:

```bash
for repeat in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.backward \
    --seqlen 4096 --kv-heads 2 --mapping mixed --mask causal \
    --repeat "$repeat" --output "results/mixed-$repeat.json"
done
python -m benchmarks.analyze results/mixed-*.json
```

The measured chain includes queue reset, backward preprocessing, the persistent
main kernel and dQ postprocessing. It excludes metadata preparation, forward,
cold compilation and CPU allocation/launch. Fixed output/scratch pools make GPU
addresses independent of variant; they are a benchmark fixture, not the public
API's output lifetime policy. The pool allocator fails on incompatible requests.

Each role/pool is warmed 12 times. Four graphs cover ABBA/BAAB and both initial
physical pool assignments. Every adjacent quad crosses physical pools; eight
cycles per graph balance both assignments. After a 10-second GPU soak, 60 paired
blocks each replay both initial layouts. Role order is balanced and shuffled
with a recorded seed. The graphs contain external CUDA timing events around
each complete backward invocation. Correctness is checked before and after
capture; the range role uses a poisoned stride-zero scalar in the legacy
shape-only ABI slot; that role receives no original row index. The index
baseline retains its independent full-index metadata for comparison.

Raw events, complete paired blocks, pool pointers, versions, compilation counts,
and per-block GPU telemetry are kept in JSON. A single stable NVML process set
is required throughout measurement. OS/NVML PID namespaces may differ.
Competing jobs that appear and disappear between checks are not detectable;
reserve the GPU for formal runs. No clocks/power limits are changed.

Analysis takes paired log-ratios within each quad, then averages complete blocks.
Circular block bootstrap lengths 1/3/5/10 provide a sensitivity envelope; three
or more independent process repeats additionally get a process-level Student-t
interval. Use the same fixture, host and GPU type in an A/B; never pool distinct
cases implicitly. For a workload suite report each case and specify its weights.

A/A calibrates the **measurement protocol**, and is not required before every
unchanged A/B. Optional `--phase index-aa` or `paired-aa` measures identical code
in both roles when introducing/changing a protocol or checking an environment.
The analyzer reports both full-data ±0.5% and two-tails-five-block trimmed ±0.3%
95% interval gates (either passes). Trimming is recomputed within each bootstrap
sample, preserves raw data, and is never applied to A/B to hide regressions.
The standalone harness must establish its own calibration before formal claims;
prior calibration of a different harness is not automatically inherited.

## External fixtures

`--input fixture.pt` accepts a `torch.save` dictionary with only tensors:
`q`, `k`, `v`, `dout` in physical order `[1,S,H,128]`, an integer
logical-to-physical `index[S]`, and boolean `logical_mask[S,S]`. Q/dO have four
heads per K/V head. All fixtures use the same default scale `128**-0.5`.
Preparation is deliberately outside graph capture. The loader uses
`weights_only=True`; private inputs/results are ignored by Git. Input IDs and
local paths in benchmark JSON are local experiment metadata, not source files.

## MAGI comparison and reports

Install a compatible [MagiAttention](https://github.com/SandAI-org/MagiAttention)
GPU library separately, following its build instructions. It is optional for the
index/paired tests. MAGI is called directly; no training framework is imported
and no MAGI kernel implementation is copied into this repository. The adapter
records the library's backward API signature and source SHA256, and fails on
incompatible APIs instead of silently switching backends.

```bash
for repeat in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.backward \
    --baseline magi --candidate paired \
    --seqlen 4096 --kv-heads 2 --mapping continuous --mask causal \
    --repeat "$repeat" --output "results/magi-causal-$repeat.json"
done
python -m benchmarks.report results/magi-causal-*.json \
  --output results/magi-report.md
```

The command generates Markdown plus a matching JSON summary. It includes A/B
median times, paired time change and 95% CI, independent process/block counts,
correctness, mask range counts and environment information. At least three
independent process repeats give a process-level Student-t interval. Reports
with fewer repeats explicitly label their weaker within-process interval.
Different fixtures, versions and comparisons are kept separate. `--candidate
index` compares MAGI against the index policy; the default command without
`--baseline magi` retains the existing index-vs-paired comparison.

This measures **backward only**. Each library first computes its own forward
output/LSE and the outputs are checked against each other. Forward, CPU metadata
preparation and compilation are excluded from timing. MAGI clears its FP32
accumulators, invokes its backward operator, and converts the three gradients
back to BF16 inside the timed chain, matching the API used in earlier tests.
The FA4 role includes its queue reset, preprocessing, main kernel and dQ scatter.
MAGI's deterministic/atomic-reduction-disabling and range-merging options are
off. Other MAGI tuning flags keep the library defaults.

The same ABBA/BAAB CUDA Graph engine, warmup, soak, balanced order, process
checks and raw event recording are reused. Shared BF16 output addresses cross
between roles, while MAGI's FP32 workspace and C++ scratch remain library-owned.
The JSON protocol is `shared-outputs-private-scratch-v1`; it must not be pooled
with the original shared-whole-workspace protocol. Optional `--phase baseline-aa`
or `candidate-aa` checks a measurement setup; unchanged setups do not require a
fresh A/A before every A/B.

Masks are transformed into physical row order before MAGI preparation. Full and
standard causal masks use one compact range; general masks use disjoint exact
rectangles, coalesced across adjacent Q rows. This generic encoder prioritizes
correctness and may yield many fragmented ranges; range count and source are
reported. It does not reproduce a training-specific range optimizer. For such
comparisons, external `.pt` fixtures may additionally provide `magi_q_ranges`
and `magi_k_ranges` (integer `[N,2]`, half-open physical row intervals) and
`magi_attn_type_map` (integer `[N]`, full=0, bottom-right causal=1 or top-left inverse causal=2). All three
must be present. Their exact mask coverage and absence of overlap are validated
before any GPU work. Keep mask planning quality separate from kernel speed.

CPU metadata/report checks need NumPy and SciPy:

```bash
pip install -e '.[analysis]'
python -m unittest discover
```

For a single-command run (default: three independent processes and a report):

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.compare_magi \
  --output-dir results/magi-run --seqlen 4096 --kv-heads 2 \
  --mapping continuous --mask causal --candidate paired
```

Additional arguments are forwarded to `benchmarks.backward`. The directory must
be new or empty. Failures stop the suite, leave raw evidence in place, and do
not produce a success report. Use `python -m tests.check_magi` for independent
FP32-reference and graph-replay checks of the optional adapter.

## Sparse-plan preparation

`prepare_attention` packs sparse index tables on CPU and uploads each table in
one transfer. The empty-table path preserves the original unit last stride
required by CuTe and skips CPU packing. F-strided masks use separate axis
reductions without copying the full mask. This changes preparation only; the
attention kernels, Full/Partial classification and list order are unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.check_sparse_planning
for repeat in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.planning \
    --seqlen 4096 --mask causal --repeat "$repeat" \
    --output "results/planning-$repeat.json"
done
python -m benchmarks.analyze results/planning-*.json
```

The default is direct A/B against the frozen row-wise constructor in
`benchmarks.reference_sparse_plan`; unchanged protocols reuse established
calibration. The Linux process pins itself to one available CPU and records
that affinity. Each of 60 balanced ABBA/BAAB blocks measures 16 constructions
per entry, including both forward/backward plans, allocations and H2D copies,
with synchronized wall time. GPU process checks and raw block data are retained.
Choose fresh output paths. Other inputs are `--mask full`, `empty` or `sparse`;
`--layout f` checks a Fortran-strided CPU mask. Layout is included in the fixture
fingerprint so analysis cannot silently pool the two layouts.

The input mask is already resident on CPU. Mask encoding, row-map preparation,
forward and attention backward are excluded. These percentages describe sparse
list construction, not training throughput or the complete preparation API.
