# Operator experiments

Default A/B is **index baseline → current paired range/index kernel**. There is
no intermediate range implementation. A negative `time_change_pct` means less
GPU time. No speedup is assumed for an arbitrary synthetic case.

## Correctness

```bash
python -m unittest tests.test_planning
CUDA_VISIBLE_DEVICES=0 python -m tests.check_gpu
```

The CPU test covers every two-range split and exclusive payload sizing. The
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
