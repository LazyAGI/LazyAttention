# Independent operator improvements — 2026-11-03

## Outcome and scope

Sparse-plan construction is accepted in the measured scope and integrated into
`lazy_attention/planning.py`. It changes CPU metadata construction/H2D upload;
it does not change the attention kernels or establish a training speedup.
Workspace, equivalent-layout Stats and shared gradient-address experiments remain
experimental because their measured gains are absent or input dependent.

## Baseline and environment

- Clean LazyAttention parent: `423d0e75734b5aab2be135995a2ec30da21dd30c`.
- Accepted planning source SHA256:
  `fc84411a2e362aaeade7a92d6b11c8f8a56334db724de0e1735273e8bc76da82`
  (final layout-aware v6); v4 C-layout evidence used
  `2acbbd599e420577147d20bbef66e285d99e9a943798f1aa7025a394a24d9180`.
- Single-host H100 80GB HBM3; each A/B stays on GPU6. Planning benchmark
  CPU affinity is `[64]`. The selected SSH alias, resolved endpoint and GPU UUID
  are recorded in the external experiment evidence.
- Python 3.12 environment, PyTorch `2.13.0+cu126`, CUDA 12.6. Detailed versions
  and GPU telemetry accompany raw results outside the repository.
- The training repositories were not modified or synchronized. Existing dirty
  changes were preserved; their manifests differ from the current local HEADs.
- Following the user's correction, no further A/A was run. Formal comparisons
  use direct A/B with fixed scopes, balanced order and independent processes.

## EXP01 — caller-owned scratch

A prototype reuses the queue counter and FP32 scratch while returning fresh
outputs. Required reset, preprocessing and clears remain. It rejects shape,
device and stream mismatches. The 96-configuration lifecycle/reference/replay
suite and bounded-allocation check passed.

At S4096/Hq8/Hkv2/D128/BF16, mixed mapping/causal mask, three processes with
60 blocks and 32 calls/entry gave synchronized eager time change −0.0437%,
conservative 95% interval [−0.5003%, +0.2500%]. No stable net benefit; this
prototype is not integrated. CPU profiling confirms allocations were removed,
while the original main kernel occupies about 96% of GPU kernel time. The
result does not reject scratch reuse for workloads limited by CPU dispatch.

## EXP02 — bulk sparse-plan construction

### Implementation and iterations

The baseline builds Python lists and performs per-group CUDA allocations,
zeroing and indexed assignments. The candidate classifies tiles on CPU, packs
sorted int32 tables there and uploads complete tables. Dtypes, dimensions,
strides, padding and Full/Partial membership are preserved.

| Iteration | Finding | Correction |
| --- | --- | --- |
| v2 | Large case faster, but consumer rejected empty NumPy-derived index tensors with last stride 0 | Allocate zero-width tensors directly with `torch.empty`, retaining unit last stride |
| v3 | Exact metadata and full operators passed; S128 empty preparation regressed about 6% | Skip CPU list packing entirely when a bucket has width 0 |
| v4 | Empty regression removed; all four measured C-layout cases improved | Initial accepted candidate |
| v5 diagnostic | Merely transposing tile axes does not eliminate multi-axis reduction traversal; queued diagnosis was interrupted after overlap detection | No formal result used |
| v6 | F-strided masks need separate axis reductions; retain original C-layout reductions, avoid full-mask copy | Final candidate; C/F direct A/B and exact ABI recheck |

### Correctness

- v3: 50 exact forward/backward metadata comparisons plus 48 full operator,
  FP32-reference and CUDA Graph configurations passed.
- v4: 50 comparisons additionally checked exact strides; four complete
  empty/full forward/backward/reference/replay checks passed.
- Cases include S128/256/384/512/4096, full/empty/causal/random partial masks
  and transposed CPU masks. `tests.check_sparse_planning` retains an independent
  tile-by-tile oracle and specifically checks the unit-stride empty ABI.

### Fixed measurement and final v4 results

Three independent processes per case, 60 ABBA/BAAB blocks/process,
16 complete forward+backward plan constructions/entry. Synchronized wall
includes CPU classification, allocation and H2D. Input masks are already on
CPU. Mask encoding, index descriptors, attention forward/backward, cold imports
and compilation are outside this scope. No A/B samples were trimmed.

The two time columns are arithmetic descriptions; percentage changes use paired
log-ratios and need not equal the ratio of those columns. Intervals envelope
circular moving-block bootstrap lengths 1/3/5/10 and the
process-level Student-t interval. Each row is a separate comparison; these are
pointwise intervals, not a simultaneous all-shape nonregression certificate.

| S / mask | Row-wise ms | Bulk ms | Time change | 95% interval |
| --- | ---: | ---: | ---: | --- |
| 128 / empty | 0.08572 | 0.08208 | −4.818% | [−8.335%, −0.306%] |
| 128 / full | 0.13307 | 0.10520 | −20.998% | [−22.995%, −19.081%] |
| 512 / sparse | 0.41691 | 0.28131 | −32.446% | [−34.138%, −30.711%] |
| 4096 / causal | 11.17662 | 8.52975 | −23.642% | [−25.992%, −21.217%] |

### Final layout-aware v6 confirmation

Same protocol, S4096 causal, three fresh processes for each input layout.
The final code uses separate axis reductions for F-strided masks without
copying the full mask. C-layout retains the original v4 reduction operations.
All three repeats have identical source/fixture/environment hashes within each
layout and match the integrated planning source byte for byte.

| Layout | Row-wise ms | Bulk ms | Time change | Conservative 95% interval |
| --- | ---: | ---: | ---: | --- |
| C | 11.57791 | 8.77974 | −24.090% | [−31.524%, −15.849%] |
| F | 12.56927 | 3.07536 | −75.533% | [−76.023%, −75.033%] |

Final v6 again passed all 50 exact metadata/stride comparisons and four public
forward/backward FP32-reference/Graph cases (S128/512, full/empty masks).
The local CPU unit suite passed all 10 tests; exported CLI/analysis smoke passed. Its C/F
implementation and performance records are separate from v4. An attempted
queued diagnostic overlapped at a process boundary and was rejected by the
GPU process guard; it produced no usable result and was not included. The
final C/F sweep ran sequentially in one shell and passed all process checks.

### Reproduction

Run `python -m tests.check_sparse_planning`, then the planning commands in
`benchmarks/README.md`. The frozen pre-optimization constructor lives only in
the benchmark package. Default public preparation uses the accepted code.

## EXP03 — equivalent-layout Stats

Three implementations were examined rather than treating the first timing as
proof against the idea. Plain Triton showed a reduction-to-store layout
conversion; a masked 2D store still retained it (40 registers, 128 B shared).
Explicit Gluon warp layouts removed that conversion: 32 registers, 0 B shared,
no spill. All preserve head-major LSE/dPsum, required FP32 dQ clearing and PDL.
The final implementation passed 52 reference/Graph checks.

Final complete GPU backward A/B, three processes × 60 blocks × 8 cycles:

| S / Hkv / mask | Time change | Conservative 95% interval |
| --- | ---: | --- |
| 512 / 2 / causal | −1.666% | [−2.769%, −0.550%] |
| 4096 / 2 / causal | +0.134% | [−0.080%, +0.349%] |
| 8192 / 8 / block-causal | +1.206% | [+1.028%, +1.385%] |

The block-causal fixture uses causal diagonal blocks of 4096 tokens; Hq32,
D128, BF16. No global replacement is integrated. A future group should compare
preprocess tile/warp layout together with the main kernel's statistics layout
and address calculation. Shared-memory elimination alone does not establish a
net speedup. Plain-Triton repeat 0 was excluded from formal aggregation because
its startup overlapped a source revision; its raw diagnostic record is retained.

## EXP04 — shared dK/dV address mapping

The candidate decodes each noncontiguous physical row once and shares it between
dK/dV stores, retaining 128-bit vector writes, thread ownership, barriers and
the contiguous TMA path. 48 reference/Graph cases and four mapping codegen checks
passed. Baseline/candidate each use 168 registers and the same local load/store
counts; no additional resource pressure was found by these static checks.

| S / mapping | Time change | Conservative 95% interval |
| --- | ---: | --- |
| 512 / index | −1.006% | [−1.075%, −0.930%] |
| 4096 / mixed | −0.189% | [−0.226%, −0.150%] |
| 512 / continuous | +0.048% | [−0.029%, +0.129%] |
| 512 / two ranges | +0.109% | [+0.029%, +0.186%] |

Same complete backward protocol as EXP03, three processes per row. Keep this
as an input-dependent candidate, not a global replacement. Small regressions
can arise from code organization even when the corresponding TMA fast path is
unchanged; the exact hardware cause remains unproven.

## Revised experiment groups

| Class | Variables and disposition |
| --- | --- |
| Independently adoptable | CPU sparse-plan packing: accepted. Caller-owned scratch: tested, no stable benefit in this workload; retain for CPU-dispatch-limited cases. Metadata reuse already exists. |
| Addressing group | index vs single/two-range representation × continuity-condition source × statistics addressing/hoisting × compact vs unrolled head loop × shared dK/dV scatter. Preserve conditional index loads; do not introduce unconditional index/base reads to remove branches. |
| Statistics layout group | existing CuTe vs explicit warp layout × preprocess tile/warps × producer output layout and main-kernel consumer addressing; include dQ clear/reset/postprocess in the timed chain. |
| Persistent scheduling group | resident CTA count × task claim timing × pipeline stages/buffering. Do not select CTA count independently of workload occupancy and task distribution. Existing delayed task claiming is already part of the baseline. |

For grouped experiments, pick the best complete combination within each input
bucket, then validate that choice on separate fixtures. Candidate bucket keys:
S, Hq/Hkv, number of active/full/partial tiles, continuous/two/index block
fractions and range lengths. Use metadata already computed during preparation;
account for dispatch/preparation cost and fall back to the baseline when a
bucket lacks evidence. No dynamic selector is implemented by this sweep, and
no exhaustive Cartesian search or private 240-dump replay is claimed.

Historical observations made before the stable crossover timing protocol are
hypotheses to retest, not evidence for fixing defaults. Keep the measurement
scheme unchanged and run direct A/B; do not repeat A/A for each candidate.

## Which historical comparisons need repeating

- Pre-calibration hot64/21-field formats, condition-source-only changes,
  affine/hoisted statistics, rolled/unrolled loops and KV TMA/barrier changes:
  retest as addressing/transfer groups; old timings do not fix defaults.
- The stable-protocol 2026-11-02 QKV-range and all-range experiments already
  have direct A/B evidence. Do not repeat them merely because another variable
  is being considered. Reuse those records; run new combinations when producer,
  barrier or gradient-store organization changes.
- Old 44-field linear run construction is a preparation-only candidate for the
  old run-cache representation. The current compact descriptor does not use
  that ABI; only revisit it together with a run-cache design, not as a drop-in
  independent toggle on the current operator.
- Q/dO paired loading has its own later stable-protocol/full-input report and
  is already present in the current baseline; do not treat it as untested.

## Existing baseline features

Prepared metadata is already caller-owned and reused, Q/dO paired loading is
already shared, and the scheduler already claims its next task after the
required rendezvous. These are retained as existing features rather than
invented new performance switches. No resident-CTA scan was performed.

## Raw evidence

Workspace artifacts: `artifacts/independent_operators_20261103/` in the XLM
workspace, outside this repository. `planning-v4-summary.json`, `planning-v6-summary.json` and `gpu-summary.json`
contain the aggregate calculations; `plan-v4-<S>-<mask>-<repeat>.json` holds every block,
GPU process/clock telemetry, affinity, correctness and source hashes. Earlier
failures and intermediate results remain available and are not pooled with v4.
No TP/GTP full-network speed or MFU claim is made.
