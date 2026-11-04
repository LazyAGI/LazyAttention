# Joint2x2 policy

`plan_tasks(..., reorder=True, split=True, policy="auto")` enables a one-time
choice among none/reorder/split/both. The two switches are permissions in auto
mode. Use `policy="manual"` for legacy explicit controls; no caller
shape whitelist is needed. Both switches false returnsNone before inspecting
inputs or allocating task metadata. The returned Tensor table is reused across
42 activations; execution does not rerun the decision.

Feature inputs are sequence/head geometry, active tasks relative to SM count,
mask-derived max/mean Q-tile counts, fraction above the split cap, Full fraction,
and whether document-group reordering would change task order. p95/CV and group
count are logged for diagnosis. No activation values are inspected.

The current experimental rule follows the full240-case H100 matrix:

- Empty mask or an unchanged task permutation: none.
- High occupancy and nonidentity document-group ordering: reorder only. The
  former max-Q<=128 bypass is removed: it lost0.16% aggregate time on this corpus.
- The previously measured underfilled-GPU route remains separate: active tasks
  below half the SM count and mean active work>=4 can split into minimum2-tile
  chunks chosen from total work and SM count. Its old short-reorder restriction
  is retained; the240-case corpus does not cover this small-shape regime.
- Long/full-mask thresholds no longer automatically enable splitting. Manual
  splitting and the split kernel remain available for future implementation work.

For the240 cases, planning/42 plus backward averaged14.9056ms without planning,
13.5146ms with reorder,19.4227ms with split,13.5877ms with both. One process ran
all cases and four corners in one round; no A/A calibration or three-process
repetition was added. A perfect per-case lookup could save only another0.44%
relative to reorder. Simple feature rules did not show useful held-out gain.
This is a result for the captured Hopper D128/GQA4 workloads, not a universal
claim against splitting. See the workspace full240 experiment report for raw data.

`policy="manual"` retains the historical short-workload bypass for reproduction.
The full240 factorial harness explicitly calls `build_task_order(...,
preserve_short=False)` for the reorder treatment, including short work, to avoid
letting that old bypass pre-decide the experiment.

Pass the whole `PreparedAttention` to reuse immutable CPU counts already produced
by ordinary mask preparation. Feature analysis, grouping, selection and uploads
are still charged as planning. Passing a raw backward sparse plan also works,
with the required GPU-to-CPU count transfer charged to planning.

`diagnostics={}` optionally receives the selected strategy, reason, features,
actual split size and count source. It is produced only during preparation;
`None` still means no table argument or reads in the compiled operator.

# Experimental backward task planning

Two independent switches live in `lazy_attention.task_planning.plan_tasks`:

- `reorder`: the previous coarse planner, now called **task reordering**. It orders sample groups by coarse Q-tile cost and retains the historical short-task bypass in manual mode.
- `split`: partitions a KV/head task's existing Partial+Full Q64 iterator into pieces of at most `split_size` tiles (default128). It does not change the four Q heads inside each tile.

```python
from lazy_attention import backward
from lazy_attention.task_planning import plan_tasks

# documents: int32/int64 logical-row document IDs; padding=-1.
# Prepare once, before graph capture; reuse while mask/mapping/geometry match.
plan = plan_tasks(prepared, documents, q.shape[1], k.shape[2],
                  reorder=True, split=True, split_size=128)
# Each activation/layer invocation uses its own q/k/v/out/dout/lse.
dq, dk, dv = backward(q, k, v, out, dout, lse, prepared, task_plan=plan)
```

With both switches false, `plan_tasks` immediately returns `None` without reading inputs, allocating metadata or launching GPU work. `backward(..., task_plan=None)` uses the unplanned compiled path and carries no task-order argument. A requested strategy can also return `None` when its workload bypass applies. Neither feature is enabled by default.

Split plans hold immutable task bounds and sparse reduction indices. Unsplit tasks retain the existing BF16 gradient stores. Truly split tasks store private FP32 dK/dV fragments; a final FP32 reduction maps them back to physical rows and converts to BF16. dQ retains the existing FP32 atomic accumulation. Scratch gradients are recomputed on every execution; only metadata is reused. The public API supports graph replay; planning contains CPU work/device synchronization and must happen outside capture.

Current experimental scope is the existing Hopper D128/GQA4 self-attention geometry, 64x128 sparse plans, at most1024 KV tiles. Splitting currently requires the paired row-map variant. Preparing with CPU or GPU documents is supported; all metadata copies and synchronization belong to preparation cost. There is no implicit cache, shape whitelist or automatic global enablement.

Benchmark entry point: `python -m benchmarks.splitting --mode add-split ...` compares reordering against reordering plus splitting; `--mode split-only` compares no plan against splitting alone. Both sides are charged their own synchronized preparation time /42, plus the complete GPU chain including dQ postprocessing and dK/dV split reduction. Existing graph event, pairing and buffer-crossover mechanics are unchanged. Benchmark allocation/CPU launch exclusions remain those of the existing operator harness. Single-process diagnostics are not three-process acceptance.
