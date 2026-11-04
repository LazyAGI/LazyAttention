# Experimental backward task planning

Two independent switches live in `lazy_attention.task_planning.plan_tasks`:

- `reorder`: the previous coarse planner, now called **task reordering**. It orders sample groups by coarse Q-tile cost and retains the existing short-task bypass.
- `split`: partitions a KV/head task's existing Partial+Full Q64 iterator into pieces of at most `split_size` tiles (default128). It does not change the four Q heads inside each tile.

```python
from lazy_attention import backward
from lazy_attention.task_planning import plan_tasks

# documents: int32/int64 logical-row document IDs; padding=-1.
# Prepare once, before graph capture; reuse while mask/mapping/geometry match.
plan = plan_tasks(prepared.backward_plan, documents, q.shape[1], k.shape[2],
                  reorder=True, split=True, split_size=128)
# Each activation/layer invocation uses its own q/k/v/out/dout/lse.
dq, dk, dv = backward(q, k, v, out, dout, lse, prepared, task_plan=plan)
```

With both switches false, `plan_tasks` immediately returns `None` without reading inputs, allocating metadata or launching GPU work. `backward(..., task_plan=None)` uses the unplanned compiled path and carries no task-order argument. A requested strategy can also return `None` when its workload bypass applies. Neither feature is enabled by default.

Split plans hold immutable task bounds and sparse reduction indices. Unsplit tasks retain the existing BF16 gradient stores. Truly split tasks store private FP32 dK/dV fragments; a final FP32 reduction maps them back to physical rows and converts to BF16. dQ retains the existing FP32 atomic accumulation. Scratch gradients are recomputed on every execution; only metadata is reused. The public API supports graph replay; planning contains CPU work/device synchronization and must happen outside capture.

Current experimental scope is the existing Hopper D128/GQA4 self-attention geometry, 64x128 sparse plans, at most1024 KV tiles. Splitting currently requires the paired row-map variant. Preparing with CPU or GPU documents is supported; all metadata copies and synchronization belong to preparation cost. There is no implicit cache, shape whitelist or automatic global enablement.

Benchmark entry point: `python -m benchmarks.splitting --mode add-split ...` compares reordering against reordering plus splitting; `--mode split-only` compares no plan against splitting alone. Both sides are charged their own synchronized preparation time /42, plus the complete GPU chain including dQ postprocessing and dK/dV split reduction. Existing graph event, pairing and buffer-crossover mechanics are unchanged. Benchmark allocation/CPU launch exclusions remain those of the existing operator harness. Single-process diagnostics are not three-process acceptance.
