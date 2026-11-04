# LazyAttention

LazyAttention is a workspace for understanding, implementing, and benchmarking optimized Attention kernels.

## Interactive tutorial

The current site explains Attention from the matrix equations to tiled Forward and Backward execution, then compares MAGI and FA4 practice with interactive CTA traversal.

- [Open the published tutorial](https://lazyagi.github.io/LazyAttention/)
- Source: [`docs/tutorials/attention-kernel-walkthrough/`](docs/tutorials/attention-kernel-walkthrough/)

## Experimental operators (`dev`)

The standalone package contains the **index baseline** and the current
**paired range/index** Hopper backward kernel. No intermediate range variant
or training-framework code is included. The website remains independent.

```text
lazy_attention/             # explicit prepare / forward / backward API
  csrc/fa4/                 # shared CuTe backward/launcher; row-map policy differs
benchmarks/                 # synthetic/replay fixtures, paired A/B and analysis
tests/                      # CPU descriptor and Hopper numerical/graph checks
licenses/                   # upstream kernel notices
```

This is a narrow experimental specialization: Hopper SM90, BF16, batch 1,
self-attention, positive S divisible by 128, head dimension 128, and four Q heads
per KV head. It exposes explicit backward rather than an autograd integration.
Unsupported shapes fail instead of silently switching implementations.

The two policies share one backward implementation, scheduler, launcher and dQ
postprocessor. Mapping and KV scatter differences use compile-time
`range_mapping`; no runtime variant dispatch is added to GPU code. The index
scheduler retains 320 barrier participants, while paired scatter requires 384.
The previous standalone implementation is preserved in commit `3047921` for
regression comparisons; it is not duplicated in the current source tree.

### Setup

Use Python 3.10+ with a CUDA-enabled PyTorch installation and an importable
[FlashAttention CuTe source tree](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
The legacy `flash-attn` package version alone does not identify the CuTe API.
Install this package and its dependencies from the repository root:

```bash
pip install -e '.[gpu,analysis]'
# If FA4 is provided as a source checkout, expose its parent directory:
export PYTHONPATH=/path/to/flash-attention:$PYTHONPATH
python -m benchmarks.environment
python -m unittest tests.test_planning
CUDA_VISIBLE_DEVICES=0 python -m tests.check_gpu
```

Tested software: Python 3.12.12, PyTorch 2.13.0+cu126, CUDA 12.6,
CUTLASS DSL 4.6.2, QuACK 0.5.3 and TVM FFI 0.1.13.post3.
The tested upstream source was an archive without Git metadata; its key file
fingerprints are recorded by `benchmarks.environment`. A different FA4 snapshot
needs compatibility/correctness validation, even if its package version matches.
The GPU extras do not install or pin that FA4 source snapshot automatically.

### Use

```python
from lazy_attention import prepare_attention, prepare_execution, forward, backward

# logical_mask: bool[S,S]; index: int logical-to-physical permutation[S].
# q/k/v/dout: physical row order [1,S,H,128], CUDA BF16.
prepared = prepare_attention(logical_mask, index, device=q.device,
                             documents=logical_document_ids, kv_heads=k.shape[2])
# Defaults: reorder=True, split=False, head_major=False,
#           reverse=True (Full-reversed then Partial-reversed), interleave=0.
out, lse = forward(q, k, v, prepared)
dq, dk, dv = backward(q, k, v, out, dout, lse, prepared)

# Independent pure-index baseline; prepare once outside measurement.
baseline = prepare_attention(logical_mask, index, device=q.device, strategy="index",
                             reorder=False, reverse=False)
dq0, dk0, dv0 = backward(q, k, v, out, dout, lse, baseline, variant="index")
```

`documents` contains document IDs in logical row order. Omit it to use each
KV128 tile as a separate scheduling group. Providing `kv_heads` finishes
planning before CUDA Graph capture; otherwise the first eager `backward` caches
one plan per KV head count. Subsequent executions do not replan. Mask, row
mapping, document IDs and prepared metadata must remain immutable across reuse.
`prepare_execution(prepared, documents=..., kv_heads=...)` configures existing
captured metadata with the same defaults. Both planning switches off cache
`None`; the kernel receives no task-order table. Auto policy can also return
`None` when it has no profitable supported change. Planning and traversal
preparation should be charged once per reuse window, e.g. cost/42.

The optimized prepared state retains no original full row index. Each logical
64-row block stores a base for one range, two bases plus a split for two ranges,
or 64 indices for fallback. Q/dO, K/V, statistics and gradient stores decode the
same descriptor. Range blocks never retain or load an additional row index;
index fallback keeps the original vector/gather loading policy. Mask metadata
and sparse tile lists are separate from row mapping. Caller-owned source
`index` may be released after preparation.

See [the benchmark protocol, MAGI comparison and report commands](benchmarks/README.md). Default A/B
is index → current paired kernel. The committed code includes no private input
dumps or performance logs, and does not claim a universal speedup.

## Local preview

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://127.0.0.1:8000/>.

The new [optimization chapter](docs/tutorials/attention-kernel-walkthrough/#optimizations) explains adopted strategies, measured gains and rejected or pending experiments.

Every push to `main` validates the static site and deploys `docs/` to GitHub Pages.
