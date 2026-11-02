# Third-party notices

The repository's original code is Apache-2.0 (see LICENSE). The shared backward, launcher, scheduler, compact mask and dQ postprocessor
under `lazy_attention/csrc/fa4/` derive from
[FlashAttention CuTe](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)
and retain its BSD-3-Clause terms; the complete notice is in
[licenses/flash-attention-BSD-3-Clause.txt](licenses/flash-attention-BSD-3-Clause.txt).
Existing per-file copyright notices are retained. Modifications implement
persistent four-query-head scheduling, compact masks, indexed row access,
exclusive range/index descriptors, paired transfers, and matching writeback.

[QuACK](https://github.com/Dao-AILab/quack) (Apache-2.0), NVIDIA CUTLASS DSL
(BSD-3-Clause), PyTorch, CUDA Python and TVM FFI are external dependencies;
their source is not vendored here. Their own distributions supply their notices.
This repository does not include pretrained models, datasets, workload dumps,
or training-framework integrations.

## Source attribution

| Files | Origin / license |
| --- | --- |
| `csrc/fa4/backward.py` | FlashAttention CuTe `flash_bwd_sm90.py`, with persistent GQA4 and row-map changes; BSD-3-Clause |
| `csrc/fa4/interface.py` | FlashAttention CuTe `interface.py`, specialized launch chain; BSD-3-Clause |
| `csrc/fa4/postprocess.py` | FlashAttention CuTe `flash_bwd_postprocess.py`, Hopper conversion and mapped stores; BSD-3-Clause |
| `csrc/fa4/scheduler.py`, `mask.py`, `post_launcher.py` | Adapted FA4 scheduling, masking and launch support; BSD-3-Clause |
| `csrc/fa4/row_map.py`, `paired_copy.py` | Project-specific row descriptors and paired transfers; Apache-2.0 |
| Python API, planning, benchmark and tests | Project code; Apache-2.0 |

Paths above are relative to `lazy_attention/`. Per-file SPDX headers identify
the applicable license. Calling an external dependency does not by itself
change the license of the caller. The source archive used during development
had no Git revision metadata; tested upstream file fingerprints are recorded
in `benchmarks/environment.py`.
