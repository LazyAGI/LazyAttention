# Third-party notices

The repository's original code is Apache-2.0 (see LICENSE). The kernels and
launchers under `lazy_attention/csrc/fa4/` derive from
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
