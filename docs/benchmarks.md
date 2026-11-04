# Reproduce the operator comparison

Install and check the dependencies as described in the repository [README](https://github.com/LazyAGI/LazyAttention/blob/dev/README.md) and [benchmark protocol](https://github.com/LazyAGI/LazyAttention/blob/dev/benchmarks/README.md).

The full captured-corpus comparison is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.magi240 \
  --capture-root /path/to/captures --output-root results/magi240
```

Captures are user supplied, not distributed with this site. The runner requires all 240 cases and an exclusive Hopper GPU with sufficient free memory. It uses a single persistent process and reserves GPU memory for case reuse. Compare on the same host/GPU; preparation is charged once per42 backward executions. `tests.check_default_execution` verifies public defaults against the measured private launcher, including42 CUDA Graph replays; it does not repeat performance A/A.
