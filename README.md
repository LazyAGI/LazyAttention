# LazyAttention

LazyAttention is a workspace for understanding, implementing, and benchmarking optimized Attention kernels.

## Interactive tutorial

The current site explains Attention from the matrix equations to tiled Forward and Backward execution, then compares MAGI and FA4 practice with interactive CTA traversal.

- [Open the published tutorial](https://lazyagi.github.io/LazyAttention/)
- Source: [`docs/tutorials/attention-kernel-walkthrough/`](docs/tutorials/attention-kernel-walkthrough/)

## Repository layout

```text
LazyAttention/
├── .github/workflows/       # CI and GitHub Pages deployment
├── docs/                    # Static documentation site
│   └── tutorials/
│       └── attention-kernel-walkthrough/
├── lazy_attention/          # Planned Python APIs and integrations
├── csrc/                    # Planned CUDA/CUTLASS/Triton kernels
├── tests/                   # Planned correctness tests
├── benchmarks/              # Planned performance and accuracy benchmarks
└── examples/                # Planned runnable examples
```

Only directories with committed content are present today. Future optimization code should keep public APIs in `lazy_attention/`, device kernels in `csrc/`, and reproducible measurements in `benchmarks/`; the independently deployable website remains under `docs/`.

## Local preview

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://127.0.0.1:8000/>.

Every push to `main` validates the static site and deploys `docs/` to GitHub Pages.
