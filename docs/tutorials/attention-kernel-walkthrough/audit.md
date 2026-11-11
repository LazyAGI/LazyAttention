# Tutorial correctness audit — 2026-11-11

The audit covers the nine navigation chapters, both MAGI/FA4 practice tabs,
shared renderers, generated mask/plan datasets and their exporters. It also
checks the README, benchmark instructions, CTA traversal/task planning notes
and the experiment/evidence appendices against the documented implementation
scope. Vendored KaTeX and the FA4 source snapshots remain reference assets.

## Corrected explanations

- **U1 temporal position is not a patch index.** The U1-style primer now gives
  all tokens of one image a constant `token_pos`, with distinct spatial `h/w`.
  Patch count and spatial coordinates may change without changing that image's
  temporal coordinate. Token ID is independent of position. Diffusion timestep
  is a separate conditioning variable.
- **Clean target copies follow GEN in the logical conversation.** The paired
  example uses a separate image-segment ID for the clean UND copy and reuses the
  GEN temporal coordinate. The physical GEN-first partition is an execution
  layout, not an autoregressive ordering rule.
- **The equal-position gate removes the entire corresponding image block.**
  Every GEN/copied-UND pair has equal temporal position. Equality passes the
  base causal comparison, so the additional gate is necessary even though the
  copy occurs later in the conversation. GEN self-attention remains Full;
  reference images remain available; later UND text can read the clean copy.
- **`duplicated_for_und` and `dup_boundary` are different.** The former identifies
  the clean image copy. The latter marks selected boundary tokens whose K
  columns are excluded for every Q. It does not globally hide the clean image
  copy from later text.
- **Synthetic diagonal holes are not U1 image examples.** The supplementary
  same-mod/incrementing-position fixture deliberately exercises a general
  predicate branch. D3 and its MAGI/FA4 consumers are labelled as artificial
  operator tests. They do not establish how real target copies are laid out.
  Their numerical ranges are retained so the algorithm demonstrations continue
  to cover Full, Causal and Inverse-causal slices.
- **Attention information flow is directional.** The architecture diagram now
  says GEN may read permitted UND K/V, while UND cannot read GEN K/V.
- **Softmax has one denominator per query/head.** For a GEN query, permitted
  keys from both branches participate in that same denominator. Reordering does
  not combine different queries' denominators, nor normalize each key branch
  independently.
- **Captured pre-partition order is a reconstruction.** A stable merge by
  temporal position produces a compatible interleaving; ties prevent recovery
  of a unique historical order. A successful round trip proves compatibility,
  not historical identity. Atlas and captured-case captions now say so.
- **Preparation is not necessarily one GPU launch.** The MAGI execution diagram
  distinguishes framework initialization from the main and zero-fill kernels.
- **42-way preparation amortization is conditional.** Reuse requires unchanged
  layout/mask, geometry, device and execution configuration. Forty-two is the
  measured experiment's assumption, not a universal model constant.
- **Exporters use explicit inputs.** Synthetic fixtures use the repository's
  frozen range converter. Historical exporters require an explicit workspace
  containing captures, rather than assuming a private sibling checkout or an
  incorrect path relative to `docs/`.

## Public model references

Model-specific statements were checked against the public SenseNova-U1 source
snapshot `9b92332171a037dce2cbec83cc29482e45e61217`:

- [Interleaved image duplication and label masking](https://github.com/OpenSenseNova/SenseNova-U1/blob/9b92332171a037dce2cbec83cc29482e45e61217/training/sensenovavl/data/multimodal_dataset.py)
- [Attention gates, boundary marking and GEN-first packing](https://github.com/OpenSenseNova/SenseNova-U1/blob/9b92332171a037dce2cbec83cc29482e45e61217/training/sensenovavl/model/sensenovavl_moe_chat/modeling_sensenovavl_chat_mot.py)

The introductory fixture omits special boundary tokens for readability and
reduces each segment to three tokens. It is an illustration of the U1 mask,
not a complete training sample or a claim about every multimodal architecture.

## Reproduce CPU validation

From the repository root, with Node.js and Python >= 3.10 plus NumPy:

```bash
python -m pip install -e .
python docs/tutorials/attention-kernel-walkthrough/build_dummy.py
python docs/tutorials/attention-kernel-walkthrough/build_fa4_plan.py
node scripts/check_tutorial_models.js
node scripts/check_bank_model.js
node scripts/check_optimization_models.js
python scripts/check_site.py
```

The model checks exercise the rendered primer's actual predicate, complete
GEN-to-clean-copy isolation, GEN self/reference access, later-text access,
boundary and document gates, integer range bands, all 69 dummy ranges' loop
coverage and sampled-key online softmax. The matrix-chain and full/causal
attention examples are checked against 144 finite-difference derivatives.
The 32K export independently compares 64,130,418 visible pairs against range
coverage, checks for overlaps and checks 100,000 reordered pairs.

Historical captures are not distributed. For a trusted local workspace with
the original `artifacts/` hierarchy, use:

```bash
python docs/tutorials/attention-kernel-walkthrough/build_mask_data.py --workspace-root /path/to/workspace
python docs/tutorials/attention-kernel-walkthrough/build_worst_case.py --workspace-root /path/to/workspace
python docs/tutorials/attention-kernel-walkthrough/build_mask_atlas.py --workspace-root /path/to/workspace
```

During this audit those exporters were rerun with the historical captures:
80 workload metadata files, 510 captured plan rows for rank 5 / batch 24,
and the 24-case selection from the 240-case corpus reproduced the checked-in
numerical data. This validates the exporters and stored metadata; it does not
repeat a GPU correctness or performance experiment. Existing H800/H100 timing
claims retain their original scope, provenance and limitations.
