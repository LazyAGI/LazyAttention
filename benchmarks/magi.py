# SPDX-License-Identifier: Apache-2.0
"""Optional MAGI adapter. Metadata preparation is outside the timed backward."""

import importlib
import inspect
import hashlib
from pathlib import Path
import numpy as np


def encode_mask(mask):
    """Disjoint physical Q/K ranges; exact, general packing, not an autotuner."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or not len(mask):
        raise ValueError("mask must be a nonempty square matrix")
    size = len(mask)
    if mask.all():
        rows = [(0, size, 0, size, 0)]
    elif np.array_equal(mask, np.tri(size, dtype=bool)):
        rows = [(0, size, 0, size, 1)]
    else:
        rows = []
        active = {}
        for q, line in enumerate(mask):
            edges = np.flatnonzero(np.diff(np.r_[False, line, False])).tolist()
            segments = set(zip(edges[::2], edges[1::2]))
            for segment in list(active):
                if segment not in segments:
                    rows.append((active.pop(segment), q, *segment, 0))
            for segment in segments:
                active.setdefault(segment, q)
        rows.extend((start, size, *segment, 0) for segment, start in active.items())
        rows.sort()
    data = np.asarray(rows, dtype=np.int32).reshape(-1, 5)
    return data[:, :2].copy(), data[:, 2:4].copy(), data[:, 4].copy()


def validate_ranges(mask, arrays):
    """Reject overlap as well as missing/extra attention entries."""
    mask = np.asarray(mask, dtype=bool)
    size = len(mask)
    qr, kr, kinds = [np.asarray(x) for x in arrays]
    if (
        qr.shape != kr.shape
        or qr.ndim != 2
        or qr.shape[1] != 2
        or kinds.shape != (len(qr),)
    ):
        raise ValueError("invalid MAGI range shapes")
    if any(not np.issubdtype(x.dtype, np.integer) for x in (qr, kr, kinds)):
        raise ValueError("MAGI ranges/types must contain integers")
    coverage = np.zeros_like(mask, dtype=bool)
    for (qs, qe), (ks, ke), kind in zip(qr, kr, kinds):
        if not (0 <= qs < qe <= size and 0 <= ks < ke <= size) or kind not in (0, 1, 2):
            raise ValueError(
                "supports bounded nonempty ranges of type full=0, causal=1 or inverse-causal=2"
            )
        tile = np.ones((qe - qs, ke - ks), dtype=bool)
        if kind == 1:
            tile = np.arange(qe - qs)[:, None] >= np.arange(ke - ks)[None, :] + (
                qe - qs
            ) - (ke - ks)
        if kind == 2:
            tile = np.arange(qe - qs)[:, None] <= np.arange(ke - ks)[None, :]
        if (coverage[qs:qe, ks:ke] & tile).any():
            raise ValueError("overlapping MAGI attention entries")
        coverage[qs:qe, ks:ke] |= tile
    if not np.array_equal(coverage, mask):
        raise ValueError("MAGI ranges do not represent the physical mask")


class MagiBackward:
    def __init__(self, fixture):
        import torch

        self.torch = torch
        try:
            self.module = importlib.import_module(
                "magi_attention.functional.flex_flash_attn"
            )
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Install a compatible MagiAttention library; see benchmarks/README.md"
            ) from exc
        self.q, self.k, self.v, self.do = [
            fixture[name][0] for name in ("q", "k", "v", "dout")
        ]
        inverse = torch.argsort(fixture["index"].cpu().long()).numpy()
        mask = fixture["logical_mask"].cpu().numpy()[inverse][:, inverse]
        names = ("magi_q_ranges", "magi_k_ranges", "magi_attn_type_map")
        supplied = [name in fixture for name in names]
        if any(supplied) and not all(supplied):
            raise ValueError("supply all three MAGI range tensors")
        arrays = (
            tuple(fixture[name].cpu().numpy() for name in names)
            if all(supplied)
            else encode_mask(mask)
        )
        validate_ranges(mask, arrays)
        self.ranges = tuple(
            torch.as_tensor(x, device=self.q.device, dtype=torch.int32) for x in arrays
        )
        out, meta = self.module.flex_flash_attn_func(
            self.q,
            self.k,
            self.v,
            *self.ranges,
            softmax_scale=128**-0.5,
            auto_range_merge=False,
            disable_fwd_atomic_reduction=False,
        )
        self.out = out
        self.lse = meta.lse if hasattr(meta, "lse") else meta
        self.workspace = {}
        signature = inspect.signature(self.module._flex_flash_attn_backward)
        self.parameters = set(signature.parameters)
        self.required = {
            name
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }
        range_digest = hashlib.sha256()
        for array in arrays:
            range_digest.update(repr((str(array.dtype), array.shape)).encode())
            range_digest.update(array.tobytes())
        self.info = dict(
            ranges_sha256=range_digest.hexdigest(),
            source_sha256=hashlib.sha256(
                Path(self.module.__file__).read_bytes()
            ).hexdigest(),
            ranges=len(arrays[0]),
            range_source="fixture" if all(supplied) else "generic_exact_encoder",
            mask_types=np.unique(arrays[2]).tolist(),
            api_signature=str(signature),
            accumulation_dtype="float32",
            output_dtype="bfloat16",
            auto_range_merge=False,
        )

    def track_compiles(self, counter):
        try:
            jit = importlib.import_module("magi_attention.common.jit.core")
            cls = jit.JitSpec
            original = cls.build
        except (ImportError, AttributeError):
            self.info["jit_build_tracking"] = False
            return lambda: None

        def build(obj, *args, **kwargs):
            counter[0] += 1
            return original(obj, *args, **kwargs)

        cls.build = build
        self.info["jit_build_tracking"] = True

        def restore():
            cls.build = original

        return restore

    def __call__(self, pool_id, outputs):
        torch = self.torch
        if pool_id not in self.workspace:
            self.workspace[pool_id] = tuple(
                torch.empty_like(x, dtype=torch.float32)
                for x in (self.q, self.k, self.v)
            )
        dq, dk, dv = self.workspace[pool_id]
        # Supplied outputs bypass MAGI's zeros_like allocations; retain all
        # required clears inside the timed chain, before atomic reductions.
        for tensor in (dq, dk, dv):
            tensor.zero_()
        kwargs = dict(
            dout=self.do,
            q=self.q,
            k=self.k,
            v=self.v,
            out=self.out,
            lse=self.lse,
            sink=None,
            sink_layout="sh",
            dq=dq,
            dk=dk,
            dv=dv,
            dsink=None,
            q_ranges=self.ranges[0],
            k_ranges=self.ranges[1],
            attn_type_map=self.ranges[2],
            softmax_scale=128**-0.5,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=False,
            sm_margin=0,
            auto_range_merge=False,
        )
        required = {
            "dout",
            "q",
            "k",
            "v",
            "out",
            "lse",
            "dq",
            "dk",
            "dv",
            "q_ranges",
            "k_ranges",
        }
        if not required <= self.parameters:
            raise RuntimeError(
                "unsupported MAGI backward API: " + self.info["api_signature"]
            )
        missing = self.required - kwargs.keys()
        if missing:
            raise RuntimeError(
                "unsupported required MAGI arguments: " + str(sorted(missing))
            )
        self.module._flex_flash_attn_backward(
            **{k: v for k, v in kwargs.items() if k in self.parameters}
        )
        for dst, src in zip(outputs, (dq, dk, dv)):
            dst.copy_(src.unsqueeze(0))
        return outputs
