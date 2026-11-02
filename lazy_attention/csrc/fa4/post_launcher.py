# SPDX-License-Identifier: BSD-3-Clause
# Derived from FlashAttention CuTe; see THIRD_PARTY_NOTICES.md.
"""Shared dQ conversion cache with policy-specific row-map shape handling."""

import threading
import cutlass
import cutlass.cute as cute
from lazy_attention.csrc.fa4.postprocess import FlashAttentionBackwardPostprocess

_lock = threading.RLock()
_cache = {}


def postprocess(accum, output, mapping, scale, *, range_mapping):
    tensors = (accum, output, mapping)
    dynamic = all(t.is_contiguous() for t in tensors)
    if dynamic:
        # Baseline index has dynamic S; compact descriptors have static size.
        layout = (
            "sequence_dynamic",
            output.shape[2],
            mapping.numel() if range_mapping else None,
        )
    else:
        layout = tuple((tuple(t.shape), tuple(t.stride())) for t in tensors)
    key = (range_mapping, layout, output.dtype)
    with _lock:
        if key not in _cache:
            obj = FlashAttentionBackwardPostprocess(range_mapping)
            views = [cute.runtime.from_dlpack(t, assumed_align=16) for t in tensors]
            if dynamic:
                for i, mode in enumerate((2, 1) if range_mapping else (2, 1, 0)):
                    views[i] = views[i].mark_compact_shape_dynamic(
                        mode=mode, stride_order=tensors[i].dim_order(), divisibility=128
                    )
            _cache[key] = cute.compile(
                obj,
                *views,
                cutlass.Float32(0.0),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        compiled = _cache[key]
    compiled(accum, output, mapping, scale)
