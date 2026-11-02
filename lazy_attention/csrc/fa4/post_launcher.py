# Derived from FlashAttention CuTe; see licenses/flash-attention-BSD-3-Clause.txt.
# Modified: persistent GQA4 scheduling and exclusive range/index row mapping.
from functools import lru_cache
import threading
import torch
import cutlass
import cutlass.cute as cute
from lazy_attention._loader import private_module
_lock = threading.RLock()
_post_cache = {}
@lru_cache(maxsize=1)
def _postprocess_module():
    return private_module('lazy_attention.csrc.fa4.postprocess', 'postprocess')

def postprocess(accum, output, index, scale, arch, dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
                hdim_multiple_of):
    # Only the contiguous padded native contract shares a sequence-dynamic
    # specialization. Keep all other layouts on their exact static signature.
    dynamic = (accum.ndim == 3 and output.ndim == 4 and index.ndim == 1
               and accum.is_contiguous() and output.is_contiguous() and index.is_contiguous()
               and output.shape[0] == 1 and output.shape[1] > 0 and output.shape[1] % 128 == 0
               and output.shape[3] == hdim == 128
               and tuple(accum.shape) == (1, output.shape[2], output.shape[1] * hdim)
               and index.numel() >= 2 * (output.shape[1] // 64)
               and accum.dtype == torch.float32 and index.dtype == torch.int32)
    if dynamic:
        layout_key = ('sequence_dynamic', output.shape[0], output.shape[2], output.shape[3], index.numel())
    else:
        layout_key = ('static', tuple(accum.shape), tuple(accum.stride()), tuple(output.shape),
                      tuple(output.stride()), tuple(index.shape), tuple(index.stride()))
    key = (layout_key, accum.dtype, output.dtype, index.dtype,
           arch, dtype, hdim, block_size, num_threads, atom_layout, swap_ab, hdim_multiple_of)
    with _lock:
        if key not in _post_cache:
            obj = _postprocess_module().FlashAttentionBackwardPostprocess(
                dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
                hdim_multiple_of=hdim_multiple_of)
            tensors = [cute.runtime.from_dlpack(t, assumed_align=16) for t in (accum, output, index)]
            if dynamic:
                # Make S and the compact strides derived from S symbolic in
                # all three arguments. Dropping S from the cache key alone
                # would incorrectly reuse a statically compiled layout.
                tensors = [view.mark_compact_shape_dynamic(
                    mode=mode, stride_order=tensor.dim_order(), divisibility=128)
                    for view, tensor, mode in zip(tensors[:2], (accum, output), (2, 1))] + tensors[2:]
            _post_cache[key] = cute.compile(
                obj, *tensors, cutlass.Float32(0.), None, None, None, None,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi')
        compiled = _post_cache[key]
    compiled(accum, output, index, scale, None, None, None, None)
