# SPDX-License-Identifier: Apache-2.0
"""One-time permutations of the existing Partial and Full Q64 index lists."""
def permute_q_lists(indices, counts, *, reverse=False, interleave=0):
    import numpy as np
    if interleave not in (0,8):
        raise ValueError('interleave must be0 or8 Q tiles')
    if not reverse and not interleave:
        return indices
    a=np.asarray(indices);c=np.asarray(counts)
    if a.shape[:-1]!=c.shape or (c<0).any() or (c>a.shape[-1]).any():
        raise ValueError('invalid sparse Q lists/counts')
    # Captured arrays can have512 columns while Partial lists use only a
    # handful. Preserve the padding but avoid running permutation arithmetic
    # and gather across it. GPU tensor shape/content remain unchanged.
    used=int(c.max(initial=0))
    out=a.copy()
    if not used:
        return out
    pos=np.arange(used,dtype=np.int32)
    src=np.broadcast_to(pos,a.shape[:-1]+(used,))
    if interleave:
        base=(pos//interleave)*interleave
        length=np.clip(c[...,None]-base,1,interleave)
        # Q lists have shape[B,H,KV-tile,Q-slot]. All four query heads of a
        # CTA share the same permutation, preserving Q-major metadata reuse.
        phase=np.arange(c.shape[-1],dtype=np.int32).reshape((1,)*(c.ndim-1)+(c.shape[-1],1))%length
        src=base+(pos-base+phase)%length
    if reverse:
        src=c[...,None]-1-src
    src=np.where(pos<c[...,None],src,pos)
    out[...,:used]=np.take_along_axis(a[...,:used],src,axis=-1)
    return out


def prepare_q_traversal(prepared, *, reverse=False, interleave=0, indices_cpu=None):
    """Return immutable metadata; all copies belong to preparation cost /42."""
    if prepared.q_reverse == reverse and prepared.q_interleave == interleave:
        return prepared
    import numpy as np
    import torch
    from dataclasses import replace
    old=prepared.base_backward_plan or prepared.backward_plan
    if not reverse and not interleave:
        return replace(prepared,backward_plan=old,q_reverse=False,q_interleave=0,
                       execution_cache={})
    # GPU-built plans have no CPU copies. Keep their exact list permutation on
    # device instead of downloading both count/index arrays and uploading them.
    if indices_cpu is None and prepared.task_indices_cpu is None and old.mask_block_idx.is_cuda:
        from .csrc.fa4.q_traversal import permute_q_lists_gpu
        updates = {
            name: permute_q_lists_gpu(indices, counts, reverse=reverse, interleave=interleave)
            for name, indices, counts in (
                ('mask_block_idx', old.mask_block_idx, old.mask_block_cnt),
                ('full_block_idx', old.full_block_idx, old.full_block_cnt))
        }
        return replace(prepared, backward_plan=old._replace(**updates), base_backward_plan=old,
                       q_reverse=reverse, q_interleave=interleave, execution_cache={})
    counters=prepared.task_counts_cpu
    if counters is None:
        counters=tuple(x.cpu().numpy() for x in (old.mask_block_cnt,old.full_block_cnt))
    if indices_cpu is None:
        indices_cpu=prepared.task_indices_cpu
    if indices_cpu is None:
        indices_cpu=tuple(x.cpu().numpy() for x in (old.mask_block_idx,old.full_block_idx))
    updates={}
    for name,source,count,old_idx,old_count in zip(
        ('mask_block_idx','full_block_idx'),indices_cpu,counters,
        (old.mask_block_idx,old.full_block_idx),(old.mask_block_cnt,old.full_block_cnt)):
        if old_idx.shape[-1] == 0:
            updates[name]=old_idx  # Preserve CuTe's unit last stride for empty buckets.
            continue
        array=permute_q_lists(source,np.asarray(count).reshape(tuple(old_count.shape)),reverse=reverse,interleave=interleave)
        updates[name]=torch.from_numpy(array).to(old_idx.device)
    return replace(prepared,backward_plan=old._replace(**updates),base_backward_plan=old,
                   q_reverse=reverse,q_interleave=interleave,execution_cache={})
