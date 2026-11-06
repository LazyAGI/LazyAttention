"""Exact GPU/CPU permutation parity, including padding and uneven lists."""
import numpy as np
import torch
from unittest.mock import patch
from dataclasses import dataclass
from collections import namedtuple
from lazy_attention.planning import PreparedAttention
from lazy_attention.q_traversal import prepare_q_traversal
from lazy_attention.q_traversal import permute_q_lists
from lazy_attention.csrc.fa4.q_traversal import permute_q_lists_gpu

torch.manual_seed(42)
cases=0
for dtype in (torch.int32, torch.int64):
    for width in (0, 1, 7, 8, 9, 129, 512):
        counts=torch.randint(width+1, (2, 3, 17), dtype=torch.int32)
        counts.flatten()[0]=0
        counts.flatten()[-1]=width
        indices=torch.randint(10000, (2, 3, 17, width), dtype=dtype)
        for reverse,interleave in ((True,0),(False,8),(True,8)):
            expected=permute_q_lists(indices.numpy(),counts.numpy(),reverse=reverse,interleave=interleave)
            actual=permute_q_lists_gpu(indices.cuda(),counts.cuda(),reverse=reverse,interleave=interleave)
            np.testing.assert_array_equal(actual.cpu().numpy(),expected)
            assert actual.dtype==dtype and actual.shape==indices.shape
            cases+=1
print(f'PASS {cases} exact GPU/CPU sparse-list permutation cases')

# Exercise the public preparation route, not only the primitive. Guard against
# accidentally reintroducing Tensor.cpu() in the GPU-built-plan path.
Plan=namedtuple('Plan','mask_block_cnt full_block_cnt mask_block_idx full_block_idx')
c=torch.tensor([[[0,1,7,8,9,17]]],dtype=torch.int32,device='cuda')
i=torch.arange(6*17,dtype=torch.int32,device='cuda').reshape(1,1,6,17)
plan=Plan(c,c,i,i)
prepared=PreparedAttention(768,None,(),(),None,plan,{})
with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('unexpected D2H')):
    actual=prepare_q_traversal(prepared,reverse=True,interleave=0)
assert actual.q_reverse and actual.base_backward_plan is plan
np.testing.assert_array_equal(actual.backward_plan.mask_block_idx.cpu().numpy(),
    permute_q_lists(i.cpu().numpy(),c.cpu().numpy(),reverse=True))
assert prepare_q_traversal(actual,reverse=True) is actual
assert prepare_q_traversal(actual,reverse=False).backward_plan is plan
print('PASS public GPU-built-plan route has no Tensor.cpu and reuses prepared state')

# Variable training lengths sharing one BLOCK must reuse the same compiled
# kernel. Width/group scalars must not specialize values or divisibility.
from lazy_attention.csrc.fa4.q_traversal import _permute
cache_sizes=[]
for width,rows in ((376,188),(428,214),(510,255)):
    c=torch.randint(width+1,(1,1,rows),dtype=torch.int32)
    i=torch.arange(rows*width,dtype=torch.int32).reshape(1,1,rows,width)
    actual=permute_q_lists_gpu(i.cuda(),c.cuda(),reverse=True)
    np.testing.assert_array_equal(actual.cpu().numpy(),permute_q_lists(i.numpy(),c.numpy(),reverse=True))
    cache_sizes.append(len(_permute.device_caches[torch.cuda.current_device()][0]))
assert len(set(cache_sizes))==1,('VARIABLE_WIDTH_RECOMPILED',cache_sizes)
print('PASS variable-width compile cache stable',cache_sizes)
