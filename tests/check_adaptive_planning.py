# SPDX-License-Identifier: Apache-2.0
"""GPU boundary checks for planner-owned bypass, then public backward/Graph."""
from types import SimpleNamespace
import torch
from lazy_attention.task_planning import build_adaptive_task_order, build_task_order
from lazy_attention import forward, backward
from benchmarks.fixtures import make_fixture, error
from lazy_attention import prepare_attention

for longest in (0,64,65,128,129,256):
    pc=torch.tensor([0,longest,1,2],device='cuda',dtype=torch.int32).view(1,1,4)
    plan=SimpleNamespace(mask_block_cnt=pc,full_block_cnt=torch.zeros_like(pc),block_size=(64,128))
    docs=torch.zeros(512,dtype=torch.int32)
    got=build_adaptive_task_order(plan,docs,512,2)
    assert (got is None)==(longest<=128)
    if got is not None:
        assert torch.equal(got,build_task_order(plan,docs,512,2))
    try: build_adaptive_task_order(plan,docs[:-1],512,2)
    except ValueError: pass
    else: raise AssertionError('invalid input was silently bypassed')

f=make_fixture(512,2,'mixed','causal')
q,k,v,do=[f[n] for n in ('q','k','v','dout')]
p=prepare_attention(f['logical_mask'],f['index'],device=q.device)
out,lse=forward(q,k,v,p)
order=build_adaptive_task_order(p.backward_plan,torch.zeros(512,dtype=torch.int32),512,2)
assert order is None
ref=backward(q,k,v,out,do,lse,p)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=stream):
    actual=backward(q,k,v,out,do,lse,p,task_order=order)
torch.cuda.current_stream().wait_stream(stream)
g.replay();torch.cuda.synchronize()
assert all(error(a,b)['rel_l2']<=.005 for a,b in zip(actual,ref))
print('six boundary policies, invalid input rejection, public None backward and Graph passed')
