# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import torch
from lazy_attention.task_planning import plan_tasks
from lazy_attention import forward,backward,prepare_attention
from benchmarks.fixtures import make_fixture,error
assert plan_tasks(None,None,0,0,reorder=False,split=False) is None
counts=[0,1,128,129,255,256,257]
pc=torch.tensor(counts,device='cuda',dtype=torch.int32).view(1,1,-1)
plan=SimpleNamespace(mask_block_cnt=pc,full_block_cnt=torch.zeros_like(pc),block_size=(64,128))
docs=torch.zeros(128*len(counts),dtype=torch.int32)
for reorder in (False,True):
    p=plan_tasks(plan,docs,len(docs),2,policy="manual",reorder=reorder,split=True)
    coverage=[[] for _ in range(2*len(counts))]
    for slot,(task,start,end,partial_slot) in enumerate(p.order.cpu().tolist()):
        assert end-start<=128
        if partial_slot>=0:
            row=p.reduction_tasks.cpu().tolist().index(task)
            assert partial_slot in p.reduction_indices[row].cpu().tolist()
        coverage[task].extend(range(start,end))
    assert all(sorted(c)==list(range(counts[t//2])) for t,c in enumerate(coverage))
for mapping in ('continuous','two','index','mixed'):
    f=make_fixture(512,2,mapping,'causal')
    q,k,v,do=[f[n] for n in ('q','k','v','dout')]
    p=prepare_attention(f['logical_mask'],f['index'],device=q.device)
    out,lse=forward(q,k,v,p)
    ref=backward(q,k,v,out,do,lse,p)
    tp=plan_tasks(p.backward_plan,torch.zeros(512,dtype=torch.int32),512,2,policy="manual",reorder=True,split=True,split_size=2)
    for _ in range(2): actual=backward(q,k,v,out,do,lse,p,task_plan=tp)
    assert all(error(a,b)['rel_l2']<=.005 and error(a,b)['max_abs']<=.0625 for a,b in zip(actual,ref))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream): actual=backward(q,k,v,out,do,lse,p,task_plan=tp)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(42): g.replay()
    torch.cuda.synchronize()
    assert all(error(a,b)['rel_l2']<=.005 for a,b in zip(actual,ref))
    print(mapping,'public split backward+42 Graph replays passed')
print('disabled identity and split boundary coverage passed')
