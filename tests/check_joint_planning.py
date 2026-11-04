# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import torch
from lazy_attention.task_planning import plan_tasks,build_task_order,TaskPlan
from benchmarks.captured import load
from benchmarks.fixtures import make_fixture,error
from lazy_attention import prepare_attention,forward,backward
import argparse
p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path,required=True);a=p.parse_args()
assert plan_tasks(None,None,0,0,reorder=False,split=False) is None

def check(f,name,expected):
    q,k,v,do=[f[n] for n in ('q','k','v','dout')]
    prepared=f.get('prepared')
    if prepared is None: prepared=prepare_attention(f['logical_mask'],f['index'],device=q.device)
    docs=f.get('documents',torch.zeros(q.shape[1],dtype=torch.int32))
    diag={};auto=plan_tasks(prepared,docs,q.shape[1],k.shape[2],reorder=True,split=True,diagnostics=diag)
    assert diag['strategy']==expected,(name,diag)
    slow_diag={}
    slow=plan_tasks(prepared.backward_plan,docs,q.shape[1],k.shape[2],reorder=True,split=True,diagnostics=slow_diag)
    assert diag['features']==slow_diag['features'] and diag['strategy']==slow_diag['strategy']
    assert diag['counts_source']=='prepared-cpu' and slow_diag['counts_source']=='gpu-copy'
    if auto is not None: assert torch.equal(auto.order,slow.order)
    manual=plan_tasks(prepared.backward_plan,docs,q.shape[1],k.shape[2],reorder=expected in ('reorder','both'),
        split=expected in ('split','both'),split_size=diag['split_size'],policy='manual')
    if expected=='reorder':
        manual=TaskPlan(build_task_order(prepared.backward_plan,docs,q.shape[1],k.shape[2],preserve_short=False))
    assert (auto is None)==(manual is None)
    if auto is not None:
        assert torch.equal(auto.order,manual.order)
        if auto.split:
            assert torch.equal(auto.reduction_indices,manual.reduction_indices)
            assert torch.equal(auto.reduction_tasks,manual.reduction_tasks)
    print(json.dumps(dict(case=name,**diag)),flush=True)
    return prepared,auto

for rank,batch,expected in [(0,6,'reorder'),(5,20,'reorder'),(5,24,'reorder'),(7,9,'reorder')]:
    f=load(a.capture_root,rank,batch);check(f,f'{rank}/{batch}',expected)
    del f;torch.cuda.empty_cache()
for mask,expected in [('causal','split'),('full','split'),('empty','none')]:
    f=make_fixture(512,2,'mixed',mask)
    prepared,tp=check(f,'small-'+mask,expected)
    q,k,v,do=[f[n] for n in ('q','k','v','dout')]
    out,lse=forward(q,k,v,prepared);ref=backward(q,k,v,out,do,lse,prepared)
    for _ in range(2): got=backward(q,k,v,out,do,lse,prepared,task_plan=tp)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): got=backward(q,k,v,out,do,lse,prepared,task_plan=tp)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(42): graph.replay()
    torch.cuda.synchronize()
    assert all(error(x,y)['rel_l2']<=.005 and error(x,y)['max_abs']<=.0625 for x,y in zip(got,ref))
print('joint strategy/table identity and42 replay checks passed',flush=True)
