# SPDX-License-Identifier: Apache-2.0
"""Public defaults versus the measured private path, including42 Graph replays."""
import argparse,json
from pathlib import Path
import torch
from lazy_attention import prepare_attention,prepare_execution,forward,backward
from lazy_attention.operator import _interface
from lazy_attention.csrc.fa4.mask import compact_mask
from lazy_attention.task_planning import plan_tasks
from lazy_attention.q_traversal import prepare_q_traversal
from benchmarks.fixtures import make_fixture,error


def check(f,name):
    q,k,v,do=[f[n] for n in ('q','k','v','dout')]
    raw=f.get('prepared')
    if raw is None:
        raw=prepare_attention(f['logical_mask'],f['index'],device=q.device,reorder=False,reverse=False)
    docs=f.get('documents',torch.zeros(q.shape[1],dtype=torch.int32))
    configured=prepare_execution(raw,documents=docs,kv_heads=k.shape[2])
    assert configured.q_reverse and not configured.head_major and configured.q_interleave==0
    tp=configured.execution_cache[k.shape[2]]
    assert tp is None or not tp.split
    measured=prepare_q_traversal(raw,reverse=True,indices_cpu=f.get('q_indices_cpu'))
    expected_plan=plan_tasks(measured,docs,q.shape[1],k.shape[2],reorder=True,split=False)
    assert (tp is None)==(expected_plan is None)
    if tp is not None: assert torch.equal(tp.order,expected_plan.order)
    for a,b in zip(configured.backward_plan,measured.backward_plan):
        if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    out,lse=forward(q,k,v,configured)
    ctas=torch.cuda.get_device_properties(q.device).multi_processor_count
    interface=_interface('paired',ctas,tp is not None,False,False,True)
    counter=torch.zeros(1,device=q.device,dtype=torch.int32)
    desc=configured.descriptor;desc.__leading_dim__,desc.__assumed_align__=0,16
    aux=[*configured.logical_aux,tp.order if tp else None,counter,counter.expand(q.shape[1]),desc]
    expected=interface._flash_attn_bwd(q,k,v,out,do,lse,softmax_scale=128**-.5,pack_gqa=False,
        V_in_regs=True,mask_mod=compact_mask,aux_tensors=aux,block_sparse_tensors=measured.backward_plan)[:3]
    got=backward(q,k,v,out,do,lse,configured)
    errors=[error(a,b) for a,b in zip(got,expected)]
    assert all(e['max_abs']<=.0625 and e['rel_l2']<=.005 for e in errors),errors
    # Cached planning must remain usable when any new planner invocation fails.
    import lazy_attention.task_planning as planning
    original=planning.plan_tasks
    def unexpected(*args,**kwargs): raise AssertionError('planning repeated during execution')
    planning.plan_tasks=unexpected
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): got=backward(q,k,v,out,do,lse,configured)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(42): graph.replay()
        torch.cuda.synchronize()
    finally: planning.plan_tasks=original
    graph_errors=[error(a,b) for a,b in zip(got,expected)]
    assert all(e['max_abs']<=.0625 and e['rel_l2']<=.005 for e in graph_errors),graph_errors
    off=prepare_execution(configured,documents=docs,kv_heads=k.shape[2],reorder=False,split=False,reverse=False)
    assert off.execution_cache[k.shape[2]] is None and not off.q_reverse
    for a,b in zip(off.backward_plan,raw.backward_plan):
        if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    print(json.dumps(dict(case=name,errors=errors,graph_errors=graph_errors,plan=tp is not None)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    for mask in ('full','causal','sparse','empty'):
        f=make_fixture(512,2,'mixed',mask)
        # Also test the ordinary prepare_attention entry point, before head count
        # is known: first eager call caches once, further calls are graph-safe.
        q,k,v,do=[f[n] for n in ('q','k','v','dout')]
        state=prepare_attention(f['logical_mask'],f['index'],device=q.device)
        assert state.q_reverse and state.execution_options[:2]==(True,False)
        out,lse=forward(q,k,v,state);backward(q,k,v,out,do,lse,state)
        assert k.shape[2] in state.execution_cache
        check(f,mask)
    if a.capture_root:
        from benchmarks.captured import load
        for rank,batch in ((0,6),(5,20),(5,24),(7,9)):
            check(load(a.capture_root,rank,batch),f'{rank}/{batch}')
    print('public defaults, measured metadata/launcher identity and42 replay checks passed',flush=True)

if __name__=='__main__': main()
