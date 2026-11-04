# SPDX-License-Identifier: Apache-2.0
"""Full captured corpus, four switch combinations, one persistent CUDA process."""
import argparse, gc, hashlib, json, math, os, random, statistics, subprocess, time
from pathlib import Path

def telemetry():
    import torch
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'): uuid='GPU-'+uuid
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    pids=sorted({x.split(',')[1].strip() for x in rows if x.split(',')[0].strip()==uuid})
    gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()
    if len(pids)!=1: raise RuntimeError(('GPU must be exclusive',pids,gpu))
    return dict(time=time.time(),uuid=uuid,pids=pids,gpu=gpu)

def case(a):
    import torch
    import cutlass.cute as cute
    from lazy_attention import prepare_attention, forward
    from lazy_attention.operator import _interface
    from lazy_attention.csrc.fa4.mask import compact_mask
    from benchmarks.fixtures import make_fixture, error
    if a.capture_root:
        from benchmarks.captured import load
        f=load(a.capture_root,a.rank,a.batch)
    elif a.input:
        f=torch.load(a.input,map_location='cpu',weights_only=True)
        for name in ('q','k','v','dout'): f[name]=f[name].cuda()
    else: f=make_fixture(a.seqlen,a.kv_heads,a.mapping,a.mask)
    if 'documents' not in f:
        f['documents']=torch.zeros(f['q'].shape[1],dtype=torch.int32)
    fixture_sha=None # immutable capture hashes retained below
    q,k,v,do=[f[name] for name in ('q','k','v','dout')]
    if a.capture_root:
        plans={'range':f['prepared']}
    else:
        plans={'range':prepare_attention(f['logical_mask'],f['index'],device=q.device, reorder=False, reverse=False)}
    from lazy_attention.task_planning import plan_tasks, build_task_order, TaskPlan
    from lazy_attention.split_reduction import reduce_partials
    plan=plans['range'].backward_plan
    decisions=[{} for _ in range(4)]
    strategies=['none','reorder','split','both']
    counts=(plan.mask_block_cnt+plan.full_block_cnt).cpu().flatten().tolist()
    def planner(role):
        strategy=strategies[role]
        # Force the reorder treatment, including short tasks. Do not let the
        # previous adaptive bypass pre-decide this factorial experiment.
        if strategy=='reorder' or (strategy=='both' and max(counts)<=a.split_size):
            decisions[role].update(strategy='reorder',reason='factorial-force-reorder')
            order=build_task_order(plan,f['documents'],q.shape[1],k.shape[2],preserve_short=False)
            order.__leading_dim__,order.__assumed_align__=0,16
            return TaskPlan(order)
        return plan_tasks(plans['range'],f['documents'],q.shape[1],k.shape[2],
            reorder=strategy in ('reorder','both','auto'),split=strategy in ('split','both','auto'),
            policy='auto' if strategy=='auto' else 'manual',split_size=a.split_size,
            diagnostics=decisions[role])
    task_plans=[planner(role) for role in range(4)]
    features={}
    diagnostic_plan=plan_tasks(plans['range'],f['documents'],q.shape[1],k.shape[2],reorder=True,split=True,policy='auto',diagnostics=features)
    del diagnostic_plan
    counts=(plan.mask_block_cnt+plan.full_block_cnt).cpu().flatten().tolist()
    planning_info=[]
    for role,tp in enumerate(task_plans):
        if tp is not None and tp.split:
            rows=tp.order.cpu().tolist()
            reduction_map=dict(zip(tp.reduction_tasks.cpu().tolist(),tp.reduction_indices.cpu().tolist()))
            coverage=[[] for _ in range(len(counts)*k.shape[2])]
            for slot,(task,start,end,partial_slot) in enumerate(rows):
                assert 0<=start<=end<=counts[task//k.shape[2]]
                coverage[task].extend(range(start,end))
                if partial_slot>=0:
                    assert partial_slot in reduction_map[task]
            assert all(sorted(c)==list(range(counts[t//k.shape[2]])) for t,c in enumerate(coverage))
            assert all(len(c)==len(set(c)) for c in coverage)
        elif tp is not None:
            assert sorted(tp.order.cpu().tolist())==list(range(len(counts)*k.shape[2]))
        planning_info.append(dict(decision=dict(decisions[role]),skipped=tp is None,split=tp is not None and tp.split,
            tasks=tp.order.shape[0] if tp is not None else len(counts)*k.shape[2],
            max_q_tiles=max(counts),split_size=decisions[role].get('split_size',a.split_size),partial_tasks=tp.partial_count if tp else 0,
            order_sha256=hashlib.sha256(tp.order.cpu().numpy().tobytes()).hexdigest() if tp else None))
    for _ in range(4):
        for role in range(4): planner(role)
    prep_samples=[]
    for _ in range(a.blocks):
        samples=[]
        for role in range(4):
            torch.cuda.synchronize(); begin=time.perf_counter_ns()
            temporary=planner(role);torch.cuda.synchronize()
            samples.append((time.perf_counter_ns()-begin)/1e6)
            del temporary
        prep_samples.append(samples)
    out,lse=forward(q,k,v,plans['range'])
    ctas=torch.cuda.get_device_properties(q.device).multi_processor_count
    backend_names=strategies
    magi=None
    if a.baseline=='magi':
        from benchmarks.magi import MagiBackward
        magi=MagiBackward(f)
        e=error(magi.out.unsqueeze(0),out)
        if not(e['max_abs']<=.0625 and e['rel_l2']<=.006): raise RuntimeError(('MAGI forward mismatch',e))
    interfaces=[_interface('paired',ctas,tp is not None,tp is not None and tp.split) for tp in task_plans]
    launches=[x._flash_attn_bwd if x else None for x in interfaces]
    scratch_tasks=max((tp.partial_count for tp in task_plans if tp is not None),default=0)
    desc=plans['range'].descriptor
    desc.__leading_dim__,desc.__assumed_align__=0,16
    poison=torch.full((1,),-1073741824,device=q.device,dtype=torch.int32).expand(q.shape[1])
    poison.__assumed_align__=16
    compile_count=[0]; original_compile=cute.compile
    def tracked(*args,**kw):
        compile_count[0]+=1
        return original_compile(*args,**kw)
    cute.compile=tracked
    restore_magi=magi.track_compiles(compile_count) if magi else lambda: None

    class FixedPool:
        # Single-thread benchmark only: each variant must make identical buffer
        # requests in identical order. No monkeypatch of torch itself.
        def __init__(self): self.slot=0; self.buffers={}
        def __getattr__(self,name): return getattr(torch,name)
        def begin(self): self.slot=0
        def get(self,kind,args,kw):
            slot=self.slot; self.slot+=1
            if kind.endswith('_like'):
                t=args[0]; sig=(kind,tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),repr(kw))
            else: sig=(kind,repr(args),repr(kw))
            if slot not in self.buffers: self.buffers[slot]=(sig,getattr(torch,kind)(*args,**kw))
            previous,t=self.buffers[slot]
            if previous != sig: raise RuntimeError(('buffer mismatch',previous,sig))
            if kind.startswith('zeros'): t.zero_()
            return t
        def empty(self,*args,**kw): return self.get('empty',args,kw)
        def zeros(self,*args,**kw): return self.get('zeros',args,kw)
        def empty_like(self,*args,**kw): return self.get('empty_like',args,kw)
        def zeros_like(self,*args,**kw): return self.get('zeros_like',args,kw)
    pools=[FixedPool(),FixedPool()]
    def fn(pool_id,role):
        variant=(0 if a.phase in ('index-aa','baseline-aa') else 1 if a.phase in ('paired-aa','candidate-aa') else role)
        pool=pools[pool_id]; pool.begin(); interface=interfaces[variant]
        backend=backend_names[variant]
        if backend=='magi':
            # Slot 0 is the FA4 queue counter. Only output slots 1..3
            # are shared across libraries; private scratch remains backend-owned.
            pool.slot=1
            outputs=tuple(pool.empty_like(x) for x in (q,k,v))
            return magi(pool_id,outputs)
        previous=interface.torch; interface.torch=pool
        try:
            counter=pool.zeros((1,),device=q.device,dtype=torch.int32)
            tp=task_plans[variant]
            aux=[*plans['range'].logical_aux,tp.order if tp else None,counter]
            aux.extend([poison,desc])
            partials=None
            if scratch_tasks:
                partials=tuple(pool.empty((scratch_tasks,128,128),device=q.device,dtype=torch.float32) for _ in range(2))
                for t in partials: t.__leading_dim__,t.__assumed_align__=2,16
                if tp is not None and tp.split: aux.extend(partials)
            result=launches[variant](q,k,v,out,do,lse,softmax_scale=128**-.5,
                pack_gqa=False,V_in_regs=True,mask_mod=compact_mask,aux_tensors=aux,
                block_sparse_tensors=plans['range'].backward_plan)[:3]
            if tp is not None and tp.split:
                reduce_partials(*partials,tp.reduction_indices,tp.reduction_tasks,desc,result[1],result[2])
            return result
        finally: interface.torch=previous
    for _ in range(3):
        for pool in (0,1):
            for role in range(4): fn(pool,role)
    torch.cuda.synchronize()
    ref=[x.clone() for x in fn(0,0)]
    correctness=[]
    def verify(got):
        errors=[error(x,y) for x,y in zip(got,ref)]
        if not all(x['max_abs']<=.0625 and x['rel_l2']<=.005 for x in errors):
            raise RuntimeError(('gradient mismatch',errors))
        correctness.append(errors)
    for pool in (0,1):
        for role in range(4): verify(fn(pool,role))
    # Four balanced rotations and their reverses; every corner shares both
    # physical output pools, and occurs equally often at each time position.
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    bundles=[]
    for rotation in range(4):
        for reverse in (False,True):
            order=[(rotation+i)%4 for i in range(4)]
            if reverse: order=order[::-1]
            timings=[]; captured=[]
            for cycle in range(a.cycles):
                for label in order:
                    pool=(cycle+label)%2
                    timings.append((label,torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True),pool))
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for label,begin,end,pool in timings:
                    begin.record(); got=fn(pool,label); end.record(); captured.append(got)
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay(); torch.cuda.synchronize()
            for got in captured[-4:]: verify(got)
            bundles.append((graph,timings))
    compiled=compile_count[0]
    initial=telemetry()
    start=time.monotonic()
    while time.monotonic()-start<a.soak_seconds:
        for graph,_ in bundles: graph.replay()
        torch.cuda.synchronize()
    schedule=list(range(8))*(a.blocks//8)
    random.Random(20261104+a.rank*31+a.batch).shuffle(schedule)
    blocks=[]
    for bi,which in enumerate(schedule[:a.blocks]):
        graph,timings=bundles[which]
        graph.replay(); torch.cuda.synchronize()
        values=[dict(label=label,ms=begin.elapsed_time(end),pool=pool) for label,begin,end,pool in timings]
        blocks.append(dict(id=bi,bundle=which,values=values))
    final=telemetry()
    if initial['pids']!=final['pids']: raise RuntimeError('GPU process changed')
    if compiled!=compile_count[0]: raise RuntimeError('compilation during measurement')
    cute.compile=original_compile
    restore_magi()
    means=[statistics.mean(x['ms'] for block in blocks for x in block['values'] if x['label']==i) for i in range(4)]
    prep=[statistics.mean(x[i] for x in prep_samples) for i in range(4)]
    effective=[x+y/42 for x,y in zip(means,prep)]
    result=dict(schema=1,rank=a.rank,batch=a.batch,strategies=strategies,split_size=a.split_size,
        protocol='four-corner-balanced-graph-shared-workspace-v1',reuse=42,blocks=blocks,
        planning_wall_ms=prep_samples,planning=planning_info,features=features,
        gpu_ms=means,effective_ms=effective,best=strategies[min(range(4),key=effective.__getitem__)],
        shape=list(q.shape),strides={name:list(f[name].stride()) for name in ('q','k','v','dout')},
        correctness=correctness,compile_count=compiled,capture_sha256=f.get('capture_sha256'),
        telemetry=[initial,final],os_pid=os.getpid(),memory=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak=torch.cuda.max_memory_allocated()))
    a.output.write_text(json.dumps(result))
    print(json.dumps({k:result[k] for k in ('rank','batch','gpu_ms','effective_ms','best','memory')}),flush=True)
    # All graphs and tensors die on return. The CUDA caching allocator survives.
    return result['best']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--limit',type=int,default=240)
    p.add_argument('--blocks',type=int,default=16)
    p.add_argument('--cycles',type=int,default=4)
    p.add_argument('--soak-seconds',type=float,default=1)
    p.add_argument('--split-size',type=int,default=128)
    a=p.parse_args()
    if a.blocks%8 or a.cycles%4: p.error('blocks multiple of8, cycles multiple of4')
    import torch, cutlass.cute as cute
    from benchmarks.environment import collect
    torch.set_num_threads(1)
    a.output_root.mkdir(parents=True,exist_ok=False)
    cases=sorted((int(path.parent.name.split('-')[1]),int(path.name.split('-')[1])) for path in a.capture_root.glob('rank-*/batch-*-index.pt'))
    if len(cases)!=240: raise RuntimeError(('expected240 captures',len(cases)))
    context_probe=torch.empty((1,),device='cuda')
    torch.cuda.synchronize()
    initial=telemetry()
    del context_probe
    free,total=torch.cuda.mem_get_info()
    # Leave 8GiB for driver/JIT/non-PyTorch workspaces. 1GiB blocks are
    # released into the caching allocator; never call empty_cache afterwards.
    target=max(0,free-8*1024**3)
    chunks=[torch.empty((1024**3,),dtype=torch.uint8,device='cuda') for _ in range(target//1024**3)]
    reserved_before=torch.cuda.memory_reserved()
    del chunks; gc.collect(); torch.cuda.synchronize()
    # CUDA Graph private pools can release ordinary cached free blocks. Hold
    # a live guard, leaving >=18GiB on an80GiB card for all case allocations.
    # This prevents Graph pool creation from returning almost the whole GPU.
    guard_gib=min(60,max(0,int(total//1024**3)-20))
    reservation_guard=[torch.empty((1024**3,),dtype=torch.uint8,device='cuda') for _ in range(guard_gib)]
    root=Path(__file__).resolve().parent.parent
    manifest=dict(environment=collect(),telemetry=initial,free_start=free,total=total,
        reserved_after_preallocation=reserved_before,live_guard_GiB=guard_gib,ssh_alias='M-CCI2',endpoint='root@10.119.31.41:22',
        cases=cases,blocks=a.blocks,cycles=a.cycles,split_size=a.split_size,os_pid=os.getpid(),
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('lazy_attention','benchmarks') for f in sorted((root/folder).rglob('*.py'))})
    (a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(start=True,cases=len(cases),reserved_GiB=reserved_before/1024**3,live_guard_GiB=guard_gib,pid=os.getpid())),flush=True)
    original_compile=cute.compile
    for i,(rank,batch) in enumerate(cases[:a.limit]):
        a.rank,a.batch=rank,batch
        a.output=a.output_root/f'rank-{rank}-batch-{batch:03d}.json'
        a.baseline='index';a.candidate='paired';a.phase='ab';a.repeat=0;a.input=None
        begin=time.time()
        try:
            case(a)
        finally:
            cute.compile=original_compile
        gc.collect(); torch.cuda.synchronize()
        if torch.cuda.memory_allocated()<guard_gib*1024**3: raise RuntimeError('reservation guard lost')
        print(json.dumps(dict(completed=i+1,total=min(a.limit,len(cases)),seconds=time.time()-begin,reserved=torch.cuda.memory_reserved())),flush=True)
    (a.output_root/'complete.json').write_text(json.dumps(dict(completed=min(a.limit,len(cases)),pid=os.getpid(),telemetry=telemetry())))

if __name__=='__main__': main()
