"""Paired CUDA Graph backward benchmark: index/MAGI versus index/paired."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',choices=['index','magi'],default='index')
    p.add_argument('--candidate',choices=['index','paired'],default='paired')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,help='torch.save dict: q,k,v,dout,index,logical_mask')
    p.add_argument('--seqlen',type=int,default=512)
    p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--mapping',choices=['continuous','two','index','mixed'],default='mixed')
    p.add_argument('--mask',choices=['causal','full','sparse','empty'],default='causal')
    p.add_argument('--blocks',type=int,default=60)
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--soak-seconds',type=float,default=10)
    p.add_argument('--phase',choices=['ab','index-aa','paired-aa','baseline-aa','candidate-aa'],default='ab')
    p.add_argument('--capture-root',type=Path)
    p.add_argument('--adaptive',action='store_true')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--batch',type=int,default=1)
    a=p.parse_args()
    if a.baseline=='magi' and a.phase in ('index-aa','paired-aa'):
        p.error('use baseline-aa or candidate-aa with a MAGI comparison')
    if a.output.exists(): p.error('output exists; use a new path')
    if a.blocks < 12 or a.cycles < 4 or a.cycles % 4 or a.soak_seconds < 0:
        p.error('blocks >=12, cycles a positive multiple of4, soak >=0 required')
    import torch
    import cutlass.cute as cute
    from lazy_attention import prepare_attention, forward
    from lazy_attention.operator import _interface
    from lazy_attention.csrc.fa4.mask import compact_mask
    from benchmarks.fixtures import make_fixture, error
    torch.set_num_threads(1)
    if a.capture_root:
        from benchmarks.captured import load
        f=load(a.capture_root,a.rank,a.batch)
    elif a.input:
        f=torch.load(a.input,map_location='cpu',weights_only=True)
        for name in ('q','k','v','dout'): f[name]=f[name].cuda()
    else: f=make_fixture(a.seqlen,a.kv_heads,a.mapping,a.mask)
    if 'documents' not in f:
        f['documents']=torch.zeros(f['q'].shape[1],dtype=torch.int32)
    digest=hashlib.sha256()
    for name in ('q','k','v','dout','index','documents'):
        tensor=f[name].detach().contiguous().cpu()
        digest.update(repr((name,str(tensor.dtype),tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    fixture_sha=digest.hexdigest()
    q,k,v,do=[f[name] for name in ('q','k','v','dout')]
    if a.capture_root:
        plans={'range':f['prepared']}
    else:
        plans={'range':prepare_attention(f['logical_mask'],f['index'],device=q.device)}
    from lazy_attention.task_planning import build_task_order, build_adaptive_task_order
    def planner(): return (build_adaptive_task_order if a.adaptive else build_task_order)(plans['range'].backward_plan,f['documents'],q.shape[1],k.shape[2])
    task_order=planner()
    from tests.check_task_planning import reference_order
    plan=plans['range'].backward_plan
    expected=reference_order((plan.mask_block_cnt+plan.full_block_cnt).cpu().flatten().tolist(),f['documents'].tolist(),k.shape[2])
    actual=task_order.cpu().tolist() if task_order is not None else list(range(len(expected)))
    if a.adaptive and task_order is None: expected=list(range(len(expected)))
    assert actual==expected
    assert sorted(actual)==list(range(len(actual)))
    if task_order is not None: task_order.__leading_dim__,task_order.__assumed_align__=0,16
    counts=(plan.mask_block_cnt+plan.full_block_cnt).cpu().flatten().tolist()
    planning_info=dict(skipped=task_order is None,policy="adaptive-three-buckets" if a.adaptive else "always",max_q_tiles=max(counts),mean_q_tiles=statistics.mean(counts),changed_slots=sum(i!=v for i,v in enumerate(actual)),tasks=len(actual),order_sha256=hashlib.sha256(bytes(str(actual),"ascii")).hexdigest())
    for _ in range(20): planner()
    prep_samples=[]
    for _ in range(a.blocks):
        torch.cuda.synchronize(); begin=time.perf_counter_ns(); temporary=planner();torch.cuda.synchronize()
        prep_samples.append((time.perf_counter_ns()-begin)/1e6)
        del temporary
    out,lse=forward(q,k,v,plans['range'])
    ctas=torch.cuda.get_device_properties(q.device).multi_processor_count
    backend_names=['paired','planned']
    magi=None
    if a.baseline=='magi':
        from benchmarks.magi import MagiBackward
        magi=MagiBackward(f)
        e=error(magi.out.unsqueeze(0),out)
        if not(e['max_abs']<=.0625 and e['rel_l2']<=.006): raise RuntimeError(('MAGI forward mismatch',e))
    interfaces=[_interface('paired',ctas,False),_interface('paired',ctas,task_order is not None)]
    launches=[x._flash_attn_bwd if x else None for x in interfaces]
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
            aux=[*plans['range'].logical_aux,task_order if backend=='planned' else counter,counter]
            aux.extend([poison,desc])
            return launches[variant](q,k,v,out,do,lse,softmax_scale=128**-.5,
                pack_gqa=False,V_in_regs=True,mask_mod=compact_mask,aux_tensors=aux,
                block_sparse_tensors=plans['range'].backward_plan)[:3]
        finally: interface.torch=previous
    for _ in range(12):
        for pool in (0,1):
            for role in (0,1): fn(pool,role)
    torch.cuda.synchronize()
    ref=[x.clone() for x in fn(0,0)]
    correctness=[]
    def verify(got):
        errors=[error(x,y) for x,y in zip(got,ref)]
        if not all(x['max_abs']<=.0625 and x['rel_l2']<=.005 for x in errors):
            raise RuntimeError(('gradient mismatch',errors))
        correctness.append(errors)
    for pool in (0,1):
        for role in (0,1): verify(fn(pool,role))
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    bundles=[]
    for graph_kind in range(4):
        layout_swap,order_kind=divmod(graph_kind,2)
        order=([0,1,1,0],[1,0,0,1])[order_kind]
        timings=[]; captured=[]
        for cycle in range(a.cycles):
            layout=layout_swap ^ (cycle%4 in (1,2))
            for label in order:
                timings.append((label,torch.cuda.Event(enable_timing=True,external=True),
                    torch.cuda.Event(enable_timing=True,external=True),int(layout)))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for label,begin,end,layout in timings:
                begin.record(); got=fn([layout,1-layout][label],label); end.record()
                captured.append(got)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay(); torch.cuda.synchronize()
        for got in captured[-4:]: verify(got)
        bundles.append((graph,timings))
    start=time.monotonic()
    while time.monotonic()-start<a.soak_seconds:
        for graph,_ in bundles: graph.replay()
        torch.cuda.synchronize()
    compiled=compile_count[0]
    # UUID resolution works with ordinal or UUID CUDA_VISIBLE_DEVICES and avoids
    # confusing process-namespace PIDs with physical NVML PIDs.
    uuid=str(torch.cuda.get_device_properties(q.device).uuid)
    if not uuid.startswith('GPU-'): uuid='GPU-'+uuid
    def telemetry():
        row=subprocess.check_output(['nvidia-smi','-i',uuid,
            '--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw',
            '--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        pids=sorted({line.split(',')[1].strip() for line in processes if line.split(',')[0].strip()==uuid})
        return dict(time=time.time(),gpu=row,pids=pids)
    initial=telemetry()
    if len(initial['pids'])!=1: raise RuntimeError(('GPU must be exclusive',initial))
    telemetry_rows=[initial]
    schedule=[0]*(a.blocks//2)+[1]*(a.blocks-a.blocks//2)
    random.Random(20261030+a.repeat).shuffle(schedule)
    blocks=[]
    for bi,order in enumerate(schedule):
        values=[]; layout_order=([0,1] if bi%2==0 else [1,0])
        before=telemetry()
        if before['pids']!=initial['pids']: raise RuntimeError(('GPU process changed',before))
        for layout in layout_order:
            graph,timings=bundles[layout*2+order]
            graph.replay(); torch.cuda.synchronize()
            values.extend(dict(label=label,ms=begin.elapsed_time(end),layout=actual)
                          for label,begin,end,actual in timings)
        ds=[]
        for offset in range(0,len(values),4):
            quad=values[offset:offset+4]
            ds.append(.5*(sum(math.log(x['ms']) for x in quad if x['label']==1)-
                          sum(math.log(x['ms']) for x in quad if x['label']==0)))
        blocks.append(dict(id=bi,order=order,layout_order=layout_order,values=values,
                           dlog=statistics.mean(ds),time=time.time()))
        telemetry_rows.append(telemetry())
        if telemetry_rows[-1]['pids']!=initial['pids']: raise RuntimeError('GPU process changed')
    if compile_count[0]!=compiled: raise RuntimeError('compilation during measurement')
    cute.compile=original_compile
    restore_magi()
    from benchmarks.environment import collect
    source_root=Path(__file__).resolve().parent.parent
    source_hashes={str(file.relative_to(source_root)):hashlib.sha256(file.read_bytes()).hexdigest()
        for folder in ('lazy_attention','benchmarks') for file in sorted((source_root/folder).rglob('*.py'))}
    effective_blocks=[]
    for block,prep in zip(blocks,prep_samples):
        vals=[{**v,'ms':v['ms']+(prep/42 if v['label'] else 0)} for v in block['values']]
        ds=[]
        for i in range(0,len(vals),4):
            quad=vals[i:i+4]
            ds.append(.5*(sum(math.log(v['ms']) for v in quad if v['label'])-sum(math.log(v['ms']) for v in quad if not v['label'])))
        effective_blocks.append({**block,'values':vals,'dlog':statistics.mean(ds)})
    result=dict(planning=planning_info,planning_wall_ms=prep_samples,reuse=42,effective_blocks=effective_blocks,capture_sha256=f.get('capture_sha256'),backends=([backend_names[0]]*2 if a.phase in ("index-aa","baseline-aa") else [backend_names[1]]*2 if a.phase in ("paired-aa","candidate-aa") else backend_names),protocol=('shared-outputs-private-scratch-v1' if magi else 'shared-workspace-v1'),magi=magi.info if magi else None,environment=collect(),fixture_sha256=fixture_sha,source_sha256=source_hashes,schema=1,phase=a.phase,repeat=a.repeat,seed=20261030+a.repeat,
        scope=('GPU backward: library-specific reset, preprocessing, kernel and final BF16 gradients' if magi else 'GPU backward: queue reset + preprocess + persistent kernel + dQ postprocess'),
        excludes=['preparation','forward','CPU launch/allocation','cold compilation'],
        config={**vars(a),'output':str(a.output),'input':str(a.input) if a.input else None,'capture_root':str(a.capture_root) if a.capture_root else None},
        shape=list(q.shape),strides={name:list(f[name].stride()) for name in ('q','k','v','dout')},
        descriptor=plans['range'].info,versions=dict(python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda),gpu_uuid=uuid,os_pid=os.getpid(),
        pool_pointers=[[t.data_ptr() for _,t in pool.buffers.values()] for pool in pools],
        correctness=correctness,compile_count=compiled,telemetry=telemetry_rows,blocks=blocks)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as file: json.dump(result,file)
    print(json.dumps(dict(output=str(a.output),time_change_pct=100*math.expm1(statistics.mean(b['dlog'] for b in blocks)),effective_time_change_pct=100*math.expm1(statistics.mean(b['dlog'] for b in effective_blocks)),planning=planning_info)))
if __name__=='__main__': main()
