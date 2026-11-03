# SPDX-License-Identifier: Apache-2.0
"""Compare unchanged sparse-plan representation with bulk H2D construction."""
import argparse,hashlib,json,math,random,statistics,time,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repeat',type=int,default=0);p.add_argument('--phase',choices=['baseline-aa','candidate-aa','ab'],default='ab');p.add_argument('--seqlen',type=int,default=4096);p.add_argument('--blocks',type=int,default=60);p.add_argument('--calls',type=int,default=16);p.add_argument('--mask',choices=['causal','full','empty','sparse'],default='causal');p.add_argument('--layout',choices=['c','f'],default='c');a=p.parse_args()
 if Path(a.output).exists():p.error('output exists; choose a fresh result path')
 if a.seqlen<128 or a.seqlen%128 or a.blocks<12 or a.calls<1:p.error('S must be a positive multiple of128; blocks>=12; calls>=1')
 import numpy as np
 import torch
 from benchmarks.reference_sparse_plan import sparse_plan as baseline
 from lazy_attention.planning import _sparse_plan as candidate
 torch.set_num_threads(1)
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[len(affinity)//2]})
 uuid=str(torch.cuda.get_device_properties(0).uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
 def telemetry():
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines();pids=sorted(r for r in rows if r.split(',')[0].strip()==uuid);assert len(pids)==1,pids
  return dict(pids=pids,gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=name,uuid,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip())
 s=a.seqlen;rng=np.random.default_rng(128);mask=np.tri(s,dtype=bool) if a.mask=='causal' else np.ones((s,s),dtype=bool) if a.mask=='full' else np.zeros((s,s),dtype=bool) if a.mask=='empty' else rng.random((s,s))<.05
 if a.layout=='f':mask=np.asfortranarray(mask)
 names=('mask_block_cnt','mask_block_idx','full_block_cnt','full_block_idx');checks=[]
 for back in (False,True):
  dims=(64,128) if back else (128,128)
  x=baseline(mask,*dims,back,'cuda');y=candidate(mask,*dims,back,'cuda')
  for n in names:
   u,v=getattr(x,n),getattr(y,n);assert u.shape==v.shape and u.dtype==v.dtype and torch.equal(u,v),(back,n);checks.append((back,n,list(u.shape)))
 def run(role):
  fn=(baseline if a.phase=='baseline-aa' else candidate if a.phase=='candidate-aa' else (baseline,candidate)[role])
  return fn(mask,128,128,False,'cuda'),fn(mask,64,128,True,'cuda')
 for _ in range(20):run(0);run(1)
 initial=telemetry();blocks=[];order=[i%2 for i in range(a.blocks)];random.Random(a.repeat+777).shuffle(order)
 for bi,o in enumerate(order):
  values=[]
  for role in ([0,1,1,0] if o==0 else [1,0,0,1]):
   run(role);torch.cuda.synchronize();start=time.perf_counter_ns()
   for _ in range(a.calls):run(role)
   torch.cuda.synchronize();values.append(dict(label=role,ms=(time.perf_counter_ns()-start)/1e6/a.calls))
  d=.5*(sum(math.log(v['ms']) for v in values if v['label'])-sum(math.log(v['ms']) for v in values if not v['label']))
  t=telemetry();assert t['pids']==initial['pids'];blocks.append(dict(id=bi,dlog=d,values=values,telemetry=t))
 from benchmarks.environment import collect
 root=Path(__file__).resolve().parent.parent
 result=dict(backends=['rowwise','bulk'],gpu_uuid=uuid,initial_telemetry=initial,affinity=sorted(os.sched_getaffinity(0)),environment={**collect(),'numpy':np.__version__},protocol='pinned-cpu-eager-v2',config={**vars(a),'mapping':'identity'},phase=a.phase,repeat=a.repeat,shape=[s],fixture_sha256=hashlib.sha256(repr((mask.shape,mask.strides,str(mask.dtype))).encode()+mask.tobytes()).hexdigest(),source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('lazy_attention/planning.py','benchmarks/reference_sparse_plan.py','benchmarks/planning.py')},blocks=blocks,correctness=checks,scope='synchronized wall: construct forward+backward sparse plans incl CPU tile analysis, allocations, H2D; input ready on CPU',gpu=str(torch.cuda.get_device_properties(0)),torch=torch.__version__)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 Path(a.output).write_text(json.dumps(result));print(json.dumps(dict(delta=100*math.expm1(statistics.mean(b['dlog'] for b in blocks)))),flush=True)
if __name__=='__main__':main()
