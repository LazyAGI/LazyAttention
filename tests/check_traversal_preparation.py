# SPDX-License-Identifier: Apache-2.0
"""Compare old/new preparation only; do not rerun the operator matrix."""
import argparse,hashlib,importlib.util,json,time,subprocess,os
from pathlib import Path
import torch
from lazy_attention import PreparedAttention
from lazy_attention.task_planning import plan_tasks
from lazy_attention.q_traversal import prepare_q_traversal
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--old-source',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('measured_q_traversal',a.old_source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(1)
apps=subprocess.check_output(['nvidia-smi','-i','6','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if {int(pid) for pid in apps.splitlines() if pid.strip()}-{os.getpid()}:raise RuntimeError(('metadata check requires exclusive GPU6',apps))
probe=torch.empty(1,device='cuda');torch.cuda.synchronize()
guard=[torch.empty(1024**3,dtype=torch.uint8,device='cuda') for _ in range(59)]
configs=[dict(reverse=r,interleave=i) for r in (False,True) for i in (0,8)];rows=[]
for result in sorted(a.results.glob('rank-*-batch-*.json')):
 measured=json.loads(result.read_text());rank,batch=measured['rank'],measured['batch'];directory=a.capture_root/f'rank-{rank}'
 m=torch.load(directory/f'batch-{batch:03d}-mask.pt',map_location='cpu',weights_only=True);ix=torch.load(directory/f'batch-{batch:03d}-index.pt',map_location='cpu',weights_only=True)
 data=ix['logical_backward_plan'];cpu=tuple(data[n].numpy().reshape(-1) for n in ('mask_block_cnt','full_block_cnt'));indices=tuple(data[n].numpy() for n in ('mask_block_idx','full_block_idx'))
 sparse=BlockSparseTensorsTorch(**{k:v.cuda() for k,v in data.items()},block_size=(64,128));s=sparse.mask_block_cnt.numel()*128;h=measured['shape'][2]//4
 prepared=PreparedAttention(s,None,(),(),None,sparse,{},task_counts_cpu=cpu)
 docs=m['metadata']['mot_document_ids'][ix['index'].long()].to(torch.int32).contiguous()
 def make(fn,config):
  qplan=fn(prepared,indices_cpu=indices,**config)
  tp=plan_tasks(qplan,docs,s,h)
  return qplan,tp
 samples=[]
 for config in configs:
  baseline,tp0=make(old.prepare_q_traversal,config);candidate,tp1=make(prepare_q_traversal,config)
  assert torch.equal(tp0.order,tp1.order)
  digest=hashlib.sha256(tp1.order.cpu().numpy().tobytes()).hexdigest();assert digest==measured['planning'][0]['order_sha256']
  assert torch.equal(baseline.backward_plan.mask_block_idx,candidate.backward_plan.mask_block_idx)
  assert torch.equal(baseline.backward_plan.full_block_idx,candidate.backward_plan.full_block_idx)
  for _ in range(2):
   make(old.prepare_q_traversal,config);make(prepare_q_traversal,config)
  paired=[]
  for sample in range(8):
   timings=[None,None]
   for side in ([0,1] if sample%2==0 else [1,0]):
    fn=(old.prepare_q_traversal,prepare_q_traversal)[side]
    torch.cuda.synchronize();t=time.perf_counter_ns();temporary=make(fn,config);torch.cuda.synchronize();timings[side]=(time.perf_counter_ns()-t)/1e6;del temporary
   paired.append(timings)
  samples.append(dict(config=config,paired_planning_ms=paired))
 rows.append(dict(rank=rank,batch=batch,table_identity=True,preparation=samples))
 if len(rows)%30==0:print(json.dumps(dict(checked=len(rows))),flush=True)
assert len(rows)==240
out=dict(cases=rows,checked=240,operator_rerun=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.old_source,Path(__file__).resolve().parent.parent/'lazy_attention/q_traversal.py',Path(__file__).resolve())})
a.output.write_text(json.dumps(out));print(json.dumps(dict(checked=240,table_identity=True,operator_rerun=False)),flush=True)
