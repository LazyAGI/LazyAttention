# SPDX-License-Identifier: Apache-2.0
"""Metadata-only validation against all240 measured GPU-emitted task orders."""
import argparse,hashlib,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import torch
from lazy_attention.task_planning import plan_tasks
p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
assert plan_tasks(None,None,0,0,reorder=False,split=False) is None
# This is a metadata identity check, not another operator performance round.
probe=torch.empty(1,device='cuda');torch.cuda.synchronize()
guard=[torch.empty(1024**3,dtype=torch.uint8,device='cuda') for _ in range(60)]
rows=[]
for result_file in sorted(a.results.glob('rank-*-batch-*.json')):
 expected=json.loads(result_file.read_text());rank,batch=expected['rank'],expected['batch'];directory=a.capture_root/f'rank-{rank}'
 m=torch.load(directory/f'batch-{batch:03d}-mask.pt',map_location='cpu',weights_only=True)
 ix=torch.load(directory/f'batch-{batch:03d}-index.pt',map_location='cpu',weights_only=True)
 cpu=tuple(ix['logical_backward_plan'][n].numpy().reshape(-1) for n in ('mask_block_cnt','full_block_cnt'))
 pc,fc=(ix['logical_backward_plan'][n].cuda() for n in ('mask_block_cnt','full_block_cnt'))
 seqlen=pc.numel()*128;heads=expected['shape'][2]//4
 prepared=SimpleNamespace(seqlen=seqlen,task_counts_cpu=cpu,backward_plan=SimpleNamespace(block_size=(64,128),mask_block_cnt=pc,full_block_cnt=fc))
 docs=m['metadata']['mot_document_ids'][ix['index'].long()].to(torch.int32).contiguous()
 def make():return plan_tasks(prepared,docs,seqlen,heads)
 for _ in range(2):actual=make()
 assert actual is not None and not actual.split
 digest=hashlib.sha256(actual.order.cpu().numpy().tobytes()).hexdigest()
 assert digest==expected['planning'][1]['order_sha256'],(rank,batch,digest,expected['planning'][1])
 slow=plan_tasks(prepared.backward_plan,docs,seqlen,heads,reorder=True,split=True)
 assert torch.equal(actual.order,slow.order)
 samples=[]
 for _ in range(8):
  torch.cuda.synchronize();begin=time.perf_counter_ns();actual=make();torch.cuda.synchronize();samples.append((time.perf_counter_ns()-begin)/1e6)
 rows.append(dict(rank=rank,batch=batch,strategy='reorder',order_sha256=digest,planning_ms=samples))
 if len(rows)%30==0:print(json.dumps(dict(checked=len(rows))),flush=True)
assert len(rows)==240
result=dict(checked=240,all_off_none=True,table_identity=True,raw_fallback_identity=True,mean_planning_ms=statistics.mean(x for r in rows for x in r['planning_ms']),cases=rows)
a.output.write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
