"""Read trusted local torch snapshots without requiring torch or executing pickle globals."""
import pickle,zipfile,io,json,collections,ast,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent; root=P.parents[1]
source=root/'artifacts/fa4_integrated_20261027/workload-audit/rank-5/batch-024-mask.pt'
z=zipfile.ZipFile(source); prefix=z.namelist()[0].split('/')[0]
def tensor(storage,offset,shape,stride,*_):
 a=np.ndarray(shape,dtype=storage.dtype,buffer=storage,offset=offset*storage.dtype.itemsize,strides=tuple(s*storage.dtype.itemsize for s in stride));return a.copy()
def tensor3(storage,offset,shape,stride,grad,hooks,dtype,*_):return tensor(storage.view(dtype),offset,shape,stride)
class Reader(pickle.Unpickler):
 def find_class(self,m,n):
  if m=='torch._utils' and n=='_rebuild_tensor_v3':return tensor3
  if (m,n)==('torch.storage','UntypedStorage'):return 'uint8'
  if m=='torch' and n=='uint32':return 'uint32'
  if (m,n)==('collections','OrderedDict'):return collections.OrderedDict
  if m=='torch._utils' and n=='_rebuild_tensor_v2':return tensor
  if m=='torch' and n in ['IntStorage','LongStorage','BoolStorage','FloatStorage','ByteStorage']:return {'IntStorage':'int32','LongStorage':'int64','BoolStorage':'bool','FloatStorage':'float32','ByteStorage':'uint8'}[n]
  raise ValueError((m,n))
 def persistent_load(self,p):return np.frombuffer(z.read(prefix+'/data/'+p[2]),dtype=p[1])
m=Reader(io.BytesIO(z.read(prefix+'/data.pkl'))).load(); meta=m['metadata']; names=list(meta)
# Captured metadata is physical GEN-first order, as used by the operator.
src=(root/'mcore-bridge/src/mcore_bridge/model/modules/mot_magi_attention.py').read_text();fn=next(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='ranges_from_metadata')
env={'np':np,'METADATA_NAMES':names};exec(fn,env);qr,kr,types=env['ranges_from_metadata'](meta,32640);ranges=np.column_stack([qr,kr,types]).tolist()
# Recover the pre GEN/UND-partition order.  The partition is stable, so the
# relative order inside GEN and UND must be retained.  Merge those two streams
# by their captured original token position; GEN wins equal-position ties
# because an image span starts at that position before following text.
doc_order=[0,1,-1]
original_order=[]
for doc in doc_order:
 sample=np.flatnonzero(meta['mot_document_ids']==doc)
 gens=sample[meta['mot_image_gen_indicators'][sample].astype(bool)].tolist()
 unds=sample[~meta['mot_image_gen_indicators'][sample].astype(bool)].tolist()
 while gens or unds:
  if not unds or (gens and meta['mot_token_pos'][gens[0]]<=meta['mot_token_pos'][unds[0]]):
   original_order.append(gens.pop(0))
  else:
   original_order.append(unds.pop(0))
original_order=np.asarray(original_order)
original_meta={name:value[original_order] for name,value in meta.items()}
gen=original_meta['mot_image_gen_indicators'].astype(bool)
physical_order=np.r_[np.flatnonzero(gen),np.flatnonzero(~gen)]
for name in names:
 assert np.array_equal(original_meta[name][physical_order],meta[name]),name
oq,ok,ot=env['ranges_from_metadata'](original_meta,32640)
original_ranges=np.column_stack([oq,ok,ot]).tolist()
def bool_runs(values):
 out=[];a=0
 for i in range(1,len(values)+1):
  if i==len(values) or values[i]!=values[a]:
   out.append([a,i,int(values[a])]);a=i
 return out
def range_pairs(rs):
 total=0
 for a,b,l,h,t in rs:
  n=b-a;width=h-l
  if t==0:total+=n*(h-l)
  elif t==1:
   delta=width-n
   total+=sum(max(0,min(width,q+delta+1)) for q in range(n))
  else:total+=sum(max(0,width-q) for q in range(n))
 return total
assert range_pairs(original_ranges)==range_pairs(ranges)
layout=json.loads(source.with_name('batch-024-layout.json').read_text())
# Use captured Full/Partial plans (128x128 forward, 64x128 backward).
# Independently derive tile counts from disjoint exact ranges.
counts=np.zeros((510,255),np.int32)
for a,b,l,h,t in ranges:
 for q in range(a,b):
  lo=l+q-a if t==2 else l;hi=min(h,l+q-a+h-l-(b-a)+1) if t==1 else h
  if lo>=hi:continue
  k0,k1=lo//128,(hi-1)//128
  counts[q//64,k0:k1+1]+=np.minimum(hi,(np.arange(k0,k1+1)+1)*128)-np.maximum(lo,np.arange(k0,k1+1)*128)
fcounts=counts.reshape(255,2,255).sum(1)
def plan(row,area):return dict(partial=np.flatnonzero((row>0)&(row<area)).tolist(),full=np.flatnonzero(row==area).tolist())
assert int(counts.sum())==layout['valid_pairs']
assert int(((counts>0)&(counts<8192)).sum())==layout['partial_tiles']
assert int((counts==8192).sum())==layout['full_tiles']
def captured_rows(captured):
 return [{kind:captured[key+'_block_idx'][0,0,i,:int(captured[key+'_block_cnt'][0,0,i])].tolist() for kind,key in [('partial','mask'),('full','full')]} for i in range(255)]
forward=captured_rows(m['forward_plan']); backward=captured_rows(m['physical_backward_plan'])
for i in range(255):
 exact=plan(counts.T[i],8192)
 assert all(sorted(backward[i][k])==exact[k] for k in exact)
 exact=plan(fcounts[i],16384)
 assert set(forward[i]['full'])<=set(exact['full'])
 assert set(exact['partial']+exact['full'])<=set(forward[i]['partial']+forward[i]['full'])
result=dict(S=32640,rank=5,batch=24,samples=layout['samples'],ranges=ranges,
 original_ranges=original_ranges,fwd=forward,bwd=backward,
 original_gen_runs=bool_runs(original_meta['mot_image_gen_indicators']),
 reordered_gen_runs=bool_runs(meta['mot_image_gen_indicators']),
 audit=dict(visible_pairs=int(counts.sum()),max_qtiles=int((counts>0).sum(0).max()),
            captured_plan_rows_checked=510,reorder_roundtrip_fields=len(names),
            original_ranges=len(original_ranges),reordered_ranges=len(ranges),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),metadata_segments=[])
# Semantic runs: document / modality / generation / image / duplicate / position progression.
keys=['mot_document_ids','mot_doc_mod_ids','mot_image_gen_indicators','mot_is_image_token','mot_dup_boundary']; a=0
for i in range(1,32641):
 if i==32640 or any(meta[k][i]!=meta[k][a] for k in keys):
  result['metadata_segments'].append(dict(start=a,end=i,doc=int(meta[keys[0]][a]),gen=bool(meta[keys[2]][a]),image=bool(meta[keys[3]][a]),dup=bool(meta[keys[4]][a])));a=i
(P/'worst-case-data.js').write_text('const worstCaseData='+json.dumps(result,separators=(',',':'))+';\n')
(P/'worst-case-verification.json').write_text(json.dumps(result['audit'],indent=2))
print(result['audit'])
