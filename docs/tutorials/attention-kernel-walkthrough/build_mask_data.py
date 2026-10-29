import io,pickle,zipfile,collections,json,ast,pathlib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).parent
class Reader(pickle.Unpickler):
 def __init__(self,z):
  self.z=z;self.prefix=z.namelist()[0].split('/')[0];super().__init__(io.BytesIO(z.read(self.prefix+'/data.pkl')))
 def find_class(self,m,n):
  if (m,n)==('collections','OrderedDict'):return collections.OrderedDict
  if m=='torch' and n in ['LongStorage','IntStorage','BoolStorage']:return {'LongStorage':'<i8','IntStorage':'<i4','BoolStorage':'?'}[n]
  if (m,n)==('torch._utils','_rebuild_tensor_v2'):return self.tensor
  raise ValueError((m,n))
 def persistent_load(self,pid):
  _,dtype,key,loc,size=pid
  return np.frombuffer(self.z.read(self.prefix+'/data/'+key),dtype=dtype)
 def tensor(self,storage,offset,shape,stride,*args):return np.lib.stride_tricks.as_strided(storage[offset:],shape=shape,strides=tuple(i*storage.itemsize for i in stride))
def load(p):
 with zipfile.ZipFile(p) as z:return Reader(z).load()
NAMES=['mot_document_ids','mot_doc_mod_ids','mot_image_gen_indicators','mot_is_image_token','mot_token_pos','mot_dup_boundary']
if __name__=='__main__':
 paths=sorted((ROOT/'artifacts/mot_fa4_common_20261020/logs/common-fa4-s32-wo0-workloads-all10').glob('*.pt'))
 hits=[]
 for p in paths:
  w=load(p);d,m,g,im,pos,dup=[w['metadata'][n] for n in NAMES]
  both=set(m[g & im].tolist()) & set(m[(~g) & im].tolist())
  if both:hits.append((p.name,list(both)))
 print(json.dumps({'files':len(paths),'same_image_gen_and_nongen':hits}))
 src=(ROOT/'mcore-bridge/src/mcore_bridge/model/modules/mot_magi_attention.py').read_text()
 fn=next(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='ranges_from_metadata')
 env={'np':np,'METADATA_NAMES':NAMES};exec(fn,env)
 path=next(p for p in paths if p.name=='workload-rank0-step3.pt');w=load(path);meta=w['metadata'];d,m,g,im,pos,dup=[meta[n] for n in NAMES];length=len(d)
 qr,kr,types=env['ranges_from_metadata'](meta,length);ranges=np.column_stack([qr,kr,types]).tolist()
 changes=np.zeros(length+1,bool);changes[[0,-1]]=True
 for a in [d,m,g,im,dup]:changes[1:-1]|=a[1:]!=a[:-1]
 edges=np.flatnonzero(changes);segments=[]
 for a,b in zip(edges[:-1],edges[1:]):
  segments.append(dict(start=int(a),end=int(b),doc=int(d[a]),mod=int(m[a]),gen=bool(g[a]),image=bool(im[a]),dup=bool(dup[a]),positions=pos[a:b].tolist(),label=('G' if g[a] else 'I' if im[a] else 'T')+str(d[a])))
 def expected(qs,qe,ks,ke):
  qp=np.arange(qs,qe)[:,None];kp=np.arange(ks,ke)[None,:]
  same_doc=d[qp]==d[kp];same_mod=m[qp]==m[kp]
  return ((same_doc&(pos[qp]>=pos[kp])) | (im[qp]&same_mod)) & ((~g[kp])|(g[qp]&same_mod)) & ((~g[qp])|g[kp]|(pos[kp]!=pos[qp])) & (~dup[kp])
 pairs=[];vis=0;err=0;overlap=0
 for qi,Q in enumerate(segments):
  for ki,K in enumerate(segments):
   qs,qe,ks,ke=Q['start'],Q['end'],K['start'],K['end']
   # Fast rejection using exact metadata gates; nonzero pairs still use the original predicate.
   if Q['doc']!=K['doc'] or (K['gen'] and not(Q['gen'] and Q['mod']==K['mod'])):continue
   a=expected(qs,qe,ks,ke);vis+=int(a.sum());cover=np.zeros(a.shape,np.uint8)
   ids=[]
   for ix,(rs,re,cs,ce,kind) in enumerate(ranges):
    y0,y1=max(qs,rs),min(qe,re);x0,x1=max(ks,cs),min(ke,ce)
    if y0>=y1 or x0>=x1:continue
    y=np.arange(y0,y1)[:,None];x=np.arange(x0,x1)[None,:]
    v=np.ones((y1-y0,x1-x0),bool) if kind==0 else (x-cs<=y-rs+(ce-cs)-(re-rs) if kind==1 else x-cs>=y-rs)
    cover[y0-qs:y1-qs,x0-ks:x1-ks]+=v;ids.append(ix)
   err+=int(((cover>0)!=a).sum());overlap+=int((cover>1).sum())
   rows=[]
   for row in a:
    dif=np.diff(np.r_[False,row,False].astype(np.int8));rows.append(list(zip(np.flatnonzero(dif==1).tolist(),np.flatnonzero(dif==-1).tolist())))
   bands=[]
   for component in range(max(map(len,rows),default=0)):
    i=0
    while i<len(rows):
     if len(rows[i])<=component:i+=1;continue
     start=i;lo,hi=rows[i][component];dl=dh=0
     if i+1<len(rows) and len(rows[i+1])>component:
      nl,nh=rows[i+1][component];dl,dh=nl-lo,nh-hi
      if dl not in (0,1) or dh not in (0,1):dl=dh=0
     j=i+1
     while j<len(rows) and len(rows[j])>component and rows[j][component]==(lo+(j-i)*dl,hi+(j-i)*dh):j+=1
     bands.append([qs+start,qs+j,ks+lo,ks+hi,dl,dh]);i=j
   if a.any():pairs.append(dict(q=qi,k=ki,bands=bands,rangeIds=ids,visible=int(a.sum()),total=int(a.size)))
 # Verify rejection did not omit any ranges.
 for r in ranges:
  assert any(seg['start']<=r[0]<seg['end'] for seg in segments)
 audit={'length':length,'visible_pairs':vis,'range_mismatches':err,'overlap_pairs':overlap,'scanned_workloads':len(paths),'workloads_with_gen_nongen_same_image':hits}
 assert err==0 and overlap==0
 assert ranges==json.load(open(ROOT/'artifacts/magi-explainer-20261020/workloads.json'))[1]['ranges']
 result={'source':str(path.relative_to(ROOT)),'segments':segments,'ranges':ranges,'pairs':pairs,'audit':audit}
 (OUT/'mask-data.js').write_text('const maskData='+json.dumps(result,separators=(',',':'))+';\n')
 (OUT/'mask-verification.json').write_text(json.dumps(audit,indent=2));print(audit)
