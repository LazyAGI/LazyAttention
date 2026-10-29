"""Select diverse captured masks and export pre/post GEN partition ranges."""
import ast,collections,hashlib,io,json,pickle,zipfile
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
AUDIT=ROOT/'artifacts/fa4_integrated_20261027/workload-audit'
NAMES=('mot_document_ids','mot_doc_mod_ids','mot_image_gen_indicators',
       'mot_is_image_token','mot_token_pos','mot_dup_boundary')

def tensor(storage,offset,shape,stride,*_):
 return np.ndarray(shape,dtype=storage.dtype,buffer=storage,offset=offset*storage.dtype.itemsize,
                   strides=tuple(s*storage.dtype.itemsize for s in stride)).copy()
def tensor3(storage,offset,shape,stride,grad,hooks,dtype,*_):
 return tensor(storage.view(dtype),offset,shape,stride)
class Reader(pickle.Unpickler):
 def find_class(self,m,n):
  if m=='torch._utils' and n=='_rebuild_tensor_v3':return tensor3
  if (m,n)==('torch.storage','UntypedStorage'):return 'uint8'
  if m=='torch' and n=='uint32':return 'uint32'
  if (m,n)==('collections','OrderedDict'):return collections.OrderedDict
  if m=='torch._utils' and n=='_rebuild_tensor_v2':return tensor
  if m=='torch' and n in ['IntStorage','LongStorage','BoolStorage','FloatStorage','ByteStorage']:
   return {'IntStorage':'int32','LongStorage':'int64','BoolStorage':'bool',
           'FloatStorage':'float32','ByteStorage':'uint8'}[n]
  raise ValueError((m,n))
 def persistent_load(self,p):
  return np.frombuffer(self.archive.read(self.prefix+'/data/'+p[2]),dtype=p[1])
def load(path):
 z=zipfile.ZipFile(path);r=Reader(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))))
 r.archive=z;r.prefix=z.namelist()[0].split('/')[0];return r.load()

source=(ROOT/'mcore-bridge/src/mcore_bridge/model/modules/mot_magi_attention.py').read_text()
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='ranges_from_metadata')
env={'np':np,'METADATA_NAMES':NAMES};exec(ast.get_source_segment(source,node),env)
def ranges(meta):
 q,k,t=env['ranges_from_metadata'](meta,len(meta[NAMES[0]]))
 return np.column_stack([q,k,t]).tolist()
def pair_count(rs):
 total=0
 for a,b,l,h,t in rs:
  n=b-a;width=h-l
  if t==0:total+=n*width
  elif t==1:total+=sum(max(0,min(width,q+width-n+1)) for q in range(n))
  else:total+=sum(max(0,width-q) for q in range(n))
 return total
def runs(v):
 out=[];a=0
 for i in range(1,len(v)+1):
  if i==len(v) or v[i]!=v[a]:out.append([a,i,int(v[a])]);a=i
 return out
def restore(meta):
 # Stable merge of the captured GEN and UND streams by original position.
 order=[]
 unique=[int(d) for d in np.unique(meta['mot_document_ids'])]
 docs=sorted(d for d in unique if d>=0)+sorted(d for d in unique if d<0)
 for doc in docs:
  sample=np.flatnonzero(meta['mot_document_ids']==doc)
  gen=sample[meta['mot_image_gen_indicators'][sample].astype(bool)].tolist()
  und=sample[~meta['mot_image_gen_indicators'][sample].astype(bool)].tolist()
  while gen or und:
   if not und or (gen and meta['mot_token_pos'][gen[0]]<=meta['mot_token_pos'][und[0]]):order.append(gen.pop(0))
   else:order.append(und.pop(0))
 order=np.asarray(order);original={n:meta[n][order] for n in NAMES}
 g=original['mot_image_gen_indicators'].astype(bool);partition=np.r_[np.flatnonzero(g),np.flatnonzero(~g)]
 assert all(np.array_equal(original[n][partition],meta[n]) for n in NAMES)
 return original

rows=[]
for p in sorted(AUDIT.glob('rank-*/batch-*-layout.json')):
 d=json.loads(p.read_text());samples=d['samples'];gen=sum(x['gen'] for x in samples);S=d['S']
 d['_path']=p;d['_features']=[np.log2(S),d['valid_pairs']/S**2,d['partial_tiles']/max(1,d['partial_tiles']+d['full_tiles']),
  d['qtiles_max']/(2*S/128),d['cross_sample_kv']/max(1,S/128),len(samples)/12,gen/S,
  max(x['length'] for x in samples)/S]
 rows.append(d)
X=np.asarray([r['_features'] for r in rows]);X=(X-X.mean(0))/np.maximum(X.std(0),1e-9)
worst=next(i for i,r in enumerate(rows) if r['rank']==5 and r['batch']==24)
chosen=[worst]
while len(chosen)<24:
 dist=np.min(((X[:,None,:]-X[chosen][None,:,:])**2).sum(2),axis=1);dist[chosen]=-1
 chosen.append(int(np.argmax(dist)))

cases=[]
for index in chosen:
 row=rows[index];rank,batch=row['rank'],row['batch'];mp=AUDIT/f'rank-{rank}'/f'batch-{batch:03d}-mask.pt'
 saved=load(mp);meta=saved['metadata'];original=restore(meta);post=ranges(meta);pre=ranges(original)
 assert pair_count(pre)==pair_count(post)==row['valid_pairs']
 cases.append(dict(rank=rank,batch=batch,S=row['S'],samples=row['samples'],ranges_before=pre,ranges_after=post,
  runs_before=runs(original['mot_image_gen_indicators']),runs_after=runs(meta['mot_image_gen_indicators']),
  metrics=dict(density=row['valid_pairs']/row['S']**2,full_tiles=row['full_tiles'],partial_tiles=row['partial_tiles'],
               qtiles_max=row['qtiles_max'],cross_sample_kv=row['cross_sample_kv'],ranges_before=len(pre),ranges_after=len(post)),
  source_sha256=hashlib.sha256(mp.read_bytes()).hexdigest()))
result=dict(corpus=240,selection='farthest-point sampling over 8 normalized structural features; rank5/batch24 forced',cases=cases)
(OUT/'mask-atlas-data.js').write_text('const maskAtlasData='+json.dumps(result,separators=(',',':'))+';\n')
(OUT/'mask-atlas-verification.json').write_text(json.dumps(dict(corpus=240,selected=len(cases),cases=[[c['rank'],c['batch']] for c in cases],roundtrip_fields=len(NAMES),pair_counts_equal=True),indent=2))
print('selected',[(c['rank'],c['batch'],c['metrics']['ranges_before'],c['metrics']['ranges_after']) for c in cases])
