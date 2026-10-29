import ast,json,pathlib,numpy as np,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=pathlib.Path(__file__).parent
names=['mot_document_ids','mot_doc_mod_ids','mot_image_gen_indicators','mot_is_image_token','mot_token_pos','mot_dup_boundary']
blocks=[]
def add(doc,label,n,image=False,gen=False,mod=0,pos=0,linear=True,dup=False):
 start=sum(b['length'] for b in blocks);blocks.append(dict(id=len(blocks),label=label,doc=doc,length=n,image=image,gen=gen,mod=doc*10+mod,pos=pos,linear=linear,dup=dup,oldStart=start,oldEnd=start+n))
add(0,'文字问题',1536);add(0,'纯文字回答',2560,pos=1536)
add(1,'多图提问',512);add(1,'输入图 A',768,True,mod=1,pos=512,linear=False);add(1,'图间文字',512,pos=513);add(1,'输入图 B',768,True,mod=2,pos=1025,linear=False);add(1,'文字回答',1536,pos=1026)
add(2,'生图提示',512);add(2,'生成图',2048,True,True,1,512,False);add(2,'文字尾部',1536,pos=513)
add(3,'编辑指令',256);add(3,'同图非 gen',1024,True,False,1,256,True);add(3,'同图 gen',1024,True,True,1,256,True);add(3,'编辑后文字',1792,pos=1280)
add(4,'多模态指令',512);add(4,'参考图',768,True,False,1,512,False);add(4,'中间文字',256,pos=513);add(4,'输出新图',1536,True,True,2,769,False);add(4,'结束文字',1024,pos=770)
add(5,'定位置提示',512);add(5,'同位置非 gen',1024,True,False,1,512,False);add(5,'同位置 gen',1024,True,True,1,512,False);add(5,'文字收尾',1536,pos=513)
add(6,'双图生成提示',512);add(6,'生成图 A',1024,True,True,1,512,False);add(6,'生成图 B',1024,True,True,2,513,False);add(6,'双图后文字',1536,pos=514)
add(7,'边界测试文字',512);add(7,'输入图',1024,True,False,1,512,False);add(7,'重复边界',128,True,False,1,512,False,True);add(7,'中段文字',1024,pos=513);add(7,'生成尾图',1408,True,True,2,1537,False)
# Deliberately vary sample lengths and introduce non-128-aligned tails.
for idx,length in {1:1533,6:2563,9:1021,13:2307,18:1281,22:1279,26:1507,31:1437}.items():blocks[idx]['length']=length
cursor=0
for b in blocks:b['oldStart']=cursor;b['oldEnd']=cursor+b['length'];cursor=b['oldEnd']
assert sum(b['length'] for b in blocks)==32768
old=[[] for _ in names]
for b in blocks:
 for a,v in zip(old,[b['doc'],b['mod'],b['gen'],b['image'],None,b['dup']]):a.extend((b['pos']+np.arange(b['length']) if b['linear'] else [b['pos']]*b['length']) if v is None else [v]*b['length'])
old=[np.asarray(a,dtype=bool if i in [2,3,5] else np.int64) for i,a in enumerate(old)]
perm=np.r_[np.flatnonzero(old[2]),np.flatnonzero(~old[2])];inv=np.argsort(perm);d,m,g,im,pos,dup=[a[perm] for a in old];not_image_gen_indicators=~g
order=sorted(range(len(blocks)),key=lambda i:not blocks[i]['gen']);cursor=0
for i in order:b=blocks[i];b['start']=cursor;b['end']=cursor+b['length'];cursor=b['end']
src=(ROOT/'mcore-bridge/src/mcore_bridge/model/modules/mot_magi_attention.py').read_text();fn=next(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='ranges_from_metadata');env={'np':np,'METADATA_NAMES':names};exec(fn,env);qr,kr,types=env['ranges_from_metadata'](dict(zip(names,[d,m,g,im,pos,dup])),32768);ranges=np.column_stack([qr,kr,types]).tolist()
# Execute the actual mask_mod body with captured arrays, independently of ranges.
source=(ROOT/'mcore-bridge/src/mcore_bridge/model/modules/mot.py').read_text();parent=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='create_flex_mask_padding_image_gen');node=next(x for x in parent.body if isinstance(x,ast.FunctionDef) and x.name=='mask_mod');masksrc=ast.unparse(node)
env2=dict(zip(['document_ids','doc_mod_ids','image_gen_indicators','is_image_token','token_pos','dup_boundary'],[d,m,g,im,pos,dup]));env2['not_image_gen_indicators']=~g;exec(masksrc,env2);maskfn=env2['mask_mod']
pairs=[];errors=overlap=visible=0
for Q in blocks:
 for K in blocks:
  qs,qe,ks,ke=Q['start'],Q['end'],K['start'],K['end'];ids=[i for i,r in enumerate(ranges) if r[0]<qe and r[1]>qs and r[2]<ke and r[3]>ks]
  if Q['doc']!=K['doc']:
   assert not ids;continue
  a=maskfn(None,None,np.arange(qs,qe)[:,None],np.arange(ks,ke)[None,:]);cover=np.zeros(a.shape,np.uint8)
  for idx in ids:
   rs,re,cs,ce,t=ranges[idx];y0,y1=max(qs,rs),min(qe,re);x0,x1=max(ks,cs),min(ke,ce);y=np.arange(y0,y1)[:,None];x=np.arange(x0,x1)[None,:];v=np.ones((y1-y0,x1-x0),bool) if t==0 else x-cs<=y-rs+(ce-cs)-(re-rs) if t==1 else x-cs>=y-rs;cover[y0-qs:y1-qs,x0-ks:x1-ks]+=v
  errors+=int(((cover>0)!=a).sum());overlap+=int((cover>1).sum());visible+=int(a.sum())
  if not a.any():continue
  rows=[]
  for row in a:
   v=np.diff(np.r_[False,row,False].astype(np.int8));rows.append(list(zip(np.flatnonzero(v==1).tolist(),np.flatnonzero(v==-1).tolist())))
  bands=[]
  for c in range(max(map(len,rows),default=0)):
   i=0
   while i<len(rows):
    if len(rows[i])<=c:i+=1;continue
    lo,hi=rows[i][c];dl=dh=0
    if i+1<len(rows) and len(rows[i+1])>c:
     nl,nh=rows[i+1][c];dl,dh=nl-lo,nh-hi
     if dl not in (0,1) or dh not in (0,1):dl=dh=0
    j=i+1
    while j<len(rows) and len(rows[j])>c and rows[j][c]==(lo+(j-i)*dl,hi+(j-i)*dh):j+=1
    bands.append([qs+i,qs+j,ks+lo,ks+hi,dl,dh]);i=j
  pairs.append(dict(q=Q['id'],k=K['id'],bands=bands,rangeIds=ids,visible=int(a.sum()),total=int(a.size)))
assert errors==overlap==0
assert np.array_equal(np.sort(perm),np.arange(32768))
# Direct old-order predicate vs permuted new-order predicate.
envold=dict(zip(['document_ids','doc_mod_ids','image_gen_indicators','is_image_token','token_pos','dup_boundary'],old));envold['not_image_gen_indicators']=~old[2];exec(masksrc,envold);rng=np.random.default_rng(42);qq=rng.integers(0,32768,100000);kk=rng.integers(0,32768,100000);assert np.array_equal(envold['mask_mod'](None,None,qq,kk),maskfn(None,None,inv[qq],inv[kk]))
# Focused rule assertions, including the hole's both sides and equal-position block.
def point(qid,kid,qo,ko):return bool(maskfn(None,None,blocks[qid]['start']+qo,blocks[kid]['start']+ko))
assert point(12,11,10,9) and point(12,11,10,11) and not point(12,11,10,10)
assert not point(21,20,0,0) and point(21,21,0,0)
assert not any(p['k']==29 for p in pairs)
assert set(types)=={0,1,2}
audit=dict(length=32768,blocks=len(blocks),documents=8,ranges=len(ranges),range_mismatches=errors,overlap_pairs=overlap,visible_pairs=visible,permutation_bijective=True,reorder_pair_checks=100000,range_types=sorted(set(types.tolist())),mask_source_sha256=hashlib.sha256(masksrc.encode()).hexdigest())
result=dict(blocks=blocks,order=order,ranges=ranges,pairs=pairs,audit=audit,cases=['纯文字输入 / 输出','多图输入、文字输出','文字到图像','同图 gen / 非 gen：挖对角线','参考图 + 新生成图','重复 position：整块挖空','多张 gen 图互相隔离','重复边界 K 屏蔽'])
(OUT/'dummy-data.js').write_text('const dummyData='+json.dumps(result,ensure_ascii=False,separators=(',',':'))+';\n');(OUT/'dummy-verification.json').write_text(json.dumps(audit,indent=2));print(audit)
