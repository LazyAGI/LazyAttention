"""Derive sparse tile lists from the audited dummy mask, without allocating N*N."""
import json
from pathlib import Path
p=Path(__file__).parent
raw=(p/'dummy-data.js').read_text();d=json.loads(raw[raw.index('{'):].strip().rstrip(';'))
counts=[[0]*256 for _ in range(512)]
for qs,qe,ks,ke,t in d['ranges']:
    for q in range(qs,qe):
        lo=max(ks,ks+q-qs) if t==2 else ks
        hi=min(ke,ks+q-qs+(ke-ks)-(qe-qs)+1) if t==1 else ke
        if lo>=hi: continue
        for k in range(lo//128,(hi-1)//128+1):
            counts[q//64][k]+=min(hi,(k+1)*128)-max(lo,k*128)
fcounts=[[counts[2*q][k]+counts[2*q+1][k] for k in range(256)] for q in range(256)]
def plan(row,area):
    return {'partial':[i for i,c in enumerate(row) if 0<c<area], 'full':[i for i,c in enumerate(row) if c==area]}
fwd=[plan(row,16384) for row in fcounts]
bwd=[plan([counts[q][k] for q in range(512)],8192) for k in range(256)]
audit={'visible_pairs':sum(map(sum,counts)),'fwd_visible_pairs':sum(map(sum,fcounts)), 'fwd_partial':sum(len(x['partial']) for x in fwd),'fwd_full':sum(len(x['full']) for x in fwd),'bwd_partial':sum(len(x['partial']) for x in bwd),'bwd_full':sum(len(x['full']) for x in bwd)}
assert audit['visible_pairs']==sum(x['visible'] for x in d['pairs'])
assert max(map(max,counts))<=8192
(p/'fa4-data.js').write_text('const fa4Data='+json.dumps({'fwd':fwd,'bwd':bwd,'audit':audit},separators=(',',':'))+';\n')
(p/'fa4-plan-verification.json').write_text(json.dumps(audit,indent=2))
print(audit)
