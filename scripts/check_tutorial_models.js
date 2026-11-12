#!/usr/bin/env node
// Numerical checks for the actual lesson functions, without a browser or GPU.
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.join(__dirname,'../docs/tutorials/attention-kernel-walkthrough');
const scope={document:{addEventListener(){}},window:{},renderMaskLesson(){},console};
vm.createContext(scope);
function load(name){vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),scope,{filename:name});}
load('u1-mask-primer.js');
vm.runInContext(`
const {tokens}=u1Example(),gen=tokens.filter(t=>t.gen),dup=tokens.filter(t=>t.group===4),ref=tokens.filter(t=>t.group===1),tail=tokens.filter(t=>t.group===5);
if(gen.length!==3||dup.length!==3)throw Error('missing paired fixture');
for(const q of gen){
 if(q.pos!==7)throw Error('image temporal positions must be constant');
 for(const k of dup)if(q.mod===k.mod||q.pos!==k.pos||u1Allowed(q,k))throw Error('GEN must not see any target-copy K');
 for(const k of gen)if(!u1Allowed(q,k))throw Error('GEN self block must be full');
 for(const k of ref)if(!u1Allowed(q,k))throw Error('reference input must remain visible');
}
for(const q of tail)for(const k of dup)if(!u1Allowed(q,k))throw Error('later text must see clean target');
for(const q of tokens){
 if(u1Allowed(q,{...tokens[0],dup:true}))throw Error('boundary K must be globally excluded');
 if(u1Allowed(q,{...tokens[0],doc:1}))throw Error('cross document access');
 for(const k of gen)if(!q.gen&&u1Allowed(q,k))throw Error('UND must not see GEN K');
}
// Token IDs and spatial coordinates do not enter this mask predicate.
for(const q of gen)for(const k of dup)if(u1Allowed({...q,id:123,h:9,w:8},{...k,id:456,h:0,w:1}))throw Error('spatial coordinates must not defeat the temporal gate');
const perm=[...tokens.filter(t=>t.gen),...tokens.filter(t=>!t.gen)];
for(const q of tokens)for(const k of tokens)if(u1Allowed(q,k)!==u1Allowed(perm[perm.indexOf(q)],perm[perm.indexOf(k)]))throw Error('partition changed visibility');
u1Mode='independent';if(u1Example().tokens.some(t=>t.mod===3))throw Error('independent mode has target copy');
`,scope);
load('plain-backward.js');load('row-backward.js');load('attention-backward.js');
// Compare accumulated lesson gradients with finite differences of dense loss.
const result=vm.runInContext(`(()=>{
 const clone=a=>a.map(r=>r.slice());
 function loss(Q,K,V,dO,mode){
  const raw=pbMul(Q,pbTranspose(K));let O;
  if(mode==='plain')O=pbMul(raw,V);
  else O=pbMul(raw.map((r,i)=>{const s=r.map((x,j)=>mode==='full'||j<=i?x/Math.sqrt(Q[0].length):-Infinity),m=Math.max(...s),e=s.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z)}),V);
  return O.reduce((v,r,i)=>v+r.reduce((a,x,j)=>a+x*dO[i][j],0),0);
 }
 let checks=0,maxError=0;
 for(const mode of ['plain','causal','full']){
  ab.mask=mode==='plain'?'full':mode;
  const accum={dQ:rbZeros(6,3),dK:rbZeros(6,3),dV:rbZeros(6,2)};
  for(let cta=0;cta<3;cta++){
   const seq=mode==='plain'?[0,1,2]:abSequence(cta);
   for(let it=0;it<seq.length;it++){
    const t=mode==='plain'?rbTile(it,cta):abTile(it,cta);
    for(const name of ['dQ','dK','dV'])for(let r=0;r<2;r++)for(let c=0;c<t[name][r].length;c++)accum[name][(name==='dQ'?t.i:t.j)+r][c]+=t[name][r][c];
   }
  }
  for(const [name,source] of [['dQ',rbQ],['dK',rbK],['dV',rbV]])for(let i=0;i<source.length;i++)for(let j=0;j<source[0].length;j++){
   const Q=clone(rbQ),K=clone(rbK),V=clone(rbV),a=name==='dQ'?Q:name==='dK'?K:V,eps=1e-5;
   a[i][j]+=eps;const plus=loss(Q,K,V,rbDO,mode);a[i][j]-=2*eps;const minus=loss(Q,K,V,rbDO,mode);
   const error=Math.abs((plus-minus)/(2*eps)-accum[name][i][j]);maxError=Math.max(maxError,error);checks++;if(error>1e-6)throw Error(mode+' '+name+' finite difference error '+error);
  }
 }
 return {checks,maxError};
})()`,scope);
assert.equal(result.checks,144);
// Full / causal / inverse range bands must exactly match integer predicates.
load('mask-data.js');load('mask-lesson.js');
vm.runInContext(`
for(const r of [[0,3,10,17,0],[0,3,10,17,1],[0,7,10,13,1],[0,7,10,13,2]]){
 const bands=rangeBand(r);
 for(let q=r[0];q<r[1];q++)for(let k=r[2];k<r[3];k++){
  const count=bands.filter(([a,z,l,h,dl,dh])=>q>=a&&q<z&&k>=l+(q-a)*dl&&k<h+(q-a)*dh).length;
  const expected=r[4]===0|| (r[4]===1?k-r[2]<=q-r[0]+r[3]-r[2]-(r[1]-r[0]):k-r[2]>=q-r[0]);
  if(count!==Number(expected))throw Error('range band mismatch');
 }
}
`,scope);
console.log(`U1 target-copy isolation, GEN self/reference access, boundary/cross-doc gates, range bands and ${result.checks} finite-difference gradients passed (max error ${result.maxError}).`);
// Every visible range tile must appear in the lesson's traversal lists.
load('dummy-data.js');load('dummy-lesson.js');load('strategy.js');load('compute-flow.js');load('backward.js');
vm.runInContext(`
const intersects=(r,qa,qz,ka,kz)=>r[4]===0||(r[4]===1?ka-r[2]<=qz-1-r[0]+r[3]-r[2]-(r[1]-r[0]):kz-1-r[2]>=qa-r[0]);
for(let ri=0;ri<dummyData.ranges.length;ri++){
 const r=dummyData.ranges[ri];
 for(let m=0;m<Math.ceil((r[1]-r[0])/128);m++){
  const ns=stNs(m,ri);stRange=ri;stM=m;
  let prior={m:-Infinity,l:0,A:0},logits=[],values=[];
  for(const n of ns){const v=cfValues(n,prior);v.scores.forEach((x,i)=>{if(v.allowed[i]){logits.push(x);values.push(v.v[i])}});prior={m:v.m,l:v.l,A:v.A};}
  if(logits.length){const max=Math.max(...logits),w=logits.map(x=>Math.exp(x-max)),z=w.reduce((a,b)=>a+b,0),o=w.reduce((a,b,i)=>a+b*values[i],0)/z;if(Math.abs(prior.A/prior.l-o)>1e-12||Math.abs(prior.m+Math.log(prior.l)-(max+Math.log(z)))>1e-12)throw Error('online softmax mismatch');}
  for(let n=0;n<Math.ceil((r[3]-r[2])/128);n++)if(intersects(r,r[0]+m*128,Math.min(r[1],r[0]+(m+1)*128),r[2]+n*128,Math.min(r[3],r[2]+(n+1)*128))&&!ns.includes(n))throw Error('forward skipped visible tile');
 }
 for(let n=0;n<Math.ceil((r[3]-r[2])/128);n++){
  const ms=bwMs({n},r,64,128);
  for(let m=0;m<Math.ceil((r[1]-r[0])/64);m++)if(intersects(r,r[0]+m*64,Math.min(r[1],r[0]+(m+1)*64),r[2]+n*128,Math.min(r[3],r[2]+(n+1)*128))&&!ms.includes(m))throw Error('backward skipped visible tile');
 }
}
`,scope);
console.log(`All ${vm.runInContext('dummyData.ranges.length',scope)} U1 tutorial ranges: traversal covers visible tiles; online softmax agrees with dense sampled-key normalization.`);

// The shared lesson fixture must contain U1 image blocks, never diagonal-hole cases.
vm.runInContext(`
if(dummyData.ranges.some(r=>r[4]===2))throw Error('inverse-causal range leaked into U1 tutorial');
const images=DB.filter(b=>b.image);
if(new Set(images.map(b=>b.mod)).size!==images.length||images.some(b=>b.linear))throw Error('invalid image segment identity/position');
for(const q of images)for(const k of images){
 const pair=dummyData.pairs.find(p=>p.q===q.id&&p.k===k.id);
 const expected=u1Allowed(q,k)?q.length*k.length:0;
 if((pair?.visible||0)!==expected)throw Error('image-image block must be wholly allowed or wholly excluded');
}
for(const copy of DB.filter(b=>b.copied_from!=null)){
 const gen=DB[copy.copied_from];
 if(!gen.gen||copy.gen||copy.oldStart<=gen.oldStart||gen.pos!==copy.pos||gen.length!==copy.length)throw Error('invalid target-copy pair');
 if(dummyData.pairs.some(p=>p.q===gen.id&&p.k===copy.id))throw Error('target copy generated a MAGI range');
 const later=DB[copy.id+1];
 if(!dummyData.pairs.some(p=>p.q===later.id&&p.k===copy.id&&p.visible===later.length*copy.length))throw Error('later text cannot read clean copy');
}
for(const name of ['self','causal','reference']){
 const ri=lessonExampleRange(name);if(ri<0||ri>=dummyData.ranges.length)throw Error('stale MAGI preset');
}
`,scope);
load('fa4-data.js');load('fa4-lesson.js');
vm.runInContext(`
let page=5;
for(const back of [false,true]){
 page=back?6:5;
 for(const name of ['self','causal','boundary']){
  faState().tile=faExampleTile(name);
  const plan=faPlan();
  if(!plan||!plan.partial.length&&!plan.full.length)throw Error('stale FA4 preset');
 }
}
`,scope);
console.log('All image-image blocks are Full or empty; clean copies have no GEN ranges; MAGI/FA4 presets resolve against regenerated data.');
