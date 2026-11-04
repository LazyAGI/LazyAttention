#!/usr/bin/env node
// Check numerical examples used by the optimization explainer, without a DOM.
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const scope={};vm.createContext(scope);vm.runInContext(fs.readFileSync(path.join(__dirname,'../docs/tutorials/attention-kernel-walkthrough/optimizations.js'),'utf8'),scope);
const c=scope.optContribution();assert.ok(Math.abs(c.parts.reduce((n,p)=>n+p.ms,0)-(c.magi-c.final))<1e-12);assert.ok(Math.abs(c.total-14.97975278410374)<1e-9);
assert.equal(c.parts.map(x=>(x.ms/c.magi*100).toFixed(2)).join('+'),'5.06+8.86+1.06');
for(const model of scope.optMappingCases()){
 assert.equal(new Set(model.physical).size,64);
 for(let r=0;r<64;r++){
  const decoded=model.header>=0?model.header+r:model.header>-64?(r<32?r:96+r-32):model.physical[r];assert.equal(decoded,model.physical[r]);
 }
}
for(const capacity of [3,4,8]){
 const asc=scope.optCacheFrames(false,capacity),desc=scope.optCacheFrames(true,capacity);
 assert.equal(asc.length,19);assert.equal(desc.length,19);
 assert.equal(scope.optCacheTrace(false).map(x=>`${x.cta}:${x.q}`).sort().join(','),scope.optCacheTrace(true).map(x=>`${x.cta}:${x.q}`).sort().join(','));
 for(const frames of [asc,desc])frames.forEach((f,i)=>{assert.equal(f.hits+f.misses,i);assert.ok(f.cache.length<=capacity);assert.equal(new Set(f.cache).size,f.cache.length);});
 if(capacity===4){assert.equal(asc[18].hits,4);assert.equal(desc[18].hits,10);}
 if(capacity===8)assert.equal(asc[18].hits,desc[18].hits);
}
assert.equal(scope.optSchedule(false).end,10);assert.equal(scope.optSchedule(true).end,8);
assert.equal(scope.optSchedule(false).rows.reduce((n,j)=>n+j.n,0),scope.optSchedule(true).rows.reduce((n,j)=>n+j.n,0));
console.log('Contribution accounting,64-row mapping,LRU capacity controls and equal-work scheduling examples passed.');
