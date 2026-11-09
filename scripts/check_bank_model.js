'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const context={document:{addEventListener(){}}};vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../docs/tutorials/attention-kernel-walkthrough/bank-explorer.js'),'utf8'),context);
vm.runInContext(`
for(const layout of ['plain','padding','swizzle']){
 const addresses=new Set();
 for(let r=0;r<32;r++)for(let c=0;c<32;c++)addresses.add(bankAddress(layout,r,c));
 if(addresses.size!==1024)throw Error(layout+' aliases logical elements');
 for(const axis of ['row','column'])for(let line=0;line<32;line++){
  const req=bankRequests(layout,axis,line),rounds=bankRounds(req);
  if(rounds!==(layout==='plain'&&axis==='column'?32:1))throw Error(layout+axis+line);
  for(let round=0;round<rounds;round++){
   const active=req.filter(t=>t.round===round);
   if(new Set(active.map(t=>t.bank)).size!==active.length)throw Error('service bank collision');
  }
  for(const t of req){if(t.bank!==t.word%32)throw Error('bank mapping');if(layout==='padding'&&t.word%33===32)throw Error('read padding');}
 }
}
`,context);
assert.equal(vm.runInContext("bankAddress('swizzle',31,31)",context),992);
console.log('All layouts: 1024 unique addresses, 32 rows/columns, bank counts and service rounds passed');
