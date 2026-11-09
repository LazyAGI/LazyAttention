'use strict';
const bankState={layout:'plain',axis:'column',line:0,round:0,lane:0};
let bankTimer=null;
function bankStop(){clearInterval(bankTimer);bankTimer=null;}
function bankAddress(layout,r,c){return layout==='padding'?r*33+c:r*32+(layout==='swizzle'?(c^r):c);}
function bankRequests(layout=bankState.layout,axis=bankState.axis,line=bankState.line){
 const queues=Array.from({length:32},()=>[]);
 return Array.from({length:32},(_,lane)=>{const r=axis==='row'?line:lane,c=axis==='row'?lane:line,word=bankAddress(layout,r,c),bank=word%32,round=queues[bank].length;queues[bank].push(word);return {lane,r,c,word,bank,round};});
}
function bankRounds(req){return Math.max(...req.map(t=>t.round))+1;}
function bankColor(b){return `hsl(${b*360/32} 58% 83%)`;}
function bankMatrix(req,layout){
 const padding=layout==='padding',rows=padding?33:32,u=20,x=48,y=45,byWord=new Map(req.map(t=>[t.word,t]));let s='';
 for(let bank=0;bank<32;bank++)s+=`<text x="${x+bank*u+10}" y="28" text-anchor="middle" font-size="10">${bank}</text>`;
 for(let row=0;row<rows;row++){
  s+=T(2,y+row*u+14,String(row*32),10);
  for(let bank=0;bank<32;bank++){
   const word=row*32+bank,isPad=padding&&word%33===32,t=byWord.get(word),active=t&&t.round===bankState.round,done=t&&t.round<bankState.round;
   const r=padding?Math.floor(word/33):row,slot=padding?word%33:bank,c=layout==='swizzle'?(slot^r):slot;
   const fill=isPad?'#596677':t?bankColor(bank):'#edf1f7';
   s+=`<g><title>word ${word} / byte ${word*4} · bank ${bank} · ${isPad?'padding':`A[${r},${c}]`}${t?' · lane '+t.lane:''}</title><rect x="${x+bank*u}" y="${y+row*u}" width="${u}" height="${u}" fill="${fill}" stroke="${active?'#152d4b':done?'#39916d':'#fff'}" stroke-width="${active?2.5:done?1.5:.6}"/>${t?`<text x="${x+bank*u+10}" y="${y+row*u+14}" text-anchor="middle" font-size="10">${t.lane}</text>`:isPad?`<text x="${x+bank*u+10}" y="${y+row*u+14}" text-anchor="middle" font-size="10" style="fill:white">P</text>`:''}</g>`;
  }
 }
 return `<div class="board bank-board"><svg viewBox="0 0 702 ${y+rows*u+18}" role="img" aria-label="按连续word地址每32槽折行的物理内存布局">${s}</svg></div>`;
}
function bankExplorer(){return `<section id="bank-explorer"><h3>32×32 矩阵：一条 warp load 到底访问哪些 bank？</h3><p>每次 32 个 lane 同时发出一条 FP32 标量读取：读行时 lane t 读 A[row,t]，读列时读 A[t,col]。下面按 bank 分组展示服务轮次，动画不是 GPU 周期实测；轮内及冲突组的顺序仅为示意。</p><div id="bank-explorer-body">${bankExplorerBody()}</div><p><small>模型：shared 基址按 128 字节对齐，32 个 bank，bank = word地址 mod 32。这里每 lane 读取不同 word；同 word 广播不构成这种冲突。XOR 是本例的显式软件布局，写入和读取必须使用同一映射；不等同于所有 TMA swizzle。BF16 / 向量指令需另按实际指令分析。依据：<a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks">NVIDIA shared memory banks</a>。</small></p></section>`;}
function bankMaxRounds(){return Math.max(...['plain','padding','swizzle'].map(layout=>bankRounds(bankRequests(layout))));}
function bankExplorerBody(){const st=bankState;
 const cards=[['plain','原始 · 32×32','r × 32 + c','行连续；列集中在同一 bank。'],['padding','Padding · 33×32','r × 33 + c','逻辑行跨物理行“小折行”；列逐行错位。'],['swizzle','XOR Swizzle · 32×32','r × 32 + (c XOR r)','逻辑行内重排；列访问显式交错。']].map(([layout,title,formula,caption])=>{
  const req=bankRequests(layout),rounds=bankRounds(req),focus=req[st.lane];
  return `<article><h4>${title}</h4><p><code>${formula}</code><br><b>${rounds===1?'无冲突 · 1 轮':rounds+' 路冲突 · '+rounds+' 轮'}</b> · ${st.round>=rounds?'已完成':'服务轮 '+(st.round+1)}</p>${bankMatrix(req,layout)}<p>${caption}</p><p class="bank-focus">L${focus.lane} → A[${focus.r},${focus.c}]<br>word ${focus.word} / byte ${focus.word*4}<br>bank ${focus.bank} · 服务轮 ${focus.round+1}</p></article>`;
 }).join('');
 return `<div class="controls"><button data-bank-axis="row" aria-pressed="${st.axis==='row'}">读一行</button><button data-bank-axis="column" aria-pressed="${st.axis==='column'}">读一列</button><label>${st.axis==='row'?'行':'列'}号 <input id="bank-line" type="number" min="0" max="31" value="${st.line}" style="width:70px"></label><button data-bank-action="reset">↺ 重播</button><button data-bank-action="round">下一服务轮</button><button data-bank-action="next">下一${st.axis==='row'?'行':'列'}</button><button data-bank-action="play">${bankTimer?'Ⅱ 暂停':'▶ 同步播放'}</button></div><p>同一次逻辑访问 · 第 ${st.line} ${st.axis==='row'?'行':'列'} · 示意服务轮 ${st.round+1}/${bankMaxRounds()}。先完成的布局保持完成状态，等待其他布局。</p><div class="bank-comparison">${cards}</div><p>三个物理网格每行均为连续 32 个 word，列号 = bank。彩色格是选中的 32 个地址，数字为 lane；深框为本轮服务，绿框为已服务，P 为 padding。</p><label>同步检查 lane <input id="bank-lane" type="range" min="0" max="31" value="${st.lane}"> ${st.lane}</label>`;
}
function bankPaint(){const host=document.getElementById('bank-explorer-body');if(host)host.innerHTML=bankExplorerBody();}
function bankAdvance(){if(bankState.round+1<bankMaxRounds())bankState.round++;else if(bankState.line<31){bankState.line++;bankState.round=0;}else bankStop();bankPaint();}
document.addEventListener('click',e=>{const b=e.target.closest('[data-bank-action],[data-bank-axis]');if(!b)return;if(b.dataset.bankAxis){bankStop();bankState.axis=b.dataset.bankAxis;bankState.round=0;bankPaint();return;}const a=b.dataset.bankAction;if(a==='play'){if(bankTimer)bankStop();else{if(bankState.line===31&&bankState.round===bankMaxRounds()-1){bankState.line=0;bankState.round=0;}bankTimer=setInterval(bankAdvance,450);}}else{bankStop();if(a==='reset'){bankState.line=0;bankState.round=0;}if(a==='round')bankState.round=(bankState.round+1)%bankMaxRounds();if(a==='next'){bankState.line=(bankState.line+1)%32;bankState.round=0;}}bankPaint();});
document.addEventListener('change',e=>{const id=e.target.id;if(!['bank-line','bank-lane'].includes(id))return;bankStop();if(id==='bank-line')bankState.line=Math.max(0,Math.min(31,Math.trunc(Number(e.target.value)||0)));if(id==='bank-lane')bankState.lane=+e.target.value;bankState.round=0;bankPaint();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)bankStop();});
