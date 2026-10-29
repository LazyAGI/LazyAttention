'use strict';
// A real captured workload, separate from the synthetic dummy used by the original lesson.
const wcState={task:0,at:0};
const wcBack=()=>page===4||page===6,wcFa=()=>page===5||page===6;
const wcBands=worstCaseData.ranges.flatMap(rangeBand);
function wcTasks(){
 const back=wcBack(),fa=wcFa(),d=worstCaseData;
 if(fa)return (back?d.bwd:d.fwd).map((p,i)=>({label:`${back?'KV':'Q parent'} ${i}`,a:i*128,z:Math.min(d.S,(i+1)*128),seq:[...(back?p.partial:[...p.partial].reverse()).map(id=>({id,kind:'Partial'})),...(back?p.full:[...p.full].reverse()).map(id=>({id,kind:'Full'}))].map(x=>({...x,a:x.id*(back?64:128),z:Math.min(d.S,(x.id+1)*(back?64:128))})),range:null}));
 return d.ranges.flatMap((r,ri)=>Array.from({length:Math.ceil((r[back?3:1]-r[back?2:0])/128)},(_,i)=>{
  const a=r[back?2:0]+i*128,z=Math.min(r[back?3:1],a+128),step=back?64:128,seq=[];
  for(let q=r[back?0:2];q<r[back?1:3];q+=step){const end=Math.min(r[back?1:3],q+step),Q=back?{start:q,end}:{start:a,end:z},K=back?{start:a,end:z}:{start:q,end};if(r[4]===0||(r[4]===1?K.start-r[2]<=Q.end-1-r[0]+r[3]-r[2]-(r[1]-r[0]):K.end-1-r[2]>=Q.start-r[0]))seq.push({id:seq.length,a:q,z:end,kind:['Full','Causal','Upper'][r[4]]})}
  if(!back)seq.reverse();
  return {label:`range ${ri} · ${back?'KV':'Q'} ${i}`,a,z,seq,range:ri};
 }));
}
function wcMaskSvg(ranges,runs,label){
 const mx=x=>45+x/worstCaseData.S*510,my=x=>45+x/worstCaseData.S*510;
 const polygons=ranges.flatMap(rangeBand).map(b=>`<polygon points="${bandPoly(b,mx,my)}" fill="#9fcdf7"/>`).join('');
 const strip=runs.map(([a,z,gen])=>`<rect x="${mx(a)}" y="570" width="${Math.max(.7,mx(z)-mx(a))}" height="18" fill="${gen?'#9573c7':'#77aa91'}"><title>${gen?'GEN':'UND'} [${a}, ${z})</title></rect>`).join('');
 return `<div><h4>${label}</h4><div class="board"><svg viewBox="0 0 600 610" role="img" aria-label="${label} mask"><text x="45" y="25" font-size="15">K →　Q ↓</text><rect x="45" y="45" width="510" height="510" fill="#eef1f5"/>${polygons}${strip}<text x="45" y="604" font-size="13">紫色 GEN · 绿色 UND</text></svg></div></div>`;
}
function wcMaskComparison(){
 return `<div class="wc-mask-comparison"><h3>UND / GEN 重排序前后</h3><p>两图使用同一套 attention 规则和相同 token。重排序只改变 token 坐标：左图保留原始交错顺序，右图把 GEN 稳定移到前部；可见 token pair 总数均为 321,784,803。</p><div class="cols">${wcMaskSvg(worstCaseData.original_ranges,worstCaseData.original_gen_runs,'重排序前 · 原始 UND / GEN 交错')}${wcMaskSvg(worstCaseData.ranges,worstCaseData.reordered_gen_runs,'重排序后 · GEN → UND')}</div></div>`;
}
function wcPaint(){
 const host=document.getElementById('wc-example');if(!host)return;
 const tasks=wcTasksCache;wcState.task=Math.min(wcState.task,tasks.length-1);const task=tasks[wcState.task];wcState.at=Math.max(0,Math.min(wcState.at,Math.max(0,task.seq.length-1)));const cur=task.seq[wcState.at],back=wcBack(),fa=wcFa();
 const Q=back?{start:cur?.a??0,end:cur?.z??0}:{start:task.a,end:task.z},K=back?{start:task.a,end:task.z}:{start:cur?.a??0,end:cur?.z??0};
 const scale=x=>70+x/32640*600,sy=x=>45+x/32640*600;
 const available=fa?wcBands:rangeBand(worstCaseData.ranges[task.range]);
 const band=clippedBands(available,back?{start:0,end:32640}:Q,back?K:{start:0,end:32640});
 let marks=wcBands.map(b=>`<polygon points="${bandPoly(b,scale,sy)}" fill="#cce5ff"/>`).join('');
 marks+=band.map(b=>`<polygon points="${bandPoly(b,scale,sy)}" fill="#baa2dd"/>`).join('');
 marks+=clippedBands(band,Q,K).map(b=>`<polygon points="${bandPoly(b,scale,sy)}" fill="#1765c1"/>`).join('');
 marks+=`<rect x="${scale(K.start)}" y="${sy(Q.start)}" width="${Math.max(2,(K.end-K.start)/32640*600)}" height="${Math.max(2,(Q.end-Q.start)/32640*600)}" fill="none" stroke="#1765c1" stroke-width="2"/>`;
 host.innerHTML=`<h3>真实最差 case · rank 5 / batch 24</h3><p>S=32,640 · Q=[1,32640,32,128] · K/V=[1,32640,8,128] · BF16 · GQA=4。样本长度：62（padding id=-1）、31,178、1,400；长样本含 GEN 14,580、UND 16,598。这里使用捕获的物理 token 顺序、真实 mask 和稀疏计划，独立于上方 dummy。历史优化版 FA4 Backward 使用 logical index 重排；此图保留物理坐标解释同一形状，不模拟该优化版的实际调度顺序。</p><div class="controls"><label>任务 <select id="wc-task">${tasks.map((t,i)=>`<option value="${i}" ${i===wcState.task?'selected':''}>${t.label} · [${t.a},${t.z}) · ${t.seq.length} 轮</option>`).join('')}</select></label><button id="wc-prev">← 上一轮</button><button id="wc-next">下一轮 →</button><button id="wc-reset">回到第一轮</button><label>跳到第几轮 <input id="wc-at" type="number" min="1" max="${Math.max(1,task.seq.length)}" value="${wcState.at+1}" style="width:85px"></label></div><div class="cols"><div><p>浅蓝：全部可见 · 紫色：当前任务 · 深蓝：本轮</p><div class="board"><svg viewBox="0 0 720 700" role="img" aria-label="真实 case 的完整 mask、当前任务和本轮访问"><text x="70" y="25" font-size="16">K → · 0 … 32640；Q ↓</text><rect x="70" y="45" width="600" height="600" fill="#eef1f5"/>${marks}<text x="70" y="675" font-size="15">真实 token 比例；极窄块以外框辅助定位</text></svg></div></div><div><h4>${fa?'FA4':'MAGI'} ${back?'Backward':'Forward'}</h4><p>固定 ${back?'KV':'Q'} [${task.a}, ${task.z})；${cur?`第 ${wcState.at+1} / ${task.seq.length} 轮：${back?'Q':'KV'} [${cur.a}, ${cur.z}) · ${cur.kind}`:'Empty：不发起有效计算'}。</p><p>${fa?(back?'固定全局 128-token KV tile，遍历 64-token Q 列表。Full / Partial 分类由真实 mask 精确导出。':'图中显示捕获的 128×128 parent mask 列表（Partial 可保守包含实际不可见的位置，仍由逐元素 mask 屏蔽）；pack_gqa 的每张子 CTA 固定其中 32 个 Q tokens × 4 heads，共四张子 CTA。每张子 CTA 使用相同 parent KV 列表。'):`range ${task.range}：${JSON.stringify(worstCaseData.ranges[task.range])}。任务从 range 起点切分，${back?'64×128':'128×128'} tile，不与全局网格强制对齐。`}</p><p>${back?'每轮重算 P，计算 dP/dS，产生 ΔdQ 并累积 dK/dV。多个 KV 任务可能贡献同一 dQ 区域，必须跨任务归约。':'每轮 QK → masked softmax → PV，维护逐行 m、ℓ、A。结束后写 O/LSE；MAGI 同一 Q 的多个 ranges 还需按 LSE 合并。'}</p><p>稀疏计划核对：321,784,803 个可见 token pairs；FA4 Backward 最长列表 488 个 Q tiles。长循环与 dQ 写回路径值得观察，不能只由图示推断硬件瓶颈。</p><p>历史 TJ-CCI H800 完整 Backward：FA4 ≈38.5 ms，MAGI ≈27.5 ms（包含预处理、主 kernel、后处理）。Forward 未在此提供该 case 的独立计时。该 H800 结果不代表当前 M-CCI H100 性能。</p><p><a href="worst-case-verification.json">数据核对与历史实验摘要</a></p></div></div>`;
 host.querySelector('#wc-at').onchange=e=>{wcState.at=Math.max(0,Math.min(task.seq.length-1,(+e.target.value||1)-1));wcPaint()};host.querySelector('#wc-task').onchange=e=>{wcState.task=+e.target.value;wcState.at=0;wcPaint()};host.querySelector('#wc-next').onclick=()=>{wcState.at=Math.max(0,Math.min(task.seq.length-1,wcState.at+1));wcPaint()};host.querySelector('#wc-prev').onclick=()=>{wcState.at=Math.max(0,wcState.at-1);wcPaint()};host.querySelector('#wc-reset').onclick=()=>{wcState.at=0;wcPaint()};
}
function wcMount(){if(![2,4,5,6].includes(page))return;const section=document.createElement('section');section.id='wc-example';section.className='st-section';$('scene').appendChild(section);const jump=document.createElement('button');jump.textContent='查看真实最差 case · 5/24 ↓';jump.onclick=()=>section.scrollIntoView({block:'start'});$('controls').appendChild(jump);wcState.task=wcFa()?(wcBack()?worstCaseData.bwd.reduce((best,p,i)=>p.full.length+p.partial.length>worstCaseData.bwd[best].full.length+worstCaseData.bwd[best].partial.length?i:best,0):worstCaseData.fwd.reduce((best,p,i)=>p.full.length+p.partial.length>worstCaseData.fwd[best].full.length+worstCaseData.fwd[best].partial.length?i:best,0)):wcTasksCache.reduce((best,t,i)=>t.seq.length>wcTasksCache[best].seq.length?i:best,0);wcState.at=0;wcPaint();if(new URLSearchParams(location.search).get('example')==='worst')setTimeout(()=>section.scrollIntoView({block:'start'}),0)}
let wcTasksCache=[];
// Append the example after existing practice rendering, preserving all original lesson state.
for(const name of ['renderStrategy','renderBackwardActual','renderFa4']){const original=window[name];window[name]=function(...args){const result=original.apply(this,args);wcTasksCache=wcTasks();wcMount();return result}}
