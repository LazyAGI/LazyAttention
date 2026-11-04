'use strict';
// Interactive examples explain dependencies; hardware measurements are in the evidence appendix.
const optState={schedule:0,mapping:0,cache:0,prefetch:0};
const optTimers={};
let optCacheCapacity=4;
const optExtent={schedule:5,mapping:7,cache:18,prefetch:4};
function optStopOne(id){clearInterval(optTimers[id]);delete optTimers[id];const b=document.querySelector(`[data-opt-play="${id}"]`);if(b)b.textContent='▶ 播放';}
function optStop(){Object.keys(optTimers).forEach(optStopOne);}
function optSvg(content,height,label){return `<div class="board ov-board"><svg viewBox="0 0 1000 ${height}" role="img" aria-label="${{schedule:'样本工作量重排前后CTA工作条',mapping:'一段、两段连续与index的逻辑物理行映射',cache:'因果mask及正倒序LRU缓存请求模型',prefetch:'共享地址解码与两路合并预取依赖'}[label]}"><defs><marker id="ov-arrow-${label}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#1765c1"/></marker></defs>${content}</svg></div>`;}
function optSchedule(sorted){const jobs=[2,1,2,1,8,7].map((n,i)=>({n,id:i}));if(sorted)jobs.sort((a,b)=>b.n-a.n||a.id-b.id);const ends=[0,0,0],rows=[];jobs.forEach((j,i)=>{const lane=ends.indexOf(Math.min(...ends));rows.push({...j,lane,start:ends[lane],i});ends[lane]+=j.n;});return {rows,end:Math.max(...ends)};}
function optContribution(){
 const magi=15.470014660131392,final=13.152644708379103;
 const reorderRatio=13.5146/14.9056,reverseRatio=13.3711/13.5383;
 const foundation=final/(reorderRatio*reverseRatio),ordered=foundation*reorderRatio;
 return {magi,final,foundation,ordered,total:(magi-final)/magi*100,
  parts:[{name:'基础实现合并贡献',ms:magi-foundation,color:'#458a89',work:'工作②＋③中的GQA复用＋④',evidence:'剩余项，未独立拆分'},
   {name:'样本工作量重排',ms:foundation-ordered,color:'#1765c1',work:'工作①',evidence:'历史比例换算'},
   {name:'Q优先下完整倒序',ms:ordered-final,color:'#7554a4',work:'工作③中的倒序',evidence:'历史比例换算'}]};
}
function optMappingCases(){
 const groups=[[4,5,6,7,8,9,10,11],[0,1,2,3,12,13,14,15],[3,12,1,9,6,14,4,11]];
 return groups.map((g,i)=>({name:['一段连续','两段连续','index（行索引表）'][i],header:[32,-32,-64][i],groups:g,
  physical:Array.from({length:64},(_,r)=>g[Math.floor(r/8)]*8+r%8),
  encoding:['header = 32 ≥ 0；base = 32，不读 payload','header = −32；split = 32；pointer → [base0=0, base1=96]','header = −64；pointer → 64 个物理行 index'][i]}));
}
function optCacheTrace(reverse){
 const streams=[0,2,4].map(k=>{const a=Array.from({length:8-k},(_,i)=>k+i);return reverse?a.reverse():a;});
 const trace=[];for(let round=0;round<8;round++)streams.forEach((s,cta)=>{if(round<s.length)trace.push({cta,q:s[round],kv:[0,2,4][cta]});});return trace;
}
function optCacheFrames(reverse,capacity){
 const frames=[{cache:[],hits:0,misses:0,event:null}];let cache=[],hits=0,misses=0;
 for(const request of optCacheTrace(reverse)){const hit=cache.includes(request.q);if(hit){hits++;cache.splice(cache.indexOf(request.q),1);}else{misses++;if(cache.length===capacity)cache.shift();}cache.push(request.q);frames.push({cache:[...cache],hits,misses,event:{...request,hit}});}return frames;
}
function optControls(id){return `<div class="controls ov-controls"><button data-opt-play="${id}">▶ 播放</button><button data-opt-next="${id}">下一步 →</button><button data-opt-reset="${id}">↺ 重置</button><label><span id="opt-label-${id}">步骤 1</span><input data-opt-slider="${id}" type="range" min="0" max="${optExtent[id]}" value="${optState[id]}" aria-label="${{schedule:'工作量重排',mapping:'三类行映射',cache:'倒序缓存复用',prefetch:'合并预取'}[id]}步骤"></label></div>`;}
function optOverview(){
 return `<section class="ov-intro-map"><h3>先把四个问题分开</h3><p>这里把一组协作的GPU线程称为<strong>CTA</strong>。先把它想成一个不断从队列里领取任务的工作组：每次领到一块K/V，再去找需要与它计算的Q和dO。dO是从后续网络传回来的输出梯度。</p><div class="opt-table-wrap"><table><thead><tr><th>先理解什么问题</th><th>接下来怎样改</th></tr></thead><tbody><tr><td>有的任务很短，有的很长，最后几组拖住收尾。</td><td>① 提前规划，让重任务先执行。</td></tr><tr><td>逻辑上相邻的数据，物理上可能分散。</td><td>② 用连续、两段连续、index三类描述完成寻址。</td></tr><tr><td>多个任务会反复访问相同的数据。</td><td>③ 一个CTA复用多个head，并调整Q的遍历顺序。</td></tr><tr><td>找地址、搬数据和计算之间存在等待。</td><td>④ 合并寻址与预取，用流水覆盖搬运延迟。</td></tr></tbody></table></div><p class="ov-reading-route">下面每一项都按同样的顺序阅读：<b>看小例子 → 改一件事 → 跟动画走 → 看省下什么 → 带着剩下的问题往下走。</b></p></section>`;
}
function optScheduleDraw(){
 let s='';const step=optState.schedule;
 for(let side=0;side<2;side++){const x=side?520:20,{rows,end}=optSchedule(Boolean(side));s+=T(x,28,side?'规划后：重任务先领取':'原顺序：重任务落在队尾',20)+T(x,52,`收尾位置 ${end} · 相同总工作量21（教学单位）`,12,'dim');
  for(let lane=0;lane<3;lane++){s+=T(x,102+lane*65,`CTA ${lane}`,12);for(let t=0;t<=10;t++)s+=`<path d="M${x+60+t*39} 77v168" stroke="#e3eaf2"/>`;}
  rows.forEach(j=>{const active=j.i===step;s+=R(x+60+j.start*39,80+j.lane*65,j.n*39-3,40,active?C.blue:C.light)+T(x+65+j.start*39,106+j.lane*65,`T${j.id}·${j.n}`,12,active?'white':'');});
 }s+=T(20,290,'只改变任务领取顺序，Q/K/V留在原处；重活提前开始，其他工作组就少等一会儿。',15);
 $('opt-board-schedule').innerHTML=optSvg(s,315,'schedule');
 $('opt-note-schedule').textContent='教学模型：同样21单位工作，3个CTA，收尾位置从10变为8；不代表实测毫秒。高亮仅标记第'+(step+1)+'个领取的任务。';
}
function optMappingDraw(){
 let s='';const step=optState.mapping;
 optMappingCases().forEach((model,i)=>{const y=i*290,logicalX=174,logicalW=87,physicalX=174,physicalW=45;
  s+=T(20,y+25,`${i+1}. ${model.name}`,20)+T(20,y+50,['物理地址 = 起点 + 逻辑行偏移','在分界前后，分别沿两个物理区间前进','从行索引表中查出每个物理落点'][i],13,'dim')+T(20,y+100,'逻辑行块',14)+T(20,y+120,'每格8行',12,'dim')+T(20,y+205,'物理内存',14)+T(20,y+225,'按地址递增',12,'dim');
  for(let r=0;r<8;r++){const x=logicalX+r*logicalW,hot=r===step;s+=R(x,y+75,logicalW-6,43,hot?C.blue:'#e6d9fa')+T(x+9,y+93,`L${r}`,12,hot?'white':'')+T(x+9,y+109,`${r*8}–${r*8+7}`,11,hot?'white':'');
   const px=physicalX+model.groups[r]*physicalW+physicalW/2,lx=x+(logicalW-6)/2;
   s+=`<path d="M${lx} ${y+119} C${lx} ${y+142},${px} ${y+152},${px} ${y+182}" fill="none" stroke="${hot?'#1765c1':'#aab6c8'}" stroke-width="${hot?3:1}" opacity="${hot?1:.45}" marker-end="url(#ov-arrow-mapping)"/>`;
  }
  for(let g=0;g<16;g++){const hot=g===model.groups[step],used=model.groups.includes(g);s+=R(physicalX+g*physicalW,y+188,physicalW-3,40,hot?C.blue:used?C.light:'#eef1f5')+T(physicalX+g*physicalW+6,y+204,`P${g}`,11,hot?'white':'')+T(physicalX+g*physicalW+4,y+219,`${g*8}–${g*8+7}`,9,hot?'white':'');}
  const r=step*8,formula=i===0?`物理行 = 起点32 + 逻辑行${r} = ${model.physical[r]}`:i===1?(r<32?`逻辑行${r}<32：第一段起点0 + ${r} = ${model.physical[r]}`:`逻辑行${r}≥32：第二段起点96 + (${r}−32) = ${model.physical[r]}`):`查行索引表：逻辑行${r} → 物理行${model.physical[r]}`;
  s+=T(174,y+257,formula,15)+T(174,y+279,i===0?'从一个起点连续搬运，不必逐行查询索引。':i===1?'先读第一段，越过逻辑分界后跳到第二段，再连续前进。':'先查表再读取相应位置；发现连续小段时仍可以成段搬运。',12,'dim');
 });
 $('opt-board-mapping').innerHTML=optSvg(s,880,'mapping');
 $('opt-note-mapping').textContent=`当前L${step}的8行被高亮。逻辑行始终是0–63；箭头改变物理落点，实际张量没有复制。P0–P15表示128行物理内存中的16组，每组8行，灰格表示本逻辑块不访问。`;
}
function optCacheDraw(){
 let s=T(20,25,'同一因果mask：相邻KV列共享较大的Q尾部',21)+T(20,50,'Q/dO按行复用；K/V按列固定在各自CTA内。蓝色尾部是多个CTA共同需要的数据。',13,'dim');
 const selected=[0,2,4],cell=28,x0=62,y0=93;
 s+=T(64,78,'KV tile →',13)+T(12,102,'Q↓',13);
 for(let r=0;r<8;r++){s+=T(32,y0+r*cell+18,`${r}`,12);for(let k=0;k<8;k++){const visible=r>=k,shared=visible&&r>=4&&selected.includes(k);s+=R(x0+k*cell,y0+r*cell,cell-2,cell-2,shared?'#b2d7fa':visible?'#dfebdf':'#eef1f5');} }
 for(let j=0;j<3;j++){const k=selected[j];s+=T(x0+k*cell+2,88,`${k}`,12)+`<rect x="${x0+k*cell-1}" y="${y0-2}" width="${cell-1}" height="${cell*8+2}" fill="none" stroke="#1765c1" stroke-width="2"/>`;}
 s+=T(320,105,'CTA A：KV0 → Q0…Q7',15)+T(320,134,'CTA B：KV2 → Q2…Q7',15)+T(320,163,'CTA C：KV4 → Q4…Q7',15);
 s+=T(320,207,'正序从不同的边界开始：Q0 / Q2 / Q4。',15)+T(320,237,'倒序都从共同尾部开始：Q7 / Q7 / Q7。',15)+T(320,271,'若领取时间接近，Q/dO更容易在被逐出前被相邻CTA复用。',14)+T(320,301,'倒序让共同尾部的读取，集中在更接近的时间。',13,'dim');
 const step=optState.cache;
 if(step){
  const f=optCacheFrames(false,optCacheCapacity)[step].event,r=optCacheFrames(true,optCacheCapacity)[step].event;
  s+=`<rect x="${x0+f.kv*cell+2}" y="${y0+f.q*cell+2}" width="${cell-6}" height="${cell-6}" fill="none" stroke="#1765c1" stroke-width="3"/><circle cx="${x0+r.kv*cell+12}" cy="${y0+r.q*cell+12}" r="9" fill="none" stroke="#7554a4" stroke-width="3"/>`;
 }
 s+=T(320,329,'蓝框：正序当前请求；紫圈：倒序当前请求。',12,'dim');
 for(let side=0;side<2;side++){const x=side?520:20,frame=optCacheFrames(Boolean(side),optCacheCapacity)[step],trace=optCacheTrace(Boolean(side));s+=T(x,369,side?'完整倒序：先共同尾部':'正序：各自从边界向后',20);
  trace.forEach((e,i)=>{const xx=x+(i%9)*50,yy=390+Math.floor(i/9)*53,hot=i===step-1,done=i<step;s+=R(xx,yy,46,45,hot?C.blue:done?C.light:'#eef1f5')+T(xx+6,yy+17,`${'ABC'[e.cta]}`,10,hot?'white':'')+T(xx+6,yy+35,`Q${e.q}`,12,hot?'white':'');});
  s+=T(x,526,`缓存：左边更早使用，右边刚使用 · 容量${optCacheCapacity}`,13);
  for(let i=0;i<optCacheCapacity;i++){const value=frame.cache[i];s+=R(x+i*54,543,49,35,value===undefined?'#eef1f5':C.light)+T(x+i*54+9,566,value===undefined?'空':`Q${value}`,12);}
  s+=T(x,611,`命中 ${frame.hits} / 请求 ${step} · ${step?(frame.hits/step*100).toFixed(1):'0.0'}%`,17);
  s+=T(x,639,frame.event?`当前 ${'ABC'[frame.event.cta]} 读 Q${frame.event.q}：${frame.event.hit?'命中，复用已在缓存中的数据':'未命中，加入缓存'}`:'尚未发出读取',13,'dim');
 }
 $('opt-board-cache').innerHTML=optSvg(s,665,'cache');
 const af=optCacheFrames(false,optCacheCapacity)[step],df=optCacheFrames(true,optCacheCapacity)[step];
 $('opt-note-cache').textContent=`这张图的缓存可放${optCacheCapacity}块Q，已读取${step}/18次；正序命中${af.hits}次，倒序命中${df.hits}次。两边需要的数据相同，访问顺序不同。这里的计数来自教学模型。`;
}
function optPrefetchDraw(){
 const step=optState.prefetch;let s=T(20,27,'先合并地址工作，再成对发出预取',22)+T(20,52,'合并的是行遍历/地址解码和发起组织；Q与dO仍是两份数据、两条异步搬运请求。',14,'dim');
 s+=T(20,90,'分别查两次位置',18)+T(520,90,'一次寻址，两份搬运',18);
 for(let lane=0;lane<2;lane++){const y=111+lane*72;s+=R(20,y,170,48,'#e6d9fa')+T(31,y+29,`查找 ${lane?'dO':'Q'} 的行地址`,13)+T(201,y+29,'→',20)+R(236,y,180,48,C.light)+T(249,y+29,`发起 ${lane?'dO':'Q'} 搬运`,14);}
 s+=R(520,142,153,55,step===0?C.blue:'#e6d9fa')+T(533,164,'同一个逻辑行',12,step===0?'white':'')+T(533,184,'只查一次物理位置',12,step===0?'white':'');
 for(let lane=0;lane<2;lane++){const y=111+lane*72;s+=`<path d="M674 169L718 ${y+24}" stroke="#1765c1" stroke-width="2" fill="none"/>`+R(720,y,122,48,step===1?C.blue:C.light)+T(730,y+29,`发起 ${lane?'dO':'Q'} 搬运`,11,step===1?'white':'')+T(850,y+29,'→',18)+R(880,y,97,48,step===2?'#7554a4':'#e6d9fa')+T(891,y+29,`${lane?'dO':'Q'} 已就绪`,12,step===2?'white':'');}
 s+=T(520,274,'两份数据都到达各自缓冲区后，计算工作组再使用它们。',13);
 s+=T(20,323,'再把下一对的预取与当前tile的计算重叠',21);
 const labels=['为Q/dO查找一次共同位置','分别发起Q、dO两份搬运','等待两份数据到达缓冲区','计算A，同时提前搬运B','计算B，同时提前搬运C'];
 s+=T(20,353,`当前：${labels[step]}`,14,'dim');
 for(let i=0;i<3;i++){const y=386+i*62;s+=T(20,y+26,`tile ${'ABC'[i]}`,14);const loadX=145+i*180,computeX=loadX+110,loading=(step===1&&i===0)||(step===3&&i===1)||(step===4&&i===2),computing=(step===3&&i===0)||(step===4&&i===1);s+=R(loadX,y,100,39,loading?C.blue:C.light)+T(loadX+9,y+25,'预取 Q + dO',12,loading?'white':'')+R(computeX,y,185,39,computing?'#7554a4':'#e6d9fa')+T(computeX+12,y+25,`计算 tile ${'ABC'[i]} · 4 heads`,12,computing?'white':'');}
 s+=T(20,603,'一次寻址，两份搬运；后续搬运与当前计算接上。图中条长仅表示先后与可重叠的部分。',14);
 $('opt-board-prefetch').innerHTML=optSvg(s,630,'prefetch');
 $('opt-note-prefetch').textContent='阶段'+(step+1)+'：'+labels[step]+'。一次寻址服务两份搬运；两份数据都准备好后，再开始对应计算。';
}
function optWorkHTML(w){
 return `<section class="ov-work" id="opt-work-${w.id}"><div class="ov-work-heading"><span class="ov-work-number">${w.number}</span><h3>${w.title}</h3></div><div class="ov-teach-step"><h4>1 · 先看一个小例子</h4>${w.example}</div><div class="ov-teach-step"><h4>2 · 再改一件事</h4>${w.change}</div><div class="ov-teach-step"><h4>3 · 跟着动画走一遍</h4><ol class="ov-walk">${w.guide.map(x=>`<li>${x}</li>`).join('')}</ol>${w.extra||''}${optControls(w.id)}<div id="opt-board-${w.id}"></div><p class="ov-scope" id="opt-note-${w.id}" aria-live="polite"></p></div><div class="ov-teach-step ov-benefit"><h4>4 · 省下了什么</h4><p>${w.benefit}</p></div><details class="ov-local-detail"><summary>实现细节</summary>${w.details}</details><div class="ov-bridge"><h4>5 · 带着下一个问题往前走</h4><p>${w.bridge}</p></div></section>`;
}
function optWorkLessons(){return [
 {id:'schedule',number:'01',title:'让重任务早一点开始',
 example:'<p>有3个工作组，依次领到长度为2、1、2、1、8、7的任务。前几个很快结束，最后两个却还在运行。<b>所有工作组完成的时刻，由最后一个长任务决定。</b>其他工作组即使已经空闲，也不能让整次计算提前结束。</p>',
 change:'<p>先大致估算每个任务要遍历多少块Q，把工作量大的任务排到前面。工作组仍然领完一项再领下一项；改变的是领取顺序。</p><p>估算来自mask：固定一块K/V后，要访问的Q块越多，通常需要做的计算也越多。这里只需要分出轻重，不必精确预测每个任务会执行多少微秒。</p>',
 guide:['先看左边：按原顺序领取，较长的T4、T5直到队列后段才开始。','再看右边：同一组任务把8和7提前，较短任务交给其他工作组陆续完成。','点击“下一步”，沿着高亮看领取位置；最后比较三条工作条的收尾，而不是只看某一项有多快。'],
 benefit:'减少的是收尾等待。总计算量没有变，但更多重活同时推进，不容易让最后一个工作组独自拖住结束时间。',
 details:'<p>当前计划按逻辑样本分组，以组内最大任务的Q tile数估计负载，按64-tile档位降序稳定排列。计划仅改变任务ID表，不移动激活张量；常驻CTA持续从队列领取任务。相同形状、mask和分组可以复用这张表。</p>',
 bridge:'现在任务更均衡了，但每个工作组开始干活时，还要找到Q、K、V所在的位置。逻辑上相邻的数据，真的存放在相邻地址吗？'},
 {id:'mapping',number:'02',title:'分清计算顺序和数据存放位置',
 example:'<p>先回到双流布局：为了分别执行GEN与UND两条分支，数据先按GEN/UND归组存放，而不是把同一样本的所有token放在一起。例如，物理内存可能依次放着<b>A的UND、B的UND、A的GEN、B的GEN</b>。样本A原本相邻的两部分，现在隔着样本B的数据。</p><p>Attention仍然要按样本和mask寻找依赖。反向计算固定一块K/V、遍历它可见的Q时，同一样本的Q/dO便可能散落在多个物理区间。如果计算也沿着GEN/UND的物理排列推进，相邻K/V任务对同一份Q/dO的读取之间，就可能穿插其他样本的数据访问。</p><p><b>这为什么会慢？</b>L2缓存只能暂存有限的数据。刚读过的Q/dO如果很久才再次用到，中间读入的其他数据可能已经把它挤出去，下次就得重新从显存取。问题不只是“地址不连续”，还包括<b>本来可以复用的数据，访问得不够集中</b>。</p>',
 change:'<p>因此，把“下一步算谁”和“去哪里读”分开：<b>计算按逻辑样本组织，数据继续保留GEN/UND物理布局</b>。同一样本需要的Q行在逻辑空间里连起来，再用行映射找到原来的物理位置；无需先复制整张张量来恢复样本顺序。哪些行可以参与计算，仍由mask决定。</p><p>例如，样本A需要的64行Q，前32行在物理行0–31，后32行在96–127。逻辑遍历从第31行走到第32行时，实际地址从31跳到96。接下来要解决的，就是怎样便宜地表达这次跳转：</p><div class="ov-three-cases"><p><b>一段连续：</b>记住起点，后续位置用“起点＋偏移”得到。</p><p><b>两段连续：</b>再记一个分界和第二个起点；到分界前沿第一段走，到分界后跳到第二段。</p><p><b>index：</b>如果分散成更多段，就保留一张表，逐行查出物理位置。这保证不规则布局也能正确访问。</p></div>',
 guide:['上排始终是同一组连续的逻辑行；下排按物理地址排列。每格代表8行，箭头连接它们。','从“一段连续”开始，注意箭头整体落在一个区间；再看“两段连续”，观察L4（第5格）怎样跳到后面的物理区间。','移动高亮到其他逻辑格，最后看index如何依靠查表定位。三个例子都没有复制或搬动整张张量。'],
 benefit:'按逻辑样本组织计算，让相关数据的访问更集中，为L2复用创造条件；用连续区间描述地址，则省去逐行索引读取与重复地址计算。两者配合，也避免了整张张量重排的搬运成本。地址描述本身不会让缓存自动命中，真正的复用还要看任务之间何时再次读到相同数据。',
 details:'<p>一个Q64块有64行。descriptor的header≥0表示单段，header本身就是base；−63到−1表示两段，其绝对值是split，另一个header指向两个base；−64表示指向64项index payload。三类编码互斥，只有需要时才读payload。</p><p>读取和梯度写回使用同一映射。连续区间可用TMA搬运；混合路径按16/4行检查更小的连续段，真正离散的小段使用gather。图中P表示物理8行分组。mask的可见性位图与行映射分开保存。</p>',
 bridge:'现在可以跨过GEN/UND的物理间隔，按样本找到数据了。接下来还要缩短重复读取之间的距离：哪些head能共用K/V？相邻K/V任务又该从哪一块Q开始读？'},
 {id:'cache',number:'03',title:'先在CTA内复用，再让相邻任务更容易复用',
 example:'<p>一个head是一组并行的Attention计算。这里4个query head共享同一份K/V，这种分组方式称为GQA。如果分成4个独立任务，每个任务都要准备这份K/V。与此同时，不同K/V任务又可能访问相同的Q和dO。</p><p>这里有两种复用：<b>同一个工作组内部的复用</b>，以及<b>不同工作组之间的复用</b>。分开理解，就容易看清多head和倒序各在解决什么。</p>',
 change:'<h5>第一小步：把共享K/V的4个head放进同一CTA</h5><p>K/V加载一次，在CTA的片上缓冲区保留。处理当前Q块时，依次计算h0、h1、h2、h3，再进入下一块Q；这就是Q优先。四个head的Q/dO数值不同，但当前Q块的行映射和mask相关信息可以接着使用。</p><div class="ov-head-reuse"><div>同一份K/V<br><b>在CTA内保留</b></div><span>→</span><div>Q7：h0 → h1 → h2 → h3<br>Q6：h0 → h1 → h2 → h3<br>…</div></div><h5>第二小步：从多个任务共同需要的Q尾部开始</h5><p>看因果mask：固定靠前的K/V列，需要访问它下面的许多Q行；相邻K/V列访问的Q尾部大幅重叠。正序时，它们分别从自己的边界开始。倒序时，它们都先读共同尾部，访问相同数据的时刻更接近。</p><p>整个GPU共享的L2缓存可以暂存刚读过的数据。一个CTA刚读入Q7/dO7，另一个CTA紧接着也读它，就更有机会直接复用。间隔太久或缓存太满时，这份数据可能已经被替换。</p>',
 guide:['先看mask中圈出的3列：它们各是一项K/V任务，蓝色尾部表示共同需要的Q行。','从第一步开始比较两条请求序列：正序的起点不同，倒序会接连读共同尾部。再看缓存里保留下了哪些Q块。','读完18次请求后，把缓存容量从4块改为8块。容量充足时，两种顺序都能留住数据；这说明复用效果还取决于容量和任务到达时间。'],
 extra:'<div class="ov-cache-setting"><span>缓存可以同时放下：</span>'+[3,4,8].map(n=>`<button data-opt-capacity="${n}" aria-pressed="${optCacheCapacity===n}">${n} 块Q</button>`).join('')+'</div><p class="ov-scope">命中计数来自这张图的教学模型，用来观察访问顺序与缓存容量的关系。</p>',
 benefit:'CTA内部减少重复的K/V准备，Q优先复用当前Q块的元数据；跨CTA则通过接近的访问时刻，争取复用L2里已有的Q/dO。实际收益会随mask、缓存压力和调度情况变化。',
 details:'<p>shared memory属于单个CTA；L2由多个CTA共享。图用8×8等宽因果教学格，按轮次交错发出3个CTA的请求，并采用最近最少使用（LRU）替换规则。每块Q代表一项抽象缓存数据，不按真实Q64/KV128或硬件cacheline绘制。</p><p>实现中的原始Q列表是[Partial, Full]；完整倒序为[Full倒序, Partial倒序]。准备时反转索引，编译时选择Full-first，各条加载、计算和存储路径使用相同顺序。四份Q/dO仍各自读取，跨K/V任务的dQ贡献仍要累积。</p>',
 bridge:'同样的数据已经尽量复用了，但工作组还会在“找地址、等数据、做计算”之间停顿。最后把这些步骤衔接起来。'},
 {id:'prefetch',number:'04',title:'一次寻址服务两份搬运，让下一块提前到达',
 example:'<p>Q与dO使用相同的行映射：为了读逻辑行r，两边都要找到同一个物理行p。如果分别组织这两条数据流，就可能把相同的地址查找重复一遍。K与V也有类似关系。</p><p>即使地址找到了，计算也不能立即开始，还要等数据从显存到达片上缓冲区。</p>',
 change:'<h5>先合并寻址和发起过程</h5><p>只查一次“r对应哪个p”，随后用p分别发起Q和dO的搬运。<b>合并的是地址工作和发起组织；两份数据仍需分别搬运。</b>它们都到达各自缓冲区后，才能进行对应计算。</p><h5>再让后续搬运与当前计算重叠</h5><p>计算块A时，就提前发起块B的搬运；计算块B时，再提前搬运块C。如果下一块及时到达，就少了一段暴露出来的等待。这就是这里要建立的流水。</p>',
 guide:['看上半图左边：相同的物理行位置分别为Q和dO查找。第一步高亮右边的一次共同寻址。','继续单步：一份地址分成两路搬运，再等待两份数据都准备好。这里没有把两份数据变成一份。','看下半图：当A进入计算，B的预取同时推进；再前进一步，观察B计算时C怎样接上。'],
 benefit:'减少重复行遍历和地址依赖；让下一块的搬运尽量覆盖在当前块的计算后面，从而减少等待。数据字节数不因“合并”自动减少。',
 details:'<p>成对reader共享Q/dO或K/V的行映射；连续段走TMA，较小连续子段继续使用TMA，真正离散的边界走cp.async gather。Q/dO保留各自pipeline和barrier，只有所需数据就绪后才消费。K/V路径可通过同一个就绪barrier累计预期搬运字节数。</p><p>producer/consumer异步流水继承FA4基础，range/index混合访问接入这一依赖协议。图中条长表达依赖与重叠关系；具体等待长度由访存、计算和缓冲区资源决定。</p>',
 bridge:'现在四项工作可以放回同一条执行流程：先准备计划，再让每个工作组按计划找数据、提前搬运、复用并计算。'}
 ];}
function optComposition(){return `<section class="ov-composition" id="opt-composition"><h3>把它们串起来</h3><p>准备阶段生成行映射、任务顺序和Q遍历列表；执行阶段按计划领取任务，解码地址、提前搬运，在CTA内复用数据并累积梯度。<b>准备结果可以复用42次，因此无需每层重新规划。</b></p><div class="ov-flow"><div><small>一次准备</small><b>行映射 · 任务顺序 · Q遍历列表</b></div><span>↓ 复用42次</span><div><small>每次执行</small><b>领任务 → 找地址 → 提前搬运 → 复用与计算 → 梯度写回</b></div></div><p>四项工作分别缩短不同的开销：规划减少收尾等待，行映射减少找地址的工作，head与访问顺序增加数据复用，合并预取和流水减少搬运等待。它们围绕同一次Attention计算配合工作。</p></section>`;}
function optResults(){const c=optContribution();return `<section id="opt-results" class="ov-results"><h3>最后快了多少？</h3><div class="ov-hero"><div><span class="ov-kicker">完整240例 · 单张H100 · backward</span><div class="ov-times"><span><small>MAGI</small>15.470 <em>ms</em></span><b>→</b><span><small>LazyAttention</small>13.153 <em>ms</em></span></div><p>平均每次节省<b>2.317 ms</b>，加速<b>1.176×</b>。</p></div><div class="ov-gain">−14.98%<small>完整backward有效耗时</small></div></div><p>这里的有效耗时是“准备时间/42＋完整GPU backward”。240个病例均有超过1%的提升，梯度检查通过。</p><h4>主要收益来自哪里</h4><div class="ov-savings" aria-label="节省时间的贡献示意">${c.parts.map(x=>`<div style="flex:${x.ms};background:${x.color}"></div>`).join('')}</div><div class="ov-contribution-key"><span><i style="background:#458a89"></i>数据组织与基础复用</span><span><i style="background:#1765c1"></i>任务重排</span><span><i style="background:#7554a4"></i>Q遍历顺序</span></div><p>这张图帮助理解主次：任务重排带来较大的收尾改善，数据组织、基础复用与倒序访问共同补充。</p><details class="ov-method"><summary>贡献图怎样换算？</summary><p>统一以MAGI的15.470 ms为分母，归因估算约为：基础实现合并5.06个百分点，任务重排8.86个百分点，Q优先下倒序1.06个百分点，合计约14.98个百分点。</p><p>重排比例r=13.5146/14.9056；倒序比例v=13.3711/13.5383。以最新终点13.152645 ms反推基础阶段≈13.152645/(r×v)=14.687787 ms，重排后≈13.317113 ms；三个阶段差额分别除以MAGI基准。</p><p>这是跨轮次的归因估算。逻辑range、成对预取、基础GQA聚合及其他基础实现差异共同归入剩余项，没有各自独立的贡献数字；顺序、交互和比例迁移也会影响分项。L2命中是访问顺序的机制解释，倒序的全部时间收益不能唯一归因于L2。</p></details></section>`;}
function optAlternatives(){return `<details class="ov-appendix"><summary>为什么没有采用其他方案？</summary><p>有些改变在理论上能缓解一个问题，却会引入另一种开销。对照结果需要看这些变化合在一起后的净收益。</p><article class="ov-alternative"><h4>局部交错：想错开热点，却多了访问与准备成本</h4><p>把相邻Q块交错访问，理论上可以让多个CTA避开同一时刻的读写热点。但交错也改变了相邻访问的局部性，并需要额外准备排列。在这批输入中，GPU执行的微小改善被准备成本抵消，没有形成足够的净收益，因此保留连续的倒序遍历。</p></article><article class="ov-alternative"><h4>长任务切分：想缩短长尾，却增加了中间结果归约</h4><p>把一个超长任务拆成几段，可以让更多CTA同时处理它。代价是各段产生局部梯度，需要写入中间缓冲并额外合并；某些数据也可能重复准备。当任务本来已足够占满GPU时，新增开销容易超过缩短长尾的收益。这批高占用输入中，切分组合弱于只重排，因此默认不切分。</p></article><article class="ov-alternative"><h4>Head优先：想连续处理同一head，却失去Q块元数据复用</h4><p>先遍历一个head的全部Q，再换下一个head，看起来更集中。K/V在当前CTA里本来就能保留；换成Head优先反而让同一个Q块的行映射和mask信息隔更久才再次使用。该循环组织在对照中更慢，因此默认让同一Q块先服务四个head。</p></article></details>`;}
function optEvidence(){return `<details class="ov-appendix" id="opt-evidence"><summary>实验与实现依据</summary><h4>测量范围</h4><p>2026-11-04；H10080GB HBM3，BF16，D128，Hq32/Hkv8，S=31616–32768。Python3.12.12、Torch2.13.0+cu126、MAGI1.1.1、CutlassDSL4.6.2。240个捕获shape/mask，同一主机GPU、同一持续进程单轮配对，16个计时块、每块4周期，CUDA Graph内部事件计时。</p><p>MAGI与最终组合采用ABBA/BAAB和输出缓冲区交叉。有效耗时=额外准备/42＋完整GPU backward，含必要清零、preprocess、主核、postprocess与BF16梯度转换；共同capture/mask稀疏计划准备、forward、冷编译、验证以及Python发射/分配不计入。全部梯度最大绝对误差0.015625、相对L2约0.00124，阈值为0.0625和0.005。</p><h4>完整规划矩阵的汇总</h4><div class="opt-table-wrap"><table><thead><tr><th>切分</th><th>重排</th><th>平均有效耗时ms</th><th>相对无规划</th></tr></thead><tbody><tr><td>关</td><td>关</td><td>14.9056</td><td>基准</td></tr><tr><td>关</td><td>开</td><td>13.5146</td><td>−9.33%</td></tr><tr><td>开</td><td>关</td><td>19.4227</td><td>＋30.30%</td></tr><tr><td>开</td><td>开</td><td>13.5877</td><td>−8.84%</td></tr></tbody></table></div><p>每个配置覆盖240例；基准已是paired range路径。切分每段上限128个Q64 tile。</p><h4>完整遍历矩阵的汇总</h4><div class="opt-table-wrap"><table><thead><tr><th>CTA外层</th><th>Q方向</th><th>8-tile交错</th><th>平均有效耗时ms</th><th>相对Q正序</th></tr></thead><tbody>${[['Q','正序','关','13.5383','基准'],['Q','正序','开','13.6478','＋0.809%'],['Q','倒序','关','13.3711','−1.235%'],['Q','倒序','开','13.3883','−1.108%'],['Head','正序','关','13.8732','＋2.474%'],['Head','正序','开','13.9164','＋2.792%'],['Head','倒序','关','14.0908','＋4.081%'],['Head','倒序','开','14.1116','＋4.234%']].map(row=>'<tr>'+row.map(x=>`<td>${x}</td>`).join('')+'</tr>').join('')}</tbody></table></div><p>每个配置覆盖240例；重排开、切分关。该矩阵采用当轮冻结准备实现，随后有效列裁剪减少了准备耗时。各轮耗时有各自基准，贡献图使用比例迁移而不是直接相减。</p><h4>实现要点</h4><p>Full/Partial/Empty由稀疏计划区分，Partial128×128 tile使用2KiB精确位图。行descriptor为单段、两段和index互斥编码；读取与梯度scatter共享映射。FP32 dQ跨KV任务累积后由专用postprocess缩放、转BF16并写回物理位置。</p><p><a href="optimization-evidence.json">测量摘要、240例MAGI对照与教学模型参数</a> · <a href="../../benchmarks.md">复现入口</a> · <a href="https://github.com/LazyAGI/LazyAttention/blob/dev/CTA_TRAVERSAL.md">遍历策略定义</a> · <a href="https://github.com/LazyAGI/LazyAttention/blob/dev/TASK_PLANNING.md">规划策略定义</a></p></details>`;}
function optUpdate(id){
 const max=optExtent[id],host=$(`opt-board-${id}`),scrollLeft=host?.querySelector('.board')?.scrollLeft||0;
 optState[id]=Math.min(max,Math.max(0,optState[id]));
 ({schedule:optScheduleDraw,mapping:optMappingDraw,cache:optCacheDraw,prefetch:optPrefetchDraw}[id])();
 const board=host?.querySelector('.board');if(board)board.scrollLeft=scrollLeft;
 const slider=document.querySelector(`[data-opt-slider="${id}"]`);if(slider)slider.value=optState[id];
 const label=$(`opt-label-${id}`);if(label)label.textContent=id==='cache'?`已读 ${optState[id]}/18`:`步骤 ${optState[id]+1}/${max+1}`;
}

function renderOptimizations(){
 $('intro').innerHTML='<h2>8 · 为什么慢，怎样一步步变快</h2><p>Attention的数学没有改变。双流布局把同一样本的数据分散到了不同位置，也让不同GPU任务的工作量相差很大。我们要解决的是：<b>怎样找到这些数据、怎样减少重复搬运，以及怎样让GPU更均衡地完成工作。</b></p>';
 $('controls').innerHTML='';$('status').textContent='';
 $('scene').innerHTML=optOverview()+`<nav class="ov-toc" aria-label="四项优化工作">${[['schedule','① 任务先后'],['mapping','② 数据在哪里'],['cache','③ 哪些数据能复用'],['prefetch','④ 怎样少等一会儿']].map(([id,label])=>`<button data-opt-jump="opt-work-${id}">${label} ↓</button>`).join('')}</nav>`+optWorkLessons().map(optWorkHTML).join('')+optComposition()+optResults();
 $('detail').innerHTML=optAlternatives()+optEvidence();
 for(const id of Object.keys(optState)){
  document.querySelector(`[data-opt-play="${id}"]`).onclick=()=>{if(optTimers[id]){optStopOne(id);return;}if(optState[id]===optExtent[id])optState[id]=0;optUpdate(id);optTimers[id]=setInterval(()=>{optState[id]++;optUpdate(id);if(optState[id]===optExtent[id])optStopOne(id);},750);document.querySelector(`[data-opt-play="${id}"]`).textContent='Ⅱ 暂停';};
  document.querySelector(`[data-opt-next="${id}"]`).onclick=()=>{optStopOne(id);optState[id]++;optUpdate(id);};
  document.querySelector(`[data-opt-reset="${id}"]`).onclick=()=>{optStopOne(id);optState[id]=0;optUpdate(id);};
  document.querySelector(`[data-opt-slider="${id}"]`).oninput=e=>{optStopOne(id);optState[id]=Number(e.target.value);optUpdate(id);};optUpdate(id);
 }
 document.querySelectorAll('[data-opt-capacity]').forEach(b=>b.onclick=()=>{optCacheCapacity=Number(b.dataset.optCapacity);document.querySelectorAll('[data-opt-capacity]').forEach(x=>x.setAttribute('aria-pressed',String(Number(x.dataset.optCapacity)===optCacheCapacity)));optUpdate('cache');});
 document.querySelectorAll('[data-opt-jump]').forEach(b=>b.onclick=()=>document.getElementById(b.dataset.optJump).scrollIntoView({block:'start'}));
}
