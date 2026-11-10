'use strict';
// Teaching timelines show dependencies, not measured GPU scheduling or timings.
const cudaSteps = { hardware: 0, arange: 0, shuffle: 0, sum: 0, bn1: 0, bn2: 0, bn3: 0 };
const cudaLimits = { hardware: 2, arange: 3, shuffle: 6, sum: 4, bn1: 2, bn2: 3, bn3: 3 };
let cudaTimer = null, cudaPlaying = null, cudaChannels = 64;
function cudaStop() {
  clearInterval(cudaTimer); cudaTimer = null; cudaPlaying = null;
  document.querySelectorAll('[data-cu-play]').forEach(b => { b.textContent = '▶ 播放'; b.setAttribute('aria-pressed', 'false'); });
}
function cudaControls(id) {
  return `<div class="cu-controls"><button data-cu-reset="${id}">↺ 重播</button><button data-cu-next="${id}">下一步 →</button><button data-cu-play="${id}" aria-pressed="false">▶ 播放</button><span id="cu-progress-${id}" class="muted"></span></div>`;
}
function cudaFrame(id) { return `${cudaControls(id)}<div id="cu-view-${id}" class="cu-view" aria-live="polite" aria-atomic="true"></div>`; }
function cudaChip(label, tone = '') { return `<span class="cu-chip ${tone}">${label}</span>`; }
function cudaPhase(labels, step) {
  return `<ol class="cu-phases">${labels.map((label,i) => `<li class="${i===step?'current':i<step?'done':''}" ${i===step?'aria-current="step"':''}>${label}</li>`).join('')}</ol>`;
}
function cudaHardware(step) {
  const sm = Array.from({length:4}, (_,i) => {
    const ids = step===0 ? [] : step===1 ? [2*i,2*i+1] : i<2 ? [8+2*i,9+2*i] : [];
    return `<div class="cu-sm"><b>SM ${i}</b><div>${ids.length?ids.map(j=>cudaChip(`CTA ${j}`,'blue')).join(''):cudaChip(step===2?'本批无任务':'等待任务')}</div></div>`;
  }).join('');
  return `${cudaPhase(['创建 grid','第一批 CTA 驻留','完成后接收下一批'],step)}<div class="cu-grid-label">一次 launch → 12 个 CTA；每 CTA 256 threads = 8 warps</div><div class="cu-queue"><b>${step===0?'等待执行':step===1?'尚未驻留':'已完成示意'}</b>${Array.from({length:step===0?12:step===1?4:8},(_,i)=>cudaChip(`CTA ${step===1?i+8:i}`)).join('')}</div><div class="cu-sm-grid">${sm}</div><p class="cu-caption">教学设备只画 4 个 SM，假设本例每 SM 可驻留 2 个 CTA。真实资源上限取决于 GPU、寄存器与 shared memory；图中分配次序不是 CUDA 的调度承诺。</p><div class="cu-warp"><b>放大 CTA 0 中的一个 warp</b><div class="cu-lanes">${Array.from({length:32},(_,i)=>`<span>${i}</span>`).join('')}</div><p>32 个 lane 各有自己的数据；以 warp 为组发射指令。若执行 <code>if (lane &lt; 16)</code>，只有对应 lane 参与该路径。一个 lane 不等于固定绑定一个 CUDA core。</p></div>`;
}
function cudaArange(step) {
  const blocks = [0,1,195,390];
  const ranges = blocks.map(b=>{
    const start=b*256,end=Math.min(start+256,100000),active=step>0&&(step!==1||b===0);
    return `<div class="cu-task ${active?'cu-task-active':''}"><b>CTA ${b}</b><code>[${start.toLocaleString('en-US')}, ${end.toLocaleString('en-US')})</code><span>${end-start} 个有效输出${b===390?' · 96 个线程越界':''}</span>${step>=2?`<small>out[${start}] = ${start}<br>out[${end-1}] = ${end-1}</small>`:'<small>各线程独立写自己的位置</small>'}</div>`;
  }).join('');
  return `${cudaPhase(['分配输出范围','展开 CTA 0','跳到中间与尾部','全部写入完成'],step)}<div class="cu-metrics">${cudaChip('N = 100,000','blue')}${cudaChip('256 threads / CTA')}${cudaChip('391 CTAs')}${cudaChip('跨 CTA 归约：0')}</div><div class="cu-task-grid">${ranges}</div><p class="cu-caption">只画 CTA 0、1、195、390，其余 387 个 CTA 省略。完整输出仍有 100,000 个元素，省略号不代表跳过计算。</p><div class="cu-lane-example"><b>${step===0?'任务映射':step===1?'放大：CTA 0，thread 7':step===2?'放大：CTA 390，thread 159 / 160':'完成条件'}</b><p>${['i = blockIdx.x × 256 + threadIdx.x；i &lt; 100000 时写 out[i] = i。','i = 0 × 256 + 7 = 7 → out[7] = 7。其他线程执行同样的指令，各写各的位置。','thread 159 → i = 99999：写入；thread 160 → i = 100000：越界，跳过。','全部 CTA 结束后，out[0…99999] 已就绪。一个 CTA 写完，不代表其他 CTA 已完成。'][step]}</p></div>`;
}
function cudaSumFlow(step, atomic) {
  const fill=step>=3?'#cce5ff':'#eef1f5', done=step===4;
  const labels=atomic?['CTA 0','CTA 1','CTA 97']:['partial[0]','partial[1]','partial[97]'];
  let nodes=labels.map((label,i)=>`<rect x="${12+i*142}" y="12" width="130" height="55" rx="7" fill="${fill}" stroke="#9caec5"/><text x="${77+i*142}" y="34" text-anchor="middle" font-size="13">${label}</text><text x="${77+i*142}" y="54" text-anchor="middle" font-size="13">${step>=3?(i===2?'672':'1024'):'待就绪'}</text>`).join('');
  nodes+=`<path d="M77 67 V88 H219 M361 67 V88 H219 M219 67 V111" fill="none" stroke="${step>=3?'#1765c1':'#9caec5'}" stroke-width="2"/><text x="235" y="106" font-size="12">${atomic?'原子累加 · 同一 kernel':'Kernel 2 · 归约'}</text><rect x="125" y="116" width="188" height="52" rx="7" fill="${done?'#dcebe1':'#eef1f5'}" stroke="#9caec5"/><text x="219" y="147" text-anchor="middle" font-size="17">out = ${done?'100000':atomic?'0 → …':'待写入'}</text>`;
  return `<svg class="cu-flow-svg" viewBox="0 0 438 180" role="img" aria-label="${atomic?'各 CTA 原子累加到同一输出':'局部结果通过第二个 kernel 汇总'}">${nodes}</svg>`;
}
function cudaSum(step) {
  const descriptions=[
    '每个线程在寄存器中累加至多 4 个元素；全部输入设为 1，便于核算。完整 CTA 每线程局部和为 4。',
    '一个完整 warp 的 32 个线程合并：32 × 4 = 128。warp 交换需要正确的参与 mask；这里用完整 warp 示意。',
    '256 threads = 8 warps；8 个 warp 和经 shared memory 与 CTA 同步合并成 1024。最后 CTA 只有 672 个有效输入，其余贡献为 0。',
    '每个 CTA 都已形成自己的局部和。两条路线从这里分开：竞争一个原子累加地址，或写入彼此独立的 partial 槽位。',
    '两条路线都得到 97 × 1024 + 672 = 100000。第二阶段只归约 98 个数，不必重新读取原来的 100000 个输入。'
  ];
  const completed=step===4;
  const atomic=step<3?'尚未提交':completed?'98 次 CTA 级原子加全部完成 → out = 100000':'每 CTA 一个线程执行 atomicAdd(out, block_sum)';
  const partial=step<3?'尚未写出':completed?'Kernel 2：归约 98 个 partial → out = 100000':'Kernel 1 写 partial[0…97]，每 CTA 独占一个槽位';
  return `${cudaPhase(['线程局部和','warp 内归约','CTA 内归约','提交 CTA 结果','最终汇总'],step)}<div class="cu-metrics">${cudaChip('N = 100,000','blue')}${cudaChip('输入：100,000 个 1')}${cudaChip('1024 元素 / CTA')}${cudaChip('98 CTAs')}</div><div class="cu-reduce-tree"><span class="${step>=0?'lit':''}">每线程 ≤ 4</span><b>→</b><span class="${step>=1?'lit':''}">每 warp ≤ 128</span><b>→</b><span class="${step>=2?'lit':''}">每 CTA ≤ 1024</span></div><p>${descriptions[step]}</p><div class="cu-compare"><article class="cu-route"><h4>A · CTA 局部归约 + atomicAdd</h4>${cudaSumFlow(step,true)}<div class="cu-memory">out = ${completed?'100000':'0（先初始化）'}</div><p>${atomic}</p><div class="cu-flow">${[0,1,97].map(i=>cudaChip(`CTA ${i}: ${i===97?672:1024}`,step>=3?'purple':'')).join('')}<span>→ 同一个地址</span></div><p class="cu-caption">仅画 3 个提交代表；实际 98 次。原子更新不会丢贡献，但同址竞争可能限制吞吐。初始化需排在归约前；下游读取需等待归约 kernel 完成。</p></article><article class="cu-route"><h4>B · 局部结果 + 二次归约 kernel</h4>${cudaSumFlow(step,false)}<div class="cu-memory">partial[98] → out</div><p>${partial}</p><div class="cu-flow">${[0,1,97].map(i=>cudaChip(`partial[${i}]${step>=3?' = '+(i===97?672:1024):''}`,step>=3?'green':'')).join('')}</div><p class="cu-caption">第二个 kernel 排在同一 stream 中，等 Kernel 1 完成后再读 partial。没有对同一 partial 槽位的写入竞争；代价是临时数组与额外 launch。</p></article></div>`;
}
function cudaBnConfig(id) {
  if(id==='bn1')return {c:4096,m:32,r:32,ct:4,parts:1,tasks:1024,shape:'N=1 · C=4096 · H=4 · W=8',label:'沿 C 拆分：每 CTA 处理 4 个完整通道'};
  if(id==='bn2')return {c:cudaChannels,m:65536,r:4096,ct:4,parts:16,tasks:cudaChannels/4*16,shape:`N=1 · C=${cudaChannels} · H=256 · W=256`,label:'同时拆 C 与 NHW：每 CTA 处理 4 通道 × 4096 位置'};
  return {c:3,m:16777216,r:65536,ct:1,parts:256,tasks:768,shape:'N=1 · C=3 · H=4096 · W=4096',label:'主要沿 NHW 拆分：每 CTA 处理 1 通道 × 65536 位置'};
}
function cudaBnDiagram(id,step,a) {
  const rows=id==='bn1'?4:8,cols=id==='bn3'?3:4;
  let cells='';
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    const channel=id==='bn3'?col:col*a.ct,chunk=id==='bn1'?0:row;
    const chosen=col===0&&(id==='bn1'||row===0),tone=step===0?'':step===1?'cu-stat':step===2?'cu-merged':'cu-output';
    cells+=`<div class="cu-bn-cell ${tone} ${chosen?'cu-selected':''}"><b>${id==='bn1'?`r ${row*8}–${row*8+7}`:`段 ${chunk}${row===rows-1?'…':''}`}</b><span>C ${channel}${a.ct>1?'–'+(channel+a.ct-1):''}</span></div>`;
  }
  return `<div class="cu-axis">横轴 C → · 纵轴 r = ((n × H) + h) × W + w ↓</div><div class="cu-bn-matrix" style="--cu-cols:${cols}">${cells}</div><p class="cu-caption">${id==='bn1'?'每列代表一组通道；竖向格子只是组内数据片段，整列属于同一个 CTA。只画前 16 通道。':`每个格子是一张独立 CTA 任务；只画前 ${rows} 个 NHW 分段${id==='bn3'?'与全部 3 个通道':'、前 16 通道'}，实际每通道有 ${a.parts} 段。`}粗框追踪第一组通道的 ${id==='bn1'?'完整任务':'第 0 段任务'}。逻辑矩阵不代表实际存储布局。</p>`;
}
function cudaBnFlow(a,step,split) {
  const ready=split?step>=2:step>=1, output=split?step===3:step===2;
  return `<div class="cu-bn-flow"><div class="${step>=1?'cu-ready':''}"><b>${split?`${a.parts} 段局部状态`:'本 CTA 输入'}</b><small>${split?`每段 count = ${a.r}`:`4 通道 × ${a.m} 位置`}</small></div><span>↓ ${split?'合并所有段':'局部统计'}</span><div class="${ready?'cu-ready':''}"><b>通道 c=0 的 μ / σ²</b><small>${ready?`最终 count = ${a.m.toLocaleString('en-US')}`:'等待统计完成'}</small></div><span>↓ ${split?'广播到输出任务':'片上复用输入'}</span><div class="${output?'cu-ready':''}"><b>${output?'y[r,0] 写回':'等待最终统计'}</b><small>每个元素用同一通道的统计</small></div></div>`;
}
function cudaBn(id,step) {
  const a=cudaBnConfig(id),split=id!=='bn1';
  const labels=split?['二维任务网格','Kernel 1：分段统计','Kernel 2：合并统计','Kernel 3：归一化输出']:['按通道分工','CTA 内完成统计','同 CTA 归一化输出'];
  let explanation;
  if(!split)explanation=[
    '4096 通道 / 每 CTA 4 通道 = 1024 CTAs。每 CTA 已拥有所负责通道的全部 32 个位置，因此不用跨 CTA 交换统计。',
    '一个示例映射：128 threads / CTA，4 个 warp 各负责一个通道；每个 lane 读取一个位置，再在 warp 内归约。没有跨 CTA 归约，仍有 warp 内归约。',
    '得到均值和方差后，各 lane 复用寄存器中的输入，归一化并写回。统计与输出可在一个 kernel 中完成；此映射是便于讲解的一种选择。'
  ][step];
  else explanation=[
    `grid = (${Math.ceil(a.c/a.ct)} 个通道组, ${a.parts} 个 NHW 分段)，共 ${a.tasks} CTAs。每段内部可以循环读取，无需用 ${a.r} 个线程。`,
    `每 CTA 为每个负责的通道写一份 (count, mean, M2)。共 ${a.c*a.parts} 份状态，分别写入独立 partial[channel, segment]。`,
    `示例用每通道一个 CTA，分别合并 ${a.parts} 份状态，得到 ${a.c} 对最终均值/方差；这里的 CTA 数较少，但处理的已是小得多的中间状态。`,
    `输出阶段重新沿元素分块：以 1024 元素一张任务为例，共 ${(a.c*a.m/1024).toLocaleString('en-US')} CTAs。每个元素读取所属通道的最终统计。统计只有 ${a.c} 组，输出仍可大量并行。`
  ][step];
  return `${cudaPhase(labels,step)}<div class="cu-metrics">${cudaChip(a.shape,'blue')}${cudaChip(`M = NHW = ${a.m.toLocaleString('en-US')}`)}${cudaChip(`统计任务：${a.tasks} CTAs`)}</div><div class="cu-bn-layout"><div>${cudaBnDiagram(id,step,a)}</div><div class="cu-bn-explain"><h4>${a.label}</h4><p>${explanation}</p>${cudaBnFlow(a,step,split)}<div class="cu-memory">${split?(step<2?`partial[C, ${a.parts}] → 最终统计[C]`:`最终统计[${a.c}] → 所有输出`):'局部输入 → 通道统计 → 本 CTA 输出'}</div><p class="cu-caption">${split?'三个 kernel 按同一 stream 顺序执行；阶段边界提供依赖顺序。动画不把 atomicAdd 当作全 grid 屏障。':'C 足够大，可以从通道轴获得大量独立任务；需要结合物理布局与资源占用验证性能。'}</p></div></div>`;
}
function cudaUpdate(id) {
  const el=document.getElementById(`cu-view-${id}`); if(!el)return;
  const step=cudaSteps[id];
  el.innerHTML=id==='hardware'?cudaHardware(step):id==='arange'?cudaArange(step):id==='sum'?cudaSum(step):id==='shuffle'?cudaShuffle(step):cudaBn(id,step);
  document.getElementById(`cu-progress-${id}`).textContent=`${step+1} / ${cudaLimits[id]+1}`;
  const next=document.querySelector(`[data-cu-next="${id}"]`); if(next)next.disabled=step===cudaLimits[id];
}
function cudaCode(source) {
  return `<pre class="cu-code"><code>${source.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</code></pre>`;
}
function cudaIndexNote(text) { return `<div class="cu-index-note"><b>ID → 数据地址</b><p>${text}</p></div>`; }
function cudaArangeCode() {
  return cudaCode(`__global__ void arange(int* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = i;  // 不读输入，只写输出
}
// Host: arange<<<391, 256, 0, stream>>>(out, 100000);`)+cudaIndexNote('<code>blockIdx.x</code> 选 256 元素的区间；<code>threadIdx.x</code> 选区间内位置。CTA 390 / thread 159 → i=99999，写 out[99999]；thread 160 → i=100000，跳过。arange 没有输入数组读取，写入值直接由 i 计算。');
}
function cudaShuffle(step) {
  const offsets=[16,8,4,2,1]; let values=Array(32).fill(4);
  for(let k=0;k<Math.min(step,5);k++) {const prev=values.slice();values=prev.map((v,i)=>i+offsets[k]<32?v+prev[i+offsets[k]]:v);}
  if(step===6)values.fill(values[0]);
  return `${cudaPhase(['线程局部值','offset=16','offset=8','offset=4','offset=2','offset=1','广播 lane 0'],step)}<div class="cu-shuffle-lanes">${values.map((v,i)=>`<div class="${i===0||step===6?'cu-ready':''}"><small>lane ${i}</small><b>${v}</b></div>`).join('')}</div><p>${step===0?'每个 lane 已从自己的 4 个输入累加出 4。下面演示一个完整 warp，32 lanes 都参与。':step<6?`本轮每个 lane 从 lane+${offsets[step-1]} 取值；来源在 0…31 内时才相加。lane 0 当前合并了 ${2**step} 个线程的贡献，值为 ${values[0]}。`:'__shfl_sync(FULL, value, 0) 让每个 lane 读取 lane 0 的最终值。现在全部 lane 都得到 128。'}</p><p class="cu-caption">只画一个完整 warp；offset=1 后只有 lane 0 保证是全 warp 的总和，其余 lane 保留部分和。shuffle 交换寄存器值，不读取输入数组，也不能访问另一个 warp。</p>`;
}
function cudaShuffleCode() {
 return `<h3>先展开 warp 内的两种 shuffle</h3><div class="cu-compare"><article class="cu-route"><h4>__shfl_down_sync：读取 lane + offset</h4><p>lane i 从同一 warp 的 lane i+offset 取一个寄存器值。<b>它只搬值，不自动做加法。</b>取回的 other 还需要执行 value += other。</p></article><article class="cu-route"><h4>__shfl_sync：读取指定 lane</h4><p><code>__shfl_sync(FULL, value, 0)</code> 让各 lane 读取 lane 0 的 value；归约结束后可以用它广播最终结果。</p></article></div>${cudaFrame('shuffle')}${cudaCode(`// 前提：一个完整 warp 的 32 lanes 都执行这些调用
__device__ float warp_sum(float value) {
  const unsigned FULL = 0xffffffffu;
  int lane = threadIdx.x % 32;
  for (int offset = 16; offset > 0; offset /= 2) {
    float other = __shfl_down_sync(FULL, value, offset);
    if (lane + offset < 32) value += other;
  }
  return value;  // 只有 lane 0 是完整和
}
// 若所有 lane 都需要完整和，在所有 32 lanes 上继续调用：
// value = __shfl_sync(0xffffffffu, value, 0);`)}${cudaIndexNote('lane=threadIdx.x%32；例如 thread 39 是 warp 1 的 lane 7，offset=16 时读取同 warp lane 23，即 thread 55 的寄存器值。不是读取 x[23]。offset 依次为 16、8、4、2、1；每轮取的是上一轮结束后的值。')}<div class="note"><b>FULL 是参与线程的位掩码，不是数组下标范围。</b>这里所有 CTA 都用完整 warp，尾块的无效输入贡献设为 0，线程仍参与 shuffle；不能在越界时直接 return。每次调用中，mask 指定的尚未退出线程必须按规则到达同一 intrinsic，读取来源也必须参与。不要把 FULL 原样用于任意分支或不完整 warp。<code>_sync</code> 不提供 shared/global memory 的内存屏障，也不做跨 warp 同步。<a href="https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html">CUDA Guide：Warp Shuffle Functions</a></div>`;
}
function cudaBlockSumCode() {
 return `<h3>跨 warp：shared memory 接力，再由 warp 0 归约</h3><p>shuffle 只在 warp 内工作。一个 256-thread CTA 有 8 个 warp，先分别得到 8 个 warp 和；各 warp 的 lane 0 写 shared memory，整个 CTA 同步后，warp 0 再读取这 8 个数。</p>${cudaCode(`// 固定 256 threads / CTA；全部线程调用，thread 0 使用返回值
__device__ float block_sum_256(float local) {
  __shared__ float warp_sums[8];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  float v = warp_sum(local);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();  // 8 个 warp 都写好，才能让 warp 0 读取
  if (warp == 0) {
    v = lane < 8 ? warp_sums[lane] : 0.0f;
    return warp_sum(v);  // warp 0 的 32 lanes 都调用
  }
  return 0;  // 其他 warp 的返回值不用
}`)}${cudaIndexNote('thread 0、32、64…224 分别把各自 warp 的和写 warp_sums[0…7]。同步后 thread 0…7 读取这 8 项；thread 8…31 取 0，但仍参加第二次 warp_sum。最终 CTA 的 thread 0 获得完整 CTA 和。每个 kernel 中此 helper 仅调用一次，不在未同步的循环中复用 shared 数组。')}`;
}
function cudaWelfordShuffleCode() {
 return `<h3>BN 的 warp_merge_all 也展开：交换三个字段，再合并</h3><p>下面把此前的教学辅助函数展开。<code>merge</code> 使用本节末尾的 Welford 公式，并处理 count=0 的空状态。它是两个状态的合并，不是把三个字段分别相加。</p>${cudaCode(`struct State { int count; float mean, M2; };
__device__ State warp_merge_lane0(State s) {
  int lane = threadIdx.x % 32;
  for (int d = 16; d > 0; d /= 2) {
    State other;
    other.count = __shfl_down_sync(0xffffffffu, s.count, d);
    other.mean  = __shfl_down_sync(0xffffffffu, s.mean, d);
    other.M2    = __shfl_down_sync(0xffffffffu, s.M2, d);
    if (lane + d < 32) s = merge(s, other);
  }
  return s;  // 只有 lane 0 是完整统计
}
__device__ State warp_merge_all(State s) {
  s = warp_merge_lane0(s);
  s.count = __shfl_sync(0xffffffffu, s.count, 0);
  s.mean  = __shfl_sync(0xffffffffu, s.mean, 0);
  s.M2    = __shfl_sync(0xffffffffu, s.M2, 0);
  return s;  // 所有 lanes 得到同一通道的完整统计
}`)}${cudaIndexNote('每个 warp 内，lane ID 决定从哪个 lane 读取 count／mean／M2。数据最初仍来自前述 offset_nchw(r,c,…)；shuffle 阶段只交换已加载并累积在寄存器中的状态。情况 1 归一化时每个 lane 都要用均值和方差，所以需要 __shfl_sync 广播。')}<h4>Kernel 2 的 block_merge：4 个 warp 通过 shared 接力</h4>${cudaCode(`// 固定 128 threads / CTA；所有线程调用，仅 thread 0 用结果
__device__ State block_merge(State s) {
  __shared__ State warp_states[4];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  s = warp_merge_lane0(s);
  if (lane == 0) warp_states[warp] = s;
  __syncthreads();
  if (warp == 0) {
    s = lane < 4 ? warp_states[lane] : empty_state();
    return warp_merge_lane0(s);
  }
  return empty_state();
}`)}${cudaIndexNote('thread 0、32、64、96 写 warp_states[0…3]；CTA 同步后，warp 0 的 lane 0…3 读这些状态，其他 lane 提供空状态，再做一次 Welford 合并。仅 thread 0 写最终 mean[c] / invstd[c]。')}`;
}

function cudaSumCode() {
  return `${cudaShuffleCode()}${cudaBlockSumCode()}<h3>关键 device 代码：线程累加 → CTA 归约 → 跨 CTA 汇总</h3><p>下面直接调用上面展开的 <code>block_sum_256</code>，不再用 CUB 隐藏细节。固定 256 threads / CTA；<code>ATOMIC</code> 是编译期参数。</p>${cudaCode(`template<bool ATOMIC>
__global__ void sum_tiles(const float* x, int N, float* dst) {
  int t = threadIdx.x;
  int base = blockIdx.x * 1024;
  float local = 0;
  for (int j = 0; j < 4; ++j) {
    int i = base + t + j * 256;
    if (i < N) local += x[i];
  }
  float total = block_sum_256(local);
  if (t == 0) {
    if constexpr (ATOMIC) atomicAdd(dst, total);
    else dst[blockIdx.x] = total;
  }
}`)}${cudaIndexNote('CTA b 读取 <code>[b×1024, min((b+1)×1024,N))</code>。thread t 读取该块的 t、t+256、t+512、t+768；同一轮 j 中，相邻线程读相邻地址。例：CTA 1 / thread 7 读取 x[1031]、x[1287]、x[1543]、x[1799]。尾块越界值不读，局部贡献保持 0。')}<div class="cu-compare"><article class="cu-route"><h4>A · 原子提交</h4>${cudaCode(`// Host：先初始化，再在同一 stream 执行
cudaMemsetAsync(out, 0, sizeof(float), stream);
sum_tiles<true><<<98, 256, 0, stream>>>(x, 100000, out);`)}<p>所有 CTA 的 thread 0 写同一个 <code>out[0]</code>，使用 atomicAdd 避免丢更新。</p></article><article class="cu-route"><h4>B · 写独立 partial，再启动 Kernel 2</h4>${cudaCode(`// Host：partial 分配 98 个 float
sum_tiles<false><<<98, 256, 0, stream>>>(x, 100000, partial);
sum_final<<<1, 256, 0, stream>>>(partial, 98, out);`)}<p>CTA b 的 thread 0 独占 <code>partial[b]</code>，不需要原子写。第二个 kernel 的读取对象是 partial。</p></article></div>${cudaCode(`__global__ void sum_final(const float* partial, int P, float* out) {
  int t = threadIdx.x;
  float local = 0;
  for (int p = t; p < P; p += blockDim.x) local += partial[p];
  float total = block_sum_256(local);
  if (t == 0) out[0] = total;
}`)}${cudaIndexNote('Kernel 2 只启动 1 个 CTA。thread t 读 partial[t]，再按 256 步长继续读；本例 P=98，所以 thread 0…97 各读一个值，98…255 不读、贡献 0，但仍参与 CTA 归约。最终仅 thread 0 写 out[0]。该单 CTA 收尾适用于本例的小 partial 数组；更大规模可继续分层。')}`;
}
function cudaBnAddressCode() {
 return `<h3>先固定读地址：以下 BN 示例使用 NCHW</h3><p>逻辑坐标用 (r,c)，其中 r 展平 NHW；指针偏移仍按真实布局计算。使用 64 位元素下标：</p>${cudaCode(`__device__ long long offset_nchw(long long r, int c, int C, int HW) {
  long long n = r / HW;
  long long hw = r % HW;
  return (n * C + c) * HW + hw;
}
// NHWC 的对应偏移则是 r * C + c。`)}${cudaIndexNote('先用 r/HW 找 batch 编号 n，再用 r%HW 找该图片中的空间位置。例：NCHW、C=64、HW=65536，r=4103、c=4 → offset=266247，读取 x[266247]；若 r 跨过 HW，公式会自动跳到下一个 batch 的同一通道。')}<p class="cu-caption">下列 BN 代码是突出索引与阶段的 device 伪代码，省略参数列表；归约辅助函数在后面展开。State=(count,mean,M2)；empty_state() 为零计数；update(s,x) 纳入一个样本；warp_merge_all(s) 合并 32 lanes 并把最终状态广播到每个 lane；block_merge(s) 合并 CTA 的全部状态，仅 thread 0 使用结果。合并数学见本节末尾，所有参与线程均需执行对应归约。</p>`;
}
function cudaBnSingleCode() {
 return `<h4>关键代码：每 warp 一个通道，CTA 内完成输出</h4>${cudaCode(`// 128 threads / CTA；本例 M=32，每 CTA 4 通道
int warp = threadIdx.x / 32;
int lane = threadIdx.x % 32;
int c = blockIdx.x * 4 + warp;
int r = lane;
State s = empty_state();
float value = 0;
if (c < C && r < M) {
  value = x[offset_nchw(r, c, C, HW)];
  s = update(s, value);
}
State all = warp_merge_all(s);  // 32 lanes 都调用
if (c < C && r < M) {
  float inv = rsqrtf(all.M2 / all.count + eps);
  y[offset_nchw(r, c, C, HW)] =
      gamma[c] * (value - all.mean) * inv + beta[c];
}`)}${cudaIndexNote('blockIdx.x 选一组 4 通道，warp ID 选组内通道，lane ID 选该通道的 NHW 位置。例：CTA 2 / thread 39 → warp=1、lane=7 → c=9、r=7；本例 HW=32，因此读 x[295]，归约后写 y[295]，使用 gamma[9] 和 beta[9]。输入 value 保留在线程寄存器中。')}`;
}
function cudaBnSplitCode(caseThree=false) {
 const cb=caseThree?1:4, tile=caseThree?65536:4096;
 return `<h4>关键代码：${caseThree?'同一读取方式，改成每 CTA 一个通道':'CTA 的 x 轴选通道组，y 轴选 NHW 分段'}</h4>${cudaCode(`// ${caseThree?'情况 3':'情况 2'}：CB=${cb}，TILE=${tile}；threads=32*CB
// grid=(ceil(C/CB), ceil(M/TILE))，S=ceil(M/TILE)
int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
int c = blockIdx.x * CB + warp;
int segment = blockIdx.y;
long long begin = (long long)segment * TILE;
long long end = min(begin + TILE, M);
State s = empty_state();
for (long long r = begin + lane; r < end; r += 32) {
  if (c < C) s = update(s, x[offset_nchw(r, c, C, HW)]);
}
State part = warp_merge_all(s);  // 同一通道的 32 lanes 合并
if (lane == 0 && c < C) partial[c * S + segment] = part;`)}${cudaIndexNote(caseThree?'CB=1，所以每 CTA 32 threads，warp=0、c=blockIdx.x。例：CTA (c=2, segment=5) / thread 7 → r 从 327687 开始，每次 +32，直到 r&lt;393216；本例 HW=16777216，首次读取 x[33882119]。lane 0 把该段结果写 partial[2×256+5]=partial[517]。':'CB=4。例：CTA (blockIdx.x=1, blockIdx.y=1) / thread 7 → c=4、lane=7、segment=1；r=4103、4135、4167…且 r&lt;8192。首次读 x[266247]；同 warp 的 lane 0 将合并结果写 partial[4×16+1]=partial[65]。不是把 HW 再拆成二维坐标。')}${caseThree?'<p class="cu-caption">每 CTA 仅用一个 warp 是突出索引的教学映射；每线程会循环读取多个位置，真实优化可让多个 warp 协同一个通道并测量收益。</p>':''}`;
}
function cudaBnFinishCode() {
 return `<h3>情况 2 / 3 共用：合并 partial，再逐元素输出</h3><h4>Kernel 2 · 每个 CTA 合并一个通道</h4>${cudaCode(`// grid=C，128 threads / CTA
int c = blockIdx.x, t = threadIdx.x;
State s = empty_state();
for (int segment = t; segment < S; segment += blockDim.x)
  s = merge(s, partial[c * S + segment]);
State all = block_merge(s);  // 全 CTA 参与
if (t == 0) {
  mean[c] = all.mean;
  invstd[c] = rsqrtf(all.M2 / all.count + eps);
}`)}${cudaIndexNote('blockIdx.x 就是通道 c；thread t 读该通道第 t、t+128…段的状态。C=64／128 时 S=16：只有前 16 个线程各读一份，其余以空状态参与归约。C=3 时 S=256：每线程读两份；例如 c=2、thread 7 读 partial[519] 和 partial[647]。thread 0 写 mean[c]、invstd[c]。')}<h4>Kernel 3 · 按 NCHW 物理连续下标分块输出</h4>${cudaCode(`// 256 threads / CTA，每 CTA 至多输出 1024 个元素
for (int j = 0; j < 4; ++j) {
  long long q = (long long)blockIdx.x * 1024 + threadIdx.x + j * 256;
  if (q < (long long)N * C * HW) {
    int c = (q / HW) % C;
    y[q] = gamma[c] * (x[q] - mean[c]) * invstd[c] + beta[c];
  }
}`)}${cudaIndexNote('q 是物理元素下标，不是逻辑 NHW 坐标 r；blockIdx.x 选连续 1024 元素，threadIdx.x 与 j 选块内位置。c=(q/HW)%C 决定读取哪份 mean／invstd／gamma／beta。例：C=64、HW=65536，CTA 256 / thread 7 / j=0 → q=262151、c=4，读 x[262151]、mean[4] 等，写 y[262151]。NCHW 下相邻线程读写相邻物理元素。')}<p>两个统计 kernel 和输出 kernel 在同一 stream 顺序启动。状态数组 <code>partial</code> 有 C×S 项；mean、invstd 各有 C 项。输出不读取未完成的局部统计。</p>`;
}

function renderCudaBasics() {
  $('intro').innerHTML=`<div class="cu-lesson"><div class="eyebrow">CUDA BASICS · 从任务拆分到跨 CTA 归约</div><h2>先回答两个问题：谁负责输出，谁来汇总？</h2><p class="cu-lead">沿着 <b>执行模型 → arange → sum → BatchNorm</b> 顺序看下去。独立输出可以直接分工；共享一个结果，就要设计归约与同步。最后再进入 Attention。</p><p class="cu-caption">本页动画展示数据归属、阶段和依赖；不是 GPU 周期模拟，也没有未经测量的速度排名。</p>
  <section class="cu-section"><div class="cu-kicker">01 / 执行模型</div><h2>CTA 是协作团队，SM 是执行硬件</h2><p>CPU 启动一个 <b>kernel</b>，创建一个 <b>grid</b>；grid 由多个 <b>CTA（Cooperative Thread Array，也称 thread block）</b>组成，每个 CTA 包含多个 thread。线程按 32 个一组构成 <b>warp</b>，以 <b>SIMT（Single Instruction, Multiple Threads）</b>方式执行：同一条指令，各线程处理自己的数据。</p><div class="cu-compare"><article class="cu-route"><h3>任务怎么组织</h3><div class="cu-hierarchy">Kernel launch → Grid → CTA → Warp → Thread</div><p>CTA 内可以通过 shared memory 交换数据，用 <code>__syncthreads()</code> 做 CTA 级屏障；参与线程必须正确到达屏障。warp 内协作也需要遵守参与 mask 与同步规则。</p></article><article class="cu-route"><h3>任务在哪执行</h3><div class="cu-hierarchy">GPU → 多个 SM → 驻留 CTA / warps</div><p>SM（Streaming Multiprocessor，流式多处理器）是执行硬件。一个 CTA 的线程在同一个 SM 上执行；一个 SM 可以容纳多个 CTA。CTA 数量可以远大于 SM 数量，按资源可用性分批执行，不能依赖 CTA 的先后顺序。</p></article></div>${cudaFrame('hardware')}<div class="cu-storage">${cudaChip('寄存器：线程局部状态','purple')}${cudaChip('Shared memory：CTA 内交换','green')}${cudaChip('Global memory：输入 / 输出 / 跨 CTA 状态','blue')}</div><div class="note"><b>SIMT 不等于所有线程永远执行同一条路径。</b>同一 warp 遇到分支时，会对相应 lane 使能/屏蔽；路径分歧可能降低效率。<code>__syncthreads()</code> 也不能同步整个 grid。本页先讲普通独立 CTA；cooperative launch 等特殊机制不在此展开。</div></section>
  <section class="cu-section"><div class="cu-kicker">02 / 独立输出</div><h2>arange(100000)：给每个输出找一个主人</h2><p>生成 <code>[0, 1, …, 99999]</code>。每个输出只依赖下标，线程之间没有数据依赖。选 256 threads / CTA，启动 ⌈100000 / 256⌉ = <b>391 CTAs</b>。</p>${cudaArangeCode()}${cudaFrame('arange')}<div class="note">每个线程写不同位置，所以没有写入竞争，也不需要归约。更大的数据也可以让线程循环处理多个位置；线程数无需等于元素数。</div></section>
  <section class="cu-section"><div class="cu-kicker">03 / 一个结果，多份贡献</div><h2>sum：先局部归约，再跨 CTA 汇总</h2><p>对 100000 个 1 求和。每个 CTA 负责至多 1024 个输入，用 256 个线程处理：线程 t 读取本块的 t、t+256、t+512、t+768，越界贡献取 0。总计 <b>98 CTAs：97 个完整块 + 1 个 672 元素尾块</b>。</p><div class="cu-race"><b>为什么不能直接 out += value？</b><p>假设 out=0，两个 CTA 的贡献都是 1024。两边同时读到 0，各写回 1024，结果可能是 1024，正确结果应是 2048。普通读—改—写会丢贡献。</p></div>${cudaFrame('sum')}${cudaSumCode()}<div class="cu-compare"><div><h3>原子加保护更新，不提供全局屏障</h3><p><code>atomicAdd(out, block_sum)</code> 保证这次同址累加不会丢失。先在 CTA 内归约，把原子请求从 100000 次降为 98 次。一次提交结束不代表其他 CTA 已完成；不能立即据此消费最终结果。</p><p>浮点加法顺序变化可能改变末位结果；本例使用全 1 输入便于核算，不代表任意输入都能精确相加。</p></div><div><h3>二次归约以中间结果换取较少竞争</h3><p>Kernel 1 写 partial，Kernel 2 汇总；若 partial 仍然很多，可以继续多级归约。额外 launch 和临时数组也有成本，应实测选择，不能只数 kernel 数量。</p><p>这里的 reduce-add 指加法归约；<code>atomicAdd</code> 是原子更新接口，不等同于 warp 归约指令。具体指令还受架构和数据类型限制。</p></div></div></section>
  <section class="cu-section"><div class="cu-kicker">04 / BatchNorm：形状改变拆分</div><h2>每个通道一份统计，每个元素一个输出</h2><p>只讨论<b>训练前向 BN</b>：每个通道沿 N、H、W 统计均值和方差，再归一化。推理通常直接使用已有统计。令 <b>M = NHW</b>，逻辑上把数据看作 M × C：</p><div class="cu-equations">μc = Σr x[r,c] / M<br>σ²c = Σr (x[r,c] − μc)² / M<br>y[r,c] = γc × (x[r,c] − μc) / √(σ²c + ε) + βc</div><div class="note"><b>HW 可以直接逻辑展平，不必按二维图像处理。</b><code>r = ((n × H) + h) × W + w</code>。切分的是 r 的区间；它可以跨图像的行边界甚至 batch 边界。通道独立，归约方向是 NHW。</div><div class="cu-compare"><article class="cu-route"><h3>NCHW：优先利用连续的空间位置</h3><p>地址 = (n × C + c) × HW + h × W + w。同一通道的一张图内部 HW 连续，跨 batch 时有步长；逻辑 NHW 展平并不保证物理上始终连续。</p></article><article class="cu-route"><h3>NHWC：优先利用连续的通道</h3><p>地址 = r × C + c。同一位置的 C 连续。一个 CTA 同时处理多个通道，往往更容易让相邻线程访问相邻地址；具体线程映射需结合布局设计。</p></article></div>
  ${cudaBnAddressCode()}<article class="cu-bn-case"><h3>情况 1 · C 很大，NHW 很小：只沿 C 拆</h3><p>C=4096，M=32。每 CTA 负责 4 个完整通道，通道轴本身已经提供 1024 张任务。<b>不用跨 CTA 归约，但仍要在 CTA 内完成统计。</b></p>${cudaFrame('bn1')}${cudaBnSingleCode()}</article>
  <article class="cu-bn-case"><h3>情况 2 · C=64 / 128：同时拆 C 和 NHW</h3><p>固定 H=W=256，M=65536。<b>这次明确沿两个轴拆：</b>通道分组，每组 4 通道；NHW 每 4096 个位置一段，每通道 16 段。只沿 C 拆会得到较少且较重的任务；二维任务网格增加并行工作，同时引入跨 CTA 统计合并。</p><div class="cu-controls" aria-label="通道数示例"><span>同样的拆分，比较通道数：</span><button data-cu-channels="64" aria-pressed="true">C = 64</button><button data-cu-channels="128" aria-pressed="false">C = 128</button></div>${cudaFrame('bn2')}${cudaBnSplitCode()}</article>
  <article class="cu-bn-case"><h3>情况 3 · C=3，HW 巨大：主要沿 NHW 拆</h3><p>H=W=4096。只沿 C 分配就只有 3 个 CTA，难以充分利用 GPU。每通道 NHW 分成 256 段，产生 <b>768 张统计任务</b>；汇总后，输出阶段继续沿元素切成大量任务。</p>${cudaFrame('bn3')}${cudaBnSplitCode(true)}</article>${cudaBnFinishCode()}${cudaWelfordShuffleCode()}
  <h3>BN 的局部状态怎么合并？</h3><p>采用可合并的 Welford 状态：每段保存 <code>(count, mean, M2)</code>，M2 是相对该段均值的平方偏差和。两段均值不同时，不能直接把局部方差相加。下面给出合并关系，零计数状态需单独跳过。这是前面 <code>merge</code> / 归约辅助函数使用的寄存器内数学；A、B 两份状态来自上述 ID 映射读取的 partial，本段不再用线程 ID 读取数组：</p><pre class="cu-code"><code>δ = meanB − meanA;  count = countA + countB
mean = meanA + δ × countB / count
M2 = M2A + M2B + δ² × countA × countB / count
variance = M2 / count</code></pre><p>本页归一化使用总体方差；框架的 running variance 更新约定需另行对齐。三个字段是关联状态，不能分别 atomicAdd 就当作完成 Welford 合并。本页用二次归约显式表达依赖。</p><div class="note"><b>优化要同时看并行度与访存。</b>能留住输入时，可减少重读；切成更多 CTA 能增加并行任务，也要多写局部状态、合并和启动 kernel。这里是可解释的拆分起点，不是所有 GPU / 布局下的最优配置。</div></section>
  <section class="cu-section"><h2>带着这三个问题进入 Attention</h2><div class="cu-takeaways"><div><b>1 · 输出归谁？</b><p>一个 CTA 负责哪些行、哪些 tile？</p></div><div><b>2 · 局部状态放哪里？</b><p>线程寄存器、CTA shared，还是 global？</p></div><div><b>3 · 拆开后怎么合？</b><p>直接写、原子累加，还是另起 kernel？</p></div></div><p>Attention 也有归约，但 softmax 引入共同分母；后面会看到局部结果为什么不能直接相加。</p><p><a href="#attention">进入第 1 章：Forward 原理 →</a></p><p class="cu-caption">参考：<a href="https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html">CUDA Programming Guide · Programming Model</a> · <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html">Writing SIMT Kernels</a> · <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html">NVIDIA · Memory-Limited Layers / BatchNorm</a>。形状示例与动画拆分为本教程设计。</p></section></div>`;
  $('controls').innerHTML=''; $('scene').innerHTML=''; $('status').textContent=''; $('detail').innerHTML='';
  Object.keys(cudaSteps).forEach(cudaUpdate);
  document.querySelectorAll('[data-cu-channels]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.cuChannels)===cudaChannels)));
}
document.addEventListener('click',e=>{
  const b=e.target.closest('[data-cu-next],[data-cu-reset],[data-cu-play],[data-cu-channels]'); if(!b)return;
  if(b.dataset.cuChannels){cudaStop();cudaChannels=Number(b.dataset.cuChannels);document.querySelectorAll('[data-cu-channels]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cudaUpdate('bn2');return;}
  const id=b.dataset.cuNext||b.dataset.cuReset||b.dataset.cuPlay;
  if(b.dataset.cuPlay){
    if(cudaPlaying===id){cudaStop();return;}
    cudaStop(); if(cudaSteps[id]===cudaLimits[id])cudaSteps[id]=0;
    cudaPlaying=id;b.textContent='Ⅱ 暂停';b.setAttribute('aria-pressed','true');cudaUpdate(id);
    cudaTimer=setInterval(()=>{cudaSteps[id]++;cudaUpdate(id);if(cudaSteps[id]===cudaLimits[id])cudaStop();},1400);return;
  }
  cudaStop();cudaSteps[id]=b.dataset.cuReset?0:Math.min(cudaLimits[id],cudaSteps[id]+1);cudaUpdate(id);
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)cudaStop();});
