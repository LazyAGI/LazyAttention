'use strict';
// Pure models below are teaching examples, not GPU hardware counters.
const optCards=[
 ['保留数学，改变工作组织','GEN/UND 双流重排后，同一样本的 token 在物理内存中可能分成多段。Attention 的可见连接没有改变，但连续 range 变碎，tile 跨样本边界，任务长度也更不均。优化目标是在完全相同的 mask、Q/K/V 和梯度语义下，少做地址计算、重复读取和跨任务归约，让长尾更短。'],
 ['沿用 FA4 的片上计算基础','分块重算 P，避免把完整 S/P 写回 HBM；Tensor Core / WGMMA 做矩阵乘法；shared memory、TMA / cp.async 与 producer/consumer pipeline 组织搬运。Full 块跳过逐元素 mask，Empty 块不进入循环。这些是已有实现的基础能力，不记成我们新增的独立收益。'],
 ['逻辑连续遍历，物理按 range 访问','计算在样本内的逻辑行序上进行，Q、K、V 仍留在双流物理布局。每个 Q64 行块保存一段 base、两段 base＋split，或真正离散时的索引 payload。逻辑行 r 通过 base+r 或两段分支得到物理行；64 行连续块无需逐行读取 64 个 index。省去额外整张 Q/K/V 重排与对应梯度逆重排，且 range 路径不保留完整 row-index 向量。'],
 ['压缩mask与Full / Partial分流','逻辑分块让相同样本的可见区域更容易用规则tile表达；具体Full/Partial比例仍取决于mask。128×128的Partial tile用512个32-bit字保存，即2KiB位图；每位对应一个精确可见连接。Full通过稀疏列表表达，不读取逐元素位图；Empty没有任务迭代。lookup先定位tile位图，再提取所需bit，避免常驻完整S×S稠密布尔mask。'],
 ['成对加载，共享寻址','Q 与 dO 使用同一个物理行映射，K 与 V 也成对发起搬运；连续段走 TMA，细段按 16/4 行检查，边界或离散小段回退 cp.async gather。同一行位置只解码一次后供两个张量使用，减少重复 index load、整数地址计算和分支。V 共享寻址已在这条成对加载路径中使用，并非待实验项；K/V 的数据仍是两份，不能省去 V 本身的数据读取。'],
 ['异步搬运与预取：覆盖延迟','producer 提前发出可异步执行的 TMA / cp.async，consumer 等对应 barrier 后计算。双缓冲让后续 Q/dO tile 的搬运有机会覆盖当前 tile 的矩阵计算；TMA descriptor 也预取。KV tile 在任务内驻留并复用。目标是减少暴露的等待，不是让加载耗时归零；barrier 仍负责可见性与缓冲区生命周期。流水基础来自 FA4，range 路径接入同一依赖协议。'],
 ['一个 CTA 聚合 4 个 GQA query heads','固定一个 KV tile 和 KV head，CTA 内处理共享该 KV head 的 4 个 query heads。K/V 只加载一次，dK/dV 在任务内跨 Q/head 累积，降低独立 head 任务带来的重复搬运及写回归约压力。Q/dO 的数值仍按 head 各自加载；跨 KV 任务的 dQ 累加仍存在。Q 外层、head 内层还允许复用当前行 descriptor、Q tile 编号及 mask 相关元数据。'],
 ['预规划：让重任务先被领取','按不可变 mask 的 Partial＋Full Q tile 数估算每个 KV 任务工作量；以逻辑样本分组，用组内最大任务的 64-tile 工作量档位降序，档位相同保持稳定顺序。组内沿 head/KV 组织任务，不移动激活数据。persistent CTA 从同一队列取工作，避免最重样本留到队尾拖住收尾。它是粗工作量估计，不是精确 FLOPs 排序，也不是把样本内 token 倒过来。'],
 ['Q 优先＋完整列表倒序','原始遍历是 [Partial 列表，Full 列表]。完整倒序为 [Full 倒序，Partial 倒序]；提前反转各类 Q 索引，编译时选择 Full-first，Full/Partial 标记不变。它不是把 Q tile 编号全局重新排序。producer、MMA、dQ store 必须遍历一致，避免队列和写回错位。可能改变相邻 CTA 的读写重叠与缓存/原子冲突形态；当前没有硬件计数证据把收益唯一归因于某一种 cache 或原子等待。'],
 ['专用pre / post与直接梯度写回','沿用FA4的D_i=dot(O_i,dO_i)预处理与局部P重算。主核将dK/dV按行descriptor直接写向物理位置；跨KV任务累积的FP32 dQ由Hopper专用postprocess统一缩放、转BF16并scatter到物理输出。复用同一行映射，避免再构造完整逻辑梯度后另做整张逆重排。pre/post仍是需要计时的GPU工作，不宣称把这部分全部消除。'],
 ['条件解码与任务领取时机','连续、一拆二与离散回退互斥：只有需要payload时才读取index/第二段信息，不能为少一个分支而在热路径无条件多读元数据。persistent scheduler在必要rendezvous之后领取后续任务，保持shared与写回的生命周期安全；这一延迟领取已属于保留基线。CTA常驻数、pipeline级数与寄存器占用互相影响，尚未完成独立常驻CTA数量扫描。'],
 ['一次准备，42 次复用','行 descriptor、倒序 Q 列表与任务表只依赖形状、mask、文档和布局，可在 42 层复用。成本记为 T准备/42＋T完整backward。CPU 侧批量构造稀疏元数据，仅对实际有效的 Q 索引列做排列，padding 原样保留；减少 Python 小操作、无效排序和 CPU/GPU 往返。早期独立准备实验中，S4096 causal的批量稀疏计划构造，C布局约快24%、F布局约快76%；这是准备阶段收益，不能当成backward快24%/76%。两项规划开关全关，或规划器发现没有需要改变的任务时，返回 None，kernel 编译路径不读任务表。']
];

function optHistoricalDetails(){return `<section id="opt-section-2"><h3>做完之后提升多少？把直接实测和估算分开</h3><div class="opt-table-wrap"><table><thead><tr><th>比较</th><th>结果</th><th>能说明什么</th></tr></thead><tbody><tr><td>MAGI → 最新组合，240例直接配对</td><td><b>15.470 → 13.153 ms<br>−14.98%，1.176×</b></td><td>2026-11-04 同一H100、单进程单轮；完整梯度链＋额外准备/42；240例均提升&gt;1%。</td></tr><tr><td>不规划 → 仅重排，240×4状态</td><td>14.9056 → 13.5146 ms<br>−9.33%</td><td>已是range路径；这里只隔离重排/切分策略，不是原生未修改FA4对照。</td></tr><tr><td>Q优先正序 → 完整倒序，240×8状态</td><td>13.5383 → 13.3711 ms<br>−1.235%</td><td>已开启重排；冻结当轮准备实现。之后裁剪无用padding列，准备均值2.118 → 1.225 ms，按/42摊销。</td></tr><tr><td>成对reader整理，历史240例</td><td>总耗时约−0.17%</td><td>测的是已具备完整range能力版本上的reader整理；不能把它解释为“全部range优化只有0.17%”。</td></tr><tr><td>重排＋倒序相对未规划range基线</td><td>粗估约−10%～−11%</td><td>乘法组合不同轮次的单项效果，仅用于量级判断；不是直接端到端测量，更不能把百分比相加。</td></tr></tbody></table></div><p>当前MAGI对照的GPU链均值为15.4448 vs 13.1231 ms，一次额外准备均值1.0595 vs 1.2410 ms；摊销后得到上表15.470 vs 13.153 ms。公共capture/mask稀疏计划准备沿用旧口径，冷编译、forward、验证和Python发射/分配不计入。最大梯度绝对误差0.015625、相对L2约0.00124。</p><div class="note"><b>整网只能做预算：</b>若该backward链占训练step关键路径的比例为f，粗估step降幅≈f×14.98%；例如f=20%时约3%，f=40%时约6%。这是假设其余部分与重叠关系不变的示例预算，尚未做整网验收。</div><h4>这些开销到底有没有被消除？</h4><ul><li><b>CUDA launch：</b>一张逻辑任务不对应一次kernel launch，MAGI/FA4本身也会persistent领取任务。我们的重排主要减少CTA收尾，不把每张任务的“虚构launch开销”算作收益。队列复位、preprocess、主核和postprocess等实际kernel仍计时。</li><li><b>HBM访问与地址依赖：</b>range描述减少逐行index读取；成对加载共享地址解码；4个query heads复用KV。Q/K/V数值本身仍需读取，无法把所有访存变成零。</li><li><b>归约与同步：</b>任务内合并GQA head的dK/dV累积，减少任务间贡献；dQ仍跨KV任务累加，barrier仍保证正确性。没有把所有原子或同步删掉。</li><li><b>准备与运行：</b>排序、Q排列和策略选择移到一次准备；执行读取紧凑结果。返回None时移除任务表读取，但保留所需的稀疏Q列表与行descriptor。</li></ul></section><section id="opt-section-3"><h3>附录：试过但没有默认采用的方向</h3><p>首轮无收益先检查路径是否生效、循环是否一致、是否引入多余分配或读取。这里描述当前实现与输入范围的结论，不把它们概括成算法永远无效。</p><div class="opt-table-wrap"><table><thead><tr><th>方向 / 为什么尝试</th><th>做了什么、看到了什么</th><th>当前选择</th></tr></thead><tbody><tr><td>局部8-tile交错<br>希望错开不同CTA读写同一Q区域，缓解热点</td><td>每类Q列表按8个tile分组，以KV tile编号决定组内起点；尾组按实际长度处理，4个heads共享排列，准备结果复用42次。最优Q倒序上GPU略快，但当轮总耗时反增约0.13%；修正padding准备后组合估算额外收益仅约0.02%。</td><td><b>关</b>。没有足够净收益支持增加默认复杂度；开关保留。</td></tr><tr><td>Head外层优先<br>希望连续处理同一head有更好局部性</td><td>交换producer/MMA/store循环并验证覆盖一致、KV不重复加载、dK/dV不被清零。相对Q优先正序基准，Head优先正序慢约2.47%，Head优先倒序慢约4.08%；也失去Q tile元数据跨heads复用机会。未用profiler分离所有原因。</td><td><b>Q外层</b>。方向与head/Q有交互，不能独立叠加收益。</td></tr><tr><td>长任务切分<br>希望降低单个超长CTA任务的尾部</td><td>规划中一次切段，每段上限128个Q64 tile，42次复用；部分dK/dV写FP32 scratch再归约。240例单独切分慢30.30%，切分＋重排仅快8.84%，弱于仅重排9.33%。已检查覆盖、归约和规划成本，主要增加在GPU执行。</td><td><b>默认关</b>。保留手动实验与小形状低占用路线，不能用本批高占用结果否定所有切分。</td></tr><tr><td>复杂shape/mask动态选型<br>希望在每例选择不同遍历或切分</td><td>检查max/mean/p95 Q tile、Full占比、CV等特征。按rank留出，规划动态规则额外收益至多约0.02%；遍历规则约0.11%。事后挑最快组合还会挑中噪声。</td><td>采用统一默认，暂不增加复杂阈值或shape白名单。</td></tr><tr><td>Full块拆专用kernel<br>希望去除Partial mask分支，采用更激进的Full专用流水</td><td>主核已经有Full fast path。再拆kernel会新增launch、公共输入加载和梯度合并成本；需要证明Full占比与省下的指令足以覆盖它们。尚未完成独立完整对照。</td><td><b>待验证</b>，不能写成“测了没提升”。</td></tr></tbody></table></div></section><section><h3>附录补充：更早的独立实验与待重试项</h3><div class="opt-table-wrap"><table><thead><tr><th>方向</th><th>为什么尝试 / 做过的检查</th><th>结论</th></tr></thead><tbody><tr><td>调用方复用scratch</td><td>希望减少CUDA allocator与Python分配开销。已移除重复分配并检查跨stream/shape、生命周期和Graph；必要清零保留。S4096完整eager链约−0.044%，区间跨零，主核约占GPU时间96%。</td><td>没有稳定净收益，未纳入默认；元数据复用本身已存在，不重复记成新开关。</td></tr><tr><td>Stats预处理等价layout</td><td>希望消除reduction到store的warp layout转换。普通Triton、2D masked store后再试显式Gluon布局，做到32寄存器、0 shared、无spill。完整链小例快1.67%，S4096约+0.13%，S8192 block-causal慢1.21%。</td><td>未全局替换；需要与主核statistics读取布局、tile/warp参数一起选型。去掉shared交换不等于整体变快。</td></tr><tr><td>dK/dV写回共享地址</td><td>希望同一不连续行只解码一次供两种梯度写回；保持128-bit写、barrier与TMA路径，参考/Graph和codegen检查通过。离散例约快1.01%，mixed约快0.19%，two-range约慢0.11%。</td><td>输入相关，未默认采用。它区别于已采用的K/V读取共享寻址，不能把两者混为同一项。</td></tr><tr><td>range路径指令/代码体积整理</td><td>尝试减少重复寻址分支与冷路径膨胀。有版本静态指令少约8%，但完整链没有稳定净收益，部分例回退；检查到了部分barrier阶段新增local访问。</td><td>未作为性能替代。保留后来已通过全240例验证的成对reader，静态指令减少本身不能证明加速。</td></tr><tr><td>更早的地址格式与循环组合</td><td>hot64/较宽descriptor、affine/hoisted Stats、rolled/unrolled head循环，以及KV TMA/barrier组合曾有探索；部分记录早于稳定对照协议。旧44-field run-cache也不是当前紧凑descriptor ABI。</td><td>保留为分组重试假设，不能据旧计时设默认；常驻CTA数×领取时机×pipeline深度应组合研究。尚未做完整搜索，不伪装成“全部测过无收益”。</td></tr></tbody></table></div><p>独立实验的范围、各次实现检查与准备阶段数据见<a href="../../experiments/independent-operators-20261103.md">早期独立优化记录</a>。每项保留自己的基准与计时边界，不把它们累加成最新15%的解释性拆账。</p></section><section><h3>复现入口与默认开关</h3><pre class="opt-code">prepared = prepare_attention(mask, index, device=q.device,
    documents=logical_document_ids, kv_heads=k.shape[2])
# 默认：reorder=True, split=False, head_major=False,
#       reverse=True, interleave=0
out, lse = forward(q, k, v, prepared)
dq, dk, dv = backward(q, k, v, out, dout, lse, prepared)</pre><p>documents按逻辑行顺序提供；未提供时，调度以每个KV128 tile为独立分组。提前提供kv_heads可以在CUDA Graph capture前完成规划；否则首次eager backward缓存一次，不会在42次执行中重算。改变mask、行映射或文档归属时必须重新准备。</p><p><a href="optimization-evidence.json">实验摘要与口径</a> · <a href="../../benchmarks.md">benchmark复现说明</a> · <a href="https://github.com/LazyAGI/LazyAttention/tree/dev/lazy_attention">实现源码（dev）</a></p></section>`;}

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
 return groups.map((g,i)=>({name:['一段连续','两段连续','index 回退'][i],header:[32,-32,-64][i],groups:g,
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
 const c=optContribution(),saved=c.magi-c.final;
 return `<section id="opt-summary"><div class="ov-hero"><div><span class="ov-kicker">完整240例 · 同一张H100 · 一轮配对</span><h3>先看总收益：相同 backward，少花 14.98% 时间</h3><div class="ov-times"><span><small>MAGI</small>15.470 <em>ms</em></span><b>→</b><span><small>LazyAttention</small>13.153 <em>ms</em></span></div><p>平均每次节省 <b>2.317 ms</b>，加速 <b>1.176×</b>；240/240例提升超过1%，梯度检查通过。</p></div><div class="ov-gain">−14.98%<small>直接实测的总耗时变化</small></div></div><p class="ov-scope">统一口径：额外准备时间 / 42 ＋完整GPU backward，包含所需清零、preprocess、主核、postprocess及BF16梯度输出。公共capture/mask计划准备、forward、冷编译与Python发射/分配沿用原口径排除。</p><h3>把这 14.98 个百分点展开</h3><p>下面所有百分点都以 <b>MAGI 的15.470 ms</b> 为分母，因此可以相加。历史实验没有完整隔离每一项，采用“比例换算＋剩余项”的<b>归因估算</b>，不把它写成新的消融实测。</p><p class="ov-scope">色条仅展开已节省的2.317 ms；三种颜色共同组成这14.98个百分点。</p><div class="ov-savings" aria-label="节省时间的归因估算">${c.parts.map(x=>`<div style="flex:${x.ms};background:${x.color}" title="${x.name}"></div>`).join('')}</div><div class="ov-ledger">${c.parts.map(x=>`<article><i style="background:${x.color}"></i><small>${x.work}</small><h4>${x.name}</h4><strong>${(x.ms/c.magi*100).toFixed(2)} <em>个百分点</em></strong><span>约 ${x.ms.toFixed(3)} ms · ${x.evidence}</span></article>`).join('')}</div><div class="ov-sum">5.06 ＋ 8.86 ＋ 1.06 ≈ <b>14.98 个百分点</b><span>对应总共节省约 ${saved.toFixed(3)} ms</span></div><p class="ov-scope">青色部分共同覆盖逻辑range、成对预取、GQA任务聚合以及其他基础实现差异；这几项还没有独立消融，不能各记一次5.06。下面四项工作会明确标出它属于哪个贡献桶。</p><details class="ov-method"><summary>展开计算依据与估算边界</summary><p>重排历史比例 r = 13.5146 / 14.9056；倒序历史比例 v = 13.3711 / 13.5383。用最新终点13.152645 ms反推：基础阶段估值 = 13.152645 / (r×v) ≈ 14.687787 ms；重排后估值 ≈13.317113 ms。三个差额分别除以MAGI基准，得到5.06、8.86、1.06个百分点。</p><p>这是把不同轮次的比例迁移到同一基准的近似账本；顺序和交互都会影响归因，剩余项也包含迁移误差。总收益14.98%是直接实测，三个分项并非同一轮的独立消融结论。特别是L2命中率没有实测计数，不能把倒序的全部贡献指定为L2收益。</p></details></section>`;
}
function optScheduleDraw(){
 let s='';const step=optState.schedule;
 for(let side=0;side<2;side++){const x=side?520:20,{rows,end}=optSchedule(Boolean(side));s+=T(x,28,side?'规划后：重任务先领取':'原顺序：重任务落在队尾',20)+T(x,52,`收尾位置 ${end} · 相同总工作量21（教学单位）`,12,'dim');
  for(let lane=0;lane<3;lane++){s+=T(x,102+lane*65,`CTA ${lane}`,12);for(let t=0;t<=10;t++)s+=`<path d="M${x+60+t*39} 77v168" stroke="#e3eaf2"/>`;}
  rows.forEach(j=>{const active=j.i===step;s+=R(x+60+j.start*39,80+j.lane*65,j.n*39-3,40,active?C.blue:C.light)+T(x+65+j.start*39,106+j.lane*65,`T${j.id}·${j.n}`,12,active?'white':'');});
 }s+=T(20,290,'仅重排任务ID，不移动Q/K/V；先做重活，减少其他CTA做完后等待长任务的尾部。',15);
 $('opt-board-schedule').innerHTML=optSvg(s,315,'schedule');
 $('opt-note-schedule').textContent='教学模型：同样21单位工作，3个CTA，收尾位置从10变为8；不代表实测毫秒。高亮仅标记第'+(step+1)+'个领取的任务。';
}
function optMappingDraw(){
 let s='';const step=optState.mapping;
 optMappingCases().forEach((model,i)=>{const y=i*290,logicalX=174,logicalW=87,physicalX=174,physicalW=45;
  s+=T(20,y+25,`${i+1}. ${model.name}`,20)+T(20,y+50,model.encoding,13,'dim')+T(20,y+100,'逻辑行块',14)+T(20,y+120,'每格8行',12,'dim')+T(20,y+205,'物理内存',14)+T(20,y+225,'按地址递增',12,'dim');
  for(let r=0;r<8;r++){const x=logicalX+r*logicalW,hot=r===step;s+=R(x,y+75,logicalW-6,43,hot?C.blue:'#e6d9fa')+T(x+9,y+93,`L${r}`,12,hot?'white':'')+T(x+9,y+109,`${r*8}–${r*8+7}`,11,hot?'white':'');
   const px=physicalX+model.groups[r]*physicalW+physicalW/2,lx=x+(logicalW-6)/2;
   s+=`<path d="M${lx} ${y+119} C${lx} ${y+142},${px} ${y+152},${px} ${y+182}" fill="none" stroke="${hot?'#1765c1':'#aab6c8'}" stroke-width="${hot?3:1}" opacity="${hot?1:.45}" marker-end="url(#ov-arrow-mapping)"/>`;
  }
  for(let g=0;g<16;g++){const hot=g===model.groups[step],used=model.groups.includes(g);s+=R(physicalX+g*physicalW,y+188,physicalW-3,40,hot?C.blue:used?C.light:'#eef1f5')+T(physicalX+g*physicalW+6,y+204,`P${g}`,11,hot?'white':'')+T(physicalX+g*physicalW+4,y+219,`${g*8}–${g*8+7}`,9,hot?'white':'');}
  const r=step*8,formula=i===0?`physical(${r}) = base + ${r} = ${model.physical[r]}`:i===1?(r<32?`r=${r}<32：base0+r = ${model.physical[r]}`:`r=${r}≥32：base1+(r−32) = ${model.physical[r]}`):`读 payload[${r}] = ${model.physical[r]}，再定位该物理行`;
  s+=T(174,y+257,formula,15)+T(174,y+279,i===0?'访存：一段可直接TMA搬运；没有逐行index依赖。':i===1?'访存：两段分别寻址；16/4行子段可走TMA，跨界小段回退。':'访存：先读64项index payload；探测子段连续性，真正离散的小段gather。',12,'dim');
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
 s+=T(320,207,'正序从不同的边界开始：Q0 / Q2 / Q4。',15)+T(320,237,'倒序都从共同尾部开始：Q7 / Q7 / Q7。',15)+T(320,271,'若领取时间接近，Q/dO更容易在被逐出前被相邻CTA复用。',14)+T(320,301,'这个例子中，Full倒序→Partial倒序恰好沿尾部→边界。',13,'dim');
 const step=optState.cache;
 if(step){
  const f=optCacheFrames(false,optCacheCapacity)[step].event,r=optCacheFrames(true,optCacheCapacity)[step].event;
  s+=`<rect x="${x0+f.kv*cell+2}" y="${y0+f.q*cell+2}" width="${cell-6}" height="${cell-6}" fill="none" stroke="#1765c1" stroke-width="3"/><circle cx="${x0+r.kv*cell+12}" cy="${y0+r.q*cell+12}" r="9" fill="none" stroke="#7554a4" stroke-width="3"/>`;
 }
 s+=T(320,329,'蓝框：正序当前请求；紫圈：倒序当前请求。',12,'dim');
 for(let side=0;side<2;side++){const x=side?520:20,frame=optCacheFrames(Boolean(side),optCacheCapacity)[step],trace=optCacheTrace(Boolean(side));s+=T(x,369,side?'完整倒序：先共同尾部':'正序：各自从边界向后',20);
  trace.forEach((e,i)=>{const xx=x+(i%9)*50,yy=390+Math.floor(i/9)*53,hot=i===step-1,done=i<step;s+=R(xx,yy,46,45,hot?C.blue:done?C.light:'#eef1f5')+T(xx+6,yy+17,`${'ABC'[e.cta]}`,10,hot?'white':'')+T(xx+6,yy+35,`Q${e.q}`,12,hot?'white':'');});
  s+=T(x,526,`LRU缓存（旧 → 新），容量 ${optCacheCapacity} 个Q tile`,13);
  for(let i=0;i<optCacheCapacity;i++){const value=frame.cache[i];s+=R(x+i*54,543,49,35,value===undefined?'#eef1f5':C.light)+T(x+i*54+9,566,value===undefined?'空':`Q${value}`,12);}
  s+=T(x,611,`命中 ${frame.hits} / 请求 ${step} · ${step?(frame.hits/step*100).toFixed(1):'0.0'}%`,17);
  s+=T(x,639,frame.event?`当前 ${'ABC'[frame.event.cta]} 读 Q${frame.event.q}：${frame.event.hit?'命中，复用已在缓存中的数据':'未命中，加入缓存'}`:'尚未发出读取',13,'dim');
 }
 $('opt-board-cache').innerHTML=optSvg(s,665,'cache');
 const af=optCacheFrames(false,optCacheCapacity)[step],df=optCacheFrames(true,optCacheCapacity)[step];
 $('opt-note-cache').textContent=`LRU教学模型：容量${optCacheCapacity}个Q tile，已读${step}/18次；正序命中${af.hits}次，倒序命中${df.hits}次。两边访问集合相同，改变的是访问顺序；这不是硬件计数器结果。`;
}
function optPrefetchDraw(){
 const step=optState.prefetch;let s=T(20,27,'先合并地址工作，再成对发出预取',22)+T(20,52,'合并的是行遍历/地址解码和发起组织；Q与dO仍是两份数据、两条异步搬运请求。',14,'dim');
 s+=T(20,90,'分开寻址（概念对照）',18)+T(520,90,'成对reader（采用）',18);
 for(let lane=0;lane<2;lane++){const y=111+lane*72;s+=R(20,y,170,48,'#e6d9fa')+T(31,y+29,`解码 ${lane?'dO':'Q'} 的行地址`,13)+T(201,y+29,'→',20)+R(236,y,180,48,C.light)+T(249,y+29,`发起 ${lane?'dO':'Q'} 搬运`,14);}
 s+=R(520,142,153,55,step===0?C.blue:'#e6d9fa')+T(533,164,'同一个logical row',12,step===0?'white':'')+T(533,184,'只解码一次 physical',12,step===0?'white':'');
 for(let lane=0;lane<2;lane++){const y=111+lane*72;s+=`<path d="M674 169L718 ${y+24}" stroke="#1765c1" stroke-width="2" fill="none"/>`+R(720,y,122,48,step===1?C.blue:C.light)+T(730,y+29,`${lane?'dO':'Q'}：TMA/cp.async`,11,step===1?'white':'')+T(850,y+29,'→',18)+R(880,y,97,48,step===2?'#7554a4':'#e6d9fa')+T(891,y+29,`barrier ${lane?'dO':'Q'}`,12,step===2?'white':'');}
 s+=T(520,274,'两个buffer都就绪后，consumer才使用它们；不删除同步。',13);
 s+=T(20,323,'再把下一对的预取与当前tile的计算重叠',21);
 const labels=['同一对Q/dO解码','成对发起Q、dO','等待各自barrier','计算A，同时预取B','计算B，同时预取C'];
 s+=T(20,353,`当前：${labels[step]}`,14,'dim');
 for(let i=0;i<3;i++){const y=386+i*62;s+=T(20,y+26,`tile ${'ABC'[i]}`,14);const loadX=145+i*180,computeX=loadX+110,loading=(step===1&&i===0)||(step===3&&i===1)||(step===4&&i===2),computing=(step===3&&i===0)||(step===4&&i===1);s+=R(loadX,y,100,39,loading?C.blue:C.light)+T(loadX+9,y+25,'预取 Q + dO',12,loading?'white':'')+R(computeX,y,185,39,computing?'#7554a4':'#e6d9fa')+T(computeX+12,y+25,`计算 tile ${'ABC'[i]} · 4 heads`,12,computing?'white':'');}
 s+=T(20,603,'保持两条数据流，减少重复地址依赖；Q/dO搬运字节数不变。长度只画依赖关系，不是实测周期。',14);
 $('opt-board-prefetch').innerHTML=optSvg(s,630,'prefetch');
 $('opt-note-prefetch').textContent='阶段'+(step+1)+'：'+labels[step]+'。Q/dO共用行寻址，但仍有两条搬运和独立buffer/barrier；计算等待两路所需数据就绪。';
}
function optWorkHTML(id,number,title,contribution,body,extra=''){
 return `<section class="ov-work" id="opt-work-${id}"><div class="ov-work-heading"><span class="ov-work-number">${number}</span><div><h3>${title}</h3><span class="ov-contribution">${contribution}</span></div></div>${body}${extra}${optControls(id)}<div id="opt-board-${id}"></div><p class="ov-scope" id="opt-note-${id}" aria-live="polite"></p></section>`;
}
function optUpdate(id){
 const max=optExtent[id],host=$(`opt-board-${id}`),scrollLeft=host?.querySelector('.board')?.scrollLeft||0;
 optState[id]=Math.min(max,Math.max(0,optState[id]));
 ({schedule:optScheduleDraw,mapping:optMappingDraw,cache:optCacheDraw,prefetch:optPrefetchDraw}[id])();
 const board=host?.querySelector('.board');if(board)board.scrollLeft=scrollLeft;
 const slider=document.querySelector(`[data-opt-slider="${id}"]`);if(slider)slider.value=optState[id];
 const label=$(`opt-label-${id}`);if(label)label.textContent=id==='cache'?`已读 ${optState[id]}/18`:`步骤 ${optState[id]+1}/${max+1}`;
}

function renderOptimizations(){
 $('intro').innerHTML='<h2>8 · LazyAttention：总收益与四项优化工作</h2><p>同一份mask、同样的Attention与梯度，改变寻址、任务组织和数据复用。默认：<b>重排开、切分关、CTA内Q优先＋完整倒序、不交错。</b></p>';
 $('controls').innerHTML='';$('status').textContent='';
 $('scene').innerHTML=optOverview()+`<nav class="ov-toc" aria-label="四项优化工作">${[['schedule','① 工作量重排'],['mapping','② 逻辑连续与range'],['cache','③ 多head、Q优先与倒序'],['prefetch','④ 合并预取']].map(([id,label])=>`<button data-opt-jump="opt-work-${id}">${label} ↓</button>`).join('')}</nav>`+
 optWorkHTML('schedule','01','样本工作量降序：减少队尾长任务','归因估算：8.86个百分点',`<p>长短任务混排时，轻任务先做完，重任务可能独占队尾。预规划仅重排任务ID：按mask估计Q tile工作量，在样本组内取最大64-tile档位，按档位降序稳定排列。<b>准备一次，42次执行复用。</b></p><div class="ov-explain"><b>省在哪里</b><span>persistent CTA取任务的机制仍然保留；改变队列让重活更早启动，减少其他CTA空闲等待的收尾时间。没有假设“每个任务启动一个CUDA kernel”。</span></div>`)+
 optWorkHTML('mapping','02','逻辑连续与range：地址规律用描述符表达','共同计入基础实现5.06个百分点，尚未独立拆分',`<p>一个逻辑Q64块始终包含连续的64个逻辑行；它在物理张量中可能是一段、两段，或多段离散。<b>上排是计算顺序，下排是按地址排列的物理内存，箭头表示每组8行实际落在哪里。</b>高亮会同时移动，而张量本身不复制。</p><div class="ov-explain"><b>怎样标识</b><span>两组header分别保存类型/起点与payload指针：header≥0是一段base；−63…−1是两段的−split；−64是64项index payload。读取同一descriptor，统一用于Q/dO、K/V和梯度scatter。</span></div><p class="ov-scope">P格中的编号是物理8行分组，不是物理page。位图mask、Full/Partial稀疏列表另行保存，descriptor只描述行映射。</p>`)+
 optWorkHTML('cache','03','一个CTA多head、Q优先，再从共同尾部倒序','倒序归因估算1.06个百分点；基础GQA复用归入5.06',`<div class="ov-head-reuse"><div>固定一个 KV tile / KV head<br><b>K / V 加载一次，在CTA内驻留</b></div><span>→</span><div>Q7：h0 → h1 → h2 → h3<br>Q6：h0 → h1 → h2 → h3<br>…</div></div><p><b>Q优先</b>让同一Q tile的行/mask元数据连续服务4个head；四份Q/dO数值各自读取。<b>倒序</b>则改变相邻KV任务访问Q/dO的时间关系：在因果mask中，多个KV列都需要Q尾部，倒序先从共同尾部读起，缩短跨CTA复用距离。</p><div class="ov-explain"><b>为什么可能命中L2</b><span>CTA A刚把Q7/dO7载入L2，CTA B/C紧接着也读Q7，就有机会复用。正序起点不同，等另一个CTA读到相同Q时，数据可能已被其他tile逐出。shared memory是CTA私有；这里讨论的是跨CTA的L2。</span></div><div class="ov-cache-setting"><span>LRU教学模型容量：</span>${[3,4,8].map(n=>`<button data-opt-capacity="${n}" aria-pressed="${optCacheCapacity===n}">${n} 个Q tile</button>`).join('')}</div><p class="ov-scope">下图用8×8等宽教学格（不按真实Q64/KV128比例），固定3个CTA按轮次交错发请求，每个Q tile抽象成一个缓存项；不是H100 cacheline或实测命中率。容量足够时正倒序同样命中；实际收益还取决于CTA到达时间、mask形状和缓存压力。当前只实测了执行时间改善，没有L2计数器证据。</p>`)+
 optWorkHTML('prefetch','04','合并预取：共享寻址、成对发起，再覆盖搬运等待','共同计入基础实现5.06个百分点，尚未独立拆分',`<p>Q与dO使用同一物理行映射；K与V也是一对。把原本可重复进行的行遍历/descriptor解码合在一起，得到physical row后，<b>成对发出两条TMA或cp.async搬运</b>。连续段与16/4行子段用TMA，真正离散边界回退gather。</p><div class="ov-explain"><b>“合并”的含义</b><span>一次行寻址 → Q和dO两路预取；不是把两个tensor拼接，也不是把两笔显存交易变成一笔。Q/dO各自buffer/barrier保留，两路就绪后才能计算。随后预取下一对，与当前tile的MMA重叠。</span></div><p class="ov-scope">TMA/cp.async与producer/consumer流水基础继承FA4；我们将range/index混合访问接入成对reader。此处展示组织原理，未声称原生MAGI/FA4只有串行搬运，也未把已有流水再计算一份独立收益。</p>`);
 $('detail').innerHTML=`<details class="ov-appendix"><summary>更多实现细节：mask压缩、pre/post、条件解码与准备复用</summary><div class="opt-card-grid">${optCards.map(([title,text])=>`<article class="opt-card"><h4>${title}</h4><p>${text}</p></article>`).join('')}</div></details><details class="ov-appendix"><summary>实验附录：原始计时、未采用方案与复现入口</summary>${optHistoricalDetails()}</details>`;
 for(const id of Object.keys(optState)){
  document.querySelector(`[data-opt-play="${id}"]`).onclick=()=>{if(optTimers[id]){optStopOne(id);return;}if(optState[id]===optExtent[id])optState[id]=0;optUpdate(id);optTimers[id]=setInterval(()=>{optState[id]++;optUpdate(id);if(optState[id]===optExtent[id])optStopOne(id);},750);document.querySelector(`[data-opt-play="${id}"]`).textContent='Ⅱ 暂停';};
  document.querySelector(`[data-opt-next="${id}"]`).onclick=()=>{optStopOne(id);optState[id]++;optUpdate(id);};
  document.querySelector(`[data-opt-reset="${id}"]`).onclick=()=>{optStopOne(id);optState[id]=0;optUpdate(id);};
  document.querySelector(`[data-opt-slider="${id}"]`).oninput=e=>{optStopOne(id);optState[id]=Number(e.target.value);optUpdate(id);};optUpdate(id);
 }
 document.querySelectorAll('[data-opt-capacity]').forEach(b=>b.onclick=()=>{optCacheCapacity=Number(b.dataset.optCapacity);document.querySelectorAll('[data-opt-capacity]').forEach(x=>x.setAttribute('aria-pressed',String(Number(x.dataset.optCapacity)===optCacheCapacity)));optUpdate('cache');});
 document.querySelectorAll('[data-opt-jump]').forEach(b=>b.onclick=()=>document.getElementById(b.dataset.optJump).scrollIntoView({block:'start'}));
}
