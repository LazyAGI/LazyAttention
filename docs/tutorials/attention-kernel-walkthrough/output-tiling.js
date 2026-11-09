'use strict';
let productMode='matmul', productTile=0, productStep=0, productTimer=null, productPlaying=null;
function productStop(){clearInterval(productTimer);productTimer=null;productPlaying=null;}
function productGrid(x,y,rows,cols,label,highlight,select=false){let s=T(x,y-18,label,18);for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){const fill=highlight(i,j);let cell=R(x+j*24,y+i*24,24,24,fill);if(select){const id=productMode==='matmul'?Math.floor(i/2)*3+Math.floor(j/2):i;cell=`<g role="button" tabindex="0" data-product-tile="${id}" data-product-owner="${productMode}" aria-label="选择输出 ${productMode==='matmul'?'2×2 块':'行'} ${id}">${cell}</g>`;}s+=cell;}return s;}
function renderProduct(){
 const mm=productMode==='matmul',r=mm?Math.floor(productTile/3)*2:productTile,c=mm?(productTile%3)*2:0,k=productStep*2;
 $('intro').innerHTML='<h2>先选输出 O，再沿收缩维度累加</h2><p>从普通矩阵乘法到 Attention：观察一块输出需要哪些输入，以及中间结果能复用在哪里。点击 O 选择输出块，再播放计算。</p>';
 $('controls').innerHTML=`<button data-product-control="reset">↺ 重播</button><button data-product-control="step">下一步 →</button><button data-product-control="play">${productTimer&&productPlaying===productMode?'Ⅱ 暂停':'▶ 播放'}</button>`;
 let s=productGrid(35,85,8,4,'Q · 8 × 4',(i,j)=>i>=r&&i<r+(mm?2:1)?(mm&&j>=k&&j<k+2?C.blue:C.q):C.off);
 s+=T(148,180,'×',25);
 s+=productGrid(195,85,4,6,mm?'K · 4 × 6':'Kᵀ · 4 × 6',(i,j)=>mm?(j>=c&&j<c+2?(i>=k&&i<k+2?C.blue:C.light):C.off):(j>=k&&j<k+2?C.blue:C.light));
 if(mm){s+=T(370,180,'→',25)+productGrid(430,85,8,6,'O · 8 × 6 · 点击 2×2 块',(i,j)=>i>=r&&i<r+2&&j>=c&&j<c+2?C.q:C.off,true);}
 else{s+=T(355,180,'→',25)+productGrid(395,85,8,6,'S = QKᵀ · 8 × 6',(i,j)=>i===r?(j>=k&&j<k+2?C.blue:C.light):C.off)+T(550,180,'×',25)+productGrid(595,85,6,4,'V · 6 × 4',(i)=>i>=k&&i<k+2?C.blue:C.light)+T(710,180,'→',25)+productGrid(755,85,8,4,'O · 8 × 4 · 点击行',(i)=>i===r?C.q:C.off,true);}
 s+=T(35,320,mm?`当前累加：O[${r}:${r+2}, ${c}:${c+2}] += Q[${r}:${r+2}, ${k}:${k+2}] × K[${k}:${k+2}, ${c}:${c+2}]`:`当前 key 块 ${k}:${k+2}：Q 的第 ${r} 行 × Kᵀ 的对应列，再乘 V 的对应行`,17);
 $('scene').innerHTML=svg(s,350);
 $('status').textContent=`第 ${productStep+1} / ${mm?2:3} 个收缩块 · 深蓝是当前读取，浅蓝是完整依赖，紫色是所选输出及 Q。`;
 $('detail').innerHTML=mm?'<div class="formula">O[I,J] = Σₜ Q[I,Tₜ] K[Tₜ,J]</div><p>8×4 乘 4×6 得到 8×6。O 可切成 12 个 2×2 小块；一块只需要 Q 的两行、K 的两列，沿长度 4 的共同维度累加，无需计算整行 O。</p>':'<div class="formula">O[I,:] = Σₜ (Q[I,:] Kᵀ[:,Tₜ]) V[Tₜ,:]</div><p>先固定 O 的一行（实际可以是一组行），它对应 Q 的同一行。沿 6 个 key 分块遍历：每块产生短 S，再立刻乘对应 V，累加到同一个 O 行。S 的这段权重供该行所有输出通道复用，因此常把 O 横向整行放在一个任务里。</p><p>纯矩阵链仍可按输出列切块，也可改成 Q(KᵀV)。这里选择 (QKᵀ)V，为理解 Attention 做准备；加入 mask 和 softmax 后，不能把它换成 Q(KᵀV)。按 Q 分块是输出行分块在输入上的对应关系，并非数学上强制只能这样切。</p><div class="note">Attention：P = softmax(QKᵀ / √d + B)，O = PV。归一化覆盖这一行所有可见 key；融合实现以在线 softmax 累积最大值、分母和输出分子，不将整行 S/P 写回显存。继续向下看 Attention 逐块动画。</div>';
}
const productStates={matmul:{tile:0,step:0},chain:{tile:0,step:0}};
function productSave(){if(productStates[productMode])productStates[productMode]={tile:productTile,step:productStep};}
function productLoad(mode){productMode=mode;productTile=productStates[mode].tile;productStep=productStates[mode].step;}
function forwardCapture(){return {controls:$('controls').innerHTML,scene:$('scene').innerHTML,status:$('status').textContent,detail:$('detail').innerHTML};}
function renderForwardFlat(){
 const sections=[];
 for(const mode of ['matmul','chain']){productLoad(mode);renderProduct();const h=forwardCapture();sections.push(`<section data-product-section="${mode}"><h2>${mode==='matmul'?'① Q × K：二维切 O':'② QKᵀV：按行切 O'}</h2><div class="controls">${h.controls}</div>${h.scene}<p aria-live="polite">${h.status}</p>${h.detail}${mode==='matmul'?productMemoryLesson():''}</section>`);}
 productMode='attention';basicAttention();const h=forwardCapture();sections.push(`<section><h2>③ Attention：在线归一化</h2><div class="controls">${h.controls}</div>${h.scene}<p aria-live="polite">${h.status}</p>${h.detail}</section>`);
 $('intro').innerHTML='<h2>从一块 O 出发：矩阵乘 → 矩阵链 → Attention</h2><p>依次向下看，三个动画各自选择输出并播放。</p>';$('controls').innerHTML='';$('scene').innerHTML=sections.join('');$('status').textContent='';$('detail').innerHTML='';
}
function productAdvance(mode){productLoad(mode);const max=mode==='matmul'?1:2;if(productStep===max){productStop();render();return;}productStep++;productSave();render();}
document.addEventListener('click',e=>{const el=e.target.closest('[data-product-tile],[data-product-control]');if(!el)return;const mode=el.dataset.productOwner||el.closest('[data-product-section]').dataset.productSection;productLoad(mode);const d=el.dataset;if(d.productTile!==undefined){productStop();productTile=+d.productTile;productStep=0;}if(d.productControl==='reset'){productStop();productStep=0;}if(d.productControl==='step'){productStop();productSave();productAdvance(mode);return;}if(d.productControl==='play'){if(productTimer)productStop();else{stop();productPlaying=mode;productTimer=setInterval(()=>productAdvance(mode),900);}}productSave();render();});
document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.matches('[data-product-tile]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)productStop();});
function productMemoryLesson(){
 let reuse=T(25,28,'分块不减少乘加次数，减少重复读取',21);
 const box=(x,y,w,label,fill)=>R(x,y,w,48,fill)+T(x+12,y+29,label,16);
 reuse+=box(25,60,235,'显存：Q / K 的当前 tile',C.off)+T(280,91,'→',25)+box(325,60,260,'warp 各线程读取相邻地址',C.light)+T(605,91,'→',25)+box(650,60,350,'CTA shared memory：共同保留',C.q);
 reuse+=T(665,145,'↓ 同步就绪后，多次复用',16)+box(650,165,350,'线程 / Tensor Core → 累积 O',C.light);
 return `<section><h3>为什么矩阵乘要分块？</h3>${svg(reuse,240)}<div class="cols"><p><b>① shared memory 复用：</b>CTA 协作加载 Q/K tile，在片上重复使用。BM×BN 输出块的一轮 BK 收缩，只需加载 BM×BK + BK×BN 个输入元素；每个 Q 元素服务 BN 列，每个 K 元素服务 BM 行。</p><p><b>② 合并访存：</b>让同一个 warp 的线程读取相邻且适当对齐的显存地址，用较少事务搬入整块数据。合并发生在 warp 的一次访存指令，不是将整个 CTA 的任意访问自动合并。分块本身不保证合并，线程到地址的映射才决定它。</p></div>${bankExplorer()}</section>`;
}
