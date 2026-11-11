'use strict';
// U1-style image tokens share the temporal coordinate; spatial h/w differ.
let u1Mode='paired',u1Pair=[9,12];
function u1Example(){
 const segments=[['提示',0,false,false,0],['参考图',3,true,false,1],['回答',4,false,false,0],['生成图',7,true,true,2]];
 if(u1Mode==='paired')segments.push(['干净副本UND',7,true,false,3]);
 segments.push(['结束',8,false,false,0]);
 const tokens=segments.flatMap(([label,pos,image,gen,mod],group)=>Array.from({length:3},(_,i)=>({label:label+(i+1),pos:image?pos:pos+i,mod,image,gen,doc:0,dup:false,group,h:0,w:image?i:0})));
 return {tokens,segments};
}
function u1Allowed(q,k){const sameDoc=q.doc===k.doc,sameMod=sameDoc&&q.mod===k.mod;
 return (sameDoc&&q.pos>=k.pos||q.image&&k.image&&sameMod)&&(!k.gen||q.gen&&sameMod)&&(!q.gen||k.gen||q.pos!==k.pos)&&!k.dup;
}
function u1Reason(q,k){if(q.doc!==k.doc)return '不同样本，禁止访问。';if(k.dup)return '重复边界 K 被排除。';if(k.gen&&!(q.gen&&q.mod===k.mod))return '生成图像 K 仅对同图的生成 Q 可见；后面的结束文字也不能读取生成图。';if(q.gen&&!k.gen&&q.pos===k.pos)return '生成 Q 不能读取相同 token_pos 的非生成 K，避免同位置信息泄漏。';if(q.image&&k.image&&q.mod===k.mod)return '同一图像段内部双向可见；GEN 自身不执行同位置排除。';return q.pos>=k.pos?'同样本，K 的逻辑位置不晚于 Q，因果可见。':'K 位于未来，且不满足同图双向规则，禁止访问。';}
function u1BranchCompute(){
 const box=(x,y,w,h,label,fill='#eef4fb')=>R(x,y,w,h,fill)+`<text x="${x+w/2}" y="${y+h/2+5}" text-anchor="middle" font-size="15">${label}</text>`;
 const arrow=(x,y)=>T(x,y,'↓',23);
 let g=T(35,28,'重排前：同分支被拆成多段',21)+T(575,28,'重排后：同分支整段计算',21);
 for(let i=0;i<4;i++){g+=box(35+i*108,55,100,42,['U₁','G₁','U₂','G₂'][i],i%2?C.q:'#dcebe1');g+=box(575+i*108,55,100,42,['G₁','G₂','U₁','U₂'][i],i<2?C.q:'#dcebe1');}
 g+=arrow(235,126)+arrow(775,126);
 for(let i=0;i<4;i++)g+=box(35+i*108,145,100,62,['U₁ × Wqkvᵁ','G₁ × Wqkvᴳ','U₂ × Wqkvᵁ','G₂ × Wqkvᴳ'][i],i%2?C.q:'#dcebe1');
 g+=box(575,145,208,62,'[G₁; G₂] × Wqkvᴳ',C.q)+box(791,145,208,62,'[U₁; U₂] × Wqkvᵁ','#dcebe1');
 g+=T(35,235,'多个小 GEMM，或先 gather 再计算',15)+T(575,235,'两条连续分支 → 较大的 GEMM',15);
 g+=arrow(235,272)+arrow(775,272);
 g+=box(35,290,424,66,'QKᵀ → 每行 masked softmax → PV',C.light)+box(575,290,424,66,'QKᵀ → 每行 masked softmax → PV',C.light);
 g+=T(35,382,'每个 query/head 独立归一化，样本之间隔离',14)+T(575,382,'同一 query 的所有可见 K 共同归一化',14);
 g+=arrow(235,419)+arrow(775,419);
 for(let i=0;i<4;i++)g+=box(35+i*108,438,100,56,['U₁：Wo/FFN','G₁：Wo/FFN','U₂：Wo/FFN','G₂：Wo/FFN'][i],i%2?C.q:'#dcebe1');
 g+=box(575,438,208,56,'GEN 整段：Woᴳ → FFNᴳ',C.q)+box(791,438,208,56,'UND 整段：Woᵁ → FFNᵁ','#dcebe1');
 g+=T(35,525,'每层处理分散小段 / 抽取与写回',15)+T(575,525,'后续层保持双流布局，减少重复路由',15);
 return `<section class="u1-compute"><h3>重排的收益：同一套权重，一次处理更多 token</h3><p>G / U 表示已有样本中的 GEN / UND 段。紫色 GEN，绿色 UND；两分支各用自己的权重。</p><div class="board"><svg viewBox="0 0 1040 550" role="img" aria-label="重排前后QKV投影Attention和Wo及FFN计算对比">${g}</svg></div><div class="note"><b>收益：</b>Wq/Wk/Wv、Wo、FFN 从分散小段变为连续大矩阵，减少小算子与反复搬运；计算量基本不变。<br><b>Softmax：</b>仍逐 query/head 计算，不合并不同 query/head 的分母；同一 GEN query 可见的 GEN 与 UND keys 则共同参与一个分母。<br><b>两种合并别混淆：</b>行合并是拼 token；Wqkv 打包是将 Q/K/V 投影拼在输出通道，两种布局都可使用 QKV 打包。</div></section>`;
}
function u1ArchitecturePrimer(){
 const block=(x,y,w,h,title,detail,fill)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${fill}" stroke="#9caec5"/><text x="${x+w/2}" y="${y+27}" text-anchor="middle" font-size="18" font-weight="600">${title}</text><text x="${x+w/2}" y="${y+53}" text-anchor="middle" font-size="14">${detail}</text>`;
 const arrow=(x,y)=>`<text x="${x}" y="${y}" text-anchor="middle" font-size="25">↓</text>`;
 let diagram=block(35,20,430,76,'UND · 理解分支','提示文字 / 参考图理解 token / 回答文字','#dcebe1')+block(575,20,430,76,'GEN · 图像生成分支','待生成图片的带噪视觉 token','#e6d9fa');
 diagram+=arrow(250,128)+arrow(790,128);
 diagram+=`<rect x="15" y="142" width="1010" height="535" rx="14" fill="#f6f8fc" stroke="#9caec5" stroke-dasharray="6 4"/><text x="40" y="174" font-size="18" font-weight="600">第 ℓ 层 · ℓ = 1 … 42（每层都有下面两套参数）</text>`;
 diagram+=block(35,195,430,76,'UND RMSNorm','独立权重 γUND⁽ℓ⁾','#dcebe1')+block(575,195,430,76,'GEN RMSNorm','独立权重 γGEN⁽ℓ⁾','#e6d9fa')+arrow(250,302)+arrow(790,302);
 diagram+=block(35,315,430,76,'UND：Wq⁽ℓ⁾、Wk⁽ℓ⁾、Wv⁽ℓ⁾','得到 QUND、KUND、VUND','#dcebe1')+block(575,315,430,76,'GEN：Wq⁽ℓ⁾、Wk⁽ℓ⁾、Wv⁽ℓ⁾','得到 QGEN、KGEN、VGEN','#e6d9fa');
 diagram+=`<path d="M250 391 V418 H520 V444 M790 391 V418 H520" fill="none" stroke="#61718a" stroke-width="2"/>`;
 diagram+=block(180,445,680,76,'Attention：按 mask 允许的关系读取 K / V','GEN 可读取允许的 UND K/V；UND 不读取 GEN K/V','#cce5ff');
 diagram+=arrow(520,575)+block(180,590,680,76,'各自的 Wo → 残差 → RMSNorm → MLP → 残差','仍按 UND / GEN 分支计算，然后进入下一层','#eef4fb');
 return `<section class="u1-architecture" aria-labelledby="u1-architecture-title"><div class="eyebrow">SENSENOVA U1 · 3 分钟入门</div><h2 id="u1-architecture-title">先认识 UND / GEN：两套参数，一起做 Attention</h2><p>以 <b>SenseNova-U1-8B-MoT 的 42 层 Transformer 主干</b>为例：UND 是 Understanding（理解）分支，GEN 是 Generation（图像生成）分支。把主干想成一栋 42 层的楼，<b>每层都有 UND 和 GEN 两个工作区</b>，token 按类型进入对应工作区。</p><div class="u1-token-story" aria-label="示例序列的分支归属"><span class="u1-und">提示文字<small>UND · 理解要求</small></span><span class="u1-und">参考图理解 token<small>UND · 提供图像信息</small></span><span class="u1-und">回答文字<small>UND · 回答或推理</small></span><span class="u1-gen">生成图带噪 token<small>GEN · 训练输入为随机时间步的加噪图</small></span></div><p><b>生成文字也属于 UND。</b>这里 GEN 专指图像生成分支；分支由 token 类型决定，不能仅凭任务名称判断。</p><div class="board u1-architecture-board"><svg viewBox="0 0 1040 685" role="img" aria-labelledby="u1-diagram-title u1-diagram-desc"><title id="u1-diagram-title">42 层 Transformer 中 UND 与 GEN 的独立参数和注意力交互</title><desc id="u1-diagram-desc">每层两分支分别经过自己的 RMSNorm 和 QKV 投影，再根据 mask 参与 Attention，随后使用各自的输出投影、归一化及 MLP。不同层也使用不同权重。图中省略位置编码和生成侧时间步调制。</desc>${diagram}</svg></div><div class="note"><b>每层独立，分支也独立：</b>第 1 层 UND 的 Wq 与第 1 层 GEN 的 Wq 是不同参数，也与第 42 层 UND 的 Wq 不同。Wk、Wv 和 RMSNorm 权重同理；并不是 42 层共用两套权重。</div><p>RMSNorm 先按均方根调整数值尺度，再乘可学习的缩放权重 γ。图中只展开 Attention 前的一处；实际 Attention 前、MLP 前及 Q/K 的 RMSNorm 都有对应的 UND / GEN 权重。Wo 和 MLP 也分别保留分支参数。图为结构简图，省略位置编码、生成侧时间步调制等细节。</p><h3>两套权重，怎样一起工作？</h3><p><b>Q 是“我想找什么”，K 是“我有什么可供匹配”，V 是“实际传递的信息”。</b>GEN 想参考输入图片时，可以用自己的 QGEN 匹配 UND 的 KUND，再读取 VUND；前提是 mask 允许这次访问。</p><div class="formula">示例：score = QGEN · KUNDᵀ / √d + mask bias
QGEN 来自 GEN 的 Wq；KUND / VUND 来自 UND 的 Wk / Wv。</div><div class="note"><b>分支决定“用哪套权重”；mask 决定“能读取谁”。</b>下面先看这条序列的可见关系，再看保持这些关系不变的 GEN / UND 重排。</div><p class="muted"><small>参考：<a href="https://github.com/OpenSenseNova/SenseNova-U1" target="_blank" rel="noopener">SenseNova-U1 官方实现</a> · <a href="https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/parameter_breakdown.md" target="_blank" rel="noopener">官方参数划分</a>。本节针对 8B-MoT 的 42 层主干。</small></p></section>`;
}

function u1Primer(){const {tokens:ts,segments}=u1Example(),u=24,x=110,y=85;let s=T(30,24,'逐 token mask · Q ↓ / K →',18);
 segments.forEach(([label],g)=>{const a=x+g*3*u,b=y+g*3*u;const fill=segments[g][3]?C.q:segments[g][2]?'#ffdfb0':'#dcebe1';s+=R(a,y-24,3*u,20,fill)+`<text x="${a+36}" y="${y-10}" text-anchor="middle" font-size="10">${label}</text>`+T(5,b+36,label,11);});
 ts.forEach((q,i)=>ts.forEach((k,j)=>{const hit=u1Pair[0]===i&&u1Pair[1]===j,hole=q.gen&&!k.gen&&q.image&&k.image&&q.pos===k.pos;let cell=R(x+j*u,y+i*u,u,u,u1Allowed(q,k)?C.light:hole?'#fff':C.off,hit?'stroke-width="2" style="stroke:#7554a4"':'');if(hole)cell+=T(x+j*u+7,y+i*u+17,'×',14);s+=`<g role="button" tabindex="0" data-u1-pair="${i},${j}" aria-label="${q.label} 读取 ${k.label}">${cell}<title>${q.label}(pos=${q.pos},h=${q.h},w=${q.w}) → ${k.label}(pos=${k.pos},h=${k.h},w=${k.w})：${u1Reason(q,k)}</title></g>`;}));
 segments.forEach((_,g)=>{s+=`<rect x="${x+g*3*u}" y="${y}" width="${3*u}" height="${ts.length*u}" fill="none" stroke="#9caec5"/><rect x="${x}" y="${y+g*3*u}" width="${ts.length*u}" height="${3*u}" fill="none" stroke="#9caec5"/>`;});
 const Q=ts[u1Pair[0]]||ts[0],K=ts[u1Pair[1]]||ts[0],size=x+ts.length*u+12;
 return `<section><h2>先看一条序列：文字 causal、同图 Full、目标副本整块隔离</h2><p>同一条样本包含提示文字、参考图片、回答文字和生成图片。<b>每段展开为 3 个 token</b>，粗线分隔语义段，细格表示一个 Q/K token pair。文字段内部现在能直接看到含对角线的 causal 下三角；同图内部是 Full 方块。</p><div class="controls"><button data-u1-mode="independent" aria-pressed="${u1Mode==='independent'}">参考图 → 生成新图</button><button data-u1-mode="paired" aria-pressed="${u1Mode==='paired'}">交错生成：加入干净目标副本</button></div><p>${u1Mode==='paired'?'当前模拟交错生成：GEN 在前，干净目标副本 UND 在后。它们是不同图像段（mod=2 / 3），但全部图像 token 共享 token_pos=7。因果比较允许相等位置，随后相同位置 gate 将 GEN → 对应副本的整个 3×3 块屏蔽。后续文字仍可读取干净副本。':'参考图 mod=1、token_pos=3；生成图 mod=2、token_pos=7。GEN 可读取先前参考图，同一 GEN 图内部是 Full。此模式省略干净目标副本。'}</p><div class="mask-zoom"><div class="board u1-board"><svg viewBox="0 0 ${size} ${y+ts.length*u+20}" role="img" aria-label="文字因果同图全可见与对应干净副本整块屏蔽">${s}</svg></div><div><h3>U1 的规则如何叠加</h3><ol><li>同 doc 的文字按 token_pos 因果访问：段内下三角，前面的完整文字段可以整块可见。</li><li>同图像内部双向可见。不同图片之间仍需检查因果关系与 GEN gate，不能一律画 Full。</li><li>GEN K 只允许同图 GEN Q 读取；文字、UND 图片和不同生成图不能读取它。</li><li><b>GEN Q → 非 GEN K：屏蔽相同 token_pos。</b>同一图片段的 token_pos 恒定，对应副本复用该位置，因此白格 × 覆盖整个配对块；GEN → GEN 自身不扣。</li><li>dup_boundary 标记的 K 对所有 Q 不可见。</li></ol><p><b>点击格子：${Q.label}(pos=${Q.pos}) → ${K.label}(pos=${K.pos}) · ${u1Allowed(Q,K)?'允许':'禁止'}</b><br>${u1Reason(Q,K)}</p><p>浅蓝 = 可见；灰色 = 禁止；白色 × = 同位置排除。这里是整个配对块被排除，并非逐空间 patch 的对角线。</p></div></div><details><summary>精确规则、位置与数据含义</summary><div class="formula" data-typeset="true">same_image = same_doc ∧ imageQ ∧ imageK ∧ (modQ = modK)\nbase = (same_doc ∧ posQ ≥ posK) ∨ same_image\nM = base ∧ (¬genK ∨ (genQ ∧ same_mod_doc))\n    ∧ (¬genQ ∨ genK ∨ posQ ≠ posK) ∧ ¬dupK</div><p><b>token_pos 是序列 / temporal 位置，不是二维 patch 编号，也不是扩散时间步。</b>图像 token 的 h/w 空间坐标不同，token_pos 则在同一图内恒定；mask 的相等判断只比较 token_pos，不比较 h/w 或 token ID。切成更多 patch 会增加 token 数与 h/w 网格，不会让图内 token_pos 逐 patch 递增。逻辑位置随 token 重排，不能用物理行号替代。这里省略了边界符；dup_boundary 只标记指定边界，不是整个干净副本。</p><p>副本属于 UND，但普通输入参考图也属于 UND；只有由目标图复制出来的那份才标记 duplicated_for_und。副本用于后续理解 / 续写，自己的图像占位不计算文本 CE，也不承担 GEN 的图像生成 loss。普通文生图 / 编辑不必插入副本。下方 D3 的递增位置是人为的算子边界测试，不代表 U1 图片；D5 才展示恒定位置的整块排除。</p><p>公开实现依据：<a href="https://github.com/OpenSenseNova/SenseNova-U1/blob/9b92332171a037dce2cbec83cc29482e45e61217/training/sensenovavl/data/multimodal_dataset.py">交错图像副本与监督</a>、<a href="https://github.com/OpenSenseNova/SenseNova-U1/blob/9b92332171a037dce2cbec83cc29482e45e61217/training/sensenovavl/model/sensenovavl_moe_chat/modeling_sensenovavl_chat_mot.py">mask 与稳定分区</a>。<a href="audit.md">本轮审计范围与复现说明</a>。</p><p>源码与独立校验：<a href="build_dummy.py">dummy 生成与校验</a>、<a href="build_mask_data.py">真实 metadata 校验</a>。</p></details><h2>再看已有多序列：为什么做 GEN / UND 稳定分区</h2><p>按 gen 标记稳定分区：<b>所有 GEN 在前，所有 UND 在后</b>，各分支内部保留原始相对顺序。排序键是分支，不是图片/文字，也不是按样本长度排序。输入图片和文字都留在 UND。</p>${u1BranchCompute()}<p><b>重排不改依赖：</b>Q/K/V 与 metadata 同步移动；mask 两轴同步置换，可见 pair 不变。</p><div class="note">下面点击语义块跟踪重排，点击 mask 查看依赖，选择 range 查看覆盖区域。</div></section>`;
}
const u1OriginalRender=renderMaskLesson;
renderMaskLesson=function(){u1OriginalRender();$('intro').insertAdjacentHTML('afterbegin',u1ArchitecturePrimer()+u1Primer());};
document.addEventListener('click',e=>{const el=e.target.closest('[data-u1-pair],[data-u1-mode]');if(!el)return;if(el.dataset.u1Mode){u1Mode=el.dataset.u1Mode;u1Pair=u1Mode==='paired'?[9,12]:[3,4];}else u1Pair=el.dataset.u1Pair.split(',').map(Number);render();});
document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.matches('[data-u1-pair]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
