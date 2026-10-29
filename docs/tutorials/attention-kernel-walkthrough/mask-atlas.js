'use strict';
function atlasSvg(ranges,runs,S,label){
 const mx=x=>24+x/S*276,my=x=>24+x/S*276;
 const mask=ranges.flatMap(rangeBand).map(b=>`<polygon points="${bandPoly(b,mx,my)}" fill="#9fcdf7"/>`).join('');
 const strip=runs.map(([a,z,g])=>`<rect x="${mx(a)}" y="307" width="${Math.max(.5,mx(z)-mx(a))}" height="10" fill="${g?'#9573c7':'#77aa91'}"><title>${g?'GEN':'UND'} [${a}, ${z})</title></rect>`).join('');
 return `<div><b>${label}</b><svg viewBox="0 0 324 330" role="img" aria-label="${label} mask"><rect x="24" y="24" width="276" height="276" fill="#eef1f5"/>${mask}${strip}<text x="24" y="326" font-size="10">紫 GEN · 绿 UND</text></svg></div>`;
}
function atlasCard(c,i){
 const m=c.metrics,gen=c.samples.reduce((n,s)=>n+s.gen,0),tag=i===0?'<span class="viz-badge">最复杂案例</span>':'';
 return `<article class="atlas-case" data-atlas-case="${c.rank}-${c.batch}"><header><div><h3>${i+1}. rank ${c.rank} / batch ${c.batch}</h3>${tag}</div><p>S=${c.S.toLocaleString()} · ${c.samples.length} samples · GEN ${(100*gen/c.S).toFixed(1)}% · 可见密度 ${(100*m.density).toFixed(1)}%</p></header><div class="atlas-pair">${atlasSvg(c.ranges_before,c.runs_before,c.S,'重排序前')}${atlasSvg(c.ranges_after,c.runs_after,c.S,'重排序后')}</div><p class="atlas-meta">ranges ${m.ranges_before} → ${m.ranges_after} · Full ${m.full_tiles.toLocaleString()} · Partial ${m.partial_tiles.toLocaleString()} · max Q tiles ${m.qtiles_max} · 跨样本 KV ${m.cross_sample_kv}</p></article>`;
}
function renderMaskAtlas(){
 document.querySelector('header>small').innerHTML='算子优化 workload 快照<br>240 组结构化筛选 · 展示 24 组<br>重排序前后使用相同 attention 规则';
 $('intro').innerHTML='<h2>Mask 图谱：UND / GEN 重排序前后</h2><p>从 240 组完整捕获中，按长度、稀疏度、Partial 比例、最长任务、跨样本块、样本数、GEN 比例和最长样本占比做多样性抽样。固定包含最复杂的 rank 5 / batch 24。</p>';
 $('controls').innerHTML='<label>定位案例 <select id="atlas-select">'+maskAtlasData.cases.map((c,i)=>`<option value="${c.rank}-${c.batch}">${i+1}. rank ${c.rank} / batch ${c.batch}</option>`).join('')+'</select></label>';
 $('scene').innerHTML='<div class="atlas-legend"><span>浅蓝：允许 attention</span><span>灰色：mask</span><span>紫条：GEN</span><span>绿条：UND</span></div><div class="atlas-grid">'+maskAtlasData.cases.map(atlasCard).join('')+'</div>';
 $('status').textContent='重排序是稳定的 GEN-first 分区：token 内容和可见 pair 数不变，二维坐标形状发生改变。';
 $('detail').innerHTML='<p><a href="mask-atlas-verification.json">查看选择与校验清单</a></p>';
 $('atlas-select').onchange=e=>document.querySelector(`[data-atlas-case="${e.target.value}"]`).scrollIntoView({block:'start'});
}
