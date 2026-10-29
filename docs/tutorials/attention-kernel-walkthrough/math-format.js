/* Local KaTeX assets: formulas remain available without a CDN. */
'use strict';
function lessonTex(source){
  return source.trim()
    .replace(/Σ_Q,range,group-head/g,'\\sum_{I,r,h}')
    .replace(/Σₖ/g,'\\sum_k ').replace(/Σ_([A-Za-z]+)/g,'\\sum_{$1} ').replace(/Σ/g,'\\sum ')
    .replace(/([A-Za-z]+)_([A-Za-z]+)/g,(_,a,b)=>a+'_{'+(b.length>2?'\\mathrm{'+b+'}':b)+'}')
    .replace(/ᵀ/g,'^{\\mathsf T}').replace(/′/g,"'")
    .replace(/√(128|4|d)/g,'\\sqrt{$1}')
    .replace(/ℓ/g,'\\ell ').replace(/⊙/g,'\\odot ').replace(/Δ/g,'\\Delta ')
    .replace(/−/g,'-').replace(/∞/g,'\\infty ').replace(/α/g,'\\alpha ')
    .replace(/→/g,'\\to ').replace(/←/g,'\\leftarrow ').replace(/·/g,'\\cdot ')
    .replace(/\b(logaddexp|rowsum|softmax|dot|score)\b/g,'\\operatorname{$1}')
    .replace(/\b(exp|max|log|ln)\b/g,'\\$1 ')
    .replace(/LSE/g,'\\operatorname{LSE}')
    .replace(/softmax_\{row\}/g,'softmax_{\\mathrm{row}}');
}
const mathExact=new Map([
 ['S = QKᵀ / √d + B',String.raw`S=\frac{QK^{\mathsf T}}{\sqrt d}+B`],
 ['S_IJ = Q_I K_Jᵀ/√d + B_IJ',String.raw`S_{IJ}=\frac{Q_I K_J^{\mathsf T}}{\sqrt d}+B_{IJ}`],
 ['O_I = Σ_J P_IJ V_J',String.raw`O_I=\sum_J P_{IJ}V_J`],
 ['P = softmax_row(S)',String.raw`P=\operatorname{softmax}_{\mathrm{row}}(S)`],
 ['S[q,k] = dot(Q[q,:], K[k,:]) / √d + B[q,k]',String.raw`S_{qk}=\frac{\langle Q_{q,:},K_{k,:}\rangle}{\sqrt d}+B_{qk}`],
 ['P[q,:] = softmax(S[q,:])',String.raw`P_{q,:}=\operatorname{softmax}(S_{q,:})`],
 ['O[q,:] = Σₖ P[q,k] · V[k,:]',String.raw`O_{q,:}=\sum_k P_{qk}V_{k,:}`],
 ['dQ = dS K/√d',String.raw`dQ=\frac{dS\,K}{\sqrt d}`],
 ['dK = dSᵀQ/√d',String.raw`dK=\frac{dS^{\mathsf T}Q}{\sqrt d}`],
 ['dK = dSᵀ Q/√d',String.raw`dK=\frac{dS^{\mathsf T}Q}{\sqrt d}`],
 ['P_IJ = exp(Q_I K_Jᵀ/√d + B_IJ − LSE_I)',String.raw`P_{IJ}=\exp\!\left(\frac{Q_I K_J^{\mathsf T}}{\sqrt d}+B_{IJ}-\operatorname{LSE}_I\right)`],
 ['ΔdQ_I = dS_IJ K_J/√d',String.raw`\Delta dQ_I=\frac{dS_{IJ}K_J}{\sqrt d}`],
 ['ΔdK_J = dS_IJᵀ Q_I/√d',String.raw`\Delta dK_J=\frac{dS_{IJ}^{\mathsf T}Q_I}{\sqrt d}`],
 ['O_slice = A/ℓ',String.raw`O_{\mathrm{slice}}=\frac A\ell`],
 ['O=A/ℓ',String.raw`O=\frac A\ell`],
 ['1/√d',String.raw`\frac{1}{\sqrt d}`],
 ['S = QKᵀ / √128 + B',String.raw`S=\frac{QK^{\mathsf T}}{\sqrt{128}}+B`],
]);
function mathMarkup(src,display=false){const tex=mathExact.get(src.trim())??lessonTex(src);return `<span class="lesson-math${display?' math-display':''}" data-math-source="${src.replace(/&/g,'&amp;').replace(/"/g,'&quot;')}">${katex.renderToString(tex,{displayMode:display,throwOnError:true,strict:'ignore',output:'htmlAndMathml'})}</span>`}
// Explicit formula fragments, longest first. Never parse user input or code listings.
const inlineMath=[...mathExact.keys(),'dQ = Σ_K ΔdQ','dK = Σ_Q,range,group-head ΔdK','dV = Σ_Q,range,group-head ΔdV','D_i = Σ_j P_ij dP_ij = Σ_d O_id dO_id','D_i = Σ_d dO_id O_id','dS = P ⊙ (dP − D)','dS = P ⊙ (dP−D)','dP=dO Vᵀ','D=rowsum(O⊙dO)','dS=P⊙(dP−D)','ΔdQ=dS K','ΔdK=dSᵀQ','ΔdV=PᵀdO','dV += PᵀdO','dK += dSᵀQ/√d','dQ += dS K/√d','P = exp(S−LSE)','P = exp(QKᵀ/√d + mask − LSE)','S = QKᵀ/√d + mask','w=exp(score−m′)','exp(m−m′)','dK_J','dV_J','dQ_I','Q_I/dO_I','Q_i','K_j','√d'].sort((a,b)=>b.length-a.length);
const mathPattern=new RegExp(inlineMath.map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
function formatLessonMath(){
  for(const el of document.querySelectorAll('.formula:not([data-typeset])')){
    el.dataset.typeset='true';const text=el.textContent;
    if(text.startsWith('同步重排')){el.classList.add('formula-note');continue}
    el.replaceChildren();
    for(const line of text.split('\n').filter(s=>s.trim())){
      const row=document.createElement('div');row.className='math-row';
      for(const part of line.split(/　{2,}|，|；|\s+←\s*/)){
        if(!part.trim())continue;
        if(/[\u4e00-\u9fff]/.test(part)){const span=document.createElement('span');span.className='math-caption';span.textContent=part;row.append(span)}
        else row.insertAdjacentHTML('beforeend',mathMarkup(part.replace(/^(Forward|Backward):\s*/,''),true));
      }el.append(row);
    }
  }
  const walker=document.createTreeWalker(document.querySelector('main'),NodeFilter.SHOW_TEXT);const nodes=[];
  while(walker.nextNode()){const n=walker.currentNode;if(!n.parentElement.closest('svg,.formula,.lesson-math,script,style,button,select,code,pre'))nodes.push(n)}
  for(const n of nodes){const s=n.textContent;mathPattern.lastIndex=0;const matches=[...s.matchAll(mathPattern)];if(!matches.length)continue;const f=document.createDocumentFragment();let last=0;for(const m of matches){f.append(s.slice(last,m.index));const span=document.createElement('span');span.innerHTML=mathMarkup(m[0]);f.append(...span.childNodes);last=m.index+m[0].length}f.append(s.slice(last));n.replaceWith(f)}
  // Keep animated SVG labels as native SVG text. Replacing them with HTML
  // foreignObject nodes breaks the SVG coordinate system during redraws.
}
let mathFrame=null;
const mathObserver=new MutationObserver(()=>{if(mathFrame!==null)return;mathFrame=requestAnimationFrame(()=>{mathFrame=null;mathObserver.disconnect();try{formatLessonMath()}finally{mathObserver.observe(document.querySelector('main'),{childList:true,subtree:true})}})});
mathObserver.observe(document.querySelector('main'),{childList:true,subtree:true});
