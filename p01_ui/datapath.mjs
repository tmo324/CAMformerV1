export function bf16(x){const buffer=new ArrayBuffer(4),f=new Float32Array(buffer),u=new Uint32Array(buffer);f[0]=x;const bits=u[0];u[0]=(bits+0x7fff+((bits>>>16)&1))&0xffff0000;return f[0];}
export function compute(q,keys,values,{k=4,tile=4,local=2,scale=1,temperature=1}={}){
 if(!q.length||!keys.length||values.length!==keys.length||keys.some(r=>r.length!==q.length)||values.some(r=>r.length!==values[0].length)||!values[0].length)throw Error('Q, K and V dimensions do not agree.');
 if([...q,...keys.flat()].some(x=>x!==0&&x!==1)||values.flat().some(x=>!Number.isFinite(x)||Math.abs(x)>1e6))throw Error('Use binary Q/K and finite V values between −1,000,000 and 1,000,000.');
 if(!Number.isInteger(k)||k<1||k>keys.length||!Number.isInteger(tile)||tile<1||!Number.isInteger(local)||local<1||local>tile||!Number.isFinite(scale)||!Number.isFinite(temperature)||scale<1e-6||scale>1e6||temperature<1e-6||temperature>1e6)throw Error('Invalid selection or score settings.');
 const matches=keys.map(r=>r.reduce((s,b,j)=>s+(b===q[j]),0));
 const rank=ids=>ids.sort((a,b)=>matches[b]-matches[a]||a-b);
 const candidates=[];for(let start=0;start<keys.length;start+=tile)candidates.push(...rank(Array.from({length:Math.min(tile,keys.length-start)},(_,j)=>start+j)).slice(0,local));
 if(k>candidates.length)throw Error(`Only ${candidates.length} local candidates survive. Reduce global top-k.`);
 const selected=rank([...candidates]).slice(0,k),logits=matches.map(m=>m*scale/temperature),max=Math.max(...selected.map(i=>logits[i]));
 const exps=logits.map((s,i)=>selected.includes(i)?Math.exp(s-max):0),z=exps.reduce((a,b)=>a+b,0),weights=exps.map(x=>x/z);
 const stored=values.map(r=>r.map(bf16)),products=stored.map((r,i)=>r.map(v=>v*weights[i]));
 const output=stored[0].map((_,j)=>products.reduce((s,r)=>s+r[j],0));
 return {matches,candidates,selected,logits,weights,stored,products,output};
}
export function parseInputs(qText,kText,vText){
 const binary=s=>s.trim().replace(/[\s,]+/g,'');
 const q=[...binary(qText)].map(c=>c==='0'?0:c==='1'?1:NaN);
 const keys=kText.trim().split(/\n/).filter(s=>s.trim()).map(s=>[...binary(s)].map(c=>c==='0'?0:c==='1'?1:NaN));
 const values=vText.trim().split(/\n/).filter(s=>s.trim()).map(s=>s.trim().split(/[\s,]+/).map(Number));
 if(q.length<1||q.length>64||keys.length>32||values[0]?.length>8)throw Error('Display limits: 1–64 query bits, up to 32 key/value rows, and up to 8 output dimensions.');
 return {q,keys,values};
}
