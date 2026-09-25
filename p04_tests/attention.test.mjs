import assert from 'node:assert/strict';
import {attention,keys,values,initialQuery} from '../p01_ui/attention.mjs';
let differing=0;
for(let bits=0;bits<256;bits++){
 const q=Array.from({length:8},(_,j)=>(bits>>j)&1);
 for(let k=1;k<=8;k++)for(const mode of ['hierarchical','exact','dense'])for(const temp of [.5,1,2]){
  const r=attention(q,k,mode,temp);
  assert.equal(r.selected.length,mode==='dense'?16:k);
  assert.equal(new Set(r.selected).size,r.selected.length);
  assert(Math.abs(r.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
  assert(r.weights.every(w=>w>=0&&w<=1));
  for(let j=0;j<3;j++){
   const component=values.map(v=>v[j]);
   assert(r.output[j]>=Math.min(...component)-1e-12&&r.output[j]<=Math.max(...component)+1e-12);
  }
  if(mode==='hierarchical')assert(r.selected.every(id=>r.candidates.includes(id)));
  if(mode==='exact'){
   const chosen=r.selected.map(id=>r.scores[id].matches);
   const excluded=r.scores.filter(s=>!r.selected.includes(s.id)).map(s=>s.matches);
   assert(Math.min(...chosen)>=Math.max(...excluded));
  }
 }
 if(attention(q).output.some((v,j)=>Math.abs(v-attention(q,4,'exact').output[j])>1e-6))differing++;
}
const base=attention(initialQuery),q=[...initialQuery];q[0]^=1;
assert.notDeepEqual(attention(q).output,base.output);
assert.equal(base.scores[0].matches,8);
assert(differing>0);
console.log('PASS: 18,432 combinations; normalization, selection, convex outputs, bit propagation, pruning differences.');
