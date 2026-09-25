export const initialQuery = [1,0,1,1,0,0,1,0];
export const keys = ['10110010','10110110','00110011','11100010','01001101','10100000','11010010','00111010','10110000','11001001','00110010','11111111','10010010','01010101','10111010','00000000'].map(s=>[...s].map(Number));
export const values = keys.map((_,i)=>[Math.sin(i*.8),Math.cos(i*.6),((i*3)%11)/5-1]);
export function attention(query, k=4, mode='hierarchical', temperature=1){
 const scores=keys.map((key,i)=>({id:i,matches:key.reduce((s,b,j)=>s+(b===query[j]),0)}));
 const rank=rows=>[...rows].sort((a,b)=>b.matches-a.matches||a.id-b.id);
 const candidates=Array.from({length:4},(_,tile)=>rank(scores.slice(tile*4,tile*4+4)).slice(0,2)).flat();
 const selected=rank(mode==='hierarchical'?candidates:scores).slice(0,mode==='dense'?16:k);
 const max=selected[0].matches;
 const raw=selected.map(s=>Math.exp((s.matches-max)/temperature));
 const sum=raw.reduce((a,b)=>a+b,0);
 const weights=scores.map(s=>{const idx=selected.findIndex(x=>x.id===s.id);return idx<0?0:raw[idx]/sum;});
 const output=[0,1,2].map(j=>weights.reduce((sum,w,i)=>sum+w*values[i][j],0));
 const exact=rank(scores).slice(0,k).map(s=>s.id);
 return {scores,candidates:candidates.map(s=>s.id),selected:selected.map(s=>s.id),weights,output,recall:selected.filter(s=>exact.includes(s.id)).length/k};
}
