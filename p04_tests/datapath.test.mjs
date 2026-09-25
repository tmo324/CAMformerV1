import assert from 'node:assert/strict';
import {compute,parseInputs,bf16} from '../p01_ui/datapath.mjs';
const q=[1,0],keys=[[1,0],[1,0],[0,1],[0,1]],values=[[2,4],[6,8],[100,100],[200,200]];
const r=compute(q,keys,values,{k:2,tile:4,local:2});
assert.deepEqual(r.selected,[0,1]);assert.deepEqual(r.weights,[.5,.5,0,0]);assert.deepEqual(r.output,[4,6]);
assert.deepEqual(r.products,[[1,2],[3,4],[0,0],[0,0]]);
const changed=compute(q,keys,[[4,4],...values.slice(1)],{k:2,tile:4,local:2});assert.deepEqual(changed.output,[5,6]);
assert.equal(bf16(1.00390625),1);assert.equal(bf16(1.01171875),1.015625);
assert.throws(()=>compute(q,keys,values,{k:4,tile:4,local:2}),/Only/);
assert.throws(()=>compute(q,keys,[[1]],{}),/dimensions/);
assert.throws(()=>compute([2,0],keys,values),/binary/);
assert.throws(()=>compute(q,keys,[[NaN,0],...values.slice(1)]),/finite/);
assert.deepEqual(parseInputs('10','10\n01','2, 4\n6 8'),{q,keys:[[1,0],[0,1]],values:[[2,4],[6,8]]});
for(let x=0;x<256;x++){
 const query=Array.from({length:8},(_,i)=>(x>>i)&1);
 const ks=Array.from({length:8},(_,j)=>Array.from({length:8},(_,i)=>((x+j*13)>>i)&1));
 const vs=ks.map((_,i)=>[i,-i,1]);const a=compute(query,ks,vs);
 assert(Math.abs(a.weights.reduce((s,v)=>s+v,0)-1)<1e-12);
 assert(Math.abs(a.output[2]-1)<1e-12);
 assert(Math.abs(a.output[0]+a.output[1])<1e-12);
 assert(a.output.every(Number.isFinite));
}
console.log('PASS: hand-calculated output, custom V propagation, BF16 ties, invalid input rejection, 256 conservation checks.');
