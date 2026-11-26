/* CombinatoryPlaySol — continuous recurrent associative reader. */
const Play = (function factory(){
 const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
 const unit=a=>{const n=Math.sqrt(dot(a,a));if(n<1e-10)return Array.from(a,()=>0);return Array.from(a,v=>v/n)};
 const mix=(d,terms)=>{const a=Array(d).fill(0);for(const [v,w] of terms)for(let j=0;j<d;j++)a[j]+=v[j]*w;return unit(a)};
 const mean=(b,ids)=>mix(b.d,ids.map(i=>[b.x[i],1]));
 const rng=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296};
 const grains=b=>[...new Set([Math.min(16,b.d),Math.min(64,b.d),b.d])];
 function prepare(raw){
  const words=raw.words,v=raw.vectors;
  if(!Array.isArray(words)||!Array.isArray(v)||words.length!==v.length||words.length<4||new Set(words).size!==words.length||words.some(w=>typeof w!=="string"))throw Error("Invalid bank words/vectors");
  const d=v[0].length;if(d<2||v.some(x=>x.length!==d||x.some(z=>typeof z!=="number"||!Number.isFinite(z))||dot(x,x)<1e-18))throw Error("Invalid bank vectors");
  const x=v.map(unit),b={words:words.slice(),x,d,meta:raw.meta||{},index:new Map(words.map((w,i)=>[w,i])),views:new Map()};
  for(const g of grains(b))b.views.set(g,g===d?x:x.map(a=>unit(a.slice(0,g))));
  return b;
 }
 function ids(b,s){const a=Array.isArray(s)?s:String(s||"").split(/[,+]/).map(x=>x.trim()).filter(Boolean);return [...new Set(a.map(w=>{if(!b.index.has(w))throw Error("Not in bank: "+w);return b.index.get(w)}))]}
 function rank(b,q,k,g=b.d,exclude=new Set(),penalty=null){
  const v=b.views.get(g);if(!v)throw Error("Unsupported grain "+g);
  const u=unit(q.slice(0,g)),rows=[];
  for(let i=0;i<v.length;i++)if(!exclude.has(i)){const sim=dot(u,v[i]);rows.push({i,sim,score:sim-(penalty?penalty[i]:0)})}
  rows.sort((a,b)=>b.score-a.score||a.i-b.i);return rows.slice(0,k);
 }
 function coherence(b,ids){let s=0,n=0;for(let i=0;i<ids.length;i++)for(let j=0;j<i;j++){s+=dot(b.x[ids[i]],b.x[ids[j]]);n++}return n?s/n:null}
 function overlap(a,b){const x=new Set(a),y=new Set(b);let n=0;for(const i of x)if(y.has(i))n++;return n/(new Set([...x,...y]).size||1)}
 function episodes(b,raw){return raw.map((e,j)=>{let k=ids(b,e.key),t=ids(b,e.tag),v=ids(b,e.value);if(!k.length||!t.length||!v.length)throw Error("Episode "+j+" needs key, tag and value");return {name:e.name||"episode "+j,key:mean(b,k),tag:mean(b,t),value:mean(b,v),keyIds:k,tagIds:t,valueIds:v}})}
 class Player{
  constructor(b,cues,{tag=[],memories=[],mode="recurrent",grain=b.d,k=12,feedback=.48,anchor=.18,tagBias=0,memoryStrength=.45,temperature=.09,coverage=.16,decay=.10,heat=true,threshold=.2}={}){
   if(!cues.length||cues.length>=b.words.length)throw Error("Need at least one cue and remaining bank words");
   if(!["recurrent","static","no-feedback","discrete"].includes(mode)||!Number.isInteger(k)||k<1||![feedback,anchor,tagBias,memoryStrength,coverage,decay,threshold].every(x=>Number.isFinite(x)&&x>=0&&x<=1)||feedback+anchor>1||!Number.isFinite(temperature)||temperature<=0||!b.views.has(grain)&&grain!=="cycle")throw Error("Invalid player options");
   this.b=b;this.cues=cues.slice();this.seed=mean(b,cues);this.x=this.seed.slice();this.tag=tag.length?mean(b,tag):Array(b.d).fill(0);this.tagIds=tag.slice();this.memories=memories;this.opts={mode,grain,k,feedback,anchor,tagBias,memoryStrength,temperature,coverage,decay,heat,threshold};this.T=temperature;this.cover=new Float64Array(b.words.length);this.history=[];this.seen=new Set();this.tick=0;this.exclude=new Set(cues);this.active=[];this.baseline=new Set(cues.flatMap(i=>rank(b,b.x[i],k,b.d,new Set([i])).map(z=>z.i)));
  }
  setTag(tag){this.tagIds=tag.slice();this.tag=tag.length?mean(this.b,tag):Array(this.b.d).fill(0)}
  setState(v){if(v.length!==this.b.d||v.some(x=>!Number.isFinite(x)))throw Error("Bad state");this.x=unit(v)}
  step(){
   const b=this.b,o=this.opts,old=this.x.slice(),g=o.grain==="cycle"?grains(b)[this.tick%grains(b).length]:o.grain;
   const active=o.mode==="static"||!this.tagIds.length?[]:this.memories.map(e=>{const a=dot(old,e.key),t=dot(this.tag,e.tag);return {e,a,t,w:a>=o.threshold&&t>=o.threshold?a*t:0}}).filter(z=>z.w>0);
   const m=active.length?mix(b.d,active.map(z=>[z.e.value,z.w])):Array(b.d).fill(0);
   const q=o.mode==="static"?this.seed:mix(b.d,[[old,1],[this.tag,o.tagBias],[m,o.memoryStrength]]);
   const pen=o.mode==="static"?null:Array.from(this.cover,z=>o.coverage*z);
   const rows=rank(b,q,o.k,g,this.exclude,pen);if(!rows.length)throw Error("No workspace candidates");
   const max=rows[0].score,ww=rows.map(z=>Math.exp((z.score-max)/this.T)),sum=ww.reduce((a,b)=>a+b,0);
   const workspace=rows.map((z,i)=>({...z,weight:ww[i]/sum})),wids=rows.map(z=>z.i);
   const prev=this.active,repeat=prev.length?overlap(prev,wids):0,novel=wids.filter(i=>!this.seen.has(i)).length;
   const unlocked=wids.filter(i=>!this.baseline.has(i));
   const y=mix(b.d,workspace.map(z=>[b.x[z.i],z.weight]));
   if(o.mode==="recurrent")this.x=mix(b.d,[[old,1-o.feedback-o.anchor],[y,o.feedback],[this.seed,o.anchor]]);
   else if(o.mode==="discrete")this.x=b.x[rows[0].i].slice();
   else this.x=this.seed.slice();
   if(o.mode!=="static"){for(let i=0;i<this.cover.length;i++)this.cover[i]*=1-o.decay;for(const i of wids)this.cover[i]+=1/o.k;}for(const i of wids)this.seen.add(i);
   if(o.heat&&o.mode!=="static")this.T=repeat>.6?Math.min(o.temperature*4,this.T*1.35):Math.max(o.temperature,this.T*.90);
   this.tick++;this.active=wids;
   const rec={tick:this.tick,grain:g,temperature:this.T,workspace,unlocked,novel,repeat,coherence:coherence(b,wids),unlockedCoherence:coherence(b,unlocked),cueRetention:dot(this.x,this.seed),stateMotion:1-dot(old,this.x),memory:active.map(z=>({name:z.e.name,keyFit:z.a,tagFit:z.t,weight:z.w})),state:this.x.slice(),query:q,tag:this.tagIds.slice()};
   this.history.push(rec);return rec;
  }
 }
 function demo(){
  const words=["bird","shadow","winter","sky","star","forest","river","rain","beach","mountain","moon","storm","wing","sun","summer","friend","family","home","letter","photo","garden","music","piano","guitar","song","voice","concert","ticket","memory","love","travel","train","station","city","road","hotel","book","paper","desk","oak","wood","chair","tree","leaf","branch","cloud","ocean","snow"];
  const r=rng(43),groups=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,0,0,0];
  const vectors=words.map((w,i)=>Array.from({length:12},(_,j)=>(r()-.5)*.6+(j===groups[i]?1.3:0)+(j===5&&[0,12,39,42,43,44].includes(i)?.7:0)));
  return {words,vectors,meta:{source:"Synthetic toy, not Qwen",pcs:12}};
 }
 function run(b,cues,options={},ticks=30){const p=new Player(b,cues,options);for(let i=0;i<ticks;i++)p.step();return p}
 function summary(p){const h=p.history,all=new Set(h.flatMap(r=>r.workspace.map(z=>z.i))),newSet=new Set(h.flatMap(r=>r.unlocked)),avg=k=>h.reduce((a,r)=>a+(r[k]??0),0)/h.length;return {ticks:h.length,distinct:all.size,unlockedDistinct:newSet.size,meanCoherence:avg("coherence"),meanCueRetention:avg("cueRetention"),meanRepeat:avg("repeat"),meanStateMotion:avg("stateMotion"),memoryActiveTicks:h.filter(r=>r.memory.length).length}}
 function experiment(b,cues,options={},ticks=30){
  const arms={};for(const mode of ["static","no-feedback","discrete","recurrent"]){const p=run(b,cues,{...options,mode},ticks);arms[mode]={summary:summary(p),trace:p.history}}
  const p=run(b,cues,{...options,mode:"recurrent",tag:[],memories:[],tagBias:0},ticks);arms["recurrent-no-context"]={summary:summary(p),trace:p.history};
  return arms;
 }
 function stateIntervention(b,cues,options={},ticks=20){
  // Two histories, then reset all receiving-machine fields except the injected x.
  // Both receivers have the same previous workspace and zero coverage.
  const a=run(b,cues,{...options,mode:"recurrent"},ticks);
  const c=run(b,cues,{...options,mode:"no-feedback"},ticks);
  const common={...options,mode:"recurrent",tag:[],memories:[],tagBias:0,coverage:0,heat:false,grain:b.d};
  const left=new Player(b,cues,common),right=new Player(b,cues,common);
  const visible=a.active.slice();left.active=visible.slice();right.active=visible.slice();
  left.setState(a.x);right.setState(c.x);
  const l=left.step(),r=right.step();
  return {priorWorkspace:visible,stateCosine:dot(a.x,c.x),nextOverlap:overlap(l.workspace.map(z=>z.i),r.workspace.map(z=>z.i)),left:l,right:r};
 }
 function selfTest(){
  const b=prepare(demo()),c=ids(b,"bird,shadow,winter"),before=JSON.stringify(b.x),p=new Player(b,c,{k:6,temperature:.1});
  const a=p.step();if(a.workspace.length!==6||Math.abs(a.workspace.reduce((s,z)=>s+z.weight,0)-1)>1e-9)throw Error("parallel workspace");
  if(b.x.some(v=>dot(v,p.x)>1-1e-10))throw Error("state snapped to word");
  const s=run(b,c,{mode:"static",k:6},3);if(overlap(s.history[0].workspace.map(z=>z.i),s.active)!==1||dot(s.x,s.seed)<1-1e-9)throw Error("static baseline");
  if(JSON.stringify(b.x)!==before)throw Error("bank changed");
  const mem=episodes(b,[{key:"bird,shadow",tag:"music",value:"song,guitar"},{key:"bird,shadow",tag:"family",value:"home,photo"}]);
  const p1=new Player(b,c,{k:6,tag:ids(b,"music"),memories:mem,threshold:.5}),p2=new Player(b,c,{k:6,tag:ids(b,"family"),memories:mem,threshold:.5});
  const x=p1.step(),y=p2.step();if(x.memory[0]?.name!=="episode 0"||y.memory[0]?.name!=="episode 1"||dot(x.query,y.query)>1-1e-8)throw Error("context gate");
  const same=new Player(b,c,{k:6});same.active=p.active.slice();same.setState(p.x);if(dot(same.x,p.x)<1-1e-9)throw Error("state transfer");
  const z=experiment(b,c,{k:6},5);if(Object.keys(z).length!==5||Object.values(z).some(a=>a.trace.length!==5))throw Error("matched arms");
  return ["parallel weighted workspace","continuous unsnapped state","static control stable","frozen bank","context-selective episodic gate","state transfer","matched control arms"];
 }
 return {dot,unit,mix,prepare,ids,rank,coherence,overlap,episodes,Player,demo,run,summary,experiment,stateIntervention,selfTest,grains};
})();
if(typeof module !== 'undefined') module.exports=Play;
