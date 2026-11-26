const fs=require("node:fs"),P=require("./play.js");
const args=process.argv.slice(2);
if(args.includes("--self-test")){console.log(P.selfTest());process.exit(0)}
const a={};for(let i=0;i<args.length;i+=2){if(!args[i].startsWith("--")||args[i+1]===undefined)throw Error("Use --key value");a[args[i].slice(2)]=args[i+1]}
const valid=new Set(["bank","cues","tag","episodes","ticks","k","grain","feedback","anchor","tag-bias","memory-strength","temperature","coverage","out","switch-tag","switch-at"]);for(const k of Object.keys(a))if(!valid.has(k))throw Error("Unknown option "+k);
const b=P.prepare(a.bank==="demo"?P.demo():JSON.parse(fs.readFileSync(a.bank||"../SweepSol/bank.json","utf8")));
const cues=P.ids(b,a.cues||"bird,shadow,winter"),tag=P.ids(b,a.tag||""),raw=a.episodes?JSON.parse(fs.readFileSync(a.episodes,"utf8")):[],memories=P.episodes(b,raw),n=(k,d)=>a[k]===undefined?d:Number(a[k]),ticks=n("ticks",40);
if(!Number.isInteger(ticks)||ticks<1||ticks>10000)throw Error("Invalid ticks");
const opts={tag,memories,k:n("k",12),grain:a.grain==="cycle"?"cycle":n("grain",b.d),feedback:n("feedback",.48),anchor:n("anchor",.18),tagBias:n("tag-bias",0),memoryStrength:n("memory-strength",.45),temperature:n("temperature",.09),coverage:n("coverage",.16)};
const arms=P.experiment(b,cues,opts,ticks);
if(a["switch-tag"]!==undefined){const p=new P.Player(b,cues,opts),sw=n("switch-at",Math.floor(ticks/2));if(!Number.isInteger(sw)||sw<0||sw>=ticks)throw Error("Invalid switch-at");for(let i=0;i<ticks;i++){if(i===sw)p.setTag(P.ids(b,a["switch-tag"]));p.step()}arms["context-switch"]={summary:P.summary(p),trace:p.history}}
console.table(Object.entries(arms).map(([arm,v])=>({arm,...v.summary})));
for(const [arm,v] of Object.entries(arms)){console.log("\n"+arm);for(const r of v.trace.slice(0,12))console.log(r.tick,r.workspace.map(w=>b.words[w.i]).join(", "))}
const out=a.out||"play-results.json";fs.writeFileSync(out,JSON.stringify({meta:b.meta,words:b.words,cues,options:{...opts,memories:undefined},episodes:raw,arms},null,2));console.log("Wrote",out);
