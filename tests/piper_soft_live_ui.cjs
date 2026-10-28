const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=process.cwd(),web=path.join(root,'sim/piper_compliance/web');
class Element{
 constructor(){this.children=[];this.elements=new Map();this.listeners={};this.clientWidth=320;this.clientHeight=300;this.width=300;this.height=150;this.value='';this.checked=true;this.dataset={};this.classList={add(){},remove(){}};}
 append(...e){this.children.push(...e)} replaceChildren(){this.children=[]} addEventListener(k,f){this.listeners[k]=f} setPointerCapture(){} getBoundingClientRect(){return {left:0,top:0,width:320,height:300}}
 querySelector(s){if(!this.elements.has(s))this.elements.set(s,new Element());return this.elements.get(s)}
 getContext(){return new Proxy({}, {get:()=>()=>{}})}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
get('noise').value='clean';get('amplitude').value='2';get('plane').value='screen';
const buttons=['+X','-X','+Y','-Y','+Z','-Z'].map(dir=>{const e=new Element();e.dataset.push=dir;return e});
const calls=[],timers=[];let simTime=0,draw;const data=JSON.parse(fs.readFileSync(path.join(root,'outputs/piper_compliance/softmimic_official/replay/data.json')));const records=data.modes.clean.slice(0,3);
class Renderer{
 constructor(){this.domElement=new Element();this.ratio=1}setPixelRatio(n){this.ratio=n}getPixelRatio(){return this.ratio}setSize(w,h){this.domElement.width=w*this.ratio;this.domElement.height=h*this.ratio}render(scene,camera){scene.updateMatrixWorld(true);camera.updateMatrixWorld(true)}dispose(){}forceContextLoss(){}
}
const context=vm.createContext({console,AbortController,AbortSignal,devicePixelRatio:1,performance:{now:()=>0},
 requestAnimationFrame(fn){draw=fn},setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},
 testWorlds:[],window:{addEventListener(){}},document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:()=>buttons,addEventListener(){}},data,
 fetch:async (url,options)=>{
  if(url.startsWith('/api/')){const body=JSON.parse(options.body);calls.push({url,body});simTime=body.reset?0:simTime+.1;
   return {ok:true,json:async()=>({session:'token',time:simTime,reference:records[0].reference,force:body.paused?[0,0,0]:(body.force||[0,0,0]),stopped:false,controllers:records.map(r=>({sha256:r.checkpoint_sha256,q:r.cases['+Z'].q[0],tcp:r.cases['+Z'].tcp[0]}))})};}
  const b=fs.readFileSync(path.join(root,'src/piper_ros/src/piper_description/meshes',path.basename(url)));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};
 }});
(async()=>{
 const core=new vm.SourceTextModule(fs.readFileSync(path.join(web,'vendor/three.core.js'),'utf8'),{context});await core.link(()=>{});await core.evaluate();
 context.T={...core.namespace,WebGLRenderer:Renderer};
 const code=fs.readFileSync(path.join(web,'replay.js'),'utf8').replace(/^import .*\n/,'').replace(/main\(\)\.catch[\s\S]*$/,'');
 vm.runInContext(code,context);
 await vm.runInContext('payload=data;Promise.all([...new Set(payload.robot.visuals.map(v=>v.mesh))].map(meshGeometry))',context);
 const api=vm.runInContext('(function(){'+fs.readFileSync(path.join(web,'soft-live.js'),'utf8').replace(/^export /gm,'')+';return {startLive,dragForce}})()',context);context.api=api;
 vm.runInContext("api.startLive({T,buildWorld:(r,h)=>{const w=buildWorld(r,h);testWorlds.push(w);return w},renderPose,initial:data.modes.clean.slice(0,3)})",context);
 await new Promise(setImmediate);await new Promise(setImmediate);
 assert.equal(get('worlds').children.length,3);assert.equal(get('connect').textContent,'已连接');draw(1000);
 const world=context.testWorlds[0],point=world.tcp.position.clone().project(world.camera);
 const x=(point.x+1)*160,y=(1-point.y)*150;
 world.stage.listeners.pointerdown({clientX:x,clientY:y,pointerId:4});
 world.stage.listeners.pointermove({clientX:x+50,clientY:y-20});await timers.pop()();
 assert(Math.abs(Math.hypot(...calls.at(-1).body.force)-2)<1e-8);
 world.stage.listeners.pointerup();await new Promise(setImmediate);assert.deepEqual(calls.at(-1).body.force,[0,0,0]);
 buttons[0].listeners.pointerdown({pointerId:1});await new Promise(setImmediate);
 assert.deepEqual(calls.at(-1).body.force,[2,0,0]);
 get('amplitude').value='3.5';get('amplitude').oninput();await timers.pop()();assert.deepEqual(calls.at(-1).body.force,[3.5,0,0]);
 buttons[0].listeners.pointerup();await new Promise(setImmediate);assert.deepEqual(calls.at(-1).body.force,[0,0,0]);
 get('live-pause').onclick();await new Promise(setImmediate);assert.equal(calls.at(-1).body.paused,true);
 get('noise').value='noisy';get('noise').onchange();await new Promise(setImmediate);assert.equal(calls.at(-1).body.reset,true);assert.equal(calls.at(-1).body.noise,true);
 for(const plane of ['screen','X','Y','Z']){
  const f=api.dragForce(40,-20,new context.T.Quaternion(),plane,3.5,context.T);
  assert(Math.abs(Math.hypot(...f)-3.5)<1e-9);
  if(plane!=='screen')assert.equal(f.filter(x=>Math.abs(x)>1e-9).length,1);
 }
 assert.deepEqual(Array.from(api.dragForce(1,1,new context.T.Quaternion(),'screen',5,context.T)),[0,0,0]);
 console.log('PASS: three live windows; same force; amplitude update; pointer release; pause; noise reset; TCP hit-test/drag/release and camera-to-world force');
})().catch(e=>{console.error(e);process.exitCode=1});
