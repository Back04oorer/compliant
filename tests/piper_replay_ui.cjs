const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=process.cwd(),web=path.join(root,'sim/piper_compliance/web');
class Element{
 constructor(){this.children=[];this.elements=new Map();this.listeners={};this.clientWidth=320;this.clientHeight=300;this.width=300;this.height=150;this.value='';this.checked=true;this.classList={add(){}};}
 append(...e){this.children.push(...e)} replaceChildren(){this.children=[]} addEventListener(k,f){this.listeners[k]=f} setPointerCapture(){}
 querySelector(s){if(!this.elements.has(s))this.elements.set(s,new Element());return this.elements.get(s)}
 getContext(){return new Proxy({}, {get:()=>()=>{}})}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
get('direction').value='+Z';get('noise').value='clean';get('speed').value='1';
class Renderer{
 constructor(){this.domElement=new Element();this.ratio=1}setPixelRatio(n){this.ratio=n}getPixelRatio(){return this.ratio}setSize(w,h){this.domElement.width=w*this.ratio;this.domElement.height=h*this.ratio}render(scene){scene.updateMatrixWorld(true)}dispose(){}forceContextLoss(){}
}
const context=vm.createContext({console,AbortController,devicePixelRatio:1,requestAnimationFrame(){},document:{getElementById:get,createElement:()=>new Element()},
 fetch:async url=>{const file=url==='/media/replay.json'?path.join(root,'outputs/piper_compliance/softmimic_official/replay/data.json'):url.startsWith('/media/rollouts/')?path.join(root,'outputs/piper_compliance/softmimic_official/replay/forces',path.basename(url)):path.join(root,'src/piper_ros/src/piper_description/meshes',path.basename(url));const b=fs.readFileSync(file);return {ok:true,json:async()=>JSON.parse(b),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};}});
(async()=>{
 const core=new vm.SourceTextModule(fs.readFileSync(path.join(web,'vendor/three.core.js'),'utf8'),{context});await core.link(()=>{});await core.evaluate();
 context.T={...core.namespace,WebGLRenderer:Renderer};
 let code=fs.readFileSync(path.join(web,'replay.js'),'utf8').replace(/^import .*\n/,'').replace('main().catch(', 'globalThis.ready=main().catch(');
 vm.runInContext(code,context);await context.ready;
 if(get('force-level').onchange){get('force-level').value='5';await get('force-level').onchange();assert(get('rollout-description').textContent.includes('50 mm'));}
 assert.equal(get('worlds').children.length,4);assert.equal(get('play').disabled,false);
 get('seek').value='4.8';get('seek').oninput();vm.runInContext('frame(1000)',context);
 assert.equal(get('play').textContent,'播放');assert(get('clock').textContent.startsWith('4.80'));
 assert(get('force').textContent.includes(get('force-level').onchange?'5.00':'2.00'));
 // Verify the actual Three.js hierarchy at the same replay instant, not just Python FK.
 const error=vm.runInContext(`Math.max(...worlds.map(w=>{const j=w.robot.joints.find(j=>j.index===5);return j.node.localToWorld(new T.Vector3(0,0,.16)).distanceTo(w.tcp.position)}))`,context);
 assert(error<5e-5,error);
 get('noise').value='noisy';get('noise').onchange();assert.equal(get('worlds').children.length,get('force-level').onchange?4:3);
 get('direction').value='-X';get('direction').onchange();get('seek').value='12';get('seek').oninput();vm.runInContext('frame(1100)',context);assert.equal(get('phase').textContent,'撤力恢复');
 get('restart').onclick();vm.runInContext('frame(1200)',context);assert.equal(get('play').textContent,'暂停');
 get('loop').checked=false;vm.runInContext('time=11.99;frame(1300)',context);assert.equal(get('play').textContent,'播放');
 console.log('PASS: loaded all meshes; 4/3 windows; seek/pause/restart/end; noise/direction switch; actual Three.js TCP error',error);
})().catch(e=>{console.error(e);process.exitCode=1});
