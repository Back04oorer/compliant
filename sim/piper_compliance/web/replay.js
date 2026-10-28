import * as T from './vendor/three.module.js';

const $=id=>document.getElementById(id);
let payload, worlds=[], time=0, playing=true, last=0;
let forceManifest=null,selectedForce=2;
let yaw=2.35, elevation=.35, distance=1.15;
const geometries=new Map(), v=new T.Vector3();
const originQuaternion=rpy=>new T.Quaternion().setFromEuler(new T.Euler(...rpy,'ZYX'));
const sample=(a,b,t)=>a.map((value,i)=>value+(b[i]-value)*t);

async function meshGeometry(name){
  const response=await fetch('/meshes/'+name); if(!response.ok)throw Error('网格加载失败：'+name);
  const buffer=await response.arrayBuffer(), view=new DataView(buffer), n=view.getUint32(80,true);
  if(buffer.byteLength!==84+50*n)throw Error('不支持的 STL 网格：'+name);
  const vertices=new Float32Array(n*9), normals=new Float32Array(n*9);
  for(let i=0;i<n;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++){
    vertices[i*9+j*3+k]=view.getFloat32(84+i*50+12+j*12+k*4,true);
    normals[i*9+j*3+k]=view.getFloat32(84+i*50+k*4,true);
  }
  const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.BufferAttribute(vertices,3));geometry.setAttribute('normal',new T.BufferAttribute(normals,3));
  geometries.set(name,geometry);
}

function makeRobot(description){
  const root=new T.Group(), links=new Map(), joints=[];
  function link(name){if(!links.has(name))links.set(name,new T.Group());return links.get(name);}
  for(const j of description.joints){
    const child=link(j.child);child.position.fromArray(j.xyz);child.quaternion.copy(originQuaternion(j.rpy));link(j.parent).add(child);
    if(j.type==='revolute')joints.push({node:child,index:Number(j.name.replace('joint',''))-1,base:child.quaternion.clone(),axis:new T.Vector3(...j.axis)});
  }
  for(const [name,node] of links)if(!description.joints.some(j=>j.child===name))root.add(node);
  const material=new T.MeshStandardMaterial({color:0xc4cbd4,metalness:.45,roughness:.48});
  for(const visual of description.visuals){
    const mesh=new T.Mesh(geometries.get(visual.mesh),material);mesh.position.fromArray(visual.xyz);mesh.quaternion.copy(originQuaternion(visual.rpy));mesh.scale.fromArray(visual.scale);link(visual.link).add(mesh);
  }
  return {root,joints,material};
}

function buildWorld(record,handlers=null){
  const card=document.createElement('article');card.className='world';
  card.innerHTML='<h2></h2><div class="stage" aria-label="机械臂三维回放"></div><div class="metrics"><div><small>沿施力方向让开</small><strong class="displacement"></strong></div><div><small>三维目标误差</small><strong class="error-value"></strong></div></div><canvas class="plot" aria-label="末端位移与目标曲线"></canvas><div class="legend">实线：让开距离 · 虚线：20 mm 目标</div>';
  card.querySelector('h2').textContent=record.label;$('worlds').append(card);
  const stage=card.querySelector('.stage'),scene=new T.Scene();scene.background=new T.Color('#edf1eb');
  const camera=new T.PerspectiveCamera(38,1,.01,10);camera.up.set(0,0,1);
  const renderer=new T.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));stage.append(renderer.domElement);
  scene.add(new T.HemisphereLight(0xffffff,0x728375,2.5));const light=new T.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);
  const grid=new T.GridHelper(1.2,12,0xb7c7b7,0xd8e1d5);grid.rotation.x=Math.PI/2;grid.position.z=-.004;scene.add(grid);
  const robot=makeRobot(payload.robot);scene.add(robot.root);
  const reference=new T.Mesh(new T.SphereGeometry(.006),new T.MeshBasicMaterial({color:0x258369}));reference.position.fromArray(record.reference);scene.add(reference);
  const target=new T.Mesh(new T.SphereGeometry(.009,16,12),new T.MeshBasicMaterial({color:0xe69125,wireframe:true}));scene.add(target);
  const tcp=new T.Mesh(new T.SphereGeometry(.004),new T.MeshBasicMaterial({color:record.color}));scene.add(tcp);
  const arrow=new T.ArrowHelper(new T.Vector3(0,0,1),new T.Vector3(),.08,0xe17c20,.02,.01);scene.add(arrow);
  let drag=null,forceDrag=false;
  stage.addEventListener('pointerdown',e=>{forceDrag=!!handlers?.down(e,{stage,camera,tcp});if(!forceDrag)drag=[e.clientX,e.clientY];stage.setPointerCapture(e.pointerId);});
  stage.addEventListener('pointermove',e=>{if(forceDrag){handlers.move(e,{stage,camera,tcp});return;}if(!drag)return;yaw-=(e.clientX-drag[0])*.008;elevation=Math.max(-.1,Math.min(1.3,elevation+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];});
  const end=()=>{if(forceDrag)handlers.end();forceDrag=false;drag=null;};
  stage.addEventListener('pointerup',end);stage.addEventListener('pointercancel',end);stage.addEventListener('lostpointercapture',end);
  stage.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(.55,Math.min(2,distance*Math.exp(e.deltaY*.001)));},{passive:false});
  return {record,card,stage,scene,camera,renderer,robot,reference,target,tcp,arrow};
}

function renderPose(w,q,tcp,force){
  for(const j of w.robot.joints)j.node.quaternion.copy(j.base).multiply(new T.Quaternion().setFromAxisAngle(j.axis,q[j.index]));
  w.tcp.position.fromArray(tcp);w.target.position.fromArray(w.record.reference.map((p,i)=>p+force[i]/100));
  const strength=Math.hypot(...force);w.arrow.visible=strength>.001;
  if(w.arrow.visible){w.arrow.position.fromArray(tcp);w.arrow.setDirection(v.fromArray(force).normalize());w.arrow.setLength(strength*.045,.018,.009);}
  const width=w.stage.clientWidth,height=w.stage.clientHeight;
  if(w.renderer.domElement.width!==Math.floor(width*w.renderer.getPixelRatio())||w.renderer.domElement.height!==Math.floor(height*w.renderer.getPixelRatio()))w.renderer.setSize(width,height,false);
  w.camera.aspect=width/height;w.camera.updateProjectionMatrix();
  w.camera.position.set(.16+distance*Math.cos(elevation)*Math.cos(yaw),distance*Math.cos(elevation)*Math.sin(yaw),.25+distance*Math.sin(elevation));w.camera.lookAt(.16,0,.25);
  w.renderer.render(w.scene,w.camera);
}

function changeMode(){
  const shared=new Set(geometries.values());
  for(const w of worlds){w.renderer.dispose();w.renderer.forceContextLoss();w.robot.material.dispose();w.scene.traverse(obj=>{if(obj.geometry&&!shared.has(obj.geometry)){obj.geometry.dispose();obj.material.dispose();}});}
  $('worlds').replaceChildren();worlds=payload.modes[$('noise').value].map(buildWorld);time=0;
  $('notice').textContent=$('noise').value==='noisy'?'含噪声：策略使用训练同级观测噪声，解析基线读取真实状态。拖动旋转视角，滚轮缩放。':'拖动任一机械臂窗口旋转全部视角；滚轮缩放。绿色圆点为原始目标，橙色圆环为受力后的柔顺目标。';
  for(const w of worlds)w.card.querySelector('.legend').textContent=`实线：让开距离 · 虚线：${selectedForce*10} mm 目标`;
  $('rollout-description').textContent=`真实评估轨迹同步回放 · 固定夹爪 · ${selectedForce} N 推力 · 柔顺目标 ${selectedForce*10} mm`;
}

async function loadForce(value){
  const old=selectedForce;
  $('force-level').disabled=true;$('play').disabled=true;$('restart').disabled=true;
  $('phase').textContent='正在加载 '+value+' N rollout…';
  try{
    const response=await fetch(forceManifest.files[String(Number(value))]);if(!response.ok)throw Error('该力量的轨迹加载失败');
    const data=await response.json();if(data.force_N!==Number(value))throw Error('轨迹力量与选择不一致');
    payload=data;selectedForce=Number(value);$('force-level').value=String(selectedForce);changeMode();
  }catch(error){$('force-level').value=String(old);$('notice').textContent=error.message;}
  finally{$('force-level').disabled=false;$('play').disabled=false;$('restart').disabled=false;}
}

function plot(w,c,index){
  const canvas=w.card.querySelector('.plot'),width=canvas.clientWidth,height=140,dpr=Math.min(devicePixelRatio,2);
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==height*dpr){canvas.width=Math.round(width*dpr);canvas.height=height*dpr;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
  const axis='XYZ'.indexOf($('direction').value[1]),sign=$('direction').value[0]==='+'?1:-1;
  const displacement=c.tcp.map(p=>(p[axis]-c.pre[axis])*sign*1000);
  const low=Math.min(-2,...displacement),high=Math.max(10,selectedForce*12,...displacement);
  const x=t=>30+t/12*(width-42),y=mm=>115-(mm-low)/(high-low)*100;
  ctx.font='10px system-ui';ctx.fillStyle='#718174';ctx.strokeStyle='#e5ebe2';
  for(const value of [0,selectedForce*10,Math.ceil(high)]){ctx.beginPath();ctx.moveTo(30,y(value));ctx.lineTo(width-12,y(value));ctx.stroke();ctx.fillText(value,5,y(value)+3);}
  ctx.fillText('mm',5,12);ctx.fillText('0 s',30,130);ctx.fillText('12 s',width-35,130);
  ctx.setLineDash([4,4]);ctx.strokeStyle='#a28356';ctx.beginPath();c.force.forEach((f,i)=>{const px=x(w.record.time[i]),py=y(f[axis]*sign/100*1000);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle=w.record.color;ctx.lineWidth=1.8;ctx.beginPath();c.tcp.forEach((p,i)=>{const px=x(w.record.time[i]),py=y((p[axis]-c.pre[axis])*sign*1000);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.lineWidth=1;
  ctx.strokeStyle='#344536';ctx.beginPath();ctx.moveTo(x(time),10);ctx.lineTo(x(time),115);ctx.stroke();
}

function frame(now){
  requestAnimationFrame(frame);if(now-last<33)return;const delta=last?Math.min((now-last)/1000,.1):0;last=now;
  if(playing){time+=delta*Number($('speed').value);if(time>12){if($('loop').checked)time=0;else{time=12;playing=false;$('play').textContent='播放';}}}
  const index=Math.min(599,Math.floor(time/.02)),mix=Math.min(1,(time-index*.02)/.02),direction=$('direction').value;
  $('seek').value=time;$('clock').textContent=time.toFixed(2)+' / 12.00 s';
  $('phase').textContent=time<3?'保持姿态':time<3.5?'逐渐施力':time<5.5?'持续受推':time<6?'逐渐撤力':'撤力恢复';
  for(const w of worlds){
    const c=w.record.cases[direction],q=sample(c.q[index],c.q[index+1],mix),tcp=sample(c.tcp[index],c.tcp[index+1],mix),force=sample(c.force[index],c.force[index+1],mix);
    renderPose(w,q,tcp,force);const strength=Math.hypot(...force);
    const axis='XYZ'.indexOf(direction[1]),sign=direction[0]==='+'?1:-1;
    w.card.querySelector('.displacement').textContent=((tcp[axis]-c.pre[axis])*sign*1000).toFixed(1)+' mm';
    w.card.querySelector('.error-value').textContent=(w.tcp.position.distanceTo(w.target.position)*1000).toFixed(1)+' mm';
    w.card.querySelector('.legend').textContent=c.metrics?.task_failure_over_30cm?'此工况曾偏离参考位置超过 30 cm · 失败轨迹完整保留':`实线：让开距离 · 虚线：${selectedForce*10} mm 目标`;
    $('force').textContent='外力 '+strength.toFixed(2)+' N';
    plot(w,c,index);
  }
}

async function main(){
  const response=await fetch('/media/replay.json');if(!response.ok)throw Error('评估回放数据尚未生成');payload=await response.json();
  await Promise.all([...new Set(payload.robot.visuals.map(v=>v.mesh))].map(meshGeometry));
  if(document.body?.dataset?.mode==='live'){
    const {startLive}=await import('./soft-live.js');
    $('camera').onclick=()=>{yaw=2.35;elevation=.35;distance=1.15;};
    startLive({T,buildWorld,renderPose,initial:payload.modes.clean.filter(r=>!r.label.includes('解析'))});return;
  }
  try{const manifest=await fetch('/media/rollouts/manifest.json');if(manifest.ok)forceManifest=await manifest.json();}catch{/* Keep the original 2 N recording available if examples are not generated yet. */}
  if(forceManifest){$('force-level').value='2';await loadForce(2);$('force-level').onchange=()=>loadForce($('force-level').value);}else changeMode();
  $('play').disabled=false;$('restart').disabled=false;
  $('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';};
  $('restart').onclick=()=>{time=0;playing=true;$('play').textContent='暂停';};
  $('seek').oninput=()=>{time=Number($('seek').value);playing=false;$('play').textContent='播放';};
  $('direction').onchange=()=>time=0;$('noise').onchange=changeMode;
  $('camera').onclick=()=>{yaw=2.35;elevation=.35;distance=1.15;};
  requestAnimationFrame(frame);
}
main().catch(error=>{$('phase').textContent='加载失败：'+error.message;$('phase').classList.add('error');console.error(error);});
