const $=id=>document.getElementById(id);

export function dragForce(dx,dy,cameraQuaternion,plane,amplitude,T){
  if(Math.hypot(dx,dy)<5)return [0,0,0];
  const direction=new T.Vector3(dx,-dy,0).applyQuaternion(cameraQuaternion);
  if(plane!=='screen'){
    const axis='XYZ'.indexOf(plane),basis=new T.Vector3().setComponent(axis,1).applyQuaternion(cameraQuaternion.clone().invert());
    let projection=dx*basis.x-dy*basis.y;
    // An axis pointing into the screen uses vertical drag to keep both signs reachable.
    if(Math.hypot(basis.x,basis.y)<.15)projection=-dy;
    direction.set(0,0,0).setComponent(axis,Math.sign(projection));
  }
  return direction.normalize().multiplyScalar(amplitude).toArray();
}

export function startLive({T,buildWorld,renderPose,initial}){
  let session=null,connected=false,busy=false,paused=false,reset=false,wanted=[0,0,0],latest=null,timer=null,gesture=null,buttonDirection=null;
  let revision=0,received=0,previous=null;
  const worlds=[],histories=[];
  const amplitude=()=>Math.max(0,Math.min(5,Number($('amplitude').value)||0));
  function clearForce(){wanted=[0,0,0];gesture=null;buttonDirection=null;revision++;
    document.querySelectorAll('[data-push]').forEach(b=>b.classList.remove('held'));
    worlds.forEach(w=>w.stage.classList.remove('pushing'));
  }
  function release(){clearForce();if(connected&&!busy)poll();}
  function refreshForce(){
    if(gesture)wanted=dragForce(gesture.dx,gesture.dy,gesture.camera.quaternion,$('plane').value,amplitude(),T);
    else if(buttonDirection){wanted=[0,0,0];wanted['XYZ'.indexOf(buttonDirection[1])]=amplitude()*(buttonDirection[0]==='+'?1:-1);}
    revision++;
  }
  const handlers={
    down(e,{stage,camera,tcp}){
      if(!connected||paused||latest?.stopped)return false;
      const rect=stage.getBoundingClientRect(),point=tcp.position.clone().project(camera);
      if(Math.hypot(e.clientX-rect.left-(point.x+1)*rect.width/2,e.clientY-rect.top-(1-point.y)*rect.height/2)>32)return false;
      clearForce();gesture={x:e.clientX,y:e.clientY,dx:0,dy:0,camera};stage.classList.add('pushing');return true;
    },
    move(e){if(!gesture)return;gesture.dx=e.clientX-gesture.x;gesture.dy=e.clientY-gesture.y;refreshForce();},end:release
  };
  initial.forEach(record=>{
    const w=buildWorld(record,handlers);w.tcp.scale.setScalar(2.3);
    w.card.querySelector('.metrics small').textContent='偏离原始姿态';
    w.card.querySelector('.legend').textContent='实线：参考位置误差 · 虚线：柔顺目标位移';
    worlds.push(w);histories.push([]);
  });
  function buttons(){['live-pause','live-reset','release'].forEach(id=>$(id).disabled=!connected);$('connect').disabled=connected||busy;$('connect').textContent=connected?'已连接':'连接实时仿真';}
  async function request(op,data){
    const response=await fetch('/api/soft-live/'+op,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:AbortSignal.timeout(14000)});
    const body=await response.json();if(!response.ok)throw Error(body.error||'实时服务连接失败');return body;
  }
  function accept(data){
    if(data.controllers.length!==initial.length||data.controllers.some((c,i)=>c.sha256!==initial[i].checkpoint_sha256))throw Error('服务端检查点与页面版本不一致，请刷新页面');
    if(latest&&data.time<latest.time)histories.forEach(h=>h.length=0);
    previous=latest;latest=data;received=performance.now();
    data.controllers.forEach((c,i)=>{const error=Math.hypot(...c.tcp.map((x,j)=>x-data.reference[j]))*1000,h=histories[i];h.push([data.time,error,Math.hypot(...data.force)*10]);while(h.length>240)h.shift();});
    if(data.stopped){paused=true;clearForce();$('live-pause').textContent='继续';}
    $('phase').textContent=data.stopped?'偏离超过 30 cm，三组已停止；请重置':paused?'已暂停 · 外力归零':Math.hypot(...data.force)>.001?'三个版本同时受力':'无外力 · 保持 / 恢复';
    $('clock').textContent=data.time.toFixed(2)+' s';$('force').textContent='实际外力 ['+data.force.map(x=>x.toFixed(2)).join(', ')+'] N';
  }
  async function connect(){
    if(busy||connected)return;busy=true;buttons();$('phase').textContent='正在连接 Isaac 仿真…';
    try{const data=await request('create',{noise:$('noise').value==='noisy'});session=data.session;connected=true;paused=false;clearForce();accept(data);$('live-pause').textContent='暂停';}
    catch(error){$('phase').textContent=error.message;}
    finally{busy=false;buttons();if(connected)timer=setTimeout(poll,100);}
  }
  async function poll(){
    clearTimeout(timer);if(!connected||busy)return;busy=true;
    const sentRevision=revision,doReset=reset;reset=false;
    try{accept(await request('step',{session,force:paused?[0,0,0]:wanted,paused,reset:doReset,noise:$('noise').value==='noisy'}));}
    catch(error){connected=false;clearForce();$('phase').textContent='连接中断，仿真撤力暂停：'+error.message;}
    finally{busy=false;buttons();if(connected)timer=setTimeout(poll,revision!==sentRevision?0:80);}
  }
  $('connect').onclick=connect;
  $('release').onclick=release;
  $('live-pause').onclick=()=>{paused=!paused;release();$('live-pause').textContent=paused?'继续':'暂停';};
  $('live-reset').onclick=()=>{reset=true;paused=false;histories.forEach(h=>h.length=0);release();$('live-pause').textContent='暂停';};
  $('noise').onchange=()=>{reset=true;release();};$('plane').onchange=release;
  $('amplitude').oninput=()=>{$('amplitude-slider').value=amplitude();refreshForce();};
  $('amplitude').onchange=()=>{$('amplitude').value=amplitude();};
  $('amplitude-slider').oninput=()=>{$('amplitude').value=$('amplitude-slider').value;refreshForce();};
  document.querySelectorAll('[data-push]').forEach(button=>{
    button.addEventListener('pointerdown',e=>{if(!connected||paused||latest?.stopped)return;clearForce();buttonDirection=button.dataset.push;button.classList.add('held');button.setPointerCapture(e.pointerId);refreshForce();if(!busy)poll();});
    for(const event of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(event,release);
    button.addEventListener('keydown',e=>{if((e.key===' '||e.key==='Enter')&&!e.repeat&&connected&&!paused){e.preventDefault();clearForce();buttonDirection=button.dataset.push;refreshForce();if(!busy)poll();}});
    button.addEventListener('keyup',e=>{if(e.key===' '||e.key==='Enter')release();});
  });
  window.addEventListener('blur',release);window.addEventListener('keydown',e=>{if(e.key==='Escape')release();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){paused=true;release();$('live-pause').textContent='继续';}});
  window.addEventListener('pagehide',()=>{clearForce();connected=false;clearTimeout(timer);if(session)fetch('/api/soft-live/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session}),keepalive:true}).catch(()=>{});});
  function chart(w,h){
    const canvas=w.card.querySelector('.plot'),width=canvas.clientWidth,dpr=Math.min(devicePixelRatio,2);
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==140*dpr){canvas.width=Math.round(width*dpr);canvas.height=140*dpr;}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,140);
    const end=latest?.time||0,start=Math.max(0,end-12),max=Math.max(50,...h.map(p=>p[1]+5)),x=t=>30+(t-start)/12*(width-40),y=n=>115-n/max*100;
    ctx.font='10px system-ui';ctx.fillStyle='#68776b';ctx.strokeStyle='#e0e7dc';
    for(const value of [0,max/2,max]){ctx.beginPath();ctx.moveTo(30,y(value));ctx.lineTo(width-10,y(value));ctx.stroke();ctx.fillText(value.toFixed(0),3,y(value)+3);}
    ctx.fillText('mm',3,10);ctx.fillText(start.toFixed(0)+' s',30,135);
    for(const [column,color,dash] of [[1,w.record.color,[]],[2,'#c77723',[4,4]]]){ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.beginPath();let first=true;for(const p of h){if(p[0]<start)continue;if(first){ctx.moveTo(x(p[0]),y(p[column]));first=false;}else ctx.lineTo(x(p[0]),y(p[column]));}ctx.stroke();}ctx.setLineDash([]);
  }
  let lastDraw=0;
  function draw(now){requestAnimationFrame(draw);if(now-lastDraw<33)return;lastDraw=now;
    worlds.forEach((w,i)=>{
      const c=latest?.controllers[i],old=previous?.controllers[i],fraction=Math.min(1,(now-received)/80);
      const initialCase=w.record.cases['+Z'];
      const blend=key=>c?(old?c[key].map((x,j)=>old[key][j]+fraction*(x-old[key][j])):c[key]):initialCase[key][0];
      const q=blend('q'),tcp=blend('tcp'),force=latest?.force||[0,0,0];
      renderPose(w,q,tcp,force);
      w.card.querySelector('.displacement').textContent=(Math.hypot(...tcp.map((x,j)=>x-w.record.reference[j]))*1000).toFixed(1)+' mm';
      w.card.querySelector('.error-value').textContent=(Math.hypot(...tcp.map((x,j)=>x-w.record.reference[j]-force[j]/100))*1000).toFixed(1)+' mm';chart(w,histories[i]);
    });
  }
  requestAnimationFrame(draw);connect();
}
