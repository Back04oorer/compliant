"""Local Isaac worker for the three frozen SoftMimic policies. No robot I/O."""
import argparse
from concurrent.futures import Future
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import threading
import time

import numpy as np

from .config import OUTPUT
from .evaluate import actor_observation

CHECKPOINTS = [('BalancedPush · 20000', OUTPUT/'BalancedPush_v1/model_19999.pt', '#2563eb'),
               ('BalancedPush · 8000', OUTPUT/'BalancedPush_v1/model_8000.pt', '#c77723'),
               ('FirstPPo · 24628', OUTPUT/'FirstPPo/model_24628.pt', '#9858a0')]


def controls(payload):
    force = np.asarray(payload.get('force',[0,0,0]), dtype=float)
    if force.shape!=(3,) or not np.isfinite(force).all() or np.linalg.norm(force)>5.000001:
        raise ValueError('外力必须为有限 XYZ 向量，合力不超过 5 N')
    for key in ('paused','reset','noise'):
        if key in payload and type(payload[key]) is not bool:raise ValueError(f'{key} must be boolean')
    return force


class Clock:
    """Request-paced physics; a disconnected browser never accumulates a push."""
    def __init__(self):self.last=None;self.remainder=0.
    def ticks(self,now,paused=False):
        elapsed=0 if self.last is None else max(0,now-self.last);self.last=now
        if paused or elapsed>.5:
            self.remainder=0.;return 0
        self.remainder+=min(.1,elapsed)
        n=min(5,int((self.remainder+1e-9)/.02));self.remainder-=n*.02
        return n


class Session:
    def __init__(self):self.token=None;self.seen=0.
    def authorize(self,operation,payload,now):
        if operation=='create':
            if self.token and now-self.seen<10:raise RuntimeError('另一个窗口正在控制仿真；关闭它后等待 10 秒再连接')
            self.token=secrets.token_hex(16)
        elif not self.token or payload.get('session')!=self.token or now-self.seen>10:
            raise KeyError('Session expired')
        self.seen=now
        if operation=='close':self.token=None


class Simulator:
    def __init__(self):
        import torch
        from rsl_rl.modules import ActorCritic
        from . import env as adapter
        from .asset import prepare_urdf
        self.torch=torch;torch.set_num_threads(2);torch.manual_seed(68123);np.random.seed(68123)
        model=prepare_urdf();manifests=[]
        for _,checkpoint,_ in CHECKPOINTS:
            manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
            assert manifest['actor_observation_dim']==117
            assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['model_sha256']
            for name in ('env.py','config.py'):
                assert hashlib.sha256(Path(adapter.__file__).with_name(name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
            manifests.append(manifest)
        cfg=adapter.PiperCfg();cfg.seed=68123;cfg.scene.num_envs=3
        self.env=adapter.PiperEnv(cfg);env=self.env;robot=env.robot;device=env.device
        ids=torch.arange(3)
        robot.root_physx_view.set_masses(robot.data.default_mass.cpu(),ids)
        robot.root_physx_view.set_inertias(robot.data.default_inertia.cpu(),ids)
        materials=robot.root_physx_view.get_material_properties().clone();materials[:]=materials[0].clone()
        robot.root_physx_view.set_material_properties(materials,ids)
        robot.write_joint_damping_to_sim(torch.tensor(adapter.KD,device=device)[None].expand(3,-1).clone())
        robot.write_joint_armature_to_sim(torch.full((3,6),.01,device=device))
        robot.write_joint_friction_coefficient_to_sim(torch.zeros(3,6,device=device))
        robot.write_joint_position_limit_to_sim(robot.data.default_joint_pos_limits)
        for prop in (robot.root_physx_view.get_masses(),robot.root_physx_view.get_inertias(),
                     robot.data.joint_damping,robot.data.joint_armature,robot.data.joint_effort_limits):
            torch.testing.assert_close(prop,prop[0:1].expand_as(prop),atol=0,rtol=0)
        self.k=torch.full((3,),100.,device=device);self.kr=torch.ones(3,device=device)
        self.force=torch.zeros(3,1,3,device=device);self.torque=torch.zeros_like(self.force)
        self.reference=np.load(OUTPUT/'evaluations/xyz_push_v1/protocol.npz')['reference_pos']
        self.policies=[];self.noise=False;self.stopped=False;self.time=0.
        self.reset()
        for (_,path,_),manifest in zip(CHECKPOINTS,manifests):
            config=dict(manifest['agent']['policy']);config.pop('class_name',None)
            obs={'policy':self.obs,'critic':torch.zeros(3,793,device=device)}
            policy=ActorCritic(obs,{'policy':['policy'],'critic':['critic']},6,**config).to(device)
            policy.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model_state_dict']);policy.eval();self.policies.append(policy)
        self.metadata=[dict(label=label,color=color,checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for label,path,color in CHECKPOINTS]

    def release(self):
        self.force.zero_()
        self.env.robot.set_external_force_and_torque(self.force,self.torque,body_ids=[self.env.tcp_id],is_global=True)

    def reset(self):
        env=self.env;torch=self.torch;self.release()
        env.robot.write_joint_state_to_sim(env.qref,torch.zeros_like(env.qref))
        env.robot.set_joint_position_target(env.qref)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(.005)
        env.actions.zero_();env.previous_actions.zero_();env.hist.zero_();env.new_history[:]=True
        self.obs=actor_observation(env,self.k,self.kr,self.noise)
        self.time=0.;self.stopped=False
        actual=(env.robot.data.body_pos_w[:,env.tcp_id]-env.scene.env_origins).cpu().numpy()
        assert np.max(np.abs(actual-self.reference))<1e-4

    def step(self,force,ticks):
        env=self.env;torch=self.torch
        self.force[:]=torch.as_tensor(force,device=env.device,dtype=torch.float32)
        if self.stopped:self.release();return
        # Isaac keeps mutable tensor caches across ticks and user-triggered resets.
        with torch.no_grad():
            for _ in range(ticks):
                actions=torch.cat([policy.act_inference({'policy':self.obs[i:i+1]}) for i,policy in enumerate(self.policies)])
                if not torch.isfinite(actions).all():self.stopped=True;self.release();break
                env.previous_actions.copy_(env.actions);env.actions.copy_(actions)
                env.robot.set_joint_position_target(env.qref+.5*actions)
                for _ in range(4):
                    env.robot.set_external_force_and_torque(self.force,self.torque,body_ids=[env.tcp_id],is_global=True)
                    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(.005)
                env.common_step_counter+=1;self.time+=.02
                self.obs=actor_observation(env,self.k,self.kr,self.noise)
                tcp=env.robot.data.body_pos_w[:,env.tcp_id]-env.scene.env_origins
                if not torch.isfinite(tcp).all() or torch.any(torch.linalg.vector_norm(tcp-torch.tensor(self.reference,device=env.device),dim=-1)>.3):
                    self.stopped=True;self.release();break

    def frame(self):
        env=self.env
        return dict(time=self.time,force=self.force[0,0].cpu().tolist(),stopped=self.stopped,noise=self.noise,
                    reference=self.reference.tolist(),controllers=[dict(**meta,q=q,tcp=tcp) for meta,q,tcp in zip(self.metadata,
                    env.robot.data.joint_pos.cpu().tolist(),(env.robot.data.body_pos_w[:,env.tcp_id]-env.scene.env_origins).cpu().tolist())])


def serve(simulator,port):
    jobs=queue.Queue(maxsize=8);session=Session();clock=Clock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                if self.path not in ('/create','/step','/close'):raise ValueError('Unknown operation')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid payload size')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('Expected object')
                controls(payload)
                future=Future();jobs.put_nowait((self.path[1:],payload,future))
                body=future.result(timeout=10);status=200
            except KeyError:body={'error':'会话已过期，请重新连接'};status=410
            except (ValueError,TypeError):body={'error':'无效的施力参数（合力范围 0–5 N）'};status=400
            except Exception as error:body={'error':str(error) or '实时服务繁忙'};status=503
            encoded=json.dumps(body,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers()
            try:self.wfile.write(encoded)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'LIVE_READY 127.0.0.1:{port}',flush=True)
    try:
        while True:
            try:operation,payload,future=jobs.get(timeout=.1)
            except queue.Empty:
                if time.monotonic()-session.seen>.5:simulator.release()
                continue
            try:
                now=time.monotonic();session.authorize(operation,payload,now)
                if operation=='close':simulator.release();future.set_result({'closed':True});continue
                noise=payload.get('noise',False)
                if operation=='create' or payload.get('reset') or noise!=simulator.noise:
                    simulator.noise=noise;simulator.reset();clock=Clock()
                paused=payload.get('paused',False)
                simulator.step(np.zeros(3) if paused else controls(payload),clock.ticks(now,paused))
                result=simulator.frame();result['session']=session.token;future.set_result(result)
            except Exception as error:simulator.release();future.set_exception(error)
    finally:server.shutdown();server.server_close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--accept-eula',action='store_true');parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    if not args.accept_eula:parser.error('--accept-eula required')
    os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    simulator=None
    try:simulator=Simulator();serve(simulator,args.port)
    except KeyboardInterrupt:pass
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if simulator:simulator.env.close()
        app.close()


if __name__=='__main__':main()
