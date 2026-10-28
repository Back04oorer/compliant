"""Use the original sampler/runner/release logic with Piper-specific Mink tasks."""
import argparse
import contextlib
import importlib
import hashlib
import io
import json
import random
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim.piper_compliance.kinematics import KP, Q_REF, TORQUE_LIMIT, load_model
from .config import DATASET, KEYPOINTS, OUTPUT, UPSTREAM, UPSTREAM_COMMIT, ADAPTATIONS

DT = 1/30


def prepare_model():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    model=load_model()
    path=OUTPUT/'piper_mink.xml';mujoco.mj_saveLastXML(str(path),model)
    root=ET.parse(path).getroot()
    wrist=next(body for body in root.iter('body') if body.get('name')=='link6')
    ET.SubElement(wrist,'body',name='tcp',pos='0 0 0.16')
    # The upstream zero-wrench fallback names a torso; a massless fixed alias is a name adapter only.
    ET.SubElement(root.find('worldbody'),'body',name='torso_link')
    ET.ElementTree(root).write(path)
    return path


class PiperIK:
    def __init__(self,*args,**kwargs):
        self.model=mujoco.MjModel.from_xml_path(str(OUTPUT/'piper_mink.xml'))
        self.configuration=mink.Configuration(self.model)
        self.configuration.update(q=Q_REF)
        self.data=self.configuration.data
        self.total_mass=float(self.model.body_mass.sum())
        self.actuated_qpos_indices=list(range(6));self.num_dofs=6
        self.posture_task=mink.PostureTask(self.model,cost=1e-4)
        self.keypoint_tasks={name:mink.FrameTask(name,'body',position_cost=.01,orientation_cost=.01) for name in KEYPOINTS}
        self.force_tasks={name:mink.FrameTask(name,'body',position_cost=5.,orientation_cost=5.) for name in KEYPOINTS}
        self.limits=[mink.ConfigurationLimit(self.model)]
        self.motion_lib=types.SimpleNamespace(get_max_times=lambda _:np.array(20.))

    def get_reference_motion(self,t):
        return Q_REF.copy(),np.zeros(6),np.zeros(2)


def ik_step(solver,qref,body,force,torque,com_force,k,kr,dt,overrides=None):
    reference=mink.Configuration(solver.model);reference.update(q=qref)
    tasks=[solver.posture_task];solver.posture_task.set_target(qref)
    active=np.linalg.norm(force)>.01 or np.linalg.norm(torque)>.01
    overrides=overrides or {}
    target_pos=reference.data.body(body).xpos.copy()
    target_rot=Rotation.from_matrix(reference.data.body(body).xmat.reshape(3,3))
    for name,task in solver.keypoint_tasks.items():
        if name in overrides:
            pos,rot=overrides[name];quat=rot.as_quat()[[3,0,1,2]]
            current=solver.force_tasks[name];current.set_target(mink.SE3(np.r_[quat,pos]));tasks.append(current)
        elif not (active and name==body):
            task.set_target(reference.get_transform_frame_to_world(name,'body'));tasks.append(task)
    if active:
        target_pos+=force/max(k,1e-6)
        if np.linalg.norm(torque)>1e-4 and kr>1e-4:target_rot=Rotation.from_rotvec(torque/kr)*target_rot
        solver.force_tasks[body].set_target(mink.SE3(np.r_[target_rot.as_quat()[[3,0,1,2]],target_pos]))
        tasks.append(solver.force_tasks[body])
    elif body in overrides:target_pos,target_rot=overrides[body]
    velocity=mink.solve_ik(solver.configuration,tasks,dt,solver='daqp',limits=solver.limits,damping=1e-5)
    solver.configuration.integrate_inplace(velocity,dt);mujoco.mj_forward(solver.model,solver.data)
    return target_pos,target_rot,velocity


def feasible(data,ref,body,force,torque,k,kr,mass):
    # Same release bounds; stance feet and CoM are inapplicable to a bolted-down arm.
    pos_error=np.linalg.norm(data.body(body).xpos-ref.body(body).xpos-force/max(k,1e-6))
    valid=(np.linalg.norm(force)<=140 and np.linalg.norm(force)/k<=.7 and
           np.linalg.norm(torque)<=10 and np.linalg.norm(torque)/kr<=2 and pos_error<=.05)
    jac=np.zeros((3,6));rot=jac.copy();mujoco.mj_jacBody(data.model,data,jac,rot,data.body(body).id)
    motor=data.qfrc_bias-jac.T@force-rot.T@torque
    command=data.qpos+motor/KP
    valid=valid and np.all(np.abs(motor)<=TORQUE_LIMIT) and np.all(command>=data.model.jnt_range[:,0]) and np.all(command<=data.model.jnt_range[:,1])
    return bool(valid),{'link_tracking_error':float(pos_error)}


def official_modules():
    sys.path.insert(0,str(UPSTREAM/'compliant_motion_augmentation'))
    sys.path.insert(0,str(UPSTREAM))
    # Replace the G1-specific constructor, leaving original event generation and update code intact.
    ik=types.ModuleType('ik_solver');ik.G1_Mink_IK_Solver=PiperIK;sys.modules['ik_solver']=ik
    update=importlib.import_module('ik_update')
    update.perform_mink_ik_step=ik_step;update.KEYPOINT_BODY_NAMES=list(KEYPOINTS)
    runner=importlib.import_module('runner')
    runner.G1_Mink_IK_Solver=PiperIK;runner.FORCEABLE_LINKS=['tcp'];runner.is_ik_solution_feasible=feasible
    return runner,importlib.import_module('config').SimulationConfig


def generate(count=40):
    DATASET.mkdir(parents=True,exist_ok=True)
    if (DATASET/'manifest.json').exists():raise ValueError('Refusing to overwrite official-method dataset')
    prepare_model();runner,Config=official_modules()
    arrays=[];stats=[]
    class Adapter(runner.SimulationRunner):
        def _record_step(self,step,qref,contacts,vis,t):
            q=self.ik_solver.configuration.q.copy()
            self.all_reference_qpos.append(qref.copy())
            self.all_adapted_qpos.append(q)
            # Only root/joint column layout changes; keep original force/field metadata.
            target=self.current_event or (self.event_queue[0] if self.event_queue else None)
            if target:
                k,kr=target['stiffness'],target.get('rotational_stiffness',1.)
                link=self.ik_solver.model.body(target['link_name']).id
            else:
                k=self.stiffness_state['stiffness'] if self.config.force_mode=='zero-wrench' else 140.
                kr=self.stiffness_state['rot_stiffness'] if self.config.force_mode=='zero-wrench' else 1.;link=-1
            self.all_force_data.append(np.r_[link,vis[1],vis[2],k,kr])
            event=self.current_event
            if event:
                pos=event.get('forcefield_setpoint_pos',np.zeros(3))
                quat=event['forcefield_setpoint_rot'].as_quat() if 'forcefield_setpoint_rot' in event else [0,0,0,1]
                meta=np.r_[event.get('forcefield_stiffness',0),event.get('rotational_forcefield_stiffness',0),pos,quat,np.zeros(3)]
            else:meta=np.r_[0,0,np.zeros(3),[0,0,0,1],np.zeros(3)]
            self.all_collision_metadata.append(meta)
        def _save_outputs(self):pass
    for mode in ('forcefield','collision-emulator','zero-wrench'):
        # Release hybrid has 40 force + 40 collision + 5 zero clips repeated eight times.
        unique=count if mode!='zero-wrench' else max(1,count//8)
        for clip in range(unique):
            cfg=Config(mode='generate',model_path=str(OUTPUT/'piper_mink.xml'),motion_path='piper_fixed.csv',seed=20261027+clip+10000*len(stats),com_cost=0.,com_cost_z_factor=0.,force_mode=mode,output_dir=str(DATASET))
            with contextlib.redirect_stdout(io.StringIO()):
                experiment=Adapter(cfg,clip);experiment.run()
            q=np.asarray(experiment.all_adapted_qpos)
            refs=np.asarray(experiment.all_reference_qpos)
            force=np.asarray(experiment.all_force_data);metadata=np.asarray(experiment.all_collision_metadata)
            fk=[];quats=[]
            for pose in q:
                experiment.ik_solver.data.qpos[:]=pose;mujoco.mj_forward(experiment.ik_solver.model,experiment.ik_solver.data)
                fk.append([experiment.ik_solver.data.body(name).xpos.copy() for name in KEYPOINTS])
                quats.append([experiment.ik_solver.data.body(name).xquat.copy() for name in KEYPOINTS])
            item={'q_aug':q,'q_ref':refs,'dq_aug':np.gradient(q,DT,axis=0),'force':force[:,1:4],'torque':force[:,4:7],
                  'k':force[:,7],'kr':force[:,8],'metadata':metadata,'keypoints':np.asarray(fk),'quats':np.asarray(quats)}
            arrays.extend([item]*(8 if mode=='zero-wrench' and count>=8 else 1))
            stats.append({'mode':mode,'seed':cfg.seed,'events':len(experiment.force_profile),'active_frames':int((np.linalg.norm(item['force'],axis=1)>.01).sum())})
            print(f'{mode} {clip+1}/{unique}, active frames {stats[-1]["active_frames"]}',flush=True)
    lengths=np.array([len(a['q_aug']) for a in arrays],dtype=np.int64)
    maximum=int(lengths.max())
    data={key:np.stack([np.pad(a[key],[(0,maximum-len(a[key]))]+[(0,0)]*(a[key].ndim-1),mode='edge') for a in arrays]) for key in arrays[0]}
    data['lengths']=lengths
    np.savez_compressed(DATASET/'train.npz',**data)
    manifest={'upstream_commit':UPSTREAM_COMMIT,'adaptations':ADAPTATIONS,'frames':data['q_aug'].shape[1],
        'trajectories':len(arrays),'dt':DT,'sha256':hashlib.sha256((DATASET/'train.npz').read_bytes()).hexdigest(),'clips':stats}
    (DATASET/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=40);args=p.parse_args();generate(args.count)
