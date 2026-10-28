"""IsaacLab Piper adapter; original SoftMimic rewards and forcefield kernels."""
import json
from types import SimpleNamespace

import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.envs import mdp

from .config import OUTPUT, DATASET, KEYPOINTS
from .asset import prepare_urdf
from .upstream import module, force_methods

Q_REF=[0.,2.,-2.,.2,-.8,0.]
KP=[60.,80.,60.,20.,15.,10.];KD=[4.,5.,4.,1.5,1.,.8];LIMIT=[12.,12.,12.,5.,5.,5.]
JOINTS=[f'joint{i}' for i in range(1,7)]

def history_indices(widths,length=3):
    indices=[];offset=0;frame=sum(widths)
    for width in widths:
        indices.extend(t*frame+offset+i for t in range(length) for i in range(width))
        offset+=width
    return indices
# IsaacLab 2.1 broadcast quat_rotate over keypoints; 2.3's replacement flattens first.
# Restore only that API behavior, retaining the official reward/force equations.
def rotate_compat(q,v):
    shape=torch.broadcast_shapes(q.shape[:-1],v.shape[:-1])
    return math_utils.quat_apply(q.expand(*shape,4),v.expand(*shape,3))
compat_math=SimpleNamespace(**vars(math_utils))
compat_math.quat_rotate=rotate_compat
_force=force_methods({'torch':torch,'math_utils':compat_math})
rewards=module('official_piper_rewards','softmimic_gym/softmimic_gym/tasks/locomotion/tracking/mdp/rewards.py')
terminations=module('official_piper_terminations','softmimic_gym/softmimic_gym/tasks/locomotion/tracking/mdp/terminations.py')
rewards.math_utils=compat_math;terminations.math_utils=compat_math


@configclass
class PiperCfg(DirectRLEnvCfg):
    decimation=4
    episode_length_s=30.
    action_space=6
    observation_space=117
    state_space=793
    sim=sim_utils.SimulationCfg(dt=.005,render_interval=4)
    scene=InteractiveSceneCfg(num_envs=4096,env_spacing=2.,replicate_physics=True,clone_in_fabric=True)
    robot_cfg=ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UrdfFileCfg(asset_path=str(OUTPUT/'piper_fixed.urdf'),fix_base=True,merge_fixed_joints=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.)),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        soft_joint_pos_limit_factor=.9,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS,Q_REF))),
        actuators={name:ImplicitActuatorCfg(joint_names_expr=[name],effort_limit_sim=limit,stiffness=kp,damping=kd,armature=.01) for name,kp,kd,limit in zip(JOINTS,KP,KD,LIMIT)},
    )


def dense_forcefield_force(command):
    # Same release equations; fixed-shape masks avoid host decisions and dynamic index sizes.
    rows=torch.arange(command.num_envs,device=command.device)
    position=command.robot.data.body_pos_w[rows,command.body_indices]
    stiffness=command._ff_stiffness[:,0,0]
    origin=command._ff_origin[:,0];normal=command._ff_normal[:,0]
    is_plane=torch.linalg.norm(normal,dim=-1)>.1
    penetration=-torch.sum((position-origin)*normal,dim=-1)
    magnitude=torch.clamp(penetration,min=0.)*stiffness
    plane_force=magnitude[:,None]*normal
    setpoint_force=stiffness[:,None]*(origin-position)
    result=torch.where(is_plane[:,None],plane_force,setpoint_force)
    return torch.where(command.active_force_mask[:,None],result,torch.zeros_like(result))


def dense_forcefield_torque(command):
    rows=torch.arange(command.num_envs,device=command.device)
    quat=command.robot.data.body_quat_w[rows,command.body_indices]
    stiffness=command._ff_rotational_stiffness[:,0,0]
    target=command._ff_setpoint_rot[:,0][...,[3,0,1,2]]
    error=math_utils.quat_mul(target,math_utils.quat_inv(quat))
    angle,axis=_force['quat_to_angle_axis'](error)
    result=stiffness[:,None]*(axis*angle[:,None])
    return torch.where(command.active_force_mask[:,None],result,torch.zeros_like(result))


class Command:
    _compute_reactive_forcefield_force_w=dense_forcefield_force
    _compute_reactive_forcefield_torque_w=dense_forcefield_torque
    @property
    def forcefield_forces_w(self):return self._compute_reactive_forcefield_force_w()
    @property
    def forcefield_torques_w(self):return self._compute_reactive_forcefield_torque_w()
    @property
    def target_forces_w(self):return self._force_vector_world[:,1]
    @property
    def target_torques_w(self):return self._torque_vector_world[:,1]


class PiperEnv(DirectRLEnv):
    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot_cfg)
        sim_utils.spawn_ground_plane('/World/Ground',sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot']=self.robot
        sim_utils.DomeLightCfg(intensity=1500.).func('/World/Light',sim_utils.DomeLightCfg(intensity=1500.))

    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        assert self.robot.num_joints==6
        ids,_=self.robot.find_joints(JOINTS,preserve_order=True);assert ids==list(range(6)),self.robot.joint_names
        self.key_id_list=self.robot.find_bodies(list(KEYPOINTS),preserve_order=True)[0]
        self.key_ids=torch.tensor(self.key_id_list,device=self.device)
        self.tcp_id=int(self.key_ids[-1])
        data=np.load(DATASET/'train.npz')
        self.dataset={key:torch.as_tensor(data[key],device=self.device,dtype=torch.long if key=='lengths' else torch.float32) for key in data.files}
        self.data_dt=json.loads((DATASET/'manifest.json').read_text())['dt']
        # The official CSV library maps env id modulo files; load_motions is a no-op.
        self.clip=torch.arange(self.num_envs,device=self.device)%self.dataset['q_aug'].shape[0]
        max_frames=(((self.dataset['lengths'][self.clip]-1)*self.data_dt)/.02).long()-1
        self.phase=(torch.rand(self.num_envs,device=self.device)*max_frames).long()*.02
        self.motion_time=torch.zeros_like(self.phase)
        self._samples={};self._indices={}
        self.future_offsets=torch.tensor([s*.02 for s in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]],device=self.device)
        self.last_damping_step=torch.full((self.num_envs,),-720,device=self.device,dtype=torch.long)
        self.qref=torch.tensor(Q_REF,device=self.device).expand(self.num_envs,-1)
        self.actions=torch.zeros(self.num_envs,6,device=self.device);self.previous_actions=self.actions.clone()
        self.command=Command();c=self.command;c.device=self.device;c.robot=self.robot;c.num_envs=self.num_envs
        c.keypoint_body_ids_tensor=self.key_ids;c.body_indices=torch.full((self.num_envs,),self.tcp_id,device=self.device,dtype=torch.long)
        c.ff_anchor_rot=torch.tensor([1.,0.,0.,0.],device=self.device).expand(self.num_envs,-1)
        c.ff_anchor_pos=self.scene.env_origins;c.ff_anchor_ref_pos=self.scene.env_origins
        self.command_manager=SimpleNamespace(get_term=lambda _:self.command)
        self.action_manager=SimpleNamespace(action=self.actions,prev_action=self.previous_actions,total_action_dim=6)
        self.hist=torch.zeros(self.num_envs,3,39,device=self.device)
        self.critic_hist=torch.zeros(self.num_envs,3,91,device=self.device)
        widths=[3,3,6,6,6,6,2,1,1,3,1,1]
        self.actor_history_order=torch.tensor(history_indices(widths),device=self.device)
        self.critic_history_order=torch.tensor(history_indices(widths+[3,3,1,1,3,3,6,2,1,1,3,3,3,15,3,1]),device=self.device)
        self.new_history=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.last_obs=-1;self.obs_cache=None
        entity=SceneEntityCfg('robot',joint_names='.*');entity.resolve(self.scene);self.asset_cfg=entity
        mass_params={'asset_cfg':entity,'mass_distribution_params':(.7,1.3),'operation':'scale'}
        randomizer=mdp.randomize_rigid_body_mass(EventTermCfg(func=mdp.randomize_rigid_body_mass,params=mass_params),self)
        randomizer(self,None,**mass_params)
        material_params={'asset_cfg':entity,'static_friction_range':(.5,2.),'dynamic_friction_range':(.5,2.),'restitution_range':(0.,.5),'num_buckets':64}
        material_randomizer=mdp.randomize_rigid_body_material(EventTermCfg(func=mdp.randomize_rigid_body_material,params=material_params),self)
        material_randomizer(self,None,**material_params)
        self.robot.write_joint_armature_to_sim(torch.rand_like(self.qref)*.09+.02)
        self.robot.write_joint_friction_coefficient_to_sim(torch.rand_like(self.qref)*.01)
        limits=self.robot.data.default_joint_limits.clone()+(torch.rand_like(self.robot.data.default_joint_limits)*.04-.02)
        self.robot.write_joint_position_limit_to_sim(limits)
        self._refresh()

    def sample(self,key,offset=0.):
        cache_key=(key,offset)
        if cache_key in self._samples:return self._samples[cache_key]
        if offset not in self._indices:
            index=(self.phase+self.motion_time+offset)/self.data_dt
            lengths=self.dataset['lengths'][self.clip]-1
            index=torch.minimum(index.clamp(min=0),lengths.float())
            first=index.long();second=torch.minimum(first+1,lengths);blend=index-first
            self._indices[offset]=(first,second,blend)
        first,second,blend=self._indices[offset]
        a=self.dataset[key][self.clip,first];b=self.dataset[key][self.clip,second]
        if key in ('quats',):
            sign=torch.where((a*b).sum(-1,keepdim=True)<0,-1.,1.);b=b*sign
        blend=blend.reshape((self.num_envs,)+(1,)*(a.ndim-1))
        result=a*(1-blend)+b*blend
        if key=='quats':result=torch.nn.functional.normalize(result,dim=-1)
        self._samples[cache_key]=result
        return result

    def future_q(self):
        index=(self.phase[:,None]+self.motion_time[:,None]+self.future_offsets[None])/self.data_dt
        lengths=self.dataset['lengths'][self.clip,None]-1
        index=torch.minimum(index.clamp(min=0),lengths.float())
        first=index.long();second=torch.minimum(first+1,lengths);blend=(index-first)[:,:,None]
        a=self.dataset['q_aug'][self.clip[:,None],first];b=self.dataset['q_aug'][self.clip[:,None],second]
        return a*(1-blend)+b*blend

    def _clear_samples(self):
        self._samples.clear();self._indices.clear()

    def _refresh(self):
        c=self.command;meta=self.sample('metadata')
        c.relative_timestamp=self.motion_time
        c.dof_pos=self.sample('q_aug')[:,None].expand(-1,2,-1)
        c.keypoints=(self.sample('keypoints')+self.scene.env_origins[:,None])[:,None].expand(-1,2,-1,-1)
        c.adapted_keypoints=c.keypoints
        c.keypoint_rotations=self.sample('quats')[:,None].expand(-1,2,-1,-1)
        c.adapted_keypoint_rotations=c.keypoint_rotations
        c._force_vector_world=self.sample('force')[:,None].expand(-1,2,-1)
        c._torque_vector_world=self.sample('torque')[:,None].expand(-1,2,-1)
        c.active_force_mask=c.target_forces_w.norm(dim=-1)>1e-6
        c.force_link_id=torch.full((self.num_envs,2,1),self.tcp_id,device=self.device,dtype=torch.long)
        c._stiffness=self.sample('k')[:,None]
        c._ff_stiffness=meta[:,0,None,None];c._ff_rotational_stiffness=meta[:,1,None,None]
        c._ff_origin=(meta[:,2:5]+self.scene.env_origins)[:,None];c._ff_normal=meta[:,9:12,None].transpose(1,2)
        c._ff_setpoint_rot=meta[:,5:9,None].transpose(1,2)
        c.root_state={'root_pos':self.scene.env_origins[:,None].expand(-1,2,-1),
                      'root_rot':torch.tensor([0.,0.,0.,1.],device=self.device).expand(self.num_envs,2,4)}

    def _pre_physics_step(self,actions):
        # Official demo_reset_robot_on_clip_end starts a fresh motion and resets to its reference.
        ended=self.phase+self.motion_time >= (self.dataset['lengths'][self.clip]-1)*self.data_dt
        ids=ended.nonzero().flatten()
        if len(ids):
            self.phase[ids]=0.;self.motion_time[ids]=0.
            self._clear_samples()
            self.robot.write_joint_state_to_sim(self.sample('q_aug')[ids],self.sample('dq_aug')[ids],env_ids=ids)
        self.previous_actions.copy_(self.actions);self.actions.copy_(actions)
        # ManagerBasedRLEnv evaluates interval force events once per control step.
        self._refresh()
        self.held_force=self.command.forcefield_forces_w
        self.held_torque=self.command.forcefield_torques_w

    def _apply_action(self):
        self.robot.set_joint_position_target(self.qref+.5*self.actions)
        self.robot.set_external_force_and_torque(self.held_force[:,None],self.held_torque[:,None],body_ids=[self.tcp_id],is_global=True)

    def _get_dones(self):
        self.motion_time+=.02
        self._clear_samples()
        self._refresh()
        done=terminations.bad_keypoint_deviation_local(self,threshold=.5,adaptive=False,command_name='reference',keypoint_body_ids=self.key_id_list,asset_cfg=self.asset_cfg)
        done|=~torch.isfinite(self.robot.data.joint_pos).all(-1)
        return done,self.episode_length_buf>=self.max_episode_length-1

    def _get_rewards(self):
        c=self.command
        values={
            'force_position':3*rewards.force_link_keypoint_tracking_local(self,'reference',sigma=.1),
            'force_orientation':3*rewards.force_link_keypoint_orientation_tracking_local(self,'reference',sigma=.1),
            'force':2*rewards.force_command_tracking(self,'reference',sigma=20.),
            'torque':2*rewards.torque_command_tracking(self,'reference',sigma=2.),
            'keypoints':2*rewards.keypoint_deviation_from_command_local(self,'reference',self.key_id_list,sigma=.3),
            'keypoint_rot':2*rewards.keypoint_orientation_deviation_from_command_local(self,'reference',self.key_id_list,sigma=.4),
            'joint':2*rewards.joint_deviation_from_command_exp(self,'reference',asset_cfg=self.asset_cfg),
            'alive':1.5*(~self.reset_terminated).float(),
            'joint_limits':-10*mdp.joint_pos_limits(self,asset_cfg=self.asset_cfg),
            'velocity':-2.8e-4*self.robot.data.joint_vel.square().sum(-1),
            'action_rate':-.01*(self.actions-self.previous_actions).square().sum(-1),
        }
        self.extras['log']={f'Reward/{name}':value.mean() for name,value in values.items()}
        self.extras['log']['Metrics/target_error_m']=(self.robot.data.body_pos_w[:,self.tcp_id]-c.keypoints[:,1,-1]).norm(dim=-1).mean()
        self.extras['log']['Metrics/force_error_N']=(c.forcefield_forces_w-c.target_forces_w).norm(dim=-1).mean()
        # IsaacLab's RewardManager multiplies every reward term by control dt.
        return sum(values.values())*.02

    def _get_observations(self):
        if self.last_obs==self.common_step_counter and self.obs_cache is not None:return self.obs_cache
        self._refresh();n=self.num_envs;device=self.device;c=self.command
        zero=torch.zeros(n,3,device=device);gravity=zero.clone();gravity[:,2]=-1
        noisy=lambda x,amplitude:x+(torch.rand_like(x)*2-1)*amplitude
        parts=[noisy(zero,.2)*.25,noisy(gravity,.01),noisy(self.robot.data.joint_pos-self.qref,.01),noisy(self.robot.data.joint_vel,1.5)*.05,
               self.actions,self.qref,zero[:,:2],zero[:,:1],zero[:,:1],gravity,torch.log(self.sample('k')[:,None]+1e-6),torch.log(self.sample('kr')[:,None]+1e-6)]
        frame=torch.cat(parts,-1);assert frame.shape[1]==39
        current=self.robot.data.body_pos_w[:,self.tcp_id];target=c.keypoints[:,1,-1]
        quat=self.robot.data.body_quat_w[:,self.tcp_id]
        desired=c.keypoint_rotations[:,1,-1];errquat=math_utils.quat_mul(desired,math_utils.quat_conjugate(quat))
        errquat=torch.where(errquat[:,0,None]<0,-errquat,errquat)
        actual_f=math_utils.quat_apply_inverse(quat,c.forcefield_forces_w)/50
        actual_t=math_utils.quat_apply_inverse(quat,c.forcefield_torques_w)/5
        target_f=math_utils.quat_apply_inverse(quat,c.target_forces_w)/50
        target_t=math_utils.quat_apply_inverse(quat,c.target_torques_w)/5
        key_error=(self.robot.data.body_pos_w[:,self.key_ids]-c.keypoints[:,1]).reshape(n,-1)
        privilege=[actual_f,actual_t,torch.log(c._ff_stiffness[:,0]+1e-6),torch.log(c._ff_rotational_stiffness[:,0]+1e-6),target_f,target_t,
            self.sample('q_aug'),zero[:,:2],zero[:,:1],zero[:,:1],gravity,(target-current)*c.active_force_mask[:,None],2*errquat[:,1:],key_error,zero,zero[:,:1]]
        critic_parts=parts.copy()
        critic_parts[:4]=[noisy(zero,.2)*.25,noisy(gravity,.01),noisy(self.robot.data.joint_pos-self.qref,.01),noisy(self.robot.data.joint_vel,1.5)*.05]
        critic_frame=torch.cat([*critic_parts,*privilege],-1);assert critic_frame.shape[1]==91,critic_frame.shape
        self.hist=torch.roll(self.hist,-1,dims=1);self.hist[:,-1]=frame
        self.critic_hist=torch.roll(self.critic_hist,-1,dims=1);self.critic_hist[:,-1]=critic_frame
        self.hist[self.new_history]=frame[self.new_history,None];self.critic_hist[self.new_history]=critic_frame[self.new_history,None];self.new_history[:]=False
        # Official ObservationManager concatenates each term's history before the next term.
        actor=self.hist.reshape(n,-1)[:,self.actor_history_order]
        critic=self.critic_hist.reshape(n,-1)[:,self.critic_history_order]
        ref_future=torch.cat([self.qref[:,None].expand(-1,20,-1).reshape(n,-1),torch.zeros(n,80,device=device),gravity[:,None].expand(-1,20,-1).reshape(n,-1)],-1)
        aug_future=torch.cat([self.future_q().reshape(n,-1),torch.zeros(n,80,device=device),gravity[:,None].expand(-1,20,-1).reshape(n,-1)],-1)
        critic=torch.cat([critic,ref_future,aug_future],-1)
        assert actor.shape[1]==117 and critic.shape[1]==793,(actor.shape,critic.shape)
        self.obs_cache={'policy':actor,'critic':critic};self.last_obs=self.common_step_counter
        return self.obs_cache

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self,'clip'):return
        # Official demo_random_reset=False, demo_zero_reset=False: retain motion clock.
        self.episode_length_buf[env_ids]=0
        q=self.sample('q_aug')[env_ids]+(torch.rand(len(env_ids),6,device=self.device)*.4-.2)
        limits=self.robot.data.soft_joint_pos_limits[env_ids];q=q.clamp(limits[:,:,0],limits[:,:,1])
        velocity_limit=self.robot.data.soft_joint_vel_limits[env_ids]
        velocity=self.sample('dq_aug')[env_ids].clamp(-velocity_limit,velocity_limit)
        self.robot.write_joint_state_to_sim(q,velocity,env_ids=env_ids)
        self.robot.set_joint_position_target(q,env_ids=env_ids)
        eligible=env_ids[self.common_step_counter-self.last_damping_step[env_ids]>=720]
        if len(eligible):
            kd=torch.tensor(KD,device=self.device)[None]+torch.rand(len(eligible),6,device=self.device)*2
            self.robot.write_joint_damping_to_sim(kd,env_ids=eligible)
            self.last_damping_step[eligible]=self.common_step_counter
        self.actions[env_ids]=0;self.previous_actions[env_ids]=0
        self.new_history[env_ids]=True;self.last_obs=-1
