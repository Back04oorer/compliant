"""GPU preflight: model FK, actor privilege boundary, and finite physics/rewards."""
import argparse
import json
import os
from pathlib import Path
from .config import OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--accept-eula',action='store_true')
    p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,default=OUTPUT/'gpu_verification.json')
    args=p.parse_args()
    if not args.accept_eula:p.error('--accept-eula is required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    env=None
    try:
        import torch
        from .asset import prepare_urdf
        from . import env as adapter
        if args.dataset is not None:adapter.DATASET=args.dataset.absolute()
        PiperEnv,PiperCfg,Command,_force=adapter.PiperEnv,adapter.PiperCfg,adapter.Command,adapter._force
        prepare_urdf();cfg=PiperCfg();cfg.scene.num_envs=16;cfg.seed=0
        env=PiperEnv(cfg)
        torch.testing.assert_close(env.robot.data.joint_vel_limits[0],torch.tensor([5.,5.,5.,5.,5.,3.],device=env.device))
        env.reset()
        env.robot.write_joint_state_to_sim(env.qref,torch.zeros_like(env.qref))
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(.005)
        actual=env.robot.data.body_pos_w[0,env.key_ids]-env.scene.env_origins[0]
        reference=env.dataset['keypoints'][0,0]
        fk_error=float((actual-reference).abs().max())
        print('FK comparison',env.robot.data.joint_pos[0].tolist(),actual.tolist(),reference.tolist(),flush=True)
        assert fk_error<.0001,f'Isaac/MuJoCo FK differs by {fk_error} m'
        boundary_force_error=None
        if args.dataset is not None:
            boundary_force_error=0.
            for time in (5.99,6.01,6.5,17.99,18.01,18.5,19.01):
                env.phase[:]=time;env.motion_time[:]=0;env._clear_samples();env._refresh()
                env.robot.write_joint_state_to_sim(env.sample('q_aug'),torch.zeros_like(env.qref))
                env.scene.write_data_to_sim();env.sim.forward();env.scene.update(.005)
                error=float((env.command.forcefield_forces_w-env.command.target_forces_w).norm(dim=-1).max())
                boundary_force_error=max(boundary_force_error,error)
            assert boundary_force_error<.15,f'Interpolated forcefield target mismatch: {boundary_force_error} N'
        # Compare dense kernels with the pinned oracle, including modes absent from static-pose data.
        from types import SimpleNamespace
        command=Command();command.num_envs=16;command.device=env.device
        command.body_indices=torch.zeros(16,device=env.device,dtype=torch.long)
        position=torch.sin(torch.arange(48,device=env.device).float()).reshape(16,1,3)
        quaternion=torch.nn.functional.normalize(torch.cos(torch.arange(64,device=env.device).float()).reshape(16,1,4),dim=-1)
        command.robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=position,body_quat_w=quaternion))
        command._stiffness=torch.ones(16,1,device=env.device)*100
        command._ff_stiffness=torch.arange(16,device=env.device).float()[:,None,None]*10
        command._ff_rotational_stiffness=command._ff_stiffness*.01
        command._ff_origin=position*.5
        command._ff_normal=torch.zeros_like(position);command._ff_normal[::2,0,0]=1.
        command._ff_setpoint_rot=quaternion.roll(1,0)[...,[1,2,3,0]]
        for mask in [torch.ones(16,device=env.device,dtype=torch.bool),torch.zeros(16,device=env.device,dtype=torch.bool),torch.arange(16,device=env.device)%3==0]:
            command.active_force_mask=mask
            # Warm both script specializations: even the unchanged torque kernel varies during JIT warmup.
            for _ in range(8):
                command.forcefield_torques_w
                _force['_compute_reactive_forcefield_torque_w'](command)
            torch.testing.assert_close(command.forcefield_forces_w,_force['_compute_reactive_forcefield_force_w'](command),rtol=0,atol=0)
            torch.testing.assert_close(command.forcefield_torques_w,_force['_compute_reactive_forcefield_torque_w'](command),rtol=0,atol=0)
        obs,_=env.reset()
        original=env.sample
        buffers=[env.hist.clone(),env.critic_hist.clone(),env.new_history.clone()]
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        env.last_obs=-1;first=env._get_observations()
        env.hist.copy_(buffers[0]);env.critic_hist.copy_(buffers[1]);env.new_history.copy_(buffers[2])
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        env.sample=lambda key,offset=0.: original(key,offset)+5 if key=='force' else original(key,offset)
        env._clear_samples()
        env.last_obs=-1;second=env._get_observations()
        assert torch.equal(first['policy'],second['policy']),'Actor received privileged force information'
        assert not torch.equal(first['critic'],second['critic']),'Critic did not receive privileged information'
        env.sample=original;env._clear_samples();env.reset()
        clock=env.phase+env.motion_time;clips=env.clip.clone()
        env._reset_idx(torch.arange(env.num_envs,device=env.device))
        assert torch.equal(env.phase+env.motion_time,clock),'Episode reset changed the official motion clock'
        assert torch.equal(env.clip,clips),'Episode reset changed assigned clips'
        if args.dataset is not None:
            env.phase[:]=torch.tensor([5.75,8.,17.75,19.25]*4,device=env.device)
            env.motion_time.zero_();env._clear_samples();env._refresh()
            env._reset_idx(torch.arange(env.num_envs,device=env.device))
        rewards=[]
        for _ in range(100):
            obs,reward,terminated,truncated,_=env.step(torch.zeros(16,6,device=env.device))
            assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()
            rewards.append(float(reward.mean()))
        report={'status':'passed','num_envs':16,'physics_steps':400,'control_steps':100,
                'isaac_mujoco_max_fk_error_m':fk_error,'actor_force_blind':True,'critic_privileged':True,
                'actor_shape':list(obs['policy'].shape),'critic_shape':list(obs['critic'].shape),'mean_reward':sum(rewards)/len(rewards),
                'episode_reset_retains_clip_and_clock':True,'urdf_velocity_limits_preserved':True,
                'dense_forcefield_matches_upstream_active_inactive_plane_setpoint':True,'torque_oracle_script_warmed_up':True}
        if args.dataset is not None:
            report.update(dataset_path=str(args.dataset.absolute()),max_interpolated_boundary_force_error_N=boundary_force_error,
                          physics_stage_starts_s=[5.75,8.,17.75,19.25])
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        args.output.write_text(json.dumps({'status':'failed','error':repr(error)},indent=2)+'\n')
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
