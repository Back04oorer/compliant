"""Zero-wrench ablations of pose bias; the live training process is never modified."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import OUTPUT

MODES = ('policy', 'policy_zero_gravity', 'policy_plus_g_offset',
         'pd', 'pd_zero_gravity', 'pd_plus_g_offset')
PLANTS = ('evaluation_nominal', 'training_midpoint', 'training_randomized')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept-eula', action='store_true')
    args = parser.parse_args()
    if not args.accept_eula:
        parser.error('--accept-eula is required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'diagnose_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    try:
        import torch
        from rsl_rl.modules import ActorCritic
        from . import env as adapter
        from .asset import prepare_urdf
        from .evaluate import actor_observation

        torch.set_num_threads(2); torch.manual_seed(44381); np.random.seed(44381)
        prepare_urdf()
        manifest = json.loads((args.checkpoints[-1].parent / 'manifest.json').read_text())
        for name in ('env.py', 'config.py'):
            assert hashlib.sha256(Path(adapter.__file__).with_name(name).read_bytes()).hexdigest() == manifest['source_sha256'][name]
        assert hashlib.sha256((OUTPUT / 'piper_fixed.urdf').read_bytes()).hexdigest() == manifest['model_sha256']
        rows = [dict(checkpoint=str(path), plant=plant, replicate=replicate, mode=mode)
                for path in args.checkpoints for plant in PLANTS for replicate in range(3) for mode in MODES]
        cfg = adapter.PiperCfg(); cfg.seed = 44381; cfg.scene.num_envs = len(rows)
        env = adapter.PiperEnv(cfg)
        robot = env.robot; device = env.device; n = env.num_envs
        ids = torch.arange(n, device=device); cpu_ids = ids.cpu()
        kp = torch.tensor(adapter.KP, device=device)
        kd = torch.tensor(adapter.KD, device=device)
        k = torch.full((n,), 100., device=device); kr = torch.ones(n, device=device)
        mass = robot.root_physx_view.get_masses().clone()
        inertia = robot.root_physx_view.get_inertias().clone()
        material = robot.root_physx_view.get_material_properties().clone()
        armature = robot.data.joint_armature.clone()
        friction = robot.data.joint_friction_coeff.clone()
        limits = robot.data.joint_pos_limits.clone()
        damping = kd[None] + torch.rand(n, 6, device=device) * 2
        # Reuse each physical realization across checkpoint and controller ablations.
        sources = {}
        for i, row in enumerate(rows):
            key = row['plant'], row['replicate']
            source = sources.setdefault(key, i)
            if row['plant'] != 'training_randomized':
                mass[i] = robot.data.default_mass[i].cpu()
                inertia[i] = robot.data.default_inertia[i].cpu()
                limits[i] = robot.data.default_joint_pos_limits[i]
                if row['plant'] == 'evaluation_nominal':
                    armature[i] = .01; friction[i] = 0.; damping[i] = kd
                else:
                    armature[i] = .065; friction[i] = .005; damping[i] = kd + 1
            for prop in (mass, inertia, material, armature, friction, limits, damping):
                prop[i] = prop[source].clone()
        robot.root_physx_view.set_masses(mass, cpu_ids)
        robot.root_physx_view.set_inertias(inertia, cpu_ids)
        robot.root_physx_view.set_material_properties(material, cpu_ids)
        robot.write_joint_armature_to_sim(armature)
        robot.write_joint_friction_coefficient_to_sim(friction)
        robot.write_joint_position_limit_to_sim(limits)
        robot.write_joint_damping_to_sim(damping)
        robot.write_joint_state_to_sim(env.qref, torch.zeros_like(env.qref))
        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(.005)
        reference = (robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins).clone()
        dataset = np.load(OUTPUT / 'dataset/train.npz')
        zero_clip = int(np.flatnonzero(np.linalg.norm(dataset['force'], axis=-1).max(1) == 0)[0])
        dataset_reference = torch.tensor(dataset['keypoints'][zero_clip, 0, -1], device=device, dtype=torch.float32)
        fk_error = float((reference - dataset_reference).abs().max())
        assert fk_error < 1e-4, fk_error
        # Verify that the evaluator supplies precisely the same actor channels as training.
        # Neutral noise draws remove noise only in this isolated diagnostic process.
        original_sample = env.sample; original_rand = torch.rand_like
        def sample(key, offset=0.):
            return k if key == 'k' else kr if key == 'kr' else original_sample(key, offset)
        env.sample = sample; parity = []
        try:
            torch.rand_like = lambda value, *a, **kw: torch.full_like(value, .5)
            for random_history in (False, True):
                env.hist = torch.randn_like(env.hist) if random_history else torch.zeros_like(env.hist)
                env.new_history[:] = not random_history
                before_hist = env.hist.clone(); before_new = env.new_history.clone()
                env.last_obs = -1
                production = adapter.PiperEnv._get_observations(env)['policy'].clone()
                env.hist = before_hist; env.new_history = before_new
                evaluated = actor_observation(env, k, kr)
                torch.testing.assert_close(production, evaluated, atol=0, rtol=0)
                parity.append(True)
        finally:
            torch.rand_like = original_rand; env.sample = original_sample
        env.actions.zero_(); env.previous_actions.zero_(); env.hist.zero_(); env.new_history[:] = True
        obs = actor_observation(env, k, kr)
        agent = dict(manifest['agent']['policy']); agent.pop('class_name', None)
        policies = []
        for path in args.checkpoints:
            policy = ActorCritic({'policy': obs, 'critic': torch.zeros(n, 793, device=device)},
                                 {'policy': ['policy'], 'critic': ['critic']}, 6, **agent).to(device)
            state = torch.load(path, map_location=device, weights_only=False)
            policy.load_state_dict(state['model_state_dict']); policy.eval()
            selected = torch.tensor([i for i, row in enumerate(rows) if row['checkpoint'] == str(path)], device=device)
            policies.append((policy, selected))
        compensate = torch.tensor([row['mode'].endswith('plus_g_offset') for row in rows], device=device)
        disable_gravity = torch.tensor([row['mode'].endswith('zero_gravity') for row in rows], device=device)
        pd = torch.tensor([row['mode'].startswith('pd') for row in rows], device=device)
        body_force = torch.zeros(n, robot.num_bodies, 3, device=device)
        body_force[disable_gravity] = -mass.to(device)[disable_gravity, :, None] * torch.tensor(cfg.sim.gravity, device=device)[None, None]
        body_ids = list(range(robot.num_bodies))
        samples = {key: [] for key in ('tcp', 'q', 'dq', 'command', 'gravity_torque', 'action')}
        with torch.inference_mode():
            for step in range(1000):
                action = torch.zeros(n, 6, device=device)
                for policy, selected in policies:
                    action[selected] = policy.act_inference({'policy': obs[selected]})
                action[pd] = 0
                gravity_torque = robot.root_physx_view.get_gravity_compensation_forces()
                command = env.qref + .5 * action
                command[compensate] += gravity_torque[compensate] / kp
                env.actions.copy_(action)
                robot.set_joint_position_target(command)
                for _ in range(4):
                    robot.set_external_force_and_torque(body_force, torch.zeros_like(body_force), body_ids=body_ids, is_global=True)
                    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(.005)
                env.common_step_counter += 1
                obs = actor_observation(env, k, kr)
                for key, value in dict(tcp=robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins,
                                       q=robot.data.joint_pos, dq=robot.data.joint_vel, command=command,
                                       gravity_torque=gravity_torque, action=action).items():
                    samples[key].append(value.clone())
                if step % 250 == 249:
                    print(f'Bias diagnosis: {step + 1}/1000 steps, {n} worlds', flush=True)
        traces = {key: torch.stack(value).cpu().numpy() for key, value in samples.items()}
        time = np.arange(1, 1001) * .02
        traces.update(time=time, reference=reference.cpu().numpy(), qref=env.qref.cpu().numpy())
        np.savez_compressed(args.output / 'traces.npz', **traces)
        for i, row in enumerate(rows):
            for label, window in [('pre_push', (time >= 2.5) & (time < 3)),
                                  ('late', time >= 15.)]:
                position = traces['tcp'][window, i]
                delta = position - traces['reference'][i]
                row[label + '_bias_m'] = float(np.linalg.norm(delta.mean(0)))
                row[label + '_rmse_m'] = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=-1))))
                row[label + '_std_m'] = float(np.sqrt(np.mean(np.sum((position-position.mean(0)) ** 2, axis=-1))))
                row[label + '_offset_xyz_m'] = delta.mean(0).tolist()
            row['late_q_error_rad'] = (traces['q'][time >= 15., i] - traces['qref'][i]).mean(0).tolist()
            row['late_command_offset_rad'] = (traces['command'][time >= 15., i] - traces['qref'][i]).mean(0).tolist()
            row['late_gravity_torque_Nm'] = traces['gravity_torque'][time >= 15., i].mean(0).tolist()
            row['initial_policy_offset_rad'] = (.5 * traces['action'][0, i]).tolist()
            row['initial_gravity_required_offset_rad'] = (traces['gravity_torque'][0, i] / np.array(adapter.KP)).tolist()
        report = dict(status='complete', seed=44381, duration_s=20., stiffness=100., rotational_stiffness=1.,
                      observation_noise=False, action_noise=False, gravity=cfg.sim.gravity,
                      gravity_compensation_note='No extra gravity compensation in production actor; offset/cancellation modes are diagnostic interventions only',
                      actor_observation_bitwise_matches_training=all(parity),
                      isaac_dataset_fk_max_error_m=fk_error,
                      model_sha256=manifest['model_sha256'],
                      checkpoint_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.checkpoints},
                      cases=rows)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': 'complete', 'output': str(args.output)}), flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(error)}) + '\n')
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
