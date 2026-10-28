"""Read-only dataset audit and controlled force/feedback ablations of frozen PPOs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import DATASET, OUTPUT
from .evaluate import case_metrics, envelope


def quantiles(values):
    return dict(zip(('min', 'p10', 'median', 'p90', 'p99', 'max'),
                    np.quantile(values, (0, .1, .5, .9, .99, 1)).tolist())) if len(values) else None


def audit_dataset(path):
    with np.load(path) as data:
        valid = np.arange(data['force'].shape[1])[None] < data['lengths'][:, None]
        force = data['force']; magnitude = np.linalg.norm(force, axis=-1)
        active = valid & (magnitude > 1e-6)
        reference = data['keypoints'][np.flatnonzero(~active.any(1))[0], 0, -1]
        offset = data['keypoints'][:, :, -1] - reference
        desired = force / data['k'][:, :, None]
        result = dict(frames=int(valid.sum()), active_frames=int(active.sum()),
                      active_fraction=float(active.sum() / valid.sum()),
                      force_N=quantiles(magnitude[active]), stiffness_N_per_m=quantiles(data['k'][active]),
                      field_stiffness_N_per_m=quantiles(data['metadata'][:, :, 0][active]),
                      target_displacement_mm=quantiles(np.linalg.norm(desired[active], axis=-1) * 1000),
                      ik_target_error_mm=quantiles(np.linalg.norm((offset - desired)[active], axis=-1) * 1000),
                      directional_frames={})
        for axis, name in enumerate('XYZ'):
            for sign in (-1, 1):
                selected = active & (sign * force[:, :, axis] > .7 * magnitude)
                matched = selected & (magnitude >= 1) & (magnitude <= 3) & (data['k'] >= 60) & (data['k'] <= 160)
                result['directional_frames'][('+' if sign > 0 else '-') + name] = dict(
                    frames=int(selected.sum()), near_2N_K100_frames=int(matched.sum()),
                    median_force_N=float(np.median(magnitude[selected])) if selected.any() else None,
                    median_stiffness_N_per_m=float(np.median(data['k'][selected])) if selected.any() else None)
        return result


def split_push_error(time, position, reference, force, stiffness):
    before = position[(time >= 2.5) & (time < 3)].mean(0)
    steady = (time >= 4.5) & (time <= 5.4 + 1e-8)
    displacement = position[steady] - before
    relative_error = displacement - force[steady] / stiffness
    return dict(pre_push_bias_xyz_m=(before - reference).tolist(),
                steady_displacement_xyz_m=displacement.mean(0).tolist(),
                debiased_push_rmse_m=float(np.sqrt(np.mean(np.sum(relative_error ** 2, axis=-1)))))


def external_force(mode, fraction, peak, stiffness, field_stiffness, reference, position):
    if mode == 'frozen_zero':
        return np.zeros_like(peak)
    if mode in ('prescribed', 'frozen_command'):
        return fraction * peak
    if fraction <= 0:
        return np.zeros_like(peak)
    origin = reference + fraction * peak * (1 / stiffness + 1 / field_stiffness)
    return field_stiffness * (origin - position)


def reward_components(env, adapter):
    r = adapter.rewards
    return dict(
        force_position=3 * r.force_link_keypoint_tracking_local(env, 'reference', sigma=.1),
        force_orientation=3 * r.force_link_keypoint_orientation_tracking_local(env, 'reference', sigma=.1),
        force=2 * r.force_command_tracking(env, 'reference', sigma=20.),
        torque=2 * r.torque_command_tracking(env, 'reference', sigma=2.),
        keypoints=2 * r.keypoint_deviation_from_command_local(env, 'reference', env.key_id_list, sigma=.3),
        keypoint_rot=2 * r.keypoint_orientation_deviation_from_command_local(env, 'reference', env.key_id_list, sigma=.4),
        joint=2 * r.joint_deviation_from_command_exp(env, 'reference', asset_cfg=env.asset_cfg),
    )


def summarize_diagnosis(rows):
    summary = {}
    for checkpoint in dict.fromkeys(row['checkpoint'] for row in rows):
        axes = {}
        for axis in 'XYZ':
            selected = [r for r in rows if r['checkpoint'] == checkpoint and r['plant'] == 'nominal'
                        and r['axis'] == axis and r['sign'] == 1]
            nominal = [r for r in selected if r['stiffness_N_per_m'] == 100]
            policy = next(r for r in nominal if r['force_mode'] == 'prescribed')
            frozen = next(r for r in nominal if r['force_mode'] == 'frozen_command')
            field = next(r for r in nominal if r['force_mode'] == 'field_100')
            axes['+' + axis] = dict(
                policy_deflection_mm=policy['steady_displacement_m'] * 1000,
                frozen_command_force_increment_mm=frozen['frozen_push_increment_vs_zero_m'] * 1000,
                effective_directional_stiffness_N_per_m=2 / policy['steady_displacement_m'],
                field100_deflection_mm=field['steady_displacement_m'] * 1000,
                field100_actual_force_N=field['steady_actual_force_in_push_direction_N'],
                pre_push_bias_xyz_mm=(np.array(policy['pre_push_bias_xyz_m']) * 1000).tolist(),
                absolute_push_rmse_mm=policy['steady_target_rmse_m'] * 1000,
                debiased_push_rmse_mm=policy['debiased_push_rmse_m'] * 1000,
                stiffness_sweep=[dict(command_N_per_m=r['stiffness_N_per_m'],
                                     expected_deflection_mm=2000 / r['stiffness_N_per_m'],
                                     actual_deflection_mm=r['steady_displacement_m'] * 1000)
                                 for r in selected if r['force_mode'] == 'prescribed'])
        summary[checkpoint] = axes
    return summary


def plot_diagnosis(output, traces, rows, checkpoints):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    time = traces['time']; before = (time >= 2.5) & (time < 3)
    newest = str(checkpoints[-1])
    for sign in (1, -1):
        fig = Figure(figsize=(14, 10), dpi=140); FigureCanvasAgg(fig)
        axes = fig.subplots(3, 3)
        for column, axis in enumerate('XYZ'):
            def find(checkpoint, mode, k=100):
                return next(i for i, r in enumerate(rows) if r['checkpoint'] == checkpoint
                            and r['plant'] == 'nominal' and r['axis'] == axis and r['sign'] == sign
                            and r['stiffness_N_per_m'] == k and r['force_mode'] == mode)
            for checkpoint, color in zip(checkpoints, ('#db873a', '#3268ba', '#966caa')):
                i = find(str(checkpoint), 'prescribed')
                displacement = traces['tcp'][:, i, column] - traces['tcp'][before, i, column].mean()
                axes[0, column].plot(time, displacement * sign * 1000, color=color, label=checkpoint.stem)
                sweep = [rows[find(str(checkpoint), 'prescribed', k)]['steady_displacement_m'] * 1000 for k in (60, 100, 160)]
                axes[2, column].plot([60, 100, 160], sweep, 'o-', color=color, label=checkpoint.stem)
            frozen = find(newest, 'frozen_command'); zero = find(newest, 'frozen_zero')
            increment = (traces['tcp'][:, frozen, column] - traces['tcp'][:, zero, column]) * sign * 1000
            axes[0, column].plot(time, increment, color='#747474', label='Frozen command - zero-force twin')
            i = find(newest, 'field_100')
            displacement = traces['tcp'][:, i, column] - traces['tcp'][before, i, column].mean()
            axes[0, column].plot(time, displacement * sign * 1000, color='#438769', label='Latest PPO, reactive field Kff=100')
            axes[1, column].plot(time, traces['force'][:, i, column] * sign, color='#438769', label='Reactive field: actual force')
            axes[1, column].plot(time, envelope(time) * 2, 'k--', label='Prescribed force (2 N)')
            axes[0, column].plot(time, envelope(time) * 20, 'k--', label='F/K target (20 mm)')
            axes[2, column].plot([60, 100, 160], 2000 / np.array([60, 100, 160]), 'k--', label='F/K target')
            axes[0, column].set_title(('+' if sign > 0 else '-') + axis)
            for row in (0, 1):
                axes[row, column].axvspan(3, 6, color='#db873a', alpha=.07)
                axes[row, column].axvline(6, color='#747474', linestyle=':', linewidth=1)
                axes[row, column].set_xlabel('Time [s]')
            axes[2, column].set_xlabel('Commanded stiffness [N/m]')
            for ax in axes[:, column]:
                ax.grid(alpha=.2)
        axes[0, 0].set_ylabel('Directional deflection [mm]')
        axes[1, 0].set_ylabel('Actual directional force [N]')
        axes[2, 0].set_ylabel('Steady deflection under 2 N [mm]')
        for row in range(3):
            axes[row, 0].legend(fontsize=6)
        fig.suptitle('Force / policy-feedback diagnosis: nominal plant, zero external torque, noise off')
        fig.tight_layout()
        fig.savefig(output / f'xyz_{"positive" if sign > 0 else "negative"}.png')


def diagnose(args):
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

        torch.set_num_threads(2); torch.manual_seed(68123); np.random.seed(68123)
        prepare_urdf()
        manifest = json.loads((args.checkpoints[-1].parent / 'manifest.json').read_text())
        for path in args.checkpoints:
            other = json.loads((path.parent / 'manifest.json').read_text())
            for key in ('source_sha256', 'model_sha256', 'agent', 'dataset'):
                assert manifest[key] == other[key], key
        for name in ('env.py', 'config.py'):
            assert hashlib.sha256(Path(adapter.__file__).with_name(name).read_bytes()).hexdigest() == manifest['source_sha256'][name]
        assert hashlib.sha256((OUTPUT / 'piper_fixed.urdf').read_bytes()).hexdigest() == manifest['model_sha256']
        protocol = np.load(args.protocol)
        description = json.loads(args.protocol.with_suffix('.json').read_text())
        cases = [(i, case) for i, case in enumerate(description['cases']) if case['force_N'] == 2]
        modes = ('prescribed', 'frozen_command', 'frozen_zero', 'field_30', 'field_100', 'field_500')
        rows = [dict(**case, protocol_case=i, checkpoint=str(checkpoint), plant=plant, force_mode=mode)
                for checkpoint in args.checkpoints for plant in ('nominal', 'training_midpoint')
                for i, case in cases for mode in modes]
        cfg = adapter.PiperCfg(); cfg.seed = 68123; cfg.scene.num_envs = len(rows)
        env = adapter.PiperEnv(cfg); robot = env.robot; device = env.device; n = env.num_envs
        ids = torch.arange(n, device=device); cpu_ids = ids.cpu()
        robot.root_physx_view.set_masses(robot.data.default_mass.cpu(), cpu_ids)
        robot.root_physx_view.set_inertias(robot.data.default_inertia.cpu(), cpu_ids)
        materials = robot.root_physx_view.get_material_properties(); materials[:] = materials[0].clone()
        robot.root_physx_view.set_material_properties(materials, cpu_ids)
        middle = torch.tensor([row['plant'] == 'training_midpoint' for row in rows], device=device)
        damping = torch.tensor(adapter.KD, device=device)[None] + middle[:, None]
        robot.write_joint_damping_to_sim(damping)
        robot.write_joint_armature_to_sim(torch.where(middle, .065, .01)[:, None].expand(-1, 6).clone())
        robot.write_joint_friction_coefficient_to_sim(torch.where(middle, .005, 0.)[:, None].expand(-1, 6).clone())
        robot.write_joint_position_limit_to_sim(robot.data.default_joint_pos_limits)
        robot.write_joint_state_to_sim(env.qref, torch.zeros_like(env.qref))
        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(.005)
        reference = torch.tensor(protocol['reference_pos'], device=device, dtype=torch.float32)
        assert float((robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins - reference).abs().max()) < 1e-4
        k = torch.tensor([r['stiffness_N_per_m'] for r in rows], device=device)
        ff = torch.tensor([float(r['force_mode'].split('_')[-1]) if r['force_mode'].startswith('field_') else 0 for r in rows], device=device)
        reactive = ff > 0; freeze = torch.tensor([r['force_mode'].startswith('frozen_') for r in rows], device=device)
        zero_force = torch.tensor([r['force_mode'] == 'frozen_zero' for r in rows], device=device)
        peak = torch.tensor([np.eye(3)['XYZ'.index(r['axis'])] * r['sign'] * 2 for r in rows], device=device, dtype=torch.float32)
        env.actions.zero_(); env.hist.zero_(); env.new_history[:] = True
        obs = actor_observation(env, k, torch.ones_like(k))
        agent = dict(manifest['agent']['policy']); agent.pop('class_name', None)
        policies = []; hashes = {}
        for path in args.checkpoints:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            policy = ActorCritic({'policy': obs, 'critic': torch.zeros(n, 793, device=device)},
                                 {'policy': ['policy'], 'critic': ['critic']}, 6, **agent).to(device)
            policy.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); policy.eval()
            selected = torch.tensor([i for i, row in enumerate(rows) if row['checkpoint'] == str(path)], device=device)
            policies.append((policy, selected))
        samples = {key: [] for key in ('tcp', 'q', 'command', 'force')}
        def record(command, force):
            for key, value in dict(tcp=robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins,
                                   q=robot.data.joint_pos, command=command, force=force).items():
                samples[key].append(value.clone())
        record(env.qref, torch.zeros_like(peak)); frozen_command = env.qref.clone()
        with torch.no_grad():
            for step, t in enumerate(protocol['time'][:-1]):
                action = torch.zeros(n, 6, device=device)
                for policy, selected in policies:
                    action[selected] = policy.act_inference({'policy': obs[selected]})
                command = env.qref + .5 * action
                if t < 3:
                    frozen_command.copy_(command)
                else:
                    command[freeze] = frozen_command[freeze]
                fraction = float(envelope(t))
                position = robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins
                force = peak * fraction
                origin = reference + fraction * peak * (1 / k[:, None] + 1 / ff.clamp(min=1)[:, None])
                field_force = ff[:, None] * (origin - position) if fraction > 0 else torch.zeros_like(force)
                force[reactive] = field_force[reactive]
                force[zero_force] = 0
                env.actions.copy_((command - env.qref) / .5)
                robot.set_joint_position_target(command)
                for _ in range(4):
                    robot.set_external_force_and_torque(force[:, None], torch.zeros(n, 1, 3, device=device), body_ids=[env.tcp_id], is_global=True)
                    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(.005)
                env.common_step_counter += 1
                obs = actor_observation(env, k, torch.ones_like(k))
                record(command, force)
                if step % 150 == 149:
                    print(f'Compliance ablation {step + 1}/600 steps, {n} worlds', flush=True)
        traces = {key: torch.stack(value).cpu().numpy() for key, value in samples.items()}
        traces.update(time=protocol['time'], reference_pos=protocol['reference_pos'])
        np.savez_compressed(args.output / 'traces.npz', **traces)
        time = traces['time']; steady = (time >= 4.5) & (time <= 5.4 + 1e-8)
        before = (time >= 2.5) & (time < 3)
        for i, row in enumerate(rows):
            requested = envelope(time)[:, None] * peak[i].cpu().numpy()
            row.update(case_metrics(time, traces['tcp'][:, i], traces['reference_pos'], requested, row))
            row.update(split_push_error(time, traces['tcp'][:, i], traces['reference_pos'], requested, row['stiffness_N_per_m']))
            axis = 'XYZ'.index(row['axis'])
            row['steady_actual_force_in_push_direction_N'] = float(traces['force'][steady, i, axis].mean() * row['sign'])
            row['peak_actual_force_N'] = float(np.linalg.norm(traces['force'][:, i], axis=-1).max())
            row['command_change_norm_rad'] = float(np.linalg.norm(traces['command'][steady, i].mean(0) - traces['command'][before, i].mean(0)))
            row['frozen_push_increment_vs_zero_m'] = None
            row['frozen_zero_drift_m'] = None
            if row['force_mode'] == 'frozen_command':
                other = next(j for j, r in enumerate(rows) if r['force_mode'] == 'frozen_zero'
                             and all(r[key] == row[key] for key in ('checkpoint', 'plant', 'protocol_case')))
                np.testing.assert_allclose(traces['tcp'][before, i], traces['tcp'][before, other], atol=2e-5, rtol=0)
                row['frozen_push_increment_vs_zero_m'] = float(
                    (traces['tcp'][steady, i, axis] - traces['tcp'][steady, other, axis]).mean() * row['sign'])
                row['frozen_zero_drift_m'] = float(np.linalg.norm(
                    traces['tcp'][steady, other].mean(0) - traces['tcp'][before, other].mean(0)))
        # Evaluate the unchanged training reward kernels at two kinematic poses.
        goal_q = torch.tensor(protocol['qgoal'][175, [r['protocol_case'] for r in rows]], device=device, dtype=torch.float32)
        robot.write_joint_state_to_sim(goal_q, torch.zeros_like(goal_q))
        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(.005)
        c = env.command
        c.dof_pos = goal_q[:, None].expand(-1, 2, -1)
        c.keypoints = robot.data.body_pos_w[:, env.key_ids].clone()[:, None].expand(-1, 2, -1, -1)
        c.adapted_keypoints = c.keypoints
        c.keypoint_rotations = robot.data.body_quat_w[:, env.key_ids].clone()[:, None].expand(-1, 2, -1, -1)
        c.adapted_keypoint_rotations = c.keypoint_rotations
        c._force_vector_world = peak[:, None].expand(-1, 2, -1)
        c._torque_vector_world = torch.zeros(n, 2, 3, device=device)
        c.active_force_mask[:] = True
        reward_ff = torch.where(reactive, ff, 100.)
        c._ff_stiffness = reward_ff[:, None, None]
        c._ff_origin = (reference + peak * (1 / k[:, None] + 1 / reward_ff[:, None]) + env.scene.env_origins)[:, None]
        c._ff_normal = torch.zeros(n, 1, 3, device=device)
        c._ff_rotational_stiffness = torch.zeros(n, 1, 1, device=device)
        reward_rows = []
        for pose, q in (('augmented_goal', goal_q), ('hold_reference', env.qref)):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(.005)
            components = {key: value.detach().cpu().numpy() for key, value in reward_components(env, adapter).items()}
            for i, row in enumerate(rows):
                reward_rows.append(dict(checkpoint=row['checkpoint'], axis=row['axis'], sign=row['sign'],
                                        stiffness_N_per_m=row['stiffness_N_per_m'], plant=row['plant'],
                                        force_mode=row['force_mode'], reward_field_stiffness=float(reward_ff[i]),
                                        pose=pose, **{key: float(value[i]) for key, value in components.items()}))
        report = dict(status='complete', checkpoints_sha256=hashes,
                      protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                      production_source_sha256=manifest['source_sha256'],
                      dataset_audit=audit_dataset(DATASET / 'train.npz'), cases=rows,
                      diagnosis_summary=summarize_diagnosis(rows),
                      reward_counterfactuals=reward_rows,
                      notes=['No training changes; mean policy, observation noise off, zero external torque.',
                             'Reactive fields use isotropic F=Kff*(origin-x), same 50 Hz hold as training.',
                             'Origin=reference+envelope*Ftarget*(1/K+1/Kff); compliant equilibrium would have Ftarget.',
                             'Frozen command is held at its last pre-push value; physical PD stays enabled.',
                             'frozen_zero suppresses the nominal push; frozen_push_increment_vs_zero_m subtracts its drift.',
                             'Reward counterfactuals compare kinematics only, excluding velocity/action-rate/limit/alive terms.',
                             'Training midpoint changes joint armature/damping/friction only; mass stays nominal.'])
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        plot_diagnosis(args.output, traces, rows, args.checkpoints)
        print(f'Diagnosis complete: {args.output}', flush=True)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(error)}) + '\n')
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--protocol', type=Path, default=OUTPUT / 'evaluations/xyz_push_v1/protocol.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept-eula', action='store_true')
    args = parser.parse_args()
    if not args.accept_eula:
        parser.error('--accept-eula is required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'diagnose_source.py').write_bytes(Path(__file__).read_bytes())
    diagnose(args)


if __name__ == '__main__':
    main()
