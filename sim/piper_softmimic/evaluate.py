"""Held-out XYZ push/release evaluation in the unchanged Isaac Piper plant.

Prepare kinematic targets with the MuJoCo environment, then evaluate the PPO and
privileged analytic controller together in Isaac. Neither command trains a model.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import OUTPUT

DT = .02
TIMING = dict(push_start_s=3., full_force_s=3.5, release_start_s=5.5,
              force_removed_s=6., end_s=12., steady_window_s=[4.5, 5.4],
              recovery_window_s=[11., 12.], minimum_settle_hold_s=.5)
VARIANTS = (
    dict(name='nominal', mass_scale=1., armature=.01, extra_damping=0., friction=0.),
    dict(name='light', mass_scale=.85, armature=.04, extra_damping=.5, friction=.003),
    dict(name='heavy', mass_scale=1.15, armature=.08, extra_damping=1., friction=.007),
)


def actor_observation(env, stiffness, rotational_stiffness, noise=False):
    """The trained 117D actor input, shared by push evaluation and bias diagnosis."""
    import torch
    zero = torch.zeros(env.num_envs, 3, device=env.device)
    gravity = zero.clone(); gravity[:, 2] = -1
    noisy = lambda value, amplitude: value + (torch.rand_like(value) * 2 - 1) * amplitude if noise else value
    parts = [noisy(zero, .2) * .25, noisy(gravity, .01),
             noisy(env.robot.data.joint_pos - env.qref, .01),
             noisy(env.robot.data.joint_vel, 1.5) * .05,
             env.actions, env.qref, zero[:, :2], zero[:, :1], zero[:, :1], gravity,
             torch.log(stiffness[:, None] + 1e-6), torch.log(rotational_stiffness[:, None] + 1e-6)]
    frame = torch.cat(parts, -1)
    env.hist = torch.roll(env.hist, -1, dims=1)
    env.hist[:, -1] = frame
    env.hist[env.new_history] = frame[env.new_history, None]
    env.new_history[:] = False
    return env.hist.reshape(env.num_envs, -1)[:, env.actor_history_order]


def envelope(t):
    return np.clip(np.minimum((t - TIMING['push_start_s']) / .5,
                              (TIMING['force_removed_s'] - t) / .5), 0., 1.)


def prepare_protocol(path):
    import mujoco
    from sim.piper_compliance.env import PiperComplianceEnv, Q_REF

    if path.exists():
        raise ValueError('Refusing to overwrite an evaluation protocol')
    path.parent.mkdir(parents=True, exist_ok=True)
    env = PiperComplianceEnv()
    cases, forces, goals, rejected = [], [], [], []
    times = np.arange(round(TIMING['end_s'] / DT) + 1) * DT
    for axis in range(3):
        for sign in (-1, 1):
            for amplitude in (2., 5.):
                for stiffness in (60., 100., 160.):
                    case = dict(axis='XYZ'[axis], sign=sign, force_N=amplitude,
                                stiffness_N_per_m=stiffness)
                    peak = np.eye(3)[axis] * amplitude * sign
                    cache = {0.: Q_REF.copy()}
                    errors = []
                    try:
                        for fraction in np.arange(1, 26) / 25:
                            q = env.augmented_pose(peak * fraction, stiffness,
                                                   initial=cache[round(fraction - .04, 8)])
                            cache[round(float(fraction), 8)] = q
                            actual = env.ik_data.site_xpos[env.tcp_id]
                            target = env.reference_pos + peak * fraction / stiffness
                            errors.append(float(np.linalg.norm(actual - target)))
                        qgoal = np.stack([cache[round(float(value), 8)] for value in envelope(times)])
                    except ValueError as error:
                        rejected.append(dict(**case, reason=str(error)))
                        continue
                    case['ik_max_position_error_m'] = max(errors)
                    cases.append(case)
                    forces.append(envelope(times)[:, None] * peak)
                    goals.append(qgoal)
    if not cases:
        raise ValueError('No feasible held-out cases')
    np.savez_compressed(path, time=times, force=np.stack(forces, 1),
                        qgoal=np.stack(goals, 1), reference_pos=env.reference_pos,
                        reference_rot=env.reference_rot, qref=Q_REF)
    description = dict(timing=TIMING, dt=DT, cases=cases, excluded_cases=rejected,
                       variants=VARIANTS, training_dataset_used_for_test=False,
                       force_mode='Prescribed world-frame TCP force, zero external torque; not the reactive training field')
    path.with_suffix('.json').write_text(json.dumps(description, indent=2) + '\n')
    print(json.dumps({'protocol': str(path), 'cases': len(cases), 'excluded': rejected}), flush=True)
    env.close()


def settling_time(times, distances, tolerance):
    after = np.flatnonzero(times >= TIMING['force_removed_s'] - 1e-8)
    if not len(after) or not np.isfinite(distances[after]).all():
        return None
    outside = np.flatnonzero(distances[after] > tolerance)
    first = int(outside[-1] + 1) if len(outside) else 0
    if first >= len(after):
        return None
    index = after[first]
    if times[-1] - times[index] < TIMING['minimum_settle_hold_s'] - 1e-8:
        return None
    return float(max(0., times[index] - TIMING['force_removed_s']))


def case_metrics(times, positions, reference, force, case):
    axis = 'XYZ'.index(case['axis'])
    direction = np.eye(3)[axis] * case['sign']
    target = reference + force / case['stiffness_N_per_m']
    distance = np.linalg.norm(positions - reference, axis=-1)
    error = np.linalg.norm(positions - target, axis=-1)
    before = (times >= 2.5) & (times < 3.)
    steady = (times >= 4.5) & (times <= 5.4 + 1e-8)
    final = times >= 11.
    pre_position = positions[before].mean(0)
    relative = np.linalg.norm(positions - pre_position, axis=-1)
    displacement = (positions - pre_position) @ direction
    expected = case['force_N'] / case['stiffness_N_per_m']
    reached = np.flatnonzero((times >= 3.) & (times <= 5.5) & (displacement >= .9 * expected))
    metrics = dict(expected_displacement_m=expected,
                   pre_push_bias_m=float(np.linalg.norm(pre_position - reference)),
                   steady_displacement_m=float(displacement[steady].mean()),
                   steady_target_rmse_m=float(np.sqrt(np.mean(error[steady] ** 2))),
                   steady_cross_axis_rmse_m=float(np.sqrt(np.mean(np.sum(
                       np.delete(positions[steady] - pre_position, axis, axis=1) ** 2, axis=1)))),
                   push_response_90pct_s=float(times[reached[0]] - 3.) if len(reached) else None,
                   recovery_rmse_m=float(np.sqrt(np.mean(distance[final] ** 2))),
                   final_error_m=float(distance[-1]),
                   final_return_to_pre_push_error_m=float(relative[-1]),
                   finite=bool(np.isfinite(positions).all()))
    for mm in (2, 5, 10):
        metrics[f'recovery_{mm}mm_s'] = settling_time(times, distance, mm / 1000)
        metrics[f'return_to_pre_push_{mm}mm_s'] = settling_time(times, relative, mm / 1000)
    return metrics


def plot_results(output, traces, metadata, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    times = traces['time']
    for sign in (1, -1):
        fig = Figure(figsize=(13, 9), dpi=150)
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 3)
        for column, axis in enumerate('XYZ'):
            selected = [i for i, row in enumerate(rows) if row['axis'] == axis and row['sign'] == sign
                        and row['force_N'] == 2 and row['stiffness_N_per_m'] == 100
                        and row['variant'] == 'nominal']
            for index in selected:
                row = rows[index]
                position = traces['tcp'][:, index]
                pre = position[(times >= 2.5) & (times < 3)].mean(0)
                target = traces['reference_pos'] + traces['force'][:, index] / 100
                color = '#df6b36' if row['controller'] == 'policy' else '#408566'
                label = 'FirstPPo' if row['controller'] == 'policy' else 'Analytic (privileged)'
                axes[0, column].plot(times, (position[:, column] - pre[column]) * sign * 1000, color=color, label=label)
                axes[1, column].plot(times, np.linalg.norm(position - target, axis=-1) * 1000, color=color)
                axes[2, column].plot(times, np.linalg.norm(position - traces['reference_pos'], axis=-1) * 1000, color=color)
            desired = envelope(times) * 20
            axes[0, column].plot(times, desired, 'k--', linewidth=1, label='F/K target (20 mm)')
            axes[0, column].set_title(f'{"+" if sign > 0 else "-"}{axis}: 2 N, K=100 N/m')
            axes[2, column].set_xlabel('Time [s]')
            for ax in axes[:, column]:
                ax.axvspan(3., 6., color='#df6b36', alpha=.07)
                ax.axvline(6., color='#687168', linestyle=':', linewidth=1)
                ax.grid(alpha=.2)
        axes[0, 0].set_ylabel('Deflection from pre-push [mm]')
        axes[1, 0].set_ylabel('3D compliant target error [mm]')
        axes[2, 0].set_ylabel('3D reference pose error [mm]')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{metadata["checkpoint_name"]}: independent Isaac XYZ push / release')
        fig.tight_layout()
        fig.savefig(output / f'xyz_{"positive" if sign > 0 else "negative"}.png')


def evaluate(args):
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    os.environ['ACCEPT_EULA'] = 'Y'
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    try:
        import torch
        from rsl_rl.modules import ActorCritic
        from . import env as adapter
        from .asset import prepare_urdf

        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        prepare_urdf()
        protocol = np.load(args.protocol)
        description = json.loads(args.protocol.with_suffix('.json').read_text())
        manifest = json.loads((args.checkpoint.parent / 'manifest.json').read_text())
        if hashlib.sha256((OUTPUT / 'piper_fixed.urdf').read_bytes()).hexdigest() != manifest['model_sha256']:
            raise ValueError('Evaluation model differs from the training control URDF')
        if manifest['actor_observation_dim'] != 117:
            raise ValueError('Checkpoint actor observation shape does not match')
        for name in ('env.py', 'config.py'):
            expected = manifest['source_sha256'][name]
            if hashlib.sha256(Path(adapter.__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Training adapter changed: {name}')
        cases = description['cases']
        rows = [dict(**case, case_index=i, variant=variant['name'], controller=controller)
                for i, case in enumerate(cases) for variant in VARIANTS for controller in ('policy', 'analytic')]
        n = len(rows)

        class EvaluationEnv(adapter.PiperEnv):
            # Use the trained actuator/plant model while prescribing an unseen wrench.
            # Direct stepping below intentionally bypasses training rewards and auto-reset.
            def _get_observations(self):
                return {'policy': actor_observation(self, self.eval_k, torch.ones_like(self.eval_k), args.observation_noise),
                        'critic': torch.zeros(self.num_envs, 793, device=self.device)}

        cfg = adapter.PiperCfg(); cfg.seed = args.seed; cfg.scene.num_envs = n
        env = EvaluationEnv(cfg)
        robot = env.robot
        device = env.device
        ids = torch.arange(n, device=device)
        cpu_ids = ids.cpu()
        case_ids = torch.tensor([row['case_index'] for row in rows], device=device)
        variant_ids = [next(i for i, variant in enumerate(VARIANTS) if variant['name'] == row['variant']) for row in rows]
        scale = torch.tensor([VARIANTS[i]['mass_scale'] for i in variant_ids], device=device)
        robot.root_physx_view.set_masses(robot.data.default_mass.cpu() * scale.cpu()[:, None], cpu_ids)
        robot.root_physx_view.set_inertias(robot.data.default_inertia.cpu() * scale.cpu()[:, None, None], cpu_ids)
        materials = robot.root_physx_view.get_material_properties().clone()
        materials[1::2] = materials[::2]
        robot.root_physx_view.set_material_properties(materials, cpu_ids)
        damping = torch.tensor(adapter.KD, device=device)[None] + torch.tensor(
            [VARIANTS[i]['extra_damping'] for i in variant_ids], device=device)[:, None]
        robot.write_joint_damping_to_sim(damping)
        robot.write_joint_armature_to_sim(torch.tensor([VARIANTS[i]['armature'] for i in variant_ids], device=device)[:, None].expand(-1, 6).clone())
        robot.write_joint_friction_coefficient_to_sim(torch.tensor([VARIANTS[i]['friction'] for i in variant_ids], device=device)[:, None].expand(-1, 6).clone())
        robot.write_joint_position_limit_to_sim(robot.data.default_joint_pos_limits)
        robot.write_joint_state_to_sim(env.qref, torch.zeros_like(env.qref))
        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(.005)
        reference = torch.tensor(protocol['reference_pos'], device=device, dtype=torch.float32)
        fk_error = float((robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins - reference).abs().max())
        if fk_error > 1e-4:
            raise ValueError(f'Isaac/IK FK mismatch: {fk_error}')
        # Both controllers in each pair must have identical physical properties.
        for prop in (robot.root_physx_view.get_masses(), robot.root_physx_view.get_inertias(),
                     robot.root_physx_view.get_material_properties(), robot.data.joint_damping,
                     robot.data.joint_armature, robot.data.joint_pos_limits):
            torch.testing.assert_close(prop[::2], prop[1::2], atol=0, rtol=0)
        torch.testing.assert_close(robot.data.joint_effort_limits[0], torch.tensor(adapter.LIMIT, device=device))
        torch.testing.assert_close(robot.data.joint_vel_limits[0], torch.tensor([5., 5., 5., 5., 5., 3.], device=device))
        env.eval_k = torch.tensor([row['stiffness_N_per_m'] for row in rows], device=device)
        env.actions.zero_(); env.previous_actions.zero_(); env.hist.zero_(); env.new_history[:] = True
        obs = env._get_observations()
        agent = dict(manifest['agent']['policy']); agent.pop('class_name', None)
        policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['critic']}, 6, **agent).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint['model_state_dict']); policy.eval()
        policy_ids = torch.arange(0, n, 2, device=device)
        analytic_ids = policy_ids + 1
        force = torch.tensor(protocol['force'], device=device, dtype=torch.float32)[:, case_ids]
        qgoal = torch.tensor(protocol['qgoal'], device=device, dtype=torch.float32)[:, case_ids]
        kp = torch.tensor(adapter.KP, device=device)
        samples = {name: [] for name in ('tcp', 'q', 'dq', 'command', 'estimated_motor_torque')}
        def record(command):
            samples['tcp'].append((robot.data.body_pos_w[:, env.tcp_id] - env.scene.env_origins).clone())
            samples['q'].append(robot.data.joint_pos.clone())
            samples['dq'].append(robot.data.joint_vel.clone())
            samples['command'].append(command.clone())
            # Same PD estimate used by the implicit actuator, with the actual evaluated damping.
            samples['estimated_motor_torque'].append(kp * (command - robot.data.joint_pos) - damping * robot.data.joint_vel)
        record(env.qref)
        with torch.inference_mode():
            for step in range(len(protocol['time']) - 1):
                actions = policy.act_inference(obs)
                command = env.qref + .5 * actions
                jacobian = robot.root_physx_view.get_jacobians()[:, env.tcp_id - 1, :3, :]
                gravity = robot.root_physx_view.get_gravity_compensation_forces()
                analytic = qgoal[step] + (gravity - torch.bmm(jacobian.transpose(1, 2), force[step, :, :, None]).squeeze(-1)) / kp
                command[analytic_ids] = analytic[analytic_ids]
                env.previous_actions.copy_(env.actions)
                env.actions.copy_((command - env.qref) / .5)
                robot.set_joint_position_target(command)
                for _ in range(4):
                    robot.set_external_force_and_torque(force[step, :, None], torch.zeros(n, 1, 3, device=device), body_ids=[env.tcp_id], is_global=True)
                    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(.005)
                env.common_step_counter += 1
                obs = env._get_observations()
                record(command)
                if step % 100 == 99:
                    print(f'Evaluation: {step + 1}/600 control steps, {n} paired worlds', flush=True)
        traces = {name: torch.stack(values).cpu().numpy() for name, values in samples.items()}
        traces.update(time=protocol['time'], force=force.cpu().numpy(), reference_pos=protocol['reference_pos'])
        np.savez_compressed(args.output / 'traces.npz', **traces)
        for index, row in enumerate(rows):
            row.update(case_metrics(traces['time'], traces['tcp'][:, index], traces['reference_pos'], traces['force'][:, index], row))
            lower, upper = robot.data.default_joint_pos_limits[index].cpu().numpy().T
            q = traces['q'][:, index]
            row['max_joint_limit_violation_rad'] = float(np.maximum(np.maximum(lower - q, q - upper), 0).max())
            row['max_joint_speed_rad_s'] = float(np.abs(traces['dq'][:, index]).max())
            row['max_joint_speed_limit_ratio'] = float((np.abs(traces['dq'][:, index]) / np.array([5., 5., 5., 5., 5., 3.])).max())
            torque = traces['estimated_motor_torque'][1:, index]
            row['estimated_torque_limit_fraction'] = float(np.mean(np.any(np.abs(torque) >= np.array(adapter.LIMIT) * .99, axis=-1)))
            row['max_reference_deviation_m'] = float(np.linalg.norm(traces['tcp'][:, index] - traces['reference_pos'], axis=-1).max())
            row['task_failure_over_30cm'] = row['max_reference_deviation_m'] > .3
        report = dict(status='complete', checkpoint=str(args.checkpoint.absolute()),
                      checkpoint_name=args.checkpoint.name,
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                      model_sha256=manifest['model_sha256'],
                      completed_training_updates=int(checkpoint['iter']) + 1,
                      protocol=description, actor_observes_force=False, analytic_observes_force=True,
                      analytic_controller='Quasi-static F/K displacement IK, same joint PD; command offset (model gravity - J.T*known_force)/KP',
                      actor_action_sampling=False, actor_observation_noise=args.observation_noise, evaluation_seed=args.seed,
                      rotational_stiffness_command_Nm_per_rad=1.,
                      plant='Same Isaac Sim Piper, 0.005 s physics / 0.02 s control, fixed gripper, unchanged torque/velocity limits; paired nominal/light/heavy properties',
                      fk_max_error_m=fk_error, pair_properties_identical=True, cases=rows)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        plot_results(args.output, traces, report, rows)
        print(json.dumps({'status': 'complete', 'output': str(args.output), 'checkpoint': args.checkpoint.name, 'cases': len(rows)}), flush=True)
    except Exception as error:
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
    parser.add_argument('--prepare-protocol', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--accept-eula', action='store_true')
    parser.add_argument('--seed', type=int, default=68123, help='Fixed evaluation noise seed; does not change training')
    parser.add_argument('--observation-noise', action='store_true', help='Include the training sensor-noise model; action inference remains deterministic')
    args = parser.parse_args()
    if args.prepare_protocol:
        prepare_protocol(args.prepare_protocol)
        return
    if not args.accept_eula or not all((args.protocol, args.checkpoint, args.output)):
        parser.error('Evaluation requires --accept-eula, --protocol, --checkpoint and --output')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'evaluate_source.py').write_bytes(Path(__file__).read_bytes())
    evaluate(args)


if __name__ == '__main__':
    main()
