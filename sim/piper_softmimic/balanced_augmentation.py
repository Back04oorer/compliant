"""Balanced Piper fixed-pose CMA data; original IK tasks, field equations and rewards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import AGENT, DATASET, KEYPOINTS, OUTPUT, UPSTREAM, UPSTREAM_COMMIT

DT = 1 / 30
SECONDS = 24
STAGES = ('hold', 'push', 'release_recovery')
FORCES = (.5, 1., 2., 3., 5.)
STIFFNESSES = (60., 100., 160.)
PROFILES = ('steady', 'two_peaks', 'modulated')


def plans(seed=20261028):
    result = []
    for axis in range(3):
        for sign in (-1, 1):
            for amplitude in FORCES:
                for stiffness in STIFFNESSES:
                    # Exclude the excessive 83 mm target uniformly across directions.
                    if amplitude == 5 and stiffness == 60:
                        continue
                    for i, profile in enumerate(PROFILES):
                        result.append(dict(axis='XYZ'[axis], sign=sign, force_N=amplitude,
                                           stiffness_N_per_m=stiffness, profile=profile,
                                           field_stiffness_N_per_m=(10., 100., 500.)[
                                               (len(result) // 3 + i) % 3]))
    np.random.default_rng(seed).shuffle(result)
    return result


def schedule(profile):
    time = np.arange(round(SECONDS / DT)) * DT
    stage = np.where(time < 6, 0, np.where(time < 18, 1, 2)).astype(np.uint8)
    ramp = dict(steady=.5, two_peaks=1., modulated=.25)[profile]
    smooth = lambda value: .5 - .5 * np.cos(np.pi * np.clip(value, 0, 1))
    fraction = smooth((time - 6) / ramp) * (1 - smooth((time - 18) / ramp))
    if profile == 'two_peaks':
        fraction *= 1 - .65 * np.exp(-((time - 12) / 1.2) ** 4)
    elif profile == 'modulated':
        fraction *= .7 + .3 * np.cos(2 * np.pi * (time - 6) / 3)
    return time, stage, fraction


def audit(data, clips):
    valid = np.arange(data['stage'].shape[1])[None] < data['lengths'][:, None]
    stage = data['stage'][valid]
    fraction = {name: float(np.mean(stage == i)) for i, name in enumerate(STAGES)}
    np.testing.assert_allclose(list(fraction.values()), [.25, .5, .25], atol=1e-12)
    counts = {f'{sign}{axis}': sum(c['axis'] == axis and c['sign'] == value for c in clips)
              for axis in 'XYZ' for sign, value in (('-', -1), ('+', 1))}
    assert len(set(counts.values())) == 1, counts
    assert all(np.linalg.norm(data['force'][i], axis=-1).max() > .49 for i in range(len(clips)))
    active = np.linalg.norm(data['force'], axis=-1) > .01
    for i, clip in enumerate(clips):
        for axis in range(3):
            peak = clip['force_N'] * (clip['sign'] if axis == 'XYZ'.index(clip['axis']) else 0)
            assert np.allclose(data['force'][i, :, axis], peak * data['force_fraction'][i])
        # The loader interpolates all metadata at 50 Hz: never interpolate a world origin from zero.
        time = np.arange(data['lengths'][i]) * DT
        control_time = np.arange(0, time[-1], .02)
        force = np.stack([np.interp(control_time, time, data['force'][i, :, axis]) for axis in range(3)], -1)
        origin = np.stack([np.interp(control_time, time, data['metadata'][i, :, 2 + axis]) for axis in range(3)], -1)
        field_k = np.interp(control_time, time, data['metadata'][i, :, 0])
        target = data['keypoints'][i, 0, -1] + force / clip['stiffness_N_per_m']
        np.testing.assert_allclose(field_k[:, None] * (origin - target), force, atol=1e-9, rtol=0)
    return dict(valid_frames=int(valid.sum()), phase_fractions=fraction,
                active_force_fraction=float(active[valid].mean()), direction_clip_counts=counts,
                force_clip_counts={str(f): sum(c['force_N'] == f for c in clips) for f in FORCES},
                max_target_force_N=float(np.linalg.norm(data['force'][valid], axis=-1).max()),
                all_arrays_finite=all(np.isfinite(value).all() for value in data.values()),
                interpolated_50hz_forcefield_equilibrium_verified=True)


def generate(output, seed=20261028):
    import mujoco
    from .augmentation import PiperIK, ik_step, prepare_model
    from sim.piper_compliance.kinematics import KP, Q_REF, TORQUE_LIMIT

    if output.resolve() == DATASET.resolve():
        raise ValueError('The original dataset is immutable; choose a new directory')
    output.mkdir(parents=True, exist_ok=False)
    try:
        prepare_model(); solver = PiperIK()
        reference = solver.data.body('tcp').xpos.copy()
        reference_quat = solver.data.body('tcp').xquat.copy()
        clip_plans = plans(seed); cache = {}; arrays = []; statistics = []
        max_error = 0.; max_orientation_error = 0.; max_speed = 0.; max_motor_fraction = 0.
        for index, plan in enumerate(clip_plans):
            axis = 'XYZ'.index(plan['axis']); k = plan['stiffness_N_per_m']
            peak = np.eye(3)[axis] * plan['sign'] * plan['force_N']
            key = (axis, plan['sign'], plan['force_N'], k)
            if key not in cache:
                solver.configuration.update(q=Q_REF)
                knots = [Q_REF.copy()]
                for fraction in np.arange(1, 101) / 100:
                    ik_step(solver, Q_REF, 'tcp', peak * fraction, np.zeros(3), np.zeros(3), k, 1., DT)
                    knots.append(solver.configuration.q.copy())
                cache[key] = np.stack(knots)
            time, stage, fraction = schedule(plan['profile'])
            q = np.stack([np.interp(fraction, np.linspace(0, 1, 101), cache[key][:, j]) for j in range(6)], -1)
            q[fraction == 0] = Q_REF
            dq = np.gradient(q, DT, axis=0)
            if np.any(np.abs(dq) > np.array([5, 5, 5, 5, 5, 3])):
                raise ValueError(f'Reference velocity limit exceeded: {plan}')
            force = fraction[:, None] * peak
            goals = reference + force / k
            fk = []; quats = []; ik_error = []; orientation_error = []
            for frame, pose in enumerate(q):
                solver.data.qpos[:] = pose; solver.data.qvel[:] = 0
                mujoco.mj_forward(solver.model, solver.data)
                fk.append([solver.data.body(name).xpos.copy() for name in KEYPOINTS])
                quats.append([solver.data.body(name).xquat.copy() for name in KEYPOINTS])
                ik_error.append(float(np.linalg.norm(fk[-1][-1] - goals[frame])))
                cosine = np.clip(abs(np.dot(quats[-1][-1], reference_quat)), 0, 1)
                orientation_error.append(float(2 * np.arccos(cosine)))
                jac = np.zeros((3, 6)); mujoco.mj_jacBody(solver.model, solver.data, jac, None, solver.model.body('tcp').id)
                motor = solver.data.qfrc_bias - jac.T @ force[frame]
                command = pose + motor / KP
                if np.any(np.abs(motor) > TORQUE_LIMIT) or np.any(command < solver.model.jnt_range[:, 0]) or np.any(command > solver.model.jnt_range[:, 1]):
                    raise ValueError(f'Static actuator/command limit exceeded: {plan}, frame {frame}')
                max_motor_fraction = max(max_motor_fraction, float(np.max(np.abs(motor) / TORQUE_LIMIT)))
            if max(ik_error) > .0002 or max(orientation_error) > .002:
                raise ValueError(f'IK target fidelity failed: {plan}, {max(ik_error)} m, {max(orientation_error)} rad')
            metadata = np.zeros((len(time), 12))
            ff = plan['field_stiffness_N_per_m']
            metadata[:, 0] = ff
            metadata[:, 2:5] = reference + force * (1 / k + 1 / ff)
            metadata[:, 5:9] = reference_quat[[1, 2, 3, 0]]
            # Pure translational pushes; no torque labels or rotational environment spring.
            item = dict(q_aug=q, q_ref=np.tile(Q_REF, (len(q), 1)), dq_aug=dq,
                        force=force, torque=np.zeros_like(force), k=np.full(len(q), k), kr=np.ones(len(q)),
                        metadata=metadata, keypoints=np.array(fk), quats=np.array(quats),
                        stage=stage, force_fraction=fraction)
            arrays.append(item)
            statistics.append(dict(**plan, frames=len(q), phase_frames={name: int((stage == i).sum()) for i, name in enumerate(STAGES)},
                                   ik_max_position_error_m=max(ik_error), ik_max_orientation_error_rad=max(orientation_error)))
            max_error = max(max_error, max(ik_error)); max_orientation_error = max(max_orientation_error, max(orientation_error))
            max_speed = max(max_speed, float(np.abs(dq).max()))
            if index % 24 == 23:
                print(f'Balanced CMA: {index + 1}/{len(clip_plans)} clips', flush=True)
        data = {key: np.stack([a[key] for a in arrays]) for key in arrays[0]}
        data['lengths'] = np.array([len(a['q_aug']) for a in arrays], dtype=np.int64)
        distribution = audit(data, statistics)
        assert distribution['all_arrays_finite']
        dataset = output / 'train.npz'; np.savez_compressed(dataset, **data)
        source = Path(__file__).read_bytes(); (output / 'generation_source.py').write_bytes(source)
        manifest = dict(status='complete', schema_version=2, upstream_commit=UPSTREAM_COMMIT,
                        method='Piper fixed-pose balanced CMA: stratified force schedules, original Piper Mink tasks and F/K forcefield construction; no policy/reward changes',
                        dt=DT, frames=len(time), trajectories=len(arrays), seed=seed,
                        sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                        parent_dataset_sha256=hashlib.sha256((DATASET / 'train.npz').read_bytes()).hexdigest(),
                        reward_adapter_sha256=hashlib.sha256(Path(__file__).with_name('env.py').read_bytes()).hexdigest(),
                        upstream_rewards_sha256=hashlib.sha256((UPSTREAM / 'softmimic_gym/softmimic_gym/tasks/locomotion/tracking/mdp/rewards.py').read_bytes()).hexdigest(),
                        generation_source_sha256=hashlib.sha256(source).hexdigest(), agent=AGENT,
                        stage_names=STAGES, distribution=distribution,
                        validation=dict(max_ik_position_error_m=max_error, max_ik_orientation_error_rad=max_orientation_error,
                                        max_joint_speed_rad_s=max_speed, max_static_motor_limit_fraction=max_motor_fraction),
                        omitted_grid_cases='5 N at K=60 N/m excluded uniformly from all directions (83 mm displacement); 5 N retained at K=100/160',
                        force_application='Reactive isotropic TCP field at 50 Hz in Isaac; metadata origin=x_ref+F/K+F/Kff, zero external torque. Target force is not a cap on off-target reactive force.',
                        timing='24 s: hold [0,6), push [6,18), release/recovery [18,24); force ramp-down occurs inside recovery. Three continuous profiles; no empty collision clips.',
                        clips=statistics)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'output': str(output), 'distribution': distribution, 'validation': manifest['validation']}, indent=2), flush=True)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps({'error': repr(error)}) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'dataset_balanced_v1')
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    generate(args.output, args.seed)


if __name__ == '__main__':
    main()
