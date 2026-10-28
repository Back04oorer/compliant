"""Held-out force timing and recovery metrics, without launching Isaac."""
import numpy as np
import pytest

from sim.piper_softmimic.evaluate import DT, envelope, case_metrics, settling_time


def test_held_out_envelope_and_recovery_measure_against_commanded_pose():
    time = np.arange(601) * DT
    force = np.zeros((len(time), 3)); force[:, 0] = envelope(time) * 2
    case = dict(axis='X', sign=1, force_N=2, stiffness_N_per_m=100)
    position = force / 100
    result = case_metrics(time, position, np.zeros(3), force, case)
    assert result['steady_displacement_m'] == pytest.approx(.02)
    assert result['steady_target_rmse_m'] == 0
    assert result['recovery_2mm_s'] == 0
    assert result['push_response_90pct_s'] == pytest.approx(.46)
    # A permanent 4 mm offset returns to its pre-push pose, but fails absolute 2 mm recovery.
    position[:, 1] += .004
    result = case_metrics(time, position, np.zeros(3), force, case)
    assert result['pre_push_bias_m'] == pytest.approx(.004)
    assert result['recovery_2mm_s'] is None
    assert result['recovery_5mm_s'] == 0
    assert result['return_to_pre_push_2mm_s'] == 0
    assert result['final_error_m'] == pytest.approx(.004)


def test_settling_requires_remaining_in_band_and_minimum_hold():
    time = np.arange(601) * DT
    distances = np.zeros_like(time)
    distances[(time >= 6) & (time < 8)] = .003
    assert settling_time(time, distances, .002) == pytest.approx(2)
    distances[-10] = .003
    assert settling_time(time, distances, .002) is None
    distances[-1] = .003
    assert settling_time(time, distances, .002) is None


def test_negative_axis_uses_positive_deflection_in_push_direction():
    time = np.arange(601) * DT
    force = np.zeros((len(time), 3)); force[:, 2] = -envelope(time) * 5
    position = force / 100
    result = case_metrics(time, position, np.zeros(3), force,
                          dict(axis='Z', sign=-1, force_N=5, stiffness_N_per_m=100))
    assert result['steady_displacement_m'] == pytest.approx(.05)
    assert result['steady_cross_axis_rmse_m'] == 0
    assert result['steady_target_rmse_m'] == 0


def test_shared_evaluation_actor_reads_proprioception_and_never_external_force():
    from types import SimpleNamespace
    import torch
    from sim.piper_softmimic.evaluate import actor_observation
    env = SimpleNamespace(num_envs=2, device='cpu', qref=torch.zeros(2, 6),
                          actions=torch.zeros(2, 6), hist=torch.zeros(2, 3, 39),
                          new_history=torch.ones(2, dtype=torch.bool),
                          actor_history_order=torch.arange(117),
                          robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2, 6),
                                                                    joint_vel=torch.zeros(2, 6))))
    first = actor_observation(env, torch.ones(2) * 100, torch.ones(2))
    env.external_force = torch.ones(2, 3) * 50
    env.hist.zero_(); env.new_history[:] = True
    second = actor_observation(env, torch.ones(2) * 100, torch.ones(2))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    env.robot.data.joint_pos[0, 0] = .01
    env.hist.zero_(); env.new_history[:] = True
    third = actor_observation(env, torch.ones(2) * 100, torch.ones(2))
    assert not torch.equal(second, third)


def test_frozen_evaluation_entry_point_generates_targets_without_legacy_rl(tmp_path):
    pytest.importorskip('mujoco')
    import json
    from sim.piper_softmimic.evaluate import prepare_protocol
    path = tmp_path / 'protocol.npz'
    prepare_protocol(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert len(metadata['cases']) + len(metadata['excluded_cases']) == 36
    with np.load(path) as data:
        assert data['qgoal'].shape[1] == len(metadata['cases'])
        assert all(np.isfinite(data[key]).all() for key in data.files)
    with pytest.raises(ValueError, match='overwrite'):
        prepare_protocol(path)
