"""First-frame positioning must not bypass motion guards or reuse pre-reset images."""
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_entrypoints import ROOT, load
from test_rgb_act_execution import config, plan, pose, telemetry, fake_trial


def reset_guard(target=None):
    guard = load('rgb_act_execution').ExecutionGuard(config(), pose() if target is None else target)
    guard.arm(pose(), telemetry()[1], 10.0)
    return guard


def test_reset_is_locally_armed_and_rejects_policy_chunks_until_settled():
    guard = reset_guard()
    assert guard.phase == 'reset'
    with pytest.raises(ValueError, match='positioning is not complete'):
        guard.submit(plan(), pose(), 100.0, 10.0)
    for i in range(1, 19):
        now = 10 + i / 30
        guard.last_client = now
        guard.step(pose(), now)
    assert guard.phase == 'policy' and guard.started > 10.5
    # The ACT observation must be entirely newer than the completed positioning.
    guard.policy_after = 100.0
    with pytest.raises(ValueError, match='fresh observations after'):
        guard.submit(plan(), pose(), 100.0, now)
    guard.submit(plan(stamp=100.01), pose(), 100.01, now)
    assert len(guard.queue) == 50


def test_reset_moves_large_distance_with_existing_speed_and_acceleration_bounds():
    target = pose()
    target[0], target[6] = 1.0, 0.06  # Exceeds the policy's initial-delta threshold.
    guard, measured = reset_guard(target), pose()
    velocities = [0.0] * 14
    for i in range(1, 500):
        now = 10 + i / 30
        guard.last_client = now
        command = guard.step(measured, now)
        next_velocities = [(a - b) * 30 for a, b in zip(command, measured)]
        for v, old, vmax, amax in zip(next_velocities, velocities, guard.limits(.2, .015), guard.limits(1, .05)):
            assert abs(v) <= vmax + 1e-8
            assert abs(v - old) <= amax / 30 + 1e-8
        measured, velocities = command, next_velocities
        if guard.phase == 'policy':
            break
    assert guard.phase == 'policy'
    assert abs(measured[0] - target[0]) <= .01
    assert abs(measured[6] - target[6]) <= .002
    assert guard.started > 15  # Reset can exceed five seconds without consuming the ACT trial budget.


def test_reset_requires_stationary_feedback_for_half_a_second():
    guard = reset_guard()
    for i in range(1, 40):
        now = 10 + i / 30
        guard.last_client = now
        guard.step(pose(), now, stationary=False)
    assert guard.phase == 'reset' and guard.reset_settled_since is None


@pytest.mark.parametrize('fault', ['heartbeat', 'tracking', 'deadline', 'timeout'])
def test_reset_still_obeys_watchdogs_and_stops(fault):
    guard, measured, now = reset_guard(), pose(), 10.03
    if fault == 'heartbeat':
        guard.last_client = 9
    elif fault == 'tracking':
        measured[0] = .3
    elif fault == 'deadline':
        now = 10.2
    else:
        guard.started = -51
    with pytest.raises(ValueError):
        guard.step(measured, now)
    guard.stop('test')
    assert guard.step(measured, now) is None and guard.phase == 'stopped'


def test_dry_run_reset_has_no_actuator_output():
    trial, published = fake_trial()
    trial.guard = reset_guard()
    trial.guard.last_tick = trial.guard.last_client = trial.guard.started = trial.guard.last_plan = time.monotonic()
    trial.tick()
    trial.stop('operator stop during reset')
    assert published == {'left': [], 'right': []}
    assert not trial.guard.active


def test_bridge_marks_completion_with_new_ros_time_and_keeps_policy_queue_empty(monkeypatch):
    mod = load('rgb_act_ros_executor')
    trial, published = fake_trial()
    trial.__class__ = mod.Trial
    trial.guard = reset_guard()
    trial.guard.reset_settled_since = 9.0
    trial.ros.Time.now = lambda: SimpleNamespace(to_sec=lambda: 101.0)
    monkeypatch.setattr(mod.time, 'monotonic', lambda: 10.03)
    trial.tick()
    assert trial.guard.phase == 'policy' and trial.guard.policy_after == 101.0
    assert not trial.guard.queue and published == {'left': [], 'right': []}


def test_dataset_reset_selects_measured_positions_not_efforts_or_actions(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    data = tmp_path / 'data/chunk-000'
    data.mkdir(parents=True)
    (data / 'file-000.parquet').touch()
    state = pose()[:7] + [999] * 7 + pose()[7:] + [-999] * 7
    arrow = ModuleType('pyarrow')
    parquet = ModuleType('pyarrow.parquet')
    def read(path, *, columns, filters):
        assert 'action' not in columns
        assert filters == [('episode_index', '=', 0), ('frame_index', '=', 0)]
        return SimpleNamespace(to_pylist=lambda: [{'observation.state': state}])
    parquet.read_table = read
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, 'pyarrow', arrow)
    monkeypatch.setitem(sys.modules, 'pyarrow.parquet', parquet)
    monkeypatch.setattr(mod, 'checkpoint_dataset_root', lambda checkpoint: tmp_path)
    assert mod.dataset_first_frame(tmp_path)['positions'] == pose()
    parquet.read_table = lambda *a, **kw: SimpleNamespace(to_pylist=lambda: [])
    with pytest.raises(ValueError, match='exactly one'):
        mod.dataset_first_frame(tmp_path)


def test_out_of_bounds_dataset_reset_fails_before_loading_model_or_starting_bridge(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'validate_checkpoint_contract', lambda *a: None)
    target = pose()
    target[1] = -12.501
    monkeypatch.setattr(mod, 'dataset_first_frame', lambda path: {'positions': target})
    monkeypatch.setattr(mod, 'RGBActPredictor', lambda *a: pytest.fail('Invalid reset loaded model'))
    monkeypatch.setattr(mod, 'run_trial', lambda *a: pytest.fail('Invalid reset started bridge'))
    args = SimpleNamespace(checkpoint=tmp_path, reset_to_first_frame=True, output_dir=tmp_path)
    with pytest.raises(ValueError, match='Dataset episode 0 frame 0.*left.joint2.pos'):
        mod.preflight_trial(args)
    assert json.loads((tmp_path / 'reset_pose.json').read_text())['positions'] == target


@pytest.mark.parametrize('mit_pose', [False, True])
def test_valid_reset_defers_predictions_until_positioning_completes(monkeypatch, tmp_path, mit_pose):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'validate_checkpoint_contract', lambda *a: None)
    target = pose()
    if mit_pose:
        target[1], target[2], target[8], target[9] = -.0264, .0081, -.0336, .0301
    monkeypatch.setattr(mod, 'dataset_first_frame', lambda path: {'positions': target})
    predictor = SimpleNamespace(contract={'action_shape': [50, 14]},
                                predict=lambda obs: pytest.fail('Predicted before reset completed'))
    monkeypatch.setattr(mod, 'RGBActPredictor', lambda *a: predictor)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda *a: None))
    state = np.array(pose()[:7] + [0] * 7 + pose()[7:] + [0] * 7)
    monkeypatch.setattr(mod, 'read_ros_observation', lambda *a: ({'observation.state': state}, {}))
    calls = []
    monkeypatch.setattr(mod, 'run_trial', lambda args, model, cfg: calls.append((args.reset_pose, model)) or 0)
    args = SimpleNamespace(checkpoint=tmp_path, reset_to_first_frame=True, output_dir=tmp_path,
                           observer_url='unused', mode='dry-run', device='cpu')
    assert mod.preflight_trial(args) == 0
    assert calls == [({'positions': target}, predictor)]
    assert args.live_act is True
    assert json.loads((tmp_path / 'preflight.json').read_text())['action_preflight_passed'] is None


def test_client_waits_for_reset_and_new_observations_before_predicting(monkeypatch):
    mod = load('execute_rgb_act')
    states = iter([
        {'armed': True, 'ready': True, 'phase': 'reset'},
        {'armed': True, 'ready': True, 'phase': 'policy', 'remaining': 0, 'sequence': 0, 'policy_after': 100},
        {'armed': True, 'ready': True, 'phase': 'policy', 'remaining': 0, 'sequence': 0, 'policy_after': 100},
        {'armed': False, 'reason': 'stopped'},
    ])
    observations = iter([({'old': True}, {'source_stamps': {'left': 99.9}}),
                         ({'fresh': True}, {'source_stamps': {'left': 100.1}, 'observation_stamp': 100.1})])
    routes, predicted = [], []
    def request(session, route, payload=None):
        routes.append(route)
        return next(states) if route == '/status' else {'sequence': 1}
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod, 'read_ros_observation', lambda *a, **kw: next(observations))
    monkeypatch.setattr(mod.time, 'sleep', lambda *a: None)
    predictor = SimpleNamespace(predict=lambda obs: predicted.append(obs) or np.zeros((50, 14)))
    assert mod.run_client({'execute': False}, predictor, 'unused', config()) == 0
    assert predicted == [{'fresh': True}]
    assert routes == ['/status', '/status', '/status', '/chunk', '/status', '/stop']


def test_predictions_reject_reset_before_creating_output(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    output = tmp_path / 'unused'
    monkeypatch.setattr(sys, 'argv', ['test', '--checkpoint', str(tmp_path), '--output-dir', str(output),
                                     '--mode', 'predictions', '--reset-to-first-frame'])
    with pytest.raises(SystemExit) as error:
        mod.main()
    assert error.value.code == 2 and not output.exists()


def test_bridge_rejects_invalid_reset_before_importing_ros_or_creating_session(monkeypatch, tmp_path):
    mod = load('rgb_act_ros_executor')
    target = pose()
    target[2] = .03
    reset = tmp_path / 'reset.json'
    reset.write_text(json.dumps({'positions': target}))
    session = tmp_path / 'session'
    monkeypatch.setattr(sys, 'argv', ['bridge', '--config', str(ROOT / 'config/rgb_act_execution.json'),
                                     '--session-dir', str(session), '--reset-pose', str(reset), '--execute'])
    with pytest.raises(ValueError, match='left.joint3.pos'):
        mod.main()
    assert not session.exists()
