import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_entrypoints import load
from test_rosbag_replay import config, recording
from test_rgb_act_execution import telemetry


def rules():
    params = {'ctrl_mode': 'mit', 'control_rate': 50., 'publish_rate': 100.,
        'mit': {'enable_pos': True, 'enable_vel': False, 'enable_tor': False, 'enable_gravity': True,
                'kp': 7., 'kd': .8, 'speed': 25, 'gravity_mix_mode': 'additive'},
        'filter': {'enable': True, 'alpha_position': .7}, 'follower_gravity': {'rate_hz': 100},
        'gripper_exist': True, 'enable_gripper': True, 'enable_gripper_haptic': False,
        'gripper_reverse': False, 'gripper_val_mutiple': 1., 'gripper_range': .08,
        'gripper_cmd_limit_enable': True, 'gripper_cmd_min_raw': 0, 'gripper_cmd_max_raw': 80000,
        'gripper_cmd_deadband_raw': 200}
    drivers = {side: dict(copy.deepcopy(params), can_port=port)
               for side, port in [('left', 'can_sl'), ('right', 'can_sr')]}
    drivers['right']['mit']['speed'] = 50
    return {'schema_version': 1, 'rate_hz': 100., 'drivers': drivers}


def plan():
    data = recording()
    for stream in data['samples'].values():
        for stamp, p in zip(stream['timestamps'], stream['positions']):
            p[0], p[1], p[2], p[4] = (stamp - 100) * .8, -.03, .02, 1.3
    cfg = dict(config(), rate_hz=100, trial_seconds=5)
    result = load('replay_rosbag').build_plan(data, cfg, 0, 1, 1, rules=rules())
    return cfg, dict(result, rules=rules())


def test_recording_rules_preserve_values_rate_and_duration_above_act_trial_limits():
    cfg, p = plan()
    a = np.asarray(p['actions'])
    assert p['passed'] and not p['constrained']
    assert p['rate_hz'] == 100 and p['execution_seconds'] == 1 and len(a) == 101
    assert np.all(a[:, [1, 8]] == -.03)
    assert np.all(a[:, [2, 9]] == .02)
    assert np.all(a[:, [4, 11]] == 1.3)
    assert a[-1, 0] == pytest.approx(.8)
    assert max(p['peak_target_speed']) > .2
    with pytest.raises(ValueError, match='Position exceeds'):
        load('rgb_act_execution').ExecutionGuard(config()).position(a[0].tolist())


def test_snapshot_loader_requires_complete_recording_parameters(tmp_path):
    mod = load('recorded_replay')
    bag = tmp_path / 'episode.bag'
    with pytest.raises(ValueError, match='repro_snapshot'):
        mod.load_recording_rules(bag)
    data = {'ros': {'key_param_namespaces': {
        '/piper_ctrl_%s_node' % side: {'status': 'ok', 'value': params}
        for side, params in rules()['drivers'].items()}}}
    snapshot = tmp_path / 'repro_snapshot.json'
    snapshot.write_text(json.dumps(data))
    loaded = mod.load_recording_rules(bag)
    assert loaded['drivers'] == rules()['drivers'] and len(loaded['snapshot_sha256']) == 64
    data['ros']['key_param_namespaces']['/piper_ctrl_right_node']['status'] = 'missing'
    snapshot.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unavailable'):
        mod.load_recording_rules(bag)


@pytest.mark.parametrize('change', ['torque', 'rate', 'nan-gain', 'gripper', 'constrain'])
def test_incompatible_recording_contract_is_rejected(change):
    mod = load('replay_rosbag')
    r = rules()
    if change == 'torque':
        r['drivers']['left']['mit']['enable_tor'] = True
    elif change == 'rate':
        r['drivers']['left']['publish_rate'] = 50
    elif change == 'nan-gain':
        r['drivers']['left']['mit']['kp'] = float('nan')
    elif change == 'gripper':
        r['drivers']['left']['gripper_reverse'] = True
    with pytest.raises(ValueError):
        mod.build_plan(recording(), dict(config(), rate_hz=100), 0, 1, 1,
                       rules=r, constrain=change == 'constrain')


def test_recorded_output_is_elapsed_time_driven_without_clipping_or_shaping():
    mod = load('recorded_replay')
    cfg, p = plan()
    guard = mod.RecordedReplayGuard(cfg, p['actions'][0], p)
    _, diagnostics, _, _ = telemetry()
    measured = p['actions'][0][:]
    guard.arm(measured, diagnostics, 10.)
    assert len(guard.queue) == 101
    now = 10.
    for _ in range(300):
        now += .01
        guard.last_client = now
        command = guard.step(measured, now)
        measured = command[:]
        if guard.phase == 'policy':
            break
    assert guard.phase == 'policy'
    for _ in range(200):
        now += .01
        guard.last_client = now
        command = guard.step(measured, now)
        index = min(int((now - guard.playback_started) * 100), 100)
        assert command == p['actions'][index]
        measured = command[:]
        if guard.phase == 'finished':
            break
    assert guard.phase == 'finished' and not guard.queue
    assert 1 <= now - guard.playback_started <= 1.011
    assert command[4] == 1.3 and command[1] == -.03
    with pytest.raises(ValueError, match='preloaded'):
        guard.submit({})
    guard.last_client = now - 1
    with pytest.raises(ValueError, match='heartbeat'):
        guard.step(measured, now + .01)


def test_recording_mode_does_not_apply_act_tracking_cutoff_to_recorded_following_error():
    cfg, p = plan()
    guard = load('recorded_replay').RecordedReplayGuard(cfg, p['actions'][0], p)
    _, diagnostics, _, _ = telemetry()
    guard.arm(p['actions'][0], diagnostics, 10.)
    guard.phase = 'policy'
    measured = p['actions'][0][:]
    measured[4] -= .31  # The actual bag contains this much MIT command/feedback separation.
    guard.last_client = 10.01
    assert guard.step(measured, 10.01) == p['actions'][0]


@pytest.mark.parametrize('changed', [None, 'mit', 'follower_gravity', 'gripper_cmd_max_raw'])
def test_live_parameters_must_match_recording_before_arm(monkeypatch, changed):
    mod = load('rgb_act_ros_executor')
    cfg, p = plan()
    values = {'/piper_ctrl_%s_node/%s' % (side, key): value
              for side, params in p['rules']['drivers'].items() for key, value in params.items()}
    if changed:
        values['/piper_ctrl_left_node/' + changed] = 'changed'
    ros = SimpleNamespace(get_name=lambda: '/private_replay_test', get_param_cached=lambda key, default: values.get(key, default))
    trial = mod.Trial(cfg, None, ros, False, {}, {}, io.StringIO(), None,
                      reset_positions=p['actions'][0], states_only=True, recording_plan=p)
    monkeypatch.setattr(mod, 'graph_connections', lambda name: ({}, {}))
    if changed:
        with pytest.raises(ValueError, match='differs from recording'):
            trial.check_graph(10.)
    else:
        trial.check_graph(10.)


def test_recording_client_waits_for_completion_without_uploading_act_chunks(monkeypatch):
    mod = load('replay_rosbag')
    states = iter([
        {'armed': False, 'ready': True, 'error': None},
        {'armed': True, 'ready': True, 'phase': 'reset', 'remaining': 101},
        {'armed': True, 'ready': True, 'phase': 'policy', 'remaining': 60},
        {'armed': True, 'ready': True, 'phase': 'finished', 'remaining': 0},
        {'armed': True, 'ready': True, 'phase': 'paused', 'remaining': 101},
        {'armed': False, 'ready': True, 'local_exit_requested': True},
    ])
    calls = []
    def request(session, route, payload=None):
        if route == '/status':
            return next(states)
        calls.append(route)
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod.time, 'sleep', lambda _: None)
    monkeypatch.setattr(mod, 'read_ros_observation', lambda *a, **kw: pytest.fail('Recording replay used an ACT observation loop'))
    assert mod.run_replay_client({'recording_rules': rules()}, [], 'unused', config()) == 0
    assert calls == ['/stop']


def playing_guard():
    cfg, p = plan()
    guard = load('recorded_replay').RecordedReplayGuard(cfg, p['actions'][0], p)
    _, diagnostics, _, _ = telemetry()
    guard.arm(p['actions'][0], diagnostics, 10.)
    guard.phase = 'policy'
    return guard, p


def tick(guard, now, measured=None):
    guard.last_client = now
    return guard.step(guard.last_command if measured is None else measured, now)


def test_pause_freezes_target_and_playback_clock_without_consuming_budget():
    guard, p = playing_guard()
    tick(guard, 10.01)
    for n in range(2, 22):
        command = tick(guard, 10 + n * .01)
    cursor = guard.cursor
    guard.toggle_pause(10.21)
    for n in range(1, 1001):
        assert tick(guard, 10.21 + n * .01) == command
    assert guard.cursor == cursor and guard.phase == 'paused'
    guard.toggle_pause(20.21)
    resumed = tick(guard, 20.22)
    assert resumed == p['actions'][min(int((20.22 - guard.playback_started) * 100), 100)]
    assert guard.cursor <= cursor + 2
    guard.last_client = 19.
    guard.toggle_pause(20.22)
    with pytest.raises(ValueError, match='heartbeat'):
        guard.step(resumed, 20.23)


@pytest.mark.parametrize('phase', ['policy', 'paused', 'finished', 'reset'])
def test_enter_slowly_returns_to_absolute_first_frame_then_waits_for_space(phase):
    guard, p = playing_guard()
    guard.first_frame = p['actions'][0][:]
    guard.first_frame[0] = -.1  # Frame zero differs from the selected segment's first target.
    guard.first_frame_feedback = guard.first_frame[:]
    measured = p['actions'][-1][:]
    guard.phase = phase
    guard.rewind(measured, 10.01)
    command = tick(guard, 10.02, measured)
    assert max(abs(a - b) for a, b in zip(command, measured)) <= .002
    now = 10.02
    for _ in range(6000):
        now += .01
        command = tick(guard, now, command)
        if guard.phase == 'paused':
            break
    assert guard.phase == 'paused'
    assert abs(command[0] - guard.first_frame[0]) < .01
    assert guard.cursor == 0 and guard.playback_started is None
    guard.toggle_pause(now)
    for _ in range(6000):
        now += .01
        command = tick(guard, now, command)
        if guard.phase == 'policy':
            break
    assert guard.phase == 'policy' and abs(command[0] - p['actions'][0][0]) < .01
    now += .01
    assert tick(guard, now) == p['actions'][0]


def test_pause_during_long_initial_positioning_and_hold_after_finish():
    guard, p = playing_guard()
    guard.phase, guard.started, guard.last_tick = 'reset', 0., 10.
    guard.toggle_pause(10.)
    assert tick(guard, 10.01) == p['actions'][0]
    guard.toggle_pause(10.01)
    assert guard.phase == 'reset'
    guard.phase, guard.last_tick, guard.started = 'finished', 20., 10.
    guard.queue.clear()
    for n in range(1, 1001):
        assert tick(guard, 20 + n * .01) == p['actions'][0]
    guard.last_tick = 30.
    with pytest.raises(ValueError, match='deadline'):
        guard.step(p['actions'][0], 31.)


def test_replay_controls_do_not_arm_a_disarmed_robot():
    cfg, p = plan()
    guard = load('recorded_replay').RecordedReplayGuard(cfg, p['actions'][0], p)
    for operation in (lambda: guard.toggle_pause(10.), lambda: guard.rewind(p['actions'][0], 10.)):
        with pytest.raises(ValueError, match='ARM'):
            operation()
    assert not guard.active


def test_keyboard_space_is_immediate_and_terminal_is_restored(monkeypatch):
    import os
    import pty
    import sys
    import termios
    mod = load('rgb_act_ros_executor')
    master, slave = pty.openpty()
    saved = termios.tcgetattr(slave)
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(fileno=lambda: slave))
    try:
        with pytest.raises(RuntimeError, match='restore'):
            with mod.ReplayKeyboard() as keyboard:
                assert not termios.tcgetattr(slave)[3] & termios.ICANON
                os.write(master, b'ARM\n')
                assert keyboard.read() == ['arm']
                os.write(master, b' ')
                assert keyboard.read() == ['pause']
                os.write(master, b'\nq')
                assert keyboard.read() == ['rewind', 'quit']
                raise RuntimeError('restore')
        assert termios.tcgetattr(slave) == saved
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize('wrong_rules', [False, True])
def test_launcher_passes_preloaded_plan_and_checks_recording_session_rules(monkeypatch, tmp_path, wrong_rules):
    mod = load('test_rgb_act')
    cfg, p = plan()
    args = SimpleNamespace(output_dir=tmp_path, mode='dry-run', states_only=True,
        observer_url='http://127.0.0.1:8766/observation', recording_rules=p['rules'],
        recording_plan=tmp_path / 'recording_plan.json', reset_pose={'positions': p['actions'][0]})
    args.recording_plan.write_text(json.dumps(p))
    (tmp_path / 'reset_pose.json').write_text(json.dumps(args.reset_pose))
    commands, calls = [], []
    def launch(command):
        commands.append(command[-1])
        folder = tmp_path / 'bridge'
        folder.mkdir()
        session = folder / 'session.json'
        session.write_text(json.dumps({'schema_version': 1, 'execute': False, 'routing': 'mux', 'config': cfg,
            'states_only': True, 'reset_positions': p['actions'][0],
            'recording_rules': {} if wrong_rules else p['rules']}))
        session.chmod(0o600)
        return SimpleNamespace(poll=lambda: None, wait=lambda timeout: None, send_signal=lambda sig: None)
    def client(session, model, url, limits):
        calls.append(True)
        assert session['recording_rules'] == p['rules'] and limits['rate_hz'] == 100
        return 0
    monkeypatch.setattr(mod.subprocess, 'Popen', launch)
    monkeypatch.setattr(mod, 'request', lambda *a: None)
    if wrong_rules:
        with pytest.raises(ValueError, match='Unexpected ROS bridge session'):
            mod.run_trial(args, [], cfg, client=client)
        assert calls == []
    else:
        assert mod.run_trial(args, [], cfg, client=client) == 0 and calls == [True]
    assert '--states-only' in commands[0] and '--recording-plan' in commands[0]


@pytest.mark.parametrize('reached', [False, True])
def test_reset_uses_recorded_gripper_feedback_instead_of_open_command(reached):
    cfg, p = plan()
    for row in p['actions']:
        row[6] = row[13] = .08
    p['first_frame_positions'] = p['actions'][0][:]
    expected = p['actions'][0][:]
    expected[6], expected[13] = .06972, .06965
    p['reset_feedback_positions'] = expected[:]
    p['first_frame_feedback_positions'] = expected[:]
    guard = load('recorded_replay').RecordedReplayGuard(cfg, p['actions'][0], p)
    _, diagnostics, _, _ = telemetry()
    measured = expected[:]
    measured[6] = measured[13] = 0.
    guard.arm(measured, diagnostics, 10.)
    for n in range(1, 1201):
        now = 10. + n * .01
        guard.last_client = now
        command = guard.step(measured, now)
        measured = command[:]
        measured[6] = min(command[6], expected[6] if reached else .05)
        measured[13] = min(command[13], expected[13] if reached else .05)
        if guard.phase == 'policy':
            break
    assert guard.phase == ('policy' if reached else 'reset')
    assert command[6] == pytest.approx(.08, abs=.002)
    assert command[13] == pytest.approx(.08, abs=.002)
    progress = guard.reset_progress(measured, now)
    assert progress['gripper_expected_m'] == [.06972, .06965]
    if reached:
        assert max(progress['gripper_error_m']) < .002
        guard.rewind(measured, now + .01)
        assert guard.reset_feedback_positions == expected
        guard.toggle_pause(now + .02)
        guard.toggle_pause(now + .03)
        assert guard.reset_feedback_positions == expected
        guard.stop('test')
        guard.arm(measured, diagnostics, now + .04)
        assert guard.reset_feedback_positions == expected
