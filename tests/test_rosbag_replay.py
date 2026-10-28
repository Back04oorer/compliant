from contextlib import contextmanager
import io
import json
import copy
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_entrypoints import ROOT, load


def config():
    return json.loads((ROOT / 'config/rgb_act_execution.json').read_text())


def recording():
    names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    samples = {}
    for side in ('left', 'right'):
        t = np.arange(0, 3.01, .01)
        poses = np.tile([0, .5, -.5, 0, 0, 0, .04], (len(t), 1))
        poses[:, 0] = t * .02
        samples[side] = {'timestamps': (100 + t).tolist(), 'positions': poses.tolist()}
    return {'schema_version': 1, 'joint_names': names, 'source': 'mux', 'bag': '/recording.bag',
            'topics': {}, 'samples': samples}


def test_plan_preserves_joint_order_timing_and_slowdown():
    mod = load('replay_rosbag')
    plan = mod.build_plan(recording(), config(), .5, 1, .25)
    assert plan['passed']
    assert len(plan['actions']) == 121
    assert plan['actions'][0][0] == pytest.approx(.01)
    assert plan['actions'][-1][7] == pytest.approx(.03)
    assert plan['peak_target_speed'][0] == pytest.approx(.005)


def test_raw_violation_between_output_ticks_is_not_hidden():
    mod = load('replay_rosbag')
    data = recording()
    data['samples']['left']['positions'][1][1] = -.001
    plan = mod.build_plan(data, config(), 0, 1, 1)
    assert not plan['passed']
    assert plan['position_violations'][0]['joint'] == 'left.joint2'


def test_constrained_replay_reports_changes_without_mutating_recording_and_retimes_fast_motion():
    mod = load('replay_rosbag')
    data = recording()
    for stream in data['samples'].values():
        for stamp, p in zip(stream['timestamps'], stream['positions']):
            p[0] = (stamp - 100) * .8
            p[1], p[2], p[4] = -.03, .02, 1.3
    original = copy.deepcopy(data)
    plan = mod.build_plan(data, config(), 0, 1, 1, constrain=True)
    assert data == original and plan['passed'] and plan['constrained']
    assert {v['joint'] for v in plan['position_violations']} == {
        side + '.' + joint for side in ('left', 'right') for joint in ('joint2', 'joint3', 'joint5')}
    a = np.asarray(plan['actions'])
    assert (a >= np.asarray(config()['arm_position_min'] * 2)).all()
    assert (a <= np.asarray(config()['arm_position_max'] * 2)).all()
    assert plan['execution_seconds'] > 1
    for field, limits in [('peak_target_speed', [.2] * 6 + [.015]),
                          ('peak_target_acceleration', [1.] * 6 + [.05])]:
        assert (np.asarray(plan[field]) <= np.asarray(limits * 2) + 1e-8).all()
    assert np.all(a[-60:] == a[-1])
    assert not mod.build_plan(data, config(), 0, 1, 1)['passed']


@pytest.mark.parametrize('fault', ['gap', 'nan', 'jump'])
def test_constrained_replay_does_not_hide_corrupt_recordings(fault):
    mod = load('replay_rosbag')
    data = recording()
    if fault == 'gap':
        for stream in data['samples'].values():
            del stream['timestamps'][5:30]
            del stream['positions'][5:30]
    elif fault == 'nan':
        data['samples']['left']['positions'][5][0] = float('nan')
    else:
        data['samples']['left']['positions'][5][0] = 1.
    if fault == 'nan':
        with pytest.raises(ValueError):
            mod.build_plan(data, config(), 0, 1, 1, constrain=True)
    else:
        assert not mod.build_plan(data, config(), 0, 1, 1, constrain=True)['passed']


def test_replay_budget_does_not_extend_act_trials():
    mod = load('rgb_act_execution')
    cfg = config()
    cfg['trial_seconds'] = 120
    assert mod.ExecutionGuard(cfg, states_only=True).config['trial_seconds'] == 120
    with pytest.raises(ValueError, match='60 seconds'):
        mod.ExecutionGuard(cfg)
    cfg['trial_seconds'] = 1801
    with pytest.raises(ValueError, match='1800 seconds'):
        mod.ExecutionGuard(cfg, states_only=True)


@pytest.mark.parametrize('joint,value', [(0, 2.7), (1, -.051), (1, 3.15), (2, .051), (4, 1.23), (6, .081)])
def test_replay_feedback_tolerance_is_only_for_j2_j3_near_zero(joint, value):
    from test_rgb_act_execution import pose
    guard = load('rgb_act_execution').ExecutionGuard(config(), states_only=True)
    p = pose()
    p[joint] = value
    with pytest.raises(ValueError, match='recovery envelope'):
        guard.measured_position(p)


def test_replay_recovers_measured_start_inward_but_never_accepts_out_of_limit_targets():
    from test_rgb_act_execution import pose, telemetry
    mod = load('rgb_act_execution')
    guard = mod.ExecutionGuard(config(), reset_positions=pose(), states_only=True)
    measured = pose()
    measured[1], measured[2] = -.03, .02
    _, diagnostics, _, _ = telemetry()
    assert guard.measured_position(measured) == measured
    with pytest.raises(ValueError, match='Position exceeds'):
        guard.position(measured)
    far = measured[:]
    far[1] = -.3
    with pytest.raises(ValueError, match='recovery envelope'):
        guard.measured_position(far)
    guard.arm(measured, diagnostics, 10.)
    previous = measured[:]
    previous_velocity = [0.] * 14
    for tick in range(1, 301):
        now = 10. + tick / 30
        guard.last_client = now
        command = guard.step(measured, now)
        assert command[1] >= min(0, previous[1]) - 1e-12
        assert command[2] <= max(0, previous[2]) + 1e-12
        velocity = (np.asarray(command) - previous) * 30
        assert (np.abs(velocity) <= np.asarray(([.2] * 6 + [.015]) * 2) + 1e-8).all()
        assert (np.abs(velocity - previous_velocity) * 30 <= np.asarray(([1.] * 6 + [.05]) * 2) + 1e-6).all()
        measured = command[:]
        previous, previous_velocity = command[:], velocity
        if guard.phase == 'policy':
            break
    assert guard.phase == 'policy'


@pytest.mark.parametrize('problem', ['gap', 'fast', 'out-of-range', 'nan', 'order'])
def test_bad_trajectory_is_rejected(problem):
    mod = load('replay_rosbag')
    data = recording()
    if problem == 'gap':
        for stream in data['samples'].values():
            del stream['timestamps'][5:30]
            del stream['positions'][5:30]
    elif problem == 'fast':
        for p in data['samples']['left']['positions']:
            p[0] *= 100
    elif problem == 'nan':
        data['samples']['left']['positions'][5][0] = float('nan')
    elif problem == 'order':
        data['samples']['left']['timestamps'][5] = 100
    if problem in {'out-of-range', 'nan', 'order'}:
        with pytest.raises(ValueError):
            mod.build_plan(data, config(), 0, 10 if problem == 'out-of-range' else 1, .25)
    else:
        assert not mod.build_plan(data, config(), 0, 1, .25)['passed']


@pytest.mark.parametrize('speed,duration', [(0, 1), (2, 1), (.25, 20), (.25, float('nan'))])
def test_bad_timing_options(speed, duration):
    with pytest.raises(ValueError):
        load('replay_rosbag').build_plan(recording(), config(), 0, duration, speed)


def test_extraction_only_reads_selected_topics_and_reorders_named_joints(monkeypatch, tmp_path):
    mod = load('rosbag_trajectory')
    bag = tmp_path / 'episode.bag'
    bag.touch()
    names = list(reversed(mod.JOINT_NAMES))
    class Bag:
        def __init__(self, path, mode):
            assert path == str(bag) and mode == 'r'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_messages(self, topics):
            assert topics == ['/robot/arm_left/joint_cmd_mux', '/robot/arm_right/joint_cmd_mux']
            for stamp in (100, 100.1):
                for topic in topics:
                    msg = SimpleNamespace(name=names, position=list(range(7)),
                        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp)))
                    yield topic, msg, SimpleNamespace(to_sec=lambda: stamp)
    monkeypatch.setitem(sys.modules, 'rosbag', SimpleNamespace(Bag=Bag))
    result = mod.read_trajectory(bag, 'mux')
    assert result['samples']['left']['positions'][0] == list(reversed(range(7)))


def test_replay_waits_for_arm_and_positioning_pads_last_chunk_and_stops(monkeypatch):
    mod = load('replay_rosbag')
    def status(**kw):
        return dict(armed=True, ready=True, error=None, phase='policy', remaining=0,
                    sequence=0, policy_after=0, **kw)
    states = [dict(status(), armed=False), dict(status(), phase='reset'), status(),
              dict(status(), remaining=50, sequence=1), dict(status(), sequence=1),
              dict(status(), sequence=2)]
    calls = []
    def request(session, route, payload=None):
        if route == '/status': return states.pop(0)
        calls.append((route, payload))
        return {}
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod.time, 'sleep', lambda _: None)
    def read(*a, **kwargs):
        assert kwargs['states_only'] is True
        return {}, {'source_stamps': {'left': 1, 'right': 1}, 'observation_stamp': 1}
    monkeypatch.setattr(mod, 'read_ros_observation', read)
    actions = [[i * .001] * 14 for i in range(51)]
    assert mod.run_replay_client({}, actions, 'unused', config()) == 0
    assert [route for route, _ in calls] == ['/chunk', '/chunk', '/stop']
    assert calls[0][1]['actions'] == actions[:50]
    assert calls[1][1]['actions'] == [actions[-1]] * 50


@pytest.mark.parametrize('failure', ['readiness', 'interrupt', 'chunk'])
def test_replay_always_requests_stop_on_runtime_failure(monkeypatch, failure):
    mod = load('replay_rosbag')
    calls = []
    def request(session, route, payload=None):
        calls.append(route)
        if route == '/status':
            return {'armed': True, 'ready': failure != 'readiness', 'error': 'stale feedback',
                    'remaining': 0, 'phase': 'policy', 'sequence': 0, 'policy_after': 0}
        if route == '/chunk':
            raise ValueError('Chunk rejected')
    def observation(*a, **kw):
        if failure == 'interrupt': raise KeyboardInterrupt
        return {}, {'source_stamps': {'test': 1}, 'observation_stamp': 1}
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod, 'read_ros_observation', observation)
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else ValueError):
        mod.run_replay_client({}, [[0] * 14], 'unused', config())
    assert calls[-1] == '/stop'


@pytest.mark.parametrize('execute', [False, True])
def test_tui_dispatches_replay_without_robot_startup(monkeypatch, execute):
    mod = load('teleop_tui')
    choices, calls = iter(['b', 'q']), []
    monkeypatch.setattr(sys, 'argv', ['tui'] + (['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr('builtins.input', lambda _: next(choices))
    monkeypatch.setattr(mod, 'hardware_status', lambda _: [])
    monkeypatch.setattr(mod, 'run_data_action', lambda *a: calls.append(a))
    monkeypatch.setattr(mod, 'piper_args', lambda *a: pytest.fail('Started robot drivers'))
    assert mod.main() == 0
    assert calls[0][0] == 'bag-replay'


def test_tui_replay_command_has_no_unfiltered_rosbag_play(monkeypatch, tmp_path):
    mod = load('teleop_tui')
    bag = tmp_path / 'episode.bag'
    bag.touch()
    values = iter([str(bag), '', '', '', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(values))
    command = mod.data_command('bag-replay', '/unused')
    assert command[1].endswith('/replay_rosbag.py')
    assert command[command.index('--mode') + 1] == 'robot'
    assert command[command.index('--source') + 1] == 'mux'
    assert command[command.index('--speed') + 1] == '1.0'
    assert command[command.index('--rules') + 1] == 'recording'
    assert '--constrain' not in command
    assert '--duration' not in command


def test_whole_episode_default_and_rewind_frame_are_independent_of_segment_start():
    from test_recorded_replay import rules
    mod = load('replay_rosbag')
    cfg = dict(config(), rate_hz=100)
    full = mod.build_plan(recording(), cfg, 0, None, 1, rules=rules())
    segment = mod.build_plan(recording(), cfg, 1, None, 1, rules=rules())
    assert full['duration_s'] == pytest.approx(3)
    assert segment['duration_s'] == pytest.approx(2)
    assert segment['first_frame_positions'] == full['actions'][0]
    from recorded_replay import RecordedReplayGuard
    RecordedReplayGuard(dict(cfg, trial_seconds=full['execution_seconds'] + 3), full['actions'][0],
                        dict(full, rules=rules()))


def test_replay_mux_route_is_available_without_starting_drivers():
    mod = load('piper')
    cfg = json.loads((ROOT / 'config/hardware.example.json').read_text())
    command = mod.build_command('mux-vla', cfg)
    assert command == ['bash', str(ROOT / 'workspaces/scripts/switch_robot_mux_with_status.sh'), 'vla']
    assert load('teleop_tui').TELEOP_ACTIONS['8'][1] == 'mux-vla'


def test_replay_bridge_fault_is_not_reported_as_success(monkeypatch):
    mod = load('replay_rosbag')
    states = iter([{'armed': True, 'ready': True, 'remaining': 50, 'phase': 'policy'},
                   {'armed': False, 'reason': 'STOPPED: stale CAN feedback'}])
    calls = []
    def request(session, route, payload=None):
        if route == '/status':
            return next(states)
        calls.append(route)
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='stopped before completion.*stale CAN'):
        mod.run_replay_client({}, [[0] * 14], 'unused', config())
    assert calls == ['/stop']


def test_invalid_recording_is_blocked_before_observer_or_execution_bridge(monkeypatch, tmp_path):
    mod = load('replay_rosbag')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/rgb_act_execution.json').write_text(json.dumps(config()))
    bag = tmp_path / 'episode.bag'
    bag.touch()
    output = tmp_path / 'outputs/replay'
    data = recording()
    data['samples']['left']['positions'][1][1] = -.02
    monkeypatch.setattr(sys, 'argv', ['replay', '--rules', 'act', '--bag', str(bag), '--mode', 'robot',
                                    '--duration', '1', '--output-dir', str(output)])
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **k:
                        (output / 'recorded_trajectory.json').write_text(json.dumps(data)))
    monkeypatch.setattr(mod, 'managed_observer', lambda *a: pytest.fail('Started observation process'))
    monkeypatch.setattr(mod, 'run_trial', lambda *a, **k: pytest.fail('Started execution bridge'))
    with pytest.raises(ValueError, match='before starting any execution bridge'):
        mod.main()
    assert json.loads((output / 'preflight.json').read_text())['passed'] is False


def test_replay_launches_state_only_observer_and_bridge(monkeypatch, tmp_path):
    mod = load('replay_rosbag')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    from test_recorded_replay import rules
    monkeypatch.setattr(mod, 'load_recording_rules', lambda path: rules())
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/rgb_act_execution.json').write_text(json.dumps(config()))
    bag = tmp_path / 'episode.bag'
    bag.touch()
    output = tmp_path / 'outputs/replay'
    monkeypatch.setattr(sys, 'argv', ['replay', '--bag', str(bag), '--mode', 'robot',
                                    '--duration', '1', '--output-dir', str(output)])
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    def extract(*a, **k):
        data = recording()
        primary = output / 'recorded_trajectory.json'
        feedback = primary.exists()
        if feedback:
            data['source'] = 'state'
        for stream in data['samples'].values():
            for row in stream['positions']:
                row[6] = .0697 if feedback else .08
        (output / ('recorded_feedback.json' if feedback else 'recorded_trajectory.json')).write_text(json.dumps(data))
    monkeypatch.setattr(mod.subprocess, 'run', extract)
    @contextmanager
    def observer(path, *, states_only):
        assert path == output and states_only is True
        yield 'local-observer'
    # Left/right positions occupy separate seven-position/seven-effort blocks.
    state = np.array(([0, .5, -.5, 0, 0, 0, .04] + [0] * 7) * 2)
    def read(url, *, states_only):
        assert url == 'local-observer' and states_only is True
        return {'observation.state': state}, {}
    def trial(args, actions, limits, *, client):
        assert args.states_only is True and args.observer_url == 'local-observer'
        assert args.recording_rules == rules() and args.recording_plan.is_file()
        rp = json.loads(args.recording_plan.read_text())
        assert rp['actions'][0][6] == .08 and rp['reset_feedback_positions'][6] == .0697
        assert limits['rate_hz'] == 100
        assert client is mod.run_replay_client and len(actions) > 0
        return 0
    monkeypatch.setattr(mod, 'managed_observer', observer)
    monkeypatch.setattr(mod, 'read_ros_observation', read)
    monkeypatch.setattr(mod, 'run_trial', trial)
    assert mod.main() == 0


def test_camera_free_observation_can_arm_and_submit_but_stale_or_missing_feedback_blocks(monkeypatch):
    from test_rgb_act_execution import pose, telemetry
    observer = load('rgb_act_ros_observer')
    executor = load('rgb_act_ros_executor')
    predictor = load('predict_rgb_act')
    buffer = observer.ObservationBuffer(states_only=True)
    stamp, now = time.time(), time.monotonic()
    for side in observer.STATE_TOPICS:
        buffer.add(side, SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp)),
            name=observer.JOINT_NAMES, position=pose()[:7], effort=[0] * 7), received=now)
    server = observer.make_server(buffer, 0, lambda: stamp)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setitem(sys.modules, 'cv2', None)
    try:
        url = 'http://127.0.0.1:%d/observation' % server.server_port
        observation, metadata = predictor.read_ros_observation(url, states_only=True)
        assert set(metadata['source_stamps']) == {'left', 'right'}
        assert list(observation) == ['observation.state']
        with pytest.raises(ImportError):
            predictor.read_ros_observation(url)  # ACT still requires image decoding.
        ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: stamp)))
        pubs = {side: SimpleNamespace(publish=lambda msg: None) for side in observer.STATE_TOPICS}
        trial = executor.Trial(config(), buffer, ros, True, pubs, {}, io.StringIO(),
                               lambda: SimpleNamespace(header=SimpleNamespace()), states_only=True,
                               mux_select=lambda side, topic: topic)
        _, trial.diagnostics, trial.selected, trial.gravity = telemetry(time.monotonic(), stamp)
        trial.check_graph = lambda now: None  # Graph checks have separate coverage.
        trial.arm()
        trial.submit({'sequence': 1, 'rate_hz': 30, 'action_names': observer.ACTION_NAMES,
                      **metadata, 'actions': [pose()] * 50})
        assert trial.guard.active and len(trial.guard.queue) == 50
        with pytest.raises(ValueError, match='Stale'):
            buffer.snapshot(stamp + 1, time.monotonic())
        buffer.invalidate('left', 'feedback lost')
        assert trial.status()['ready'] is False
        trial.tick()
        assert trial.guard.active is False
        with pytest.raises(ValueError, match='Missing observations: left'):
            trial.arm()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


@pytest.mark.parametrize('fault', ['missing', 'stale', 'future', 'skew', 'before-positioning'])
def test_state_only_chunks_still_require_fresh_synchronized_follower_timestamps(fault):
    from test_rgb_act_execution import pose, telemetry
    mod = load('rgb_act_execution')
    guard = mod.ExecutionGuard(config(), states_only=True)
    _, diagnostics, _, _ = telemetry()
    guard.arm(pose(), diagnostics, 10)
    stamps = {'left': 100., 'right': 100.}
    if fault == 'missing':
        stamps.pop('left')
    elif fault == 'stale':
        stamps = dict.fromkeys(stamps, 99.)
    elif fault == 'future':
        stamps = dict.fromkeys(stamps, 101.)
    elif fault == 'skew':
        stamps['left'] = 99.9
    else:
        guard.policy_after = 100.
    with pytest.raises(ValueError):
        guard.submit({'sequence': 1, 'rate_hz': 30, 'action_names': mod.ACTION_NAMES,
                      'source_stamps': stamps, 'observation_stamp': 100., 'actions': [pose()] * 50},
                     pose(), 100., 10.)
    assert not guard.queue


def test_teleop_act_and_replay_share_one_driver_entrypoint_and_mux():
    import xml.etree.ElementTree as ET
    mod = load('piper')
    cfg = json.loads((ROOT / 'config/hardware.example.json').read_text())
    command = mod.build_command('robot', cfg)
    assert command[:3] == ['roslaunch', 'robot_setup', 'start_robot_all.launch']
    assert 'robot_control_mode:=teleop' in command
    assert 'auto_enable:=true' in command and 'enable_follower_gravity:=true' in command
    with pytest.raises(ValueError):
        mod.build_command('robot-replay', cfg)
    path = ROOT / 'src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/launch/piper_dual_robot.launch'
    launch = ET.parse(path).getroot()
    assert 'direct_replay' not in {a.attrib['name'] for a in launch.findall('arg')}
    assert 'replay_joint_cmd' not in path.read_text()
    actions = load('teleop_tui').TELEOP_ACTIONS
    assert '9' not in actions
    assert actions['1'][1] == 'robot'
    assert actions['7'][1] == 'mux-teleop' and actions['8'][1] == 'mux-vla'


@pytest.mark.parametrize('fault', [None, 'publisher', 'mux', 'subscriber', 'connection'])
def test_replay_graph_requires_exclusive_mux_connection(monkeypatch, fault):
    from test_recorded_replay import plan
    from test_rgb_act_execution import mux_graph
    mod = load('rgb_act_ros_executor')
    cfg, p = plan()
    values = {'/piper_ctrl_%s_node/%s' % (side, key): value
              for side, params in p['rules']['drivers'].items() for key, value in params.items()}
    ros = SimpleNamespace(get_name=lambda: '/replay', get_param_cached=lambda k, d: values.get(k, d))
    pubs = {side: SimpleNamespace(get_num_connections=lambda: 0 if fault == 'connection' else 1)
            for side in ('left', 'right')}
    selected = {side: SimpleNamespace(get_num_connections=lambda: 1) for side in pubs}
    trial = mod.Trial(cfg, None, ros, True, pubs, selected, io.StringIO(), None,
                      reset_positions=p['actions'][0], states_only=True, recording_plan=p)
    published, subscribed = mux_graph('/replay')
    if fault == 'publisher':
        published[mod.COMMAND_TOPICS['left']].append('/rogue')
    elif fault == 'subscriber':
        subscribed[mod.COMMAND_TOPICS['left']] = ['/piper_ctrl_left_node']
    elif fault == 'mux':
        published[mod.MUX_TOPICS['left'] + '/selected'] = ['/rogue']
    monkeypatch.setattr(mod, 'graph_connections', lambda name: (published, subscribed))
    if fault:
        with pytest.raises(ValueError):
            trial.check_graph(10.)
    else:
        trial.check_graph(10.)
        assert not trial.mux_pending
        assert trial.command_topics == mod.COMMAND_TOPICS
        assert trial.guard.driver_topics == mod.MUX_TOPICS


@pytest.mark.parametrize('fault', [None, 'old-route', 'selection', 'disabled', 'stale', 'gravity'])
def test_replay_feedback_requires_shared_driver_and_policy_mux_selection(fault):
    from test_recorded_replay import plan
    from test_rgb_act_execution import telemetry
    mod = load('rgb_act_ros_executor')
    cfg, p = plan()
    trial = mod.Trial(cfg, None, None, True, {}, {}, io.StringIO(), None,
                      reset_positions=p['actions'][0], states_only=True, recording_plan=p)
    obs, diagnostics, selected, gravity = telemetry()
    item = diagnostics['left'][0]
    if fault == 'old-route':
        item['gripper_topic'] = '/robot/arm_left/replay_joint_cmd'
    elif fault == 'selection':
        selected['left'] = '/teleop/arm_left/joint_states_single'
    elif fault == 'disabled':
        item['driver_enabled'] = False
    elif fault == 'stale':
        item['joint_feedback_age_s'][0] = 10
    elif fault == 'gravity':
        gravity['left'] = (False, 10.)
    if fault:
        with pytest.raises(ValueError):
            trial.guard.feedback(obs, diagnostics, selected, gravity, 100., 10.)
    else:
        assert trial.guard.feedback(obs, diagnostics, selected, gravity, 100., 10.)
