import json
from contextlib import nullcontext
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_entrypoints import ROOT, load


def checkpoint(root):
    path = root / 'outputs/run/checkpoints/080000/pretrained_model'
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').write_bytes(b'fixture')
    return path


@pytest.mark.parametrize('mode', ['robot', 'dry-run', 'predictions'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_tui_offers_checkpoint_and_builds_one_test_command(monkeypatch, tmp_path, mode, device):
    mod = load('teleop_tui')
    model = checkpoint(tmp_path)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    answers = iter(['', mode, device])
    monkeypatch.setattr('builtins.input', lambda label: next(answers))
    command = mod.data_command('act-test', Path('/training/bin/python'))
    assert command[:2] == ['/opt/piper-inference/bin/python', str(tmp_path/'scripts/test_rgb_act.py')]
    assert command[command.index('--checkpoint')+1] == str(model)
    assert command[command.index('--mode')+1] == mode
    assert command[command.index('--device')+1] == device
    assert ('--reset-to-first-frame' in command) is (mode != 'predictions')
    assert not list((tmp_path/'outputs').glob('model_tests/*'))


@pytest.mark.parametrize('answer', ['bad-mode', 'bad-device'])
def test_tui_rejects_unknown_test_choices(monkeypatch, tmp_path, answer):
    mod = load('teleop_tui')
    checkpoint(tmp_path)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    choices = iter(['', 'bad-mode', 'robot', 'cpu'] if answer == 'bad-mode' else ['', 'robot', 'bad-device', 'cpu'])
    monkeypatch.setattr('builtins.input', lambda label: next(choices))
    command = mod.data_command('act-test', '/training/bin/python')
    assert command[command.index('--mode') + 1] == 'robot'
    assert command[command.index('--device') + 1] == 'cpu'


@pytest.mark.parametrize('execute', [False, True])
def test_model_menu_dispatches_test_action(monkeypatch, execute):
    mod = load('teleop_tui')
    answers, calls = iter(['m', 'q']), []
    monkeypatch.setattr(sys, 'argv', ['tui'] + (['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr('builtins.input', lambda label: next(answers))
    monkeypatch.setattr(mod, 'hardware_status', lambda cfg: [])
    monkeypatch.setattr(mod, 'run_data_action', lambda *args: calls.append(args))
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **kw: pytest.fail('TUI started a robot process'))
    assert mod.main() == 0
    assert calls == [('act-test', ROOT/'.venv-training/bin/python', execute)]


def test_action_preflight_does_not_arm_or_queue_actions():
    mod = load('rgb_act_execution')
    guard = mod.ExecutionGuard(json.loads((ROOT/'config/rgb_act_execution.json').read_text()))
    pose = [0, .5, -.5, 0, 0, 0, .04] * 2
    assert guard.validate_actions([pose]*50, pose) == [pose]*50
    assert not guard.active and not guard.queue
    pose[1] = -.03
    with pytest.raises(ValueError, match=r'left\.joint2\.pos=-0\.03 outside'):
        guard.validate_actions([pose]*50, pose)


def test_predictions_mode_never_creates_actuator_bridge(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    output, calls = tmp_path/'predictions', []
    monkeypatch.setattr(sys, 'argv', ['test', '--checkpoint', str(tmp_path), '--mode', 'predictions',
                                     '--output-dir', str(output), '--observer-url', 'http://127.0.0.1:8766/observation'])
    monkeypatch.setattr(mod, 'managed_observer', lambda output, url: nullcontext(url))
    monkeypatch.setattr(mod.subprocess, 'call', lambda command: calls.append(command) or 0)
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Prediction mode started ROS bridge'))
    assert mod.main() == 0
    assert len(calls) == 1 and calls[0][1].endswith('/scripts/predict_rgb_act.py')
    assert '--execute' not in calls[0] and '--allow-execution' not in calls[0]


@pytest.mark.parametrize('mode', ['robot', 'dry-run'])
def test_bad_model_targets_fail_before_any_ros_bridge_or_hardware_process(monkeypatch, tmp_path, mode):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    config_dir = tmp_path/'config'
    config_dir.mkdir()
    (config_dir/'rgb_act_execution.json').write_text((ROOT/'config/rgb_act_execution.json').read_text())
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    output = tmp_path/'outputs/test'
    monkeypatch.setattr(sys, 'argv', ['test', '--checkpoint', str(tmp_path), '--mode', mode, '--output-dir', str(output)])
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(mod, 'validate_checkpoint_contract', lambda *args: None)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda value: None))
    pose = [0, -12.501, -.5, 0, 0, 0, .04]*2
    observation = {'observation.state': np.array(pose[:7]+[0]*7+pose[7:]+[0]*7)}
    actions = np.array([pose]*50)
    monkeypatch.setattr(mod, 'RGBActPredictor', lambda *a: SimpleNamespace(
        contract={'action_shape':[50,14]}, select_action=lambda obs: actions[0], reset=lambda: None))
    monkeypatch.setattr(mod, 'read_ros_observation', lambda *a: (observation, {'source_stamps':{}}))
    monkeypatch.setattr(mod, 'managed_observer', lambda output, url: nullcontext('http://127.0.0.1:8766/observation'))
    monkeypatch.setattr(mod, 'managed_cameras', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mod, 'run_trial', lambda *a: pytest.fail('Invalid pose started ROS bridge'))
    assert not output.exists()
    with pytest.raises(ValueError, match='left.joint2.pos'):
        mod.main()
    report = json.loads((output/'preflight.json').read_text())
    assert report['action_preflight_passed'] is False
    assert (output/'preflight_prediction.npy').exists()


@pytest.mark.parametrize('mode', ['robot', 'dry-run'])
@pytest.mark.parametrize('client_fails', [False, True])
@pytest.mark.parametrize('reset', [False, True])
@pytest.mark.parametrize('custom_client', [False, True])
@pytest.mark.parametrize('states_only,live_act', [(False, False), (True, False), (False, True)])
def test_trial_pairs_bridge_mode_with_loaded_client_and_requests_clean_quit(monkeypatch, tmp_path, mode, client_fails, reset, custom_client, states_only, live_act):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    output = tmp_path/'outputs/test'
    output.mkdir(parents=True)
    args = SimpleNamespace(mode=mode, output_dir=output, observer_url='http://127.0.0.1:8766/observation')
    args.states_only = states_only
    args.live_act = live_act
    positions = [0, .5, -.5, 0, 0, 0, .04] * 2 if reset else None
    if reset:
        args.reset_pose = {'positions': positions}
        (output/'reset_pose.json').write_text(json.dumps(args.reset_pose))
    config = json.loads((ROOT/'config/rgb_act_execution.json').read_text())
    commands, shutdown, ended = [], [], []
    predictor = object()
    process = SimpleNamespace(poll=lambda: None, send_signal=lambda sig: pytest.fail('Healthy bridge was killed'),
                              wait=lambda timeout: ended.append(timeout) or 0)
    def launch(command):
        commands.append(command)
        session = output/'bridge/session.json'
        session.parent.mkdir()
        session.write_text(json.dumps({'schema_version':1,'execute':mode=='robot', 'config':config,
                                       'reset_positions':positions, 'states_only':states_only, 'live_act': live_act, 'routing': 'mux'}))
        session.chmod(0o600)
        return process
    def client(session, model, observer, limits):
        assert model is predictor and limits == config and session['execute'] is (mode=='robot')
        if client_fails:
            raise ValueError('Client failed')
        return 0
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(mod.subprocess, 'Popen', launch)
    monkeypatch.setattr(mod, 'run_client', client)
    def shutdown_request(session, route, payload):
        assert payload == ({'immediate': True} if client_fails else {})
        shutdown.append(route)
    monkeypatch.setattr(mod, 'request', shutdown_request)
    options = {'client': client} if custom_client else {}
    if custom_client:
        monkeypatch.setattr(mod, 'run_client', lambda *a: pytest.fail('Called ACT client during replay'))
    if client_fails:
        with pytest.raises(ValueError, match='Client failed'):
            mod.run_trial(args, predictor, config, **options)
    else:
        assert mod.run_trial(args, predictor, config, **options) == 0
    launches = [command for command in commands if command[:2] == ['bash','-c']]
    assert len(launches) == 1
    assert ('--execute' in launches[0][-1]) is (mode=='robot')
    assert ('--reset-pose' in launches[0][-1]) is reset
    assert ('--states-only' in launches[0][-1]) is states_only
    assert ('--live-act' in launches[0][-1]) is live_act
    assert 'auto_enable' not in launches[0][-1] and 'roslaunch' not in launches[0][-1]
    assert shutdown == ['/quit'] and ended == [65 if live_act and reset and not client_fails else 5]
    assert all('docker' not in command for command in commands)
    assert '/usr/bin/python3' in launches[0][-1]


def test_failed_bridge_exit_preserves_watchdog_process(monkeypatch):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    commands = []
    def wait(timeout):
        raise subprocess.TimeoutExpired('bridge', timeout)
    process = SimpleNamespace(poll=lambda: None, wait=wait,
        send_signal=lambda *a: pytest.fail('Do not kill active bridge'))
    monkeypatch.setattr(mod, 'request', lambda *a: None)
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **k: commands.append(a))
    with pytest.raises(RuntimeError, match='Bridge did not exit'):
        mod.stop_bridge(process, {})
    assert not commands


def test_authenticated_quit_stops_trial_and_requests_bridge_exit():
    from test_rgb_act_execution import command_server, fake_trial
    import urllib.request
    trial, _ = fake_trial()
    with command_server(trial) as url:
        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(urllib.request.Request(url+'/quit', data=b'{}'))
        assert denied.value.code == 403 and not trial.exit_requested
        req = urllib.request.Request(url+'/quit', data=b'{}', headers={'Authorization':'Bearer test-private-token'})
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
        assert trial.exit_requested and not trial.guard.active


def test_loaded_client_sends_one_chunk_and_requests_stop_after_trial(monkeypatch):
    mod = load('execute_rgb_act')
    config = json.loads((ROOT/'config/rgb_act_execution.json').read_text())
    calls = []
    status = iter([{'armed':True,'ready':True,'remaining':0,'sequence':0},
                   {'armed':False,'reason':'STOPPED: Bounded trial time elapsed'}])
    def request(session, route, payload=None):
        calls.append((route, payload))
        return next(status) if route == '/status' else {'sequence':1}
    predictor = SimpleNamespace(predict=lambda observation: np.zeros((50,14)))
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod, 'read_ros_observation', lambda *args, **kwargs:
        ({}, {'source_stamps':{}, 'observation_stamp':1.0}))
    assert mod.run_client({'execute':False}, predictor, 'http://127.0.0.1:8766/observation', config) == 0
    assert [route for route, _ in calls] == ['/status','/chunk','/status','/stop']
    assert calls[1][1]['sequence'] == 1 and len(calls[1][1]['actions']) == 50


@pytest.mark.parametrize('condition', ['public-session', 'wrong-mode', 'wrong-limits', 'wrong-route', 'old-session', 'launch-fails'])
def test_bad_bridge_startup_never_runs_a_client(monkeypatch, tmp_path, condition):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', True)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    output = tmp_path/'outputs/trial'
    output.mkdir(parents=True)
    args = SimpleNamespace(mode='dry-run', output_dir=output, observer_url='http://127.0.0.1:8766/observation')
    config = json.loads((ROOT/'config/rgb_act_execution.json').read_text())
    cleaned = []
    def launch(command):
        if condition == 'launch-fails':
            raise OSError('Cannot launch ROS Python')
        session = output/'bridge/session.json'
        session.parent.mkdir()
        raw = {'schema_version':1,'execute':condition=='wrong-mode',
               'routing': 'direct' if condition=='wrong-route' else None if condition=='old-session' else 'mux',
               'config':{} if condition=='wrong-limits' else config}
        session.write_text(json.dumps(raw))
        session.chmod(0o644 if condition=='public-session' else 0o600)
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(mod.subprocess, 'Popen', launch)
    monkeypatch.setattr(mod, 'run_client', lambda *a: pytest.fail('Bad session ran client'))
    monkeypatch.setattr(mod, 'stop_bridge', lambda *a, **kw: cleaned.append(a))
    with pytest.raises((ValueError, OSError)):
        mod.run_trial(args, object(), config)
    assert len(cleaned) == 1


def test_model_launcher_rejects_host_before_creating_output(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'IN_CONTAINER', False)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['test', '--checkpoint', str(tmp_path), '--output-dir', str(output)])
    with pytest.raises(SystemExit) as error:
        mod.main()
    assert error.value.code == 2 and not output.exists()


def test_ros_children_use_system_python_and_rebuild_environment(tmp_path, monkeypatch):
    import shlex
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'ROOT', tmp_path / 'checkout with spaces')
    command = mod.ros_command('rgb_act_ros_observer.py', '--session-file', tmp_path / 'session file.json')
    assert command[:2] == ['bash', '-c']
    assert 'unset PYTHONPATH PYTHONHOME VIRTUAL_ENV' in command[-1]
    assert '/usr/bin/python3' in command[-1]
    assert shlex.quote(str(tmp_path / 'session file.json')) in command[-1]
    assert 'docker' not in command[-1]


@pytest.mark.parametrize('interrupted', [False, True])
def test_tui_uses_container_inference_python_and_waits_for_shutdown(monkeypatch, interrupted):
    mod = load('teleop_tui')
    exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda path: True if str(path) == '/.dockerenv' else exists(path))
    monkeypatch.setenv('PYTHONPATH', '/opt/ros/noetic/lib/python3/dist-packages')
    monkeypatch.setenv('PYTHONHOME', '/some/ros/runtime')
    command = ['/opt/piper-inference/bin/python', '/opt/piper/scripts/test_rgb_act.py']
    monkeypatch.setattr(mod, 'data_command', lambda *a: command)
    calls = []
    waits = []
    def wait():
        waits.append(True)
        if interrupted and len(waits) == 1:
            raise KeyboardInterrupt
        return 0
    monkeypatch.setattr(mod.subprocess, 'run', lambda args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda args, **kw:
                        calls.append((args, kw)) or SimpleNamespace(wait=wait))
    mod.run_data_action('act-test', '/unavailable/host-training/bin/python', True)
    assert len(calls) == 2 and all(args[0] == command[0] for args, _ in calls)
    assert calls[1][0] == command
    assert all('PYTHONPATH' not in kw['env'] and 'PYTHONHOME' not in kw['env'] for _, kw in calls)
    assert len(waits) == (2 if interrupted else 1)


def test_bridge_shutdown_keeps_waiting_after_ctrl_c(monkeypatch):
    mod = load('test_rgb_act')
    waits, routes = [], []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise KeyboardInterrupt
    process = SimpleNamespace(poll=lambda: None, wait=wait)
    monkeypatch.setattr(mod, 'request', lambda session, route, payload: routes.append(route))
    mod.stop_bridge(process, {'live_act': True, 'reset_positions': [0]*14})
    assert routes == ['/quit'] and waits == [65, 65]
