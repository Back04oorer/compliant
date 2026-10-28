from contextlib import nullcontext
import io
import json
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from test_entrypoints import load


def test_existing_observer_is_reused_without_starting_or_stopping_it(monkeypatch, tmp_path):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **k: pytest.fail('External observer ownership changed'))
    with mod.managed_observer(tmp_path/'unused', 'http://127.0.0.1:8766/observation') as url:
        assert url.endswith(':8766/observation')
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('fault', [None, 'missing-inputs', 'bad-frame', 'quit-fails', 'public-session'])
@pytest.mark.parametrize('states_only', [False, True])
def test_managed_observer_is_subscriber_only_and_cleans_up_owned_node(monkeypatch, tmp_path, fault, states_only):
    mod = load('test_rgb_act')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    output = tmp_path/'outputs/test'
    output.mkdir(parents=True)
    commands, quits, waits, terminated = [], [], [], []
    clock = [0.0]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(mod.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+6))
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    def launch(command, **kwargs):
        commands.append(command)
        assert kwargs['stdin'] == mod.subprocess.DEVNULL
        assert kwargs['start_new_session'] is True
        assert kwargs['stdout'].name.endswith('/observer.log')
        assert kwargs['stderr'] is kwargs['stdout']
        session = output/'observer_session.json'
        session.write_text(json.dumps({'schema_version':1,'prediction_only':True,
            'url':'http://127.0.0.1:39000/observation','shutdown_token':'a'*64,'pid':4242}))
        session.chmod(0o644 if fault=='public-session' else 0o600)
        return SimpleNamespace(poll=lambda: None, wait=lambda timeout: waits.append(timeout),
                               terminate=lambda: terminated.append(True))
    monkeypatch.setattr(mod.subprocess, 'Popen', launch)
    def read(*args, **kwargs):
        assert kwargs['states_only'] is states_only
        if fault=='missing-inputs':
            raise urllib.error.HTTPError(args[0],503,'missing',{},io.BytesIO(b'{"error":"Missing observations: left"}'))
        if fault=='bad-frame':
            raise ValueError('Invalid state order')
        return {}, {}
    monkeypatch.setattr(mod, 'read_ros_observation', read)
    def open_request(request, **kwargs):
        assert request.get_header('Authorization') == 'Bearer '+'a'*64
        assert request.full_url == 'http://127.0.0.1:39000/quit'
        quits.append(request.full_url)
        if fault=='quit-fails':
            raise urllib.error.URLError('Unavailable')
        return nullcontext(None)
    monkeypatch.setattr(mod.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=open_request))
    if fault in ('missing-inputs','bad-frame','public-session'):
        with pytest.raises(ValueError, match='Missing observations: left' if fault=='missing-inputs' else
                           'Invalid state order' if fault=='bad-frame' else 'private read-only observer'):
            with mod.managed_observer(output, states_only=states_only):
                pytest.fail('Invalid observer was accepted')
    else:
        with mod.managed_observer(output, states_only=states_only) as url:
            assert url == 'http://127.0.0.1:39000/observation'
    launch_command = next(command for command in commands if command[:2] == ['bash', '-c'])
    assert '--port 0' in launch_command[-1] and 'rgb_act_ros_observer.py' in launch_command[-1]
    assert ('--states-only' in launch_command[-1]) is states_only
    assert not any(word in launch_command[-1] for word in ('executor.py', 'roslaunch', 'can_sl', '--execute'))
    assert waits == [5]
    assert bool(quits) is (fault!='public-session')
    assert bool(terminated) is (fault in ('quit-fails', 'public-session'))
    assert all('docker' not in command for command in commands)
    assert '/usr/bin/python3' in launch_command[-1]


@pytest.mark.parametrize('managed', [False, True])
def test_observer_shutdown_requires_its_private_token_and_cannot_arm(managed):
    mod = load('rgb_act_ros_observer')
    server = mod.make_server(mod.ObservationBuffer(),0)
    if managed:
        server.shutdown_token = 'test-private-token'
    worker = threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    url = 'http://127.0.0.1:%d'%server.server_port
    try:
        for token in ('', 'wrong', 'é'):
            request = urllib.request.Request(url+'/quit',data=b'',headers={'Authorization':'Bearer '+token})
            with pytest.raises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(request)
            assert denied.value.code == 403 and not server.exit_requested
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(urllib.request.Request(url+'/arm',data=b''))
        assert denied.value.code == 404
        if managed:
            request = urllib.request.Request(url+'/quit',data=b'',headers={'Authorization':'Bearer test-private-token'})
            with urllib.request.urlopen(request) as response:
                assert response.status==200
            assert server.exit_requested
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def test_external_observer_does_not_start_or_stop_cameras(monkeypatch, tmp_path):
    mod=load('test_rgb_act')
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('External camera ownership changed'))
    with mod.managed_cameras(tmp_path,enabled=False):
        pass
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('failure',[None,'startup','inference','partial-json'])
def test_managed_cameras_clean_up_the_owned_supervisor_on_every_exit(monkeypatch, tmp_path, failure):
    mod=load('test_rgb_act')
    commands,signals,waits=[],[],[]
    session=tmp_path/'camera_session.json'
    def launch(command,**kwargs):
        commands.append(command)
        assert kwargs['stdin']==mod.subprocess.DEVNULL and kwargs['start_new_session'] is True
        assert kwargs['stdout'].name.endswith('cameras.log')
        session.write_text('{' if failure=='partial-json' else '{"started":["left"],"reused":["right","top"]}')
        return SimpleNamespace(poll=lambda:1 if failure=='startup' else None,
            send_signal=signals.append,wait=lambda timeout:waits.append(timeout))
    monkeypatch.setattr(mod.subprocess,'Popen',launch)
    def complete(seconds):
        session.write_text('{"started":["left"],"reused":["right","top"]}')
    monkeypatch.setattr(mod.time,'sleep',complete)
    if failure in ('startup','inference'):
        with pytest.raises(ValueError):
            with mod.managed_cameras(tmp_path):
                raise ValueError('inference failed')
    else:
        with mod.managed_cameras(tmp_path):
            pass
    assert '--reuse-existing' in commands[0][-1] and 'start_rgb_cameras.py' in commands[0][-1]
    assert '--execute' not in commands[0][-1] and 'executor.py' not in commands[0][-1]
    assert bool(signals) is (failure!='startup')
    assert waits==([] if failure=='startup' else [25])
