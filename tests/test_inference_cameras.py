import json
import sys
from types import SimpleNamespace

import pytest

from test_entrypoints import ROOT, load


CAMERAS = {'right': '/dev/video0', 'left': '/dev/video2', 'top': '/dev/video4'}


@pytest.mark.parametrize('failure', ['interrupt', 'camera-exit', 'partial-start', 'busy', 'topic-conflict'])
def test_camera_launcher_only_starts_cameras_and_cleans_up_owned_processes(monkeypatch, tmp_path, failure):
    mod = load('start_rgb_cameras')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    opened, stopped, checked = [], [], []
    published = [('/top_camera/image_raw/compressed', 'sensor_msgs/CompressedImage')] if failure == 'topic-conflict' else []
    master = SimpleNamespace(getPid=lambda: 1, getPublishedTopics=lambda ns: published)
    monkeypatch.setitem(sys.modules, 'rosgraph', SimpleNamespace(Master=lambda name: master))
    monkeypatch.setattr(sys, 'argv', ['cameras'] + [item for role, device in CAMERAS.items() for item in ('--camera', role + '=' + device)])
    monkeypatch.chdir(tmp_path)
    def check(device):
        checked.append(device)
        if failure == 'busy' and device == '/dev/video4':
            raise ValueError('Camera already in use')
    monkeypatch.setattr(mod, 'ensure_camera_free', check)
    def launch(command, **kwargs):
        assert kwargs == {'start_new_session': True}
        assert command[1].endswith('/robot_setup/scripts/top_mjpg_camera_node.py')
        assert '_framerate:=30' in command and '_output_width:=680' in command and '_output_height:=480' in command
        if failure == 'partial-start' and len(opened) == 1:
            raise OSError('Camera startup failed')
        process = SimpleNamespace(poll=lambda: 1 if failure == 'camera-exit' else None)
        opened.append((command, process))
        return process
    monkeypatch.setattr(mod.subprocess, 'Popen', launch)
    monkeypatch.setattr(mod, 'stop_cameras', lambda processes: stopped.extend(processes))
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(mod.time, 'sleep', interrupt)
    expected = KeyboardInterrupt if failure == 'interrupt' else OSError if failure == 'partial-start' else ValueError
    with pytest.raises(expected):
        mod.main()
    assert stopped == [process for _, process in opened]
    assert len(opened) == (0 if failure in ('busy', 'topic-conflict') else 1 if failure == 'partial-start' else 3)
    if len(opened) == 3:
        assert {next(arg for arg in command if arg.startswith('__ns:=')) for command, _ in opened} == {
            '__ns:=/top_camera', '__ns:=/fisheye_left/usb_cam_node', '__ns:=/fisheye_right/usb_cam_node'}
    saved = tmp_path / 'outputs/tui/inference_cameras.json'
    assert saved.exists() is (len(opened) == 3)
    if saved.exists():
        assert json.loads(saved.read_text())['cameras'] == CAMERAS
    assert not list(tmp_path.glob('**/*.bag'))


@pytest.mark.parametrize('arguments', [['--camera', 'right=/dev/video0'],
                                     ['--camera', 'right=/dev/video0', '--camera', 'left=/dev/video0', '--camera', 'top=/dev/video4'],
                                     ['--image-width', '0', '--camera', 'right=/dev/video0', '--camera', 'left=/dev/video2', '--camera', 'top=/dev/video4']])
def test_bad_camera_contract_is_rejected_before_ros_or_process_start(monkeypatch, arguments):
    mod = load('start_rgb_cameras')
    monkeypatch.setattr(sys, 'argv', ['cameras', *arguments])
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **k: pytest.fail('Invalid camera contract launched a process'))
    with pytest.raises((ValueError, SystemExit)):
        mod.main()


@pytest.mark.parametrize('active', [set(), {'left'}, {'left', 'right', 'top'}])
def test_automatic_cameras_start_only_missing_roles_and_preserve_reused_feeds(monkeypatch, tmp_path, active):
    mod = load('start_rgb_cameras')
    settings = tmp_path/'cameras.json'
    settings.write_text(json.dumps({'cameras':CAMERAS,'width':680,'height':480}))
    published=[(mod.CAMERA_BASES[role]+'/compressed','sensor_msgs/CompressedImage') for role in active]
    master=SimpleNamespace(getPid=lambda:1,getPublishedTopics=lambda ns:published)
    monkeypatch.setitem(sys.modules,'rosgraph',SimpleNamespace(Master=lambda name:master))
    session=tmp_path/'session.json'
    monkeypatch.setattr(sys,'argv',['cameras','--config',str(settings),'--reuse-existing','--session-file',str(session)])
    checked,opened,stopped=[],[],[]
    monkeypatch.setattr(mod,'ensure_camera_free',checked.append)
    def launch(command,**kwargs):
        process=SimpleNamespace(poll=lambda:None)
        opened.append(process)
        assert next(arg for arg in command if arg.startswith('__name:='))[len('__name:=record_camera_'):] not in active
        return process
    monkeypatch.setattr(mod.subprocess,'Popen',launch)
    monkeypatch.setattr(mod,'stop_cameras',lambda processes:stopped.extend(processes))
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(mod.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):
        mod.main()
    result=json.loads(session.read_text())
    assert set(result['started'])==set(CAMERAS)-active
    assert set(result['reused'])==active
    assert set(checked)=={device for role,device in CAMERAS.items() if role not in active}
    assert stopped==opened and len(opened)==3-len(active)


def test_first_automatic_use_remembers_known_live_roles_without_guessing_device_order(monkeypatch, tmp_path):
    mod=load('start_rgb_cameras')
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    bases=mod.CAMERA_BASES
    values={}
    pubs=[]
    for role,device in CAMERAS.items():
        node='/camera_'+role
        pubs.append((bases[role]+'/compressed',[node]))
        values.update({node+'/video_device':device,node+'/output_width':680,node+'/output_height':480})
    master=SimpleNamespace(getPid=lambda:1,getSystemState=lambda:(pubs,[],[]),
        getPublishedTopics=lambda ns:[(topic,'sensor_msgs/CompressedImage') for topic,_ in pubs],
        getParam=lambda name:values[name])
    monkeypatch.setitem(sys.modules,'rosgraph',SimpleNamespace(Master=lambda name:master))
    monkeypatch.setattr(sys,'argv',['cameras','--reuse-existing'])
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Reused camera restarted'))
    monkeypatch.setattr(mod,'ensure_camera_free',lambda *a:pytest.fail('Reused camera device probed'))
    monkeypatch.setattr(mod,'stop_cameras',lambda processes:None)
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(mod.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):
        mod.main()
    saved=json.loads((tmp_path/'outputs/tui/inference_cameras.json').read_text())
    assert saved=={'cameras':CAMERAS,'width':680,'height':480}


def test_missing_camera_settings_fail_before_starting_processes(monkeypatch, tmp_path):
    mod=load('start_rgb_cameras')
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    master=SimpleNamespace(getPid=lambda:1,getPublishedTopics=lambda ns:[],getSystemState=lambda:([],[],[]))
    monkeypatch.setitem(sys.modules,'rosgraph',SimpleNamespace(Master=lambda name:master))
    monkeypatch.setattr(sys,'argv',['cameras','--reuse-existing'])
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Guessed camera mapping'))
    with pytest.raises(ValueError,match='Configure all three roles'):
        mod.main()
