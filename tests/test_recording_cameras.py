import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from test_entrypoints import ROOT, load
from test_tui_data import prompts, session_dir


def test_all_nonempty_camera_combinations_only_require_selected_views():
    mod = load('record_session')
    base = {'profile_name': 'old', 'required_topics': ['/robot/state'],
            'camera_streams': [{'image_base': '/old'}], 'hz_monitor_topics': ['/old']}
    for count in (1, 2, 3):
        for roles in itertools.combinations(mod.CAMERA_BASES, count):
            cameras = {role: '/dev/video%d' % (index * 2) for index, role in enumerate(roles)}
            profile = mod.recording_profile(base, cameras)
            assert {stream['image_base'] for stream in profile['camera_streams']} == {mod.CAMERA_BASES[r] for r in roles}
            assert profile['hz_monitor_topics'] == ['/robot/state'] + [mod.CAMERA_BASES[r] for r in roles]
            assert all(not stream['camera_info_required'] for stream in profile['camera_streams'])
    assert base['camera_streams'] == [{'image_base': '/old'}]


@pytest.mark.parametrize('values', [[], ['unknown=/dev/video0'], ['left=/tmp/file'],
    ['left=/dev/video0','top=/dev/video0'], ['left=/dev/video0','left=/dev/video2']])
def test_camera_selection_rejects_invalid_or_duplicate_devices(values):
    with pytest.raises(ValueError):
        load('record_session').parse_cameras(values)


def test_record_entry_and_native_vs_output_size():
    mod = load('record_session')
    cameras = ['left=/dev/video0', 'top=/dev/video2']
    command = load('piper').build_command('record', {}, 'test_task', cameras)
    assert command[1] == str(ROOT / 'scripts/record_session.py')
    assert command.count('--camera') == 2
    assert command[command.index('--image-width') + 1] == '680'
    assert command[command.index('--image-height') + 1] == '480'
    camera = mod.camera_command('top', '/dev/video2', 680, 480)
    assert '_image_width:=640' in camera and '_image_height:=480' in camera
    assert '_output_width:=680' in camera and '_output_height:=480' in camera
    assert '__ns:=/top_camera' in camera
    assert '_auto_exposure:=true' in camera


def test_record_waits_for_child_save_after_interrupt():
    waits=[]
    def wait():
        waits.append(1)
        if len(waits) == 1:
            raise KeyboardInterrupt
        return 0
    assert load('record_session').wait_for_recording(SimpleNamespace(wait=wait)) == 0
    assert len(waits) == 2


@pytest.mark.parametrize('interrupted', [False, True])
def test_recording_form_uses_visual_roles_instead_of_device_order(monkeypatch, interrupted):
    mod=load('teleop_tui')
    entries=[Path('/sys/class/video4linux/video0'),Path('/sys/class/video4linux/video2')]
    monkeypatch.setattr(mod.Path,'glob',lambda self, pattern: entries)
    monkeypatch.setattr(mod.Path,'is_char_device',lambda self: True)
    monkeypatch.setattr(mod.Path,'read_text',lambda self: '0')
    calls=[]
    def communicate():
        calls.append(1)
        if interrupted and len(calls)==1:
            raise KeyboardInterrupt
        return json.dumps({'top':'/dev/video2','right':'/dev/video0'}), ''
    def select(command, **kwargs):
        assert command[-4:] == ['--device', '/dev/video0', '--device', '/dev/video2']
        assert any('camera_window.sh' in item for item in command)
        return SimpleNamespace(returncode=0, communicate=communicate)
    monkeypatch.setattr(mod.subprocess, 'Popen', select)
    prompts(monkeypatch,[''])
    assert mod.recording_options() == ['--camera','top=/dev/video2','--camera','right=/dev/video0',
                                      '--image-width','680','--image-height','480']


@pytest.mark.parametrize('selected', [{}, {'left':'/dev/video99'},
                                    {'left':'/dev/video0','right':'/dev/video0'},
                                    {'other':'/dev/video0'}])
def test_recording_form_rejects_invalid_visual_selection(monkeypatch, selected):
    mod=load('teleop_tui')
    monkeypatch.setattr(mod.Path,'glob',lambda self, pattern: [Path('/sys/class/video4linux/video0')])
    monkeypatch.setattr(mod.Path,'is_char_device',lambda self: True)
    monkeypatch.setattr(mod.Path,'read_text',lambda self: '0')
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:
                        SimpleNamespace(returncode=0, communicate=lambda:(json.dumps(selected),'')))
    with pytest.raises(ValueError, match='Invalid camera selection'):
        mod.recording_options()


def test_recording_form_cancellation_does_not_start_recording(monkeypatch):
    mod=load('teleop_tui')
    monkeypatch.setattr(mod.Path,'glob',lambda self, pattern: [Path('/sys/class/video4linux/video0')])
    monkeypatch.setattr(mod.Path,'is_char_device',lambda self: True)
    monkeypatch.setattr(mod.Path,'read_text',lambda self: '0')
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:
                        SimpleNamespace(returncode=2, communicate=lambda:('', 'Camera selection cancelled.')))
    with pytest.raises(mod.Cancelled, match='cancelled'):
        mod.recording_options()


def test_camera_ownership_check_prevents_reconfiguration(tmp_path, monkeypatch):
    mod=load('record_session')
    monkeypatch.setattr(mod.Path,'is_char_device',lambda self: True)
    fd=tmp_path/'123/fd'
    fd.mkdir(parents=True)
    (fd/'7').symlink_to('/dev/video0')
    with pytest.raises(ValueError, match='already in use'):
        mod.ensure_camera_free('/dev/video0', tmp_path)
    mod.ensure_camera_free('/dev/video2', tmp_path)


def test_conversion_uses_selected_camera_manifest(tmp_path, monkeypatch):
    mod=load('teleop_tui')
    session=session_dir(tmp_path)
    manifest={'width':680,'height':480,'image_topics':{'left':'/fisheye_left/usb_cam_node/image_raw/compressed',
                                                    'top':'/top_camera/image_raw/compressed'}}
    (session/'recording_cameras.json').write_text(json.dumps(manifest))
    prompts(monkeypatch,[session,tmp_path/'rgb','pick',''])
    command=mod.data_command('convert-rgb',sys.executable)
    assert command[command.index('--obs-indices')+1] == '1,2'
    assert command[command.index('--top-camera-topic')+1] == manifest['image_topics']['top']
    assert command[command.index('--image-width')+1] == '680'
    assert '--lfv-top-crop' not in command


@pytest.mark.parametrize('keys', [('observation.left_wrist_image',), ('observation.image',),
    ('observation.right_wrist_image','observation.image')])
def test_act_uses_available_camera_subset(tmp_path, keys):
    (tmp_path/'meta').mkdir()
    features={'observation.state':{'shape':[28]},'action':{'shape':[14]}}
    for key in keys:
        features[key]={'shape':[480,680,3]}
    (tmp_path/'meta/info.json').write_text(json.dumps({'features':features}))
    args=SimpleNamespace(dataset=tmp_path,output=tmp_path/'out',arm_mode='bimanual',steps=1,batch_size=1,smoke=True)
    command=load('train_act').build_command(args)
    inputs=json.loads(next(v.split('=',1)[1] for v in command if v.startswith('--policy.input_features=')))
    assert set(inputs)==set(keys)|{'observation.state'}


@pytest.mark.parametrize('output',[(680,480),(0,0)])
def test_camera_publishes_requested_size_and_uncalibrated_info(monkeypatch, output):
    frames=[]
    infos=[]
    class Frame:
        def __init__(self,w,h): self.shape=(h,w,3)
        def tobytes(self): return b'pixels'
    class Message:
        def __init__(self):
            self.header=SimpleNamespace()
            self.K=[0.0]*9
    cv2=SimpleNamespace(INTER_AREA=3,IMREAD_COLOR=1,IMWRITE_JPEG_QUALITY=1,
        imdecode=lambda data, mode: Frame(640,480),
        imencode=lambda *args: (True,SimpleNamespace(tobytes=lambda: b'resized-jpeg')),
        resize=lambda frame,size,interpolation: Frame(*size))
    monkeypatch.setitem(sys.modules,'cv2',cv2)
    monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(frombuffer=lambda *a,**kw: b'jpeg',uint8=object))
    monkeypatch.setitem(sys.modules,'rospy',SimpleNamespace(Time=SimpleNamespace(now=lambda:42)))
    monkeypatch.setitem(sys.modules,'sensor_msgs',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',SimpleNamespace(CameraInfo=Message,CompressedImage=Message,Image=Message))
    path=ROOT/'src/zeno-wholebody-intervation/zeno-wholebody-robot/robot_side/robot_setup/scripts/top_mjpg_camera_node.py'
    spec=importlib.util.spec_from_file_location('test_camera_node',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node=module.DirectMjpgCamera.__new__(module.DirectMjpgCamera)
    node.output_width,node.output_height=output
    node.frame_id='test'
    node.width,node.height=640,480
    node.image_pub=SimpleNamespace(publish=frames.append)
    node.info_pub=SimpleNamespace(publish=infos.append)
    node._publish_frame(b'original-jpeg',42)
    assert frames[0].data == (b'resized-jpeg' if output[0] else b'original-jpeg')
    assert (infos[0].width,infos[0].height)==((680,480) if output[0] else (640,480))
    assert infos[0].K == [0.0]*9
    # ROS shutdown may close the camera before spin returns its last buffer.
    node.fd = -1
    monkeypatch.setattr(module.fcntl, 'ioctl', lambda *args: pytest.fail('ioctl on closed camera'))
    node._queue_buffer(0)


@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('reject_control', [False, True])
def test_camera_reapplies_exposure_on_each_open(monkeypatch, automatic, reject_control):
    monkeypatch.setitem(sys.modules, 'rospy', SimpleNamespace(Time=object, loginfo=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'sensor_msgs', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', SimpleNamespace(CameraInfo=object, CompressedImage=object))
    path = ROOT/'src/zeno-wholebody-intervation/zeno-wholebody-robot/robot_side/robot_setup/scripts/top_mjpg_camera_node.py'
    spec = importlib.util.spec_from_file_location('test_exposure_camera', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.DirectMjpgCamera.__new__(module.DirectMjpgCamera)
    node.video_device, node.auto_exposure = '/dev/video2', automatic
    node.enforce_resolution = node.enforce_mjpg = False
    node.fps, node.buffers = 30, []
    monkeypatch.setattr(module.os, 'open', lambda *args: 42)
    for name in ('_set_format', '_set_fps', '_init_mmap', '_stream_on'):
        monkeypatch.setattr(node, name, lambda: None)
    fmt = module.V4L2Format()
    fmt.fmt.pix.width, fmt.fmt.pix.height = 640, 480
    fmt.fmt.pix.pixelformat = module._fourcc('MJPG')
    monkeypatch.setattr(node, '_get_format', lambda: fmt)
    calls = []
    def ioctl(fd, request, control, mutate):
        calls.append((fd, request, control.id, control.value))
        if reject_control:
            raise OSError('Camera rejected exposure setting')
    monkeypatch.setattr(module.fcntl, 'ioctl', ioctl)
    for _ in range(2):
        if automatic and reject_control:
            with pytest.raises(OSError, match='rejected exposure'):
                node._open_once()
        else:
            node._open_once()
    assert calls == ([(42, module.VIDIOC_S_CTRL, module.V4L2_CID_EXPOSURE_AUTO, 3)] * 2 if automatic else [])
