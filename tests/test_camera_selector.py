import time
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_entrypoints import load


def test_camera_roles_are_unique_and_not_inferred_from_device_order():
    mod=load('select_recording_cameras')
    selection=mod.CameraSelection(['/dev/video0','/dev/video2'])
    assert selection.selected == {}
    selection.assign('/dev/video2','left')
    selection.assign('/dev/video0','right')
    assert selection.selected == {'left':'/dev/video2','right':'/dev/video0'}
    selection.assign('/dev/video0','left')
    assert selection.selected == {'left':'/dev/video0'}
    selection.assign('/dev/video0',None)
    assert selection.selected == {}


def test_confirm_requires_fresh_selected_camera_frames():
    mod=load('select_recording_cameras')
    selection=mod.CameraSelection(['/dev/video0','/dev/video2'])
    assert not selection.can_confirm()
    selection.assign('/dev/video2','top')
    assert not selection.can_confirm()
    selection.update('/dev/video2',np.zeros((300,400,3),np.uint8))
    assert selection.can_confirm()  # An unselected camera need not be healthy.
    selection.errors['/dev/video2']='Device busy'
    assert not selection.can_confirm()
    selection.errors.clear()
    selection.frames['/dev/video2']=(selection.frames['/dev/video2'][0],time.monotonic()-3)
    assert not selection.can_confirm()


def test_top_selector_only_allows_one_top_and_requires_live_frame():
    mod = load('select_recording_cameras')
    selection = mod.CameraSelection(['/dev/video0', '/dev/video4'], top_only=True)
    selection.render()
    assert {button[2] for button in selection.buttons if button[1].startswith('/dev/')} == {'top', None}
    with pytest.raises(ValueError):
        selection.assign('/dev/video0', 'left')
    selection.assign('/dev/video0', 'top')
    selection.assign('/dev/video4', 'top')
    assert selection.selected == {'top': '/dev/video4'}
    assert not selection.can_confirm()
    selection.update('/dev/video4', np.zeros((300, 400, 3), np.uint8))
    selection.render()
    rect, _, _ = next(button for button in selection.buttons if button[1] == 'confirm')
    selection.click(rect[0] + 5, rect[1] + 5)
    assert selection.confirmed
    selection.errors['/dev/video4'] = 'Camera busy'
    assert not selection.can_confirm()
    selection.assign('/dev/video4', None)
    assert not selection.can_confirm()


@pytest.mark.parametrize('devices', [['/dev/video0'], ['/dev/video0','/dev/video2'],
                                   ['/dev/video0','/dev/video2','/dev/video4']])
def test_selector_buttons_render_and_select_confirm_or_cancel(devices):
    mod=load('select_recording_cameras')
    selection=mod.CameraSelection(devices)
    for device in devices:
        selection.update(device,np.zeros((300,400,3),np.uint8))
    canvas=selection.render()
    assert canvas.shape[0] <= 910 and canvas.shape[1] <= 880
    # Choose Left wrist on the last tile.
    rect,device,role=next(b for b in selection.buttons if b[1:] == (devices[-1],'left'))
    selection.click(rect[0]+5,rect[1]+5)
    assert selection.selected == {'left':devices[-1]}
    selection.render()
    rect,_,_=next(b for b in selection.buttons if b[1]=='confirm')
    selection.click(rect[0]+5,rect[1]+5)
    assert selection.confirmed
    rect,_,_=next(b for b in selection.buttons if b[1]=='cancel')
    selection.click(rect[0]+5,rect[1]+5)
    assert selection.cancelled


@pytest.mark.parametrize('count', [1, 2, 3])
def test_preview_shows_every_device_without_recording_role_buttons(count, monkeypatch):
    import cv2
    mod=load('select_recording_cameras')
    devices=['/dev/video'+str(i*2) for i in range(count)]
    preview=mod.CameraSelection(devices, preview_only=True)
    labels=[]
    draw=cv2.putText
    def capture(image,text,*args,**kwargs):
        labels.append(text)
        return draw(image,text,*args,**kwargs)
    monkeypatch.setattr(cv2,'putText',capture)
    for device in devices:
        preview.update(device,np.full((300,400,3),128,np.uint8))
    preview.render()
    assert all(any(device in label and 'FPS' in label for label in labels) for device in devices)
    assert [button[1] for button in preview.buttons]==['cancel']
    rect,_,_=preview.buttons[0]
    preview.click(rect[0]+5,rect[1]+5)
    assert preview.cancelled
    preview.frames[devices[0]]=(preview.frames[devices[0]][0],time.monotonic()-3)
    labels.clear()
    preview.render()
    assert any(devices[0] in label and 'NO LIVE IMAGE' in label for label in labels)
    assert 'Preview stale' in labels


@pytest.mark.parametrize('exit_key',[ord('q'),27,-1])
def test_preview_attaches_to_recording_and_only_stops_its_own_camera(monkeypatch,tmp_path,exit_key):
    mod=load('select_recording_cameras')
    subscribers=[]
    closed=[]
    opened=[]
    stopped=[]
    process=SimpleNamespace(poll=lambda:None)
    published=[('/fisheye_left/usb_cam_node/image_raw/compressed',['/fisheye_left/usb_cam_node/record_camera_left'])]
    def subscribe(topic,*args,**kwargs):
        subscribers.append(topic)
        return SimpleNamespace(unregister=lambda:closed.append(topic))
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(mod,'configure_display',lambda:None)
    monkeypatch.setattr(mod.CameraSelection,'render',lambda self:None)
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *args,**kwargs:process)
    monkeypatch.setitem(sys.modules,'record_session',SimpleNamespace(
        camera_command=lambda *args:['camera'],ensure_camera_free=opened.append,stop_cameras=stopped.extend))
    monkeypatch.setitem(sys.modules,'rospy',SimpleNamespace(init_node=lambda *a,**kw:None,
        Subscriber=subscribe,is_shutdown=lambda:False,signal_shutdown=lambda *a:None,
        get_param=lambda name,default:'/dev/video0'))
    monkeypatch.setitem(sys.modules,'rosgraph',SimpleNamespace(Master=lambda name:SimpleNamespace(
        getSystemState=lambda:[published,[],[]])))
    monkeypatch.setitem(sys.modules,'sensor_msgs',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',SimpleNamespace(CompressedImage=object))
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(WINDOW_AUTOSIZE=1,WND_PROP_AUTOSIZE=1,EVENT_LBUTTONDOWN=1,
        namedWindow=lambda *a:None,setMouseCallback=lambda *a:None,imshow=lambda *a:None,
        waitKey=lambda *a:exit_key,getWindowProperty=lambda *a:-1,destroyAllWindows=lambda:None))
    assert mod.select_cameras(['/dev/video0','/dev/video2'],preview_only=True) is None
    assert opened==['/dev/video2']
    assert subscribers[0]==published[0][0]
    assert len(subscribers)==2 and subscribers==closed
    assert stopped==[process]


def test_preview_ignores_stale_parameters_without_publishers_and_raw_topics():
    mod=load('select_recording_cameras')
    publishers=[('/camera/image_raw',['/camera']),('/other/image_raw/compressed',['/other']),
                ('/right/image_raw/compressed',['/right'])]
    params={'/other/video_device':'/dev/video8','/right/video_device':'/dev/video2'}
    assert mod.active_camera_topics(['/dev/video0','/dev/video2'],publishers,
                                   lambda name,default:params.get(name,default))=={'/dev/video2':'/right/image_raw/compressed'}


def test_docker_x11_compatibility_library_disables_shared_memory(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from test_entrypoints import ROOT
    compiler=shutil.which('cc')
    if not compiler:
        pytest.skip('C compiler is provided by the ROS image')
    library=tmp_path/'no_mitshm.so'
    subprocess.run([compiler,'-shared','-fPIC','-Wall','-Werror',str(ROOT/'scripts/x11_no_mitshm.c'),
                    '-Wl,--no-as-needed','-lX11','-lxcb','-ldl','-o',str(library)],check=True)
    shim=ctypes.CDLL(str(library))
    shim.XQueryExtension.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    assert shim.XQueryExtension(None,b'MIT-SHM',None,None,None)==0
    class Extension(ctypes.Structure):
        _fields_=[('name',ctypes.c_char_p),('global_id',ctypes.c_int)]
    shim.xcb_get_extension_data.argtypes=[ctypes.c_void_p,ctypes.POINTER(Extension)]
    shim.xcb_get_extension_data.restype=ctypes.POINTER(ctypes.c_ubyte)
    reply=shim.xcb_get_extension_data(None,ctypes.byref(Extension(b'MIT-SHM',0)))
    assert reply[0]==1 and reply[8]==0

