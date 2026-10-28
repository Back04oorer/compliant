import os
import socket
import threading

import pytest

from test_entrypoints import load


@pytest.fixture
def access(tmp_path, monkeypatch):
    mod=load('camera_access')
    monkeypatch.setattr(mod,'SOCKET',tmp_path/'camera.sock')
    monkeypatch.setattr(mod,'capture_devices',lambda:['/dev/null'])
    with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as server:
        server.bind(str(mod.SOCKET))
        server.listen(1)
        server.settimeout(3)
        def serve():
            peer,_=server.accept()
            with peer:
                mod.serve_peer(peer)
        thread=threading.Thread(target=serve)
        thread.start()
        yield mod
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_discovery_returns_current_host_devices_without_opening_them(access):
    response,fds=access.request({})
    assert response=={'devices':['/dev/null']} and fds==[]


def test_descriptor_remains_usable_after_host_closes_its_copy(access):
    response,fds=access.request({'device':'/dev/null'})
    try:
        assert response=={} and len(fds)==1
        assert os.read(fds[0],1)==b''
        assert not os.get_inheritable(fds[0])
    finally:
        for fd in fds:
            os.close(fd)


@pytest.mark.parametrize('device',['/etc/passwd','/dev/video1','/dev/../etc/passwd'])
def test_host_refuses_paths_outside_capture_camera_list(access,device):
    with pytest.raises(OSError,match='Not an available capture camera'):
        access.request({'device':device})


def test_unmapped_device_uses_host_descriptor(monkeypatch):
    mod=load('camera_access')
    def unavailable(*args):
        raise FileNotFoundError(2,'No device')
    monkeypatch.setattr(mod.os,'open',unavailable)
    calls=[]
    monkeypatch.setattr(mod,'request',lambda payload:(calls.append(payload) or ({},[42])))
    assert mod.open_camera('/dev/video4')==42
    assert calls==[{'device':'/dev/video4'}]


def test_tui_merges_host_cameras_with_existing_mappings(monkeypatch):
    import camera_access
    mod=load('teleop_tui')
    monkeypatch.setattr(mod.Path,'glob',lambda *args:[])
    monkeypatch.setattr(camera_access,'remote_devices',lambda:['/dev/video0','/dev/video2','/dev/video4'])
    assert mod.camera_devices()==['/dev/video0','/dev/video2','/dev/video4']


def test_recording_accepts_unmapped_host_camera_and_checks_ownership(tmp_path, monkeypatch):
    import camera_access
    mod=load('record_session')
    monkeypatch.setattr(mod.Path,'is_char_device',lambda self:False)
    monkeypatch.setattr(camera_access,'remote_devices',lambda:['/dev/video4'])
    mod.ensure_camera_free('/dev/video4',tmp_path)
    with pytest.raises(ValueError,match='unavailable'):
        mod.ensure_camera_free('/dev/video99',tmp_path)
    folder=tmp_path/'123/fd'
    folder.mkdir(parents=True)
    (folder/'7').symlink_to('/dev/video4')
    with pytest.raises(ValueError,match='already in use'):
        mod.ensure_camera_free('/dev/video4',tmp_path)
