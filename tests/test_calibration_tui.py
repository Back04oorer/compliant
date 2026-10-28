import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_entrypoints import ROOT, load
from test_tui_data import prompts


def calibration_form(monkeypatch, tmp_path, changes=None):
    mod = load("teleop_tui")
    def select(top_only=False):
        assert top_only
        return {"top": "/dev/video4"}
    monkeypatch.setattr(mod, "select_camera_roles", select)
    values = ["2560x720", "30", "", str(tmp_path / "sessions"), "8765"]
    for index, value in (changes or {}).items():
        values[index] = value
    prompts(monkeypatch, values)
    return mod


def test_preview_after_selection_does_not_launch_calibration_or_create_output(monkeypatch, tmp_path, capsys):
    mod = calibration_form(monkeypatch, tmp_path)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: pytest.fail("preview ran probe"))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: pytest.fail("preview launched GUI"))
    mod.run_calibration_action(Path("/calibration/.venv/bin/python"), False)
    text = capsys.readouterr().out
    assert "PREVIEW ONLY" in text and "--width 2560 --height 720" in text
    assert "--camera /dev/video4" in text and "--board stereo-charuco-7x5-38-a4-v1" in text
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize("board", ["stereo-charuco-7x5-38-a4-v1", "existing-20mm", "a4-30mm"])
def test_calibration_presets_remain_selectable(monkeypatch, tmp_path, capsys, board):
    mod = calibration_form(monkeypatch, tmp_path, {2: board})
    mod.run_calibration_action(Path("/calibration/python"), False)
    assert f"--board {board}" in capsys.readouterr().out


@pytest.mark.parametrize("index,value", [(0, "680x0"), (0, "2559x720"),
                                        (1, "0"), (2, "unknown"), (4, "65536")])
def test_invalid_calibration_form_returns_to_the_same_field(monkeypatch, tmp_path, capsys, index, value):
    mod = calibration_form(monkeypatch, tmp_path)
    values = ['2560x720', '30', 'a4-30mm', str(tmp_path / 'sessions'), '8765']
    values.insert(index, value)
    answers, labels = iter(values), []
    def read(label):
        labels.append(label)
        return next(answers)
    monkeypatch.setattr('builtins.input', read)
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Preview launched calibration'))
    mod.run_calibration_action(Path('/calibration/python'), False)
    assert labels[index] == labels[index + 1]
    output = capsys.readouterr().out
    assert 'Invalid:' in output and '--width 2560 --height 720' in output
    assert '--board a4-30mm' in output and '--port 8765' in output


def test_existing_sessions_are_reused_without_overwriting(monkeypatch, tmp_path):
    folder = tmp_path / "sessions"
    folder.mkdir()
    keep = folder / "existing-session.json"
    keep.write_text("existing calibration")
    mod = calibration_form(monkeypatch, tmp_path)
    mod.run_calibration_action(Path("/calibration/python"), False)
    assert keep.read_text() == "existing calibration"


@pytest.mark.parametrize("fails", [False, True])
def test_execute_isolates_ros_environment_and_waits_for_cleanup(monkeypatch, tmp_path, fails):
    mod = calibration_form(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/noetic/lib/python3/dist-packages")
    monkeypatch.setenv("PYTHONHOME", "/ros")
    calls, waits = [], []

    def probe(command, **kwargs):
        assert command[-1] == "--check"
        assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]
        if fails:
            raise FileNotFoundError("missing calibration venv")

    def wait():
        waits.append(1)
        if len(waits) == 1:
            raise KeyboardInterrupt
        return 0

    def launch(command, **kwargs):
        assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]
        calls.append(command)
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(mod.subprocess, "run", probe)
    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    if fails:
        with pytest.raises(ValueError, match="setup_calibration.sh"):
            mod.run_calibration_action(Path("/calibration/.venv/bin/python"), True)
        assert not calls
    else:
        mod.run_calibration_action(Path("/calibration/.venv/bin/python"), True)
        assert calls[0][:3] == ["/calibration/.venv/bin/python",
                               str(ROOT / "scripts/calibrate_top_camera.py"), "gui"]
        assert len(waits) == 2


@pytest.mark.parametrize("execute", [False, True])
def test_menu_dispatch_keeps_venv_python_symlink(monkeypatch, tmp_path, execute):
    mod = load("teleop_tui")
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    prompts(monkeypatch, ["k", "q"])
    monkeypatch.setattr(sys, "argv", ["tui", "--calibration-python", str(python)]
                        + (["--execute"] if execute else []))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(mod, "hardware_status", lambda cfg: [])
    calls = []
    monkeypatch.setattr(mod, "run_calibration_action", lambda *a: calls.append(a))
    assert mod.main() == 0
    assert calls == [(python, execute)]


@pytest.mark.parametrize("interrupt", [False, True])
def test_top_selection_uses_opencv_window_and_waits_for_release(monkeypatch, interrupt):
    mod = load("teleop_tui")
    monkeypatch.setattr(mod, "camera_devices", lambda: ["/dev/video0", "/dev/video4"])
    waits = []
    def communicate():
        waits.append(1)
        if interrupt and len(waits) == 1:
            raise KeyboardInterrupt
        return json.dumps({"top": "/dev/video4"}), ""
    def launch(command, **kwargs):
        assert command == ["bash", str(ROOT / "scripts/camera_window.sh"),
                           "--device", "/dev/video0", "--device", "/dev/video4", "--top-only"]
        return SimpleNamespace(returncode=0, communicate=communicate)
    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    assert mod.select_camera_roles(top_only=True) == {"top": "/dev/video4"}
    assert len(waits) == (2 if interrupt else 1)


@pytest.mark.parametrize("chosen", [{}, {"left": "/dev/video4"},
                                  {"top": "/dev/video99"},
                                  {"top": "/dev/video4", "left": "/dev/video0"}])
def test_top_selection_rejects_invalid_window_result(monkeypatch, chosen):
    mod = load("teleop_tui")
    monkeypatch.setattr(mod, "camera_devices", lambda: ["/dev/video0", "/dev/video4"])
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
        returncode=0, communicate=lambda: (json.dumps(chosen), "")))
    with pytest.raises(ValueError, match="Invalid camera selection"):
        mod.select_camera_roles(top_only=True)


def test_cancelled_top_window_stops_calibration_before_form_or_launch(monkeypatch):
    mod = load("teleop_tui")
    monkeypatch.setattr(mod, "camera_devices", lambda: ["/dev/video4"])
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=2, communicate=lambda: ("", "Camera selection cancelled"))
    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    monkeypatch.setattr(mod, "ask", lambda *a: pytest.fail("cancel must not continue to form"))
    with pytest.raises(mod.Cancelled, match="cancelled"):
        mod.run_calibration_action(Path("/calibration/python"), True)
    assert len(calls) == 1 and calls[0][-1] == "--top-only"


def camera_reader():
    path = load("record_session").CAMERA_NODE
    spec = importlib.util.spec_from_file_location("native_calibration_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fails", [False, True])
def test_native_capture_has_no_publishers_resize_or_exposure_change(monkeypatch, fails):
    mod = camera_reader()
    closed = []

    def open_camera(camera):
        assert camera.width == 2560 and camera.height == 720 and camera.fps == 30
        assert camera.enforce_resolution and camera.enforce_mjpg
        assert not camera.auto_exposure and not hasattr(camera, "image_pub")
        if fails:
            raise OSError("busy")

    monkeypatch.setattr(mod.DirectMjpgCamera, "_open_once", open_camera)
    monkeypatch.setattr(mod.DirectMjpgCamera, "close", lambda camera: closed.append(camera))
    if fails:
        with pytest.raises(OSError, match="busy"):
            mod.DirectMjpgCamera.capture_only("/dev/video4", 2560, 720)
        assert len(closed) == 1
    else:
        camera = mod.DirectMjpgCamera.capture_only("/dev/video4", 2560, 720)
        assert camera.video_device == "/dev/video4" and not closed


@pytest.mark.parametrize("data,bytesused,error", [(b"\xff\xd8raw", 5, None),
                                                (b"not-jpeg", 8, "non-JPEG"),
                                                (b"\xff\xd8raw", 30, "buffer length")])
def test_native_reader_preserves_jpeg_and_returns_buffer(monkeypatch, data, bytesused, error):
    mod = camera_reader()
    camera = mod.DirectMjpgCamera.__new__(mod.DirectMjpgCamera)
    camera.fd, camera.select_timeout_sec = 42, 1
    camera.buffers = [(data, len(data))]
    monkeypatch.setattr(mod.select, "select", lambda *a: ([42], [], []))
    queued = []
    monkeypatch.setattr(camera, "_queue_buffer", queued.append)

    def dequeue(fd, request, buffer, mutate):
        buffer.index, buffer.bytesused = 0, bytesused

    monkeypatch.setattr(mod.fcntl, "ioctl", dequeue)
    if error:
        with pytest.raises(ValueError, match=error):
            camera.read_jpeg()
    else:
        assert camera.read_jpeg() == data
    assert queued == [0]


def test_native_reader_timeout_is_bounded(monkeypatch):
    mod = camera_reader()
    camera = mod.DirectMjpgCamera.__new__(mod.DirectMjpgCamera)
    camera.fd, camera.select_timeout_sec = 42, 1
    monkeypatch.setattr(mod.select, "select", lambda *a: ([], [], []))
    monkeypatch.setattr(mod.fcntl, "ioctl", lambda *a: pytest.fail("no frame to dequeue"))
    assert camera.read_jpeg() is None
