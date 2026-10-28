import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_preview_no_ros():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/piper.py"), "robot"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "PREVIEW ONLY" in r.stdout
    assert "auto_enable:=true" in r.stdout and "enable_auto_zero_on_start:=false" in r.stdout


def test_unreviewed_config_rejected():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/piper.py"), "robot", "--execute",
                        "--config", str(ROOT / "config/hardware.example.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "hardware_reviewed" in r.stderr


def test_teleop_tui_preview_no_ros():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/teleop_tui.py")], input="1\nq\n",
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "Piper teleop TUI (PREVIEW)" in r.stdout
    assert "PREVIEW ONLY" in r.stdout


def test_missing_workspace_explains_docker_environment(tmp_path, monkeypatch, capsys):
    mod = load('piper')
    cfg = json.loads((ROOT/'config/hardware.example.json').read_text())
    cfg['hardware_reviewed'] = True
    path = tmp_path/'hardware.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['piper','robot','--execute','--config',str(path)])
    monkeypatch.setattr(mod.subprocess, 'call', lambda *a, **kw: pytest.fail('Unbuilt checkout must not launch a driver'))
    with pytest.raises(SystemExit) as error:
        mod.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(tmp_path) in message
    assert 'docker exec -it -w /opt/piper piper-control bash' in message
    assert 'native ROS Noetic' in message


@pytest.mark.parametrize("action", ["robot", "teleop"])
def test_launch_arguments_exist(action):
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg.update(enable_cameras=True, left_wrist_usb_port="a", right_wrist_usb_port="b", top_usb_port="c")
    command = load("piper").build_command(action, cfg)
    base = ROOT / "src/zeno-wholebody-intervation/zeno-wholebody-robot"
    path = base / ({"robot": "robot_side/robot_setup", "teleop": "teleop_side/teleop_setup"}[action]) / "launch" / command[2]
    valid = {e.attrib["name"] for e in ET.parse(path).getroot().findall("arg")}
    assert {v.split(":=", 1)[0] for v in command[3:]} <= valid


def test_record_name_and_profile():
    mod = load("piper")
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    with pytest.raises(ValueError):
        mod.build_command("record", cfg, "../outside")
    cmd = mod.build_command("record", cfg, "my_task")
    assert Path(cmd[cmd.index("--profile") + 1]).is_file()


@pytest.mark.parametrize("arm_mode,dims", [("single", (14, 7)), ("bimanual", (28, 14))])
@pytest.mark.parametrize("image_size", [None, 256])
def test_act_fields_and_defaults(tmp_path, arm_mode, dims, image_size):
    (tmp_path / "meta").mkdir()
    features = {"observation.state": {"shape": [dims[0]]}, "action": {"shape": [dims[1]]}}
    for key in ("observation.image", "observation.left_wrist_image", "observation.right_wrist_image"):
        features[key] = {"shape": [320, 640, 3], "names": ["height", "width", "channels"]}
    features["observation.fastfs_depth_image"] = {"shape": [320, 640]}
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": features}))
    args = SimpleNamespace(dataset=tmp_path, output=tmp_path / "out", arm_mode=arm_mode,
                           steps=80000, batch_size=16, smoke=False, image_size=image_size)
    command = load("train_act").build_command(args)
    assert "--steps=80000" in command and "--batch_size=16" in command
    assert "--policy.use_amp=true" in command
    fields = json.loads(next(c.split("=", 1)[1] for c in command if c.startswith("--policy.input_features=")))
    assert "observation.fastfs_depth_image" not in fields
    assert ("observation.left_wrist_image" in fields) == (arm_mode == "bimanual")
    expected_shape = [3, image_size, image_size] if image_size else [3, 320, 640]
    assert fields["observation.image"]["shape"] == expected_shape
    if image_size:
        assert command[1] == str(ROOT / "scripts/lerobot_train_rgb_act.py")
        assert "--image-size=256" in command


def test_act_launch_enables_bf16(tmp_path, monkeypatch, capsys):
    mod = load("train_act")
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    monkeypatch.setattr(mod, "build_command", lambda args: command)
    monkeypatch.setattr(sys, "argv", ["train_act", "--dataset", str(tmp_path),
                                     "--output", str(tmp_path / "out"), "--arm-mode", "bimanual"])
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    monkeypatch.setenv("BF16_TEST_INHERITED", "present")
    launches = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd, **kwargs: launches.append((cmd, kwargs)) or 0)
    assert mod.main() == 0
    cmd, kwargs = launches[0]
    assert cmd == command
    assert kwargs["env"]["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert kwargs["env"]["BF16_TEST_INHERITED"] == "present"
    assert "ACCELERATE_MIXED_PRECISION=bf16" in capsys.readouterr().out


def test_teleop_starts_gravity_with_configured_calibration():
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg["gravity_joint_scale"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    command = load("piper").build_command("teleop", cfg)
    params = dict(arg.split(":=", 1) for arg in command[3:])
    assert params["enable_gravity_compensation"] == "true"
    assert json.loads(params["gravity_joint_scale"]) == cfg["gravity_joint_scale"]
    assert params["gravity_urdf_package"] == "piper_description"
    assert params["gravity_urdf_relpath"] == "urdf/piper_description.urdf"
    assert params["auto_enable"] == "true"
    assert params["enable_paddle"] == "false"


@pytest.mark.parametrize("action", ["teleop", "gravity-teleop"])
@pytest.mark.parametrize("scale", [[1] * 5, [float("nan")] * 6, [float("inf")] * 6,
                                   [True] * 6, ["1"] * 6, None])
def test_teleop_gravity_rejects_invalid_calibration(action, scale):
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg["gravity_joint_scale"] = scale
    with pytest.raises(ValueError, match="six finite numbers"):
        load("piper").build_command(action, cfg)


@pytest.mark.parametrize("action", ["robot", "teleop"])
def test_foreground_driver_waits_for_shutdown_after_ctrl_c(action, tmp_path, monkeypatch):
    mod = load("piper")
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg["hardware_reviewed"] = True
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(cfg))
    setup = tmp_path / "workspaces/ws_robot/devel/setup.bash"
    setup.parent.mkdir(parents=True)
    setup.touch()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["piper", action, "--execute", "--config", str(config)])
    waits = []

    class Process:
        def wait(self):
            waits.append("wait")
            if len(waits) == 1:
                raise KeyboardInterrupt
            return 0

    def launch(command, **kwargs):
        assert kwargs == {"cwd": tmp_path}
        assert "roslaunch" in command
        return Process()

    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    assert mod.main() == 0
    assert waits == ["wait", "wait"]
