import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_preview_no_ros():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/piper.py"), "robot"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "PREVIEW ONLY" in r.stdout
    assert "auto_enable:=false" in r.stdout and "enable_auto_zero_on_start:=false" in r.stdout


def test_unreviewed_config_rejected():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/piper.py"), "robot", "--execute",
                        "--config", str(ROOT / "config/hardware.example.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "hardware_reviewed" in r.stderr


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
def test_act_fields_and_defaults(tmp_path, arm_mode, dims):
    (tmp_path / "meta").mkdir()
    features = {"observation.state": {"shape": [dims[0]]}, "action": {"shape": [dims[1]]}}
    for key in ("observation.image", "observation.left_wrist_image", "observation.right_wrist_image"):
        features[key] = {"shape": [320, 640, 3], "names": ["height", "width", "channels"]}
    features["observation.fastfs_depth_image"] = {"shape": [320, 640]}
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": features}))
    args = SimpleNamespace(dataset=tmp_path, output=tmp_path / "out", arm_mode=arm_mode,
                           steps=80000, batch_size=16, smoke=False)
    command = load("train_act").build_command(args)
    assert "--steps=80000" in command and "--batch_size=16" in command
    fields = json.loads(next(c.split("=", 1)[1] for c in command if c.startswith("--policy.input_features=")))
    assert "observation.fastfs_depth_image" not in fields
    assert ("observation.left_wrist_image" in fields) == (arm_mode == "bimanual")
    assert fields["observation.image"]["shape"] == [3, 320, 640]
