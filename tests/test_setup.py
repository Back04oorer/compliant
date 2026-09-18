import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_entrypoints import load, ROOT


def config_and_links():
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg["hardware_reviewed"] = True
    links = []
    for i, role in enumerate(load("setup_can").ROLES):
        port = f"1-{i + 1}:1.0"
        cfg["can_usb_ports"][role] = port
        links.append({"ifname": f"can{i}", "usb_port": port, "flags": [],
                      "linkinfo": {"info_kind": "can", "info_data": {}}})
    return cfg, links


def test_can_four_roles_and_commands():
    cfg, links = config_and_links()
    commands, mapping = load("setup_can").plan(cfg, links)
    assert len(commands) == 12 and len(mapping) == 4
    assert [m["target"] for m in mapping] == ["can_sl", "can_sr", "can_left", "can_right"]
    assert all(command[:4] == ["ip", "link", "set", "dev"] for command in commands)
    assert not any("down" in command for command in commands)


@pytest.mark.parametrize("error", ["unreviewed", "missing", "duplicate_usb", "duplicate_name", "collision", "active_wrong", "bus_off", "bad_rate"])
def test_can_preflight_rejects_all_before_changes(error):
    cfg, links = config_and_links()
    if error == "unreviewed":
        cfg["hardware_reviewed"] = False
    elif error == "missing":
        links.pop()
    elif error == "duplicate_usb":
        cfg["can_usb_ports"]["robot_left_can"] = cfg["can_usb_ports"]["robot_right_can"]
    elif error == "duplicate_name":
        cfg["robot_left_can"] = cfg["robot_right_can"]
    elif error == "collision":
        links.append({"ifname": "can_sl"})
    elif error == "active_wrong":
        links[0]["flags"] = ["UP"]
    elif error == "bus_off":
        links[0].update(ifname="can_sl", flags=["UP"])
        links[0]["linkinfo"]["info_data"] = {"bittiming": {"bitrate": 1000000}, "state": "BUS-OFF"}
    else:
        cfg["can_bitrate"] = 500000
    with pytest.raises(ValueError):
        load("setup_can").plan(cfg, links)


def test_can_active_correct_is_noop():
    cfg, links = config_and_links()
    for role, link in zip(load("setup_can").ROLES, links):
        link.update(ifname=cfg[role], flags=["UP"])
        link["linkinfo"]["info_data"] = {"bittiming": {"bitrate": 1000000}, "state": "ERROR-ACTIVE"}
    assert load("setup_can").plan(cfg, links)[0] == []


@pytest.mark.parametrize("execute,confirmed,expected_mutations", [(False, False, 0), (True, False, 0), (True, True, 12)])
def test_can_cli_execution_boundary(tmp_path, monkeypatch, execute, confirmed, expected_mutations):
    mod = load("setup_can")
    cfg, links = config_and_links()
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(cfg))
    active = copy.deepcopy(links)
    for role, link in zip(mod.ROLES, active):
        link.update(ifname=cfg[role], flags=["UP"])
        link["linkinfo"]["info_data"] = {"bittiming": {"bitrate": 1000000}, "state": "ERROR-ACTIVE"}
    snapshots = iter([links, links, active])
    monkeypatch.setattr(mod, "inventory", lambda: next(snapshots))
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    argv = ["setup_can", "--config", str(path)]
    if execute:
        argv.append("--execute")
    if confirmed:
        argv.append("--drivers-stopped")
    monkeypatch.setattr(sys, "argv", argv)
    if execute and not confirmed:
        with pytest.raises(SystemExit):
            mod.main()
    else:
        assert mod.main() == 0
    assert len(calls) == expected_mutations


@pytest.mark.parametrize("target", ["ros", "training"])
def test_install_preview_does_not_install(tmp_path, target):
    venv = tmp_path / "absent"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/setup_environment.py"), target,
                             "--venv", str(venv)], capture_output=True, text=True)
    assert result.returncode == 0 and "PREVIEW ONLY" in result.stdout
    assert not venv.exists()
    assert "roscore" not in result.stdout


def test_install_refuses_existing_environment(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/setup_environment.py"), "training",
                             "--venv", str(tmp_path), "--execute"], capture_output=True, text=True)
    assert result.returncode != 0 and "existing training environment" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_training_versions_and_separate_environment():
    commands = load("setup_environment").training_commands("python3.10", Path("/tmp/test-venv"), "https://download.pytorch.org/whl/cu128")
    assert "torch==2.10.0" in commands[2] and "torchvision==0.25.0" in commands[2]
    assert all(c[0] == "/tmp/test-venv/bin/python" for c in commands[1:])
    assert commands[-2][-1] == "check"
