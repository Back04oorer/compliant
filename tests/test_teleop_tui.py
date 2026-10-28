import copy
import json
from types import SimpleNamespace

import pytest

from test_entrypoints import load, ROOT


def feedback():
    return {"published_at": 100.0, "motor_enabled": [True] * 6,
            "driver_enabled": False,
            "enable_feedback_age_s": [0.1] * 6,
            "gripper_enabled": True, "gripper_feedback_age_s": 0.1}


@pytest.mark.parametrize("enabled,expected", [([True] * 6, "ENABLED (6/6)"),
                                              ([False] * 6, "DISABLED (0/6)"),
                                              ([True, False] * 3, "PARTIAL (3/6)")])
def test_enable_uses_all_six_motors(enabled, expected):
    state = feedback()
    state["motor_enabled"] = enabled
    assert load("teleop_tui").enable_status(state, 100.2) == expected


@pytest.mark.parametrize("change", [{"published_at": 97}, {"published_at": 101},
                                   {"published_at": float("nan")},
                                   {"enable_feedback_age_s": [None] * 6},
                                   {"enable_feedback_age_s": [0.1] * 5 + [2]},
                                   {"enable_feedback_age_s": [float("nan")] * 6},
                                   {"motor_enabled": [True] * 5},
                                   {"motor_enabled": [1] * 6}])
def test_stale_or_invalid_feedback_is_unknown(change):
    state = feedback()
    state.update(change)
    assert load("teleop_tui").enable_status(state, 100.2).startswith("UNKNOWN")


@pytest.mark.parametrize("link,expected", [(None, "MISSING"),
    ({"link_type": "ether"}, "NOT CAN"),
    ({"link_type": "can", "flags": []}, "DOWN"),
    ({"link_type": "can", "flags": ["UP"],
      "linkinfo": {"info_data": {"state": "ERROR-ACTIVE"}}}, "UP (ERROR-ACTIVE)"),
    ({"link_type": "can", "flags": ["UP"],
      "linkinfo": {"info_data": {"state": "BUS-OFF"}}}, "UP (BUS-OFF)")])
def test_can_state_including_bus_fault(link, expected):
    assert load("teleop_tui").can_status(link) == expected


def test_status_maps_configured_ports_and_read_only_queries(monkeypatch):
    mod = load("teleop_tui")
    cfg = json.loads((ROOT / "config/hardware.example.json").read_text())
    cfg["teleop_left_can"] = "custom_can"
    calls = []
    state = feedback()
    state["published_at"] = mod.time.time()

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] <= 3
        if command[0] == "ip":
            return SimpleNamespace(stdout=json.dumps([
                {"ifname": "custom_can", "link_type": "can", "flags": ["UP"],
                 "linkinfo": {"info_data": {"state": "ERROR-ACTIVE"}}}]))
        return SimpleNamespace(stdout=json.dumps({"teleop_left": state}))

    monkeypatch.setattr(mod.subprocess, "run", run)
    rows = mod.hardware_status(cfg)
    assert "custom_can" in rows[0] and "UP (ERROR-ACTIVE)" in rows[0] and "ENABLED (6/6)" in rows[1]
    assert "GRIP: ENABLED" in rows[1]
    assert "DRIVER: RUNNING" in rows[0] and "CTRL: DISABLED" in rows[1]
    assert "MISSING" in rows[2] and "GRIP: UNKNOWN" in rows[3]
    assert "DRIVER: NO DATA" in rows[2] and "CTRL: UNKNOWN" in rows[3]
    assert calls[0] == ["ip", "-details", "-json", "link", "show"]
    assert calls[1][-1].endswith("teleop_tui.py")
    assert "--ros-status" in calls[1][2]


def test_status_tools_unavailable_does_not_break_menu(monkeypatch):
    mod = load("teleop_tui")

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(mod.subprocess, "run", unavailable)
    rows = mod.hardware_status({})
    assert len(rows) == 8
    assert all("CAN: UNKNOWN" in row for row in rows[::2])
    assert all("ARM: UNKNOWN" in row and "GRIP: UNKNOWN" in row for row in rows[1::2])


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("menu", ["main", "teleop"])
def test_live_menu_refreshes_while_waiting_for_key(stale, menu):
    mod = load("teleop_tui")
    monitor = SimpleNamespace(rows=["CAN: DOWN ARM: UNKNOWN"], updated=mod.time.monotonic() - (3 if stale else 0))

    class Screen:
        def __init__(self):
            self.frames = []
            self.lines = []

        def timeout(self, ms):
            assert ms == 200

        def erase(self):
            self.lines = []

        def getmaxyx(self):
            return 30, 120

        def keypad(self, enabled):
            assert enabled

        def addnstr(self, row, col, text, width, attr=0):
            self.lines.append(text[:width])

        def refresh(self):
            self.frames.append(copy.copy(self.lines))

        def getch(self):
            if len(self.frames) == 1:
                monitor.rows = ["CAN: UP ARM: ENABLED"]
                monitor.updated = mod.time.monotonic()
                return -1
            return ord("b" if menu == "teleop" else "q")

    screen = Screen()
    assert mod.live_choice(screen, monitor, "PREVIEW", menu) == ("b" if menu == "teleop" else "q")
    assert any("Back to main menu" in line for line in screen.frames[0]) == (menu == "teleop")
    expected = "CAN: UNKNOWN ARM: UNKNOWN" if stale else "CAN: DOWN ARM: UNKNOWN"
    assert any(expected in line for line in screen.frames[0])
    assert any("CAN: UP ARM: ENABLED" in line for line in screen.frames[1])


def test_ros_sampler_subscribes_to_existing_diagnostics_only(monkeypatch, capsys):
    import sys

    mod = load("teleop_tui")
    topics = []
    closed = []

    def subscribe(topic, message_type, callback, callback_args, queue_size):
        topics.append(topic)
        callback(SimpleNamespace(data=json.dumps(feedback())), callback_args)
        return SimpleNamespace(unregister=lambda: closed.append(topic))

    monkeypatch.setitem(sys.modules, "rospy", SimpleNamespace(
        init_node=lambda *args, **kwargs: None, Subscriber=subscribe,
        signal_shutdown=lambda reason: None))
    monkeypatch.setitem(sys.modules, "std_msgs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "std_msgs.msg", SimpleNamespace(String=object))
    mod.ros_status()
    states = json.loads(capsys.readouterr().out)
    assert set(states) == set(mod.ARMS)
    assert topics == ["/teleop/arm_left/agent_driver_state", "/teleop/arm_right/agent_driver_state",
                      "/robot/arm_left/agent_driver_state", "/robot/arm_right/agent_driver_state"]
    assert closed == topics


@pytest.mark.parametrize("enabled,expected", [(True, "ENABLED"), (False, "DISABLED")])
def test_gripper_enable_is_independent_of_joints(enabled, expected):
    mod = load("teleop_tui")
    state = feedback()
    state.update(gripper_enabled=enabled, motor_enabled=[not enabled] * 6)
    assert mod.gripper_status(state, 100.2) == expected


@pytest.mark.parametrize("change", [{"published_at": 97}, {"published_at": 101},
    {"published_at": float("nan")}, {"gripper_feedback_age_s": None},
    {"gripper_feedback_age_s": 2}, {"gripper_feedback_age_s": -1}, {"gripper_feedback_age_s": -0.1},
    {"gripper_feedback_age_s": float("nan")}, {"gripper_enabled": 1}])
def test_gripper_stale_or_invalid_feedback_is_unknown(change):
    state = feedback()
    state.update(change)
    assert load("teleop_tui").gripper_status(state, 100.2) == "UNKNOWN"


def test_old_driver_without_gripper_fields_is_unknown():
    assert load("teleop_tui").gripper_status({"published_at": 100}, 100.2) == "UNKNOWN"


def test_teleop_submenu_navigation_does_not_launch_commands(monkeypatch, capsys):
    import sys

    mod = load("teleop_tui")
    choices = iter(["2", "b", "q"])
    calls = []
    monkeypatch.setattr(sys, "argv", ["teleop_tui"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: next(choices))
    monkeypatch.setattr(mod, "hardware_status", lambda cfg: ["CAN: UP ARM: ENABLED GRIP: ENABLED"])
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: calls.append(cmd))
    assert mod.main() == 0
    assert calls == []
    output = capsys.readouterr().out
    assert "2. Teleop" in output
    assert "Piper teleop TUI (PREVIEW) / Teleop" in output
    assert "1. Start follower drivers" in output
    assert "2. Start teleop (leader) + gravity compensation" in output
    assert "Leader gravity compensation" not in output
    assert output.count("CAN: UP ARM: ENABLED GRIP: ENABLED") == 3


@pytest.mark.parametrize("choice,action", [("1", "robot"), ("2", "teleop")])
@pytest.mark.parametrize("execute", [False, True])
def test_submenu_launches_only_selected_driver(monkeypatch, choice, action, execute):
    import sys

    mod = load("teleop_tui")
    choices = iter(["2", choice, "b", "q"])
    calls = []
    monkeypatch.setattr(sys, "argv", ["teleop_tui"] + (["--execute"] if execute else []))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: next(choices))
    monkeypatch.setattr(mod, "hardware_status", lambda cfg: [])
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd:
                        (calls.append(cmd) or SimpleNamespace(wait=lambda: 0)))
    assert mod.main() == 0
    assert len(calls) == 1
    assert calls[0][2] == action
    assert ("--execute" in calls[0]) == execute


@pytest.mark.parametrize('execute', [False, True])
def test_can_menu_uses_setup_script_without_ros_or_enable(monkeypatch, execute):
    import sys
    mod = load('teleop_tui')
    choices = iter(['i','q'])
    calls = []
    monkeypatch.setattr(sys, 'argv', ['teleop_tui'] + (['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr('builtins.input', lambda prompt: next(choices))
    monkeypatch.setattr(mod, 'hardware_status', lambda cfg: [])
    monkeypatch.setattr(mod.subprocess, 'call', lambda command: calls.append(command))
    assert mod.main() == 0
    assert calls == [[sys.executable, str(ROOT/'scripts/setup_can.py'), '--config',
                      str(ROOT/'config/hardware.local.json')] + (['--execute','--check-drivers-stopped'] if execute else [])]


@pytest.mark.parametrize('execute',[False,True])
def test_preview_menu_opens_all_devices_without_recording(monkeypatch,execute):
    import sys
    mod=load('teleop_tui')
    choices=iter(['p','q'])
    commands=[]
    monkeypatch.setattr(sys,'argv',['teleop_tui']+(['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin,'isatty',lambda:False)
    monkeypatch.setattr('builtins.input',lambda prompt:next(choices))
    monkeypatch.setattr(mod,'hardware_status',lambda cfg:[])
    monkeypatch.setattr(mod,'camera_devices',lambda:['/dev/video0','/dev/video2','/dev/video4'])
    def launch(command):
        commands.append(command)
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(mod.subprocess,'Popen',launch)
    assert mod.main()==0
    assert len(commands)==int(execute)
    if execute:
        command=commands[0]
        assert command[-7:]==['--device','/dev/video0','--device','/dev/video2',
                             '--device','/dev/video4','--preview-only']
        assert command[1]==str(ROOT/'scripts/camera_window.sh')
        assert 'record_session.py' not in ' '.join(command)


def test_camera_discovery_includes_all_mapped_cameras_and_skips_metadata(tmp_path, monkeypatch):
    import camera_access
    monkeypatch.setattr(camera_access,'remote_devices',lambda:[])
    mod=load('teleop_tui')
    sys_root=tmp_path/'sys'
    dev_root=tmp_path/'dev'
    sys_root.mkdir()
    dev_root.mkdir()
    for number in range(7):
        entry=sys_root/f'video{number}'
        entry.mkdir()
        (entry/'index').write_text(str(number % 2))
        if number < 6:
            (dev_root/entry.name).symlink_to('/dev/null')
    roots={'/sys/class/video4linux':sys_root,'/dev':dev_root}
    monkeypatch.setattr(mod,'Path',lambda value:roots[value])
    assert mod.camera_devices()==[str(dev_root/f'video{number}') for number in (0,2,4)]


class MenuScreen:
    def __init__(self, keys, size=(24, 80)):
        self.keys = iter(keys)
        self.size = size
        self.frames = []
        self.draws = []

    def keypad(self, enabled):
        pass

    def timeout(self, value):
        pass

    def erase(self):
        self.draws = []

    def getmaxyx(self):
        return self.size

    def addnstr(self, row, col, text, width, attr=0):
        assert 0 <= row < self.size[0] and 0 <= col < self.size[1]
        self.draws.append((text[:width], attr))

    def refresh(self):
        self.frames.append(list(self.draws))

    def getch(self):
        return next(self.keys)


@pytest.mark.parametrize("menu", ["main", "teleop"])
@pytest.mark.parametrize("size", [(24, 80), (12, 50)])
def test_arrow_navigation_and_scrolling(menu, size):
    mod = load("teleop_tui")
    monitor = SimpleNamespace(rows=["CAN: UP (ERROR-ACTIVE)"] * 8, updated=mod.time.monotonic())
    # Moving up from the first item wraps to Quit, which must remain visible.
    screen = MenuScreen([mod.curses.KEY_UP, 10], size)
    assert mod.live_choice(screen, monitor, "PREVIEW", menu) == "q"
    assert any("> [q] Quit" in text for text, attr in screen.frames[-1])


def test_enter_selects_highlighted_action():
    mod = load("teleop_tui")
    monitor = SimpleNamespace(rows=[], updated=mod.time.monotonic())
    screen = MenuScreen([mod.curses.KEY_DOWN, mod.curses.KEY_ENTER])
    assert mod.live_choice(screen, monitor, "PREVIEW") == "2"
    assert any("> [2] Teleop" in text and attr & mod.curses.A_REVERSE
               for text, attr in screen.frames[-1])


def test_tiny_terminal_does_not_run_invisible_action():
    mod = load("teleop_tui")
    monitor = SimpleNamespace(rows=[], updated=mod.time.monotonic())
    screen = MenuScreen([10, ord("1"), ord("q")], (8, 30))
    assert mod.live_choice(screen, monitor, "EXECUTE") == "q"
    assert len(screen.frames) == 3


@pytest.mark.parametrize("value,expected", [("UP (ERROR-ACTIVE)", "ok"),
    ("RUNNING", "ok"), ("STALE", "warn"), ("NO DATA", "warn"),
    ("UP (BUS-OFF)", "bad"), ("UP (ERROR-PASSIVE)", "warn"),
    ("MISSING", "bad"), ("DOWN", "bad"), ("UNKNOWN", "warn"),
    ("PARTIAL (3/6)", "warn"), ("DISABLED", "muted"), ("ENABLED (6/6)", "ok")])
def test_status_colors_keep_faults_distinct(value, expected):
    colors = dict(accent=1, ok=2, warn=3, bad=4, muted=5)
    assert load("teleop_tui").status_color(value, colors) == colors[expected]


def test_joint_and_gripper_colors_are_independent(monkeypatch):
    mod = load("teleop_tui")
    colors = dict(accent=1, ok=2, warn=3, bad=4, muted=5)
    monkeypatch.setattr(mod, "ui_colors", lambda: colors)
    monitor = SimpleNamespace(rows=["  ARM: ENABLED (6/6)  GRIP: DISABLED"], updated=mod.time.monotonic())
    screen = MenuScreen([ord("q")])
    mod.live_choice(screen, monitor, "EXECUTE")
    assert ("ENABLED (6/6)", colors["ok"]) in screen.frames[0]
    assert ("DISABLED", colors["muted"]) in screen.frames[0]
    assert ("[ EXECUTE ]", colors["bad"]) in screen.frames[0]


def test_monochrome_terminal_keeps_text_and_selection(monkeypatch):
    mod = load("teleop_tui")
    monkeypatch.setattr(mod.curses, "has_colors", lambda: False)
    colors = mod.ui_colors()
    assert colors["ok"] == mod.curses.A_BOLD
    assert colors["muted"] == mod.curses.A_DIM


@pytest.mark.parametrize("state,expected", [
    ({"published_at": 100}, "RUNNING"),
    ({"published_at": 98.2}, "RUNNING"),
    ({"published_at": 98}, "STALE"),
    ({"published_at": 101}, "UNKNOWN"),
    ({"published_at": float("nan")}, "UNKNOWN"),
    ({"published_at": None}, "NO DATA"),
    ({}, "NO DATA"),
])
def test_driver_liveness_uses_diagnostic_freshness(state, expected):
    assert load("teleop_tui").driver_status(state, 100.2) == expected


@pytest.mark.parametrize("change,expected", [
    ({"driver_enabled": True, "motor_enabled": [False] * 6}, "ENABLED"),
    ({"driver_enabled": False}, "DISABLED"),
    ({"driver_enabled": 1}, "UNKNOWN"),
    ({"driver_enabled": None}, "UNKNOWN"),
    ({"published_at": 97, "driver_enabled": True}, "UNKNOWN"),
])
def test_control_state_is_independent_of_motor_enable(change, expected):
    state = feedback()
    state.update(change)
    assert load("teleop_tui").control_status(state, 100.2) == expected


def test_driver_and_control_status_render_at_80_columns(monkeypatch):
    mod = load("teleop_tui")
    colors = dict(accent=1, ok=2, warn=3, bad=4, muted=5)
    monkeypatch.setattr(mod, "ui_colors", lambda: colors)
    monitor = SimpleNamespace(rows=[
        "teleop_left   can_ml     CAN: UP (ERROR-ACTIVE)  DRIVER: RUNNING",
        "  CTRL: DISABLED  ARM: ENABLED (6/6)  GRIP: ENABLED",
    ], updated=mod.time.monotonic())
    screen = MenuScreen([ord("q")])
    mod.live_choice(screen, monitor, "EXECUTE")
    assert ("RUNNING", colors["ok"]) in screen.frames[0]
    assert ("DISABLED", colors["muted"]) in screen.frames[0]
    assert any("CTRL: DISABLED" in text and "ARM: ENABLED (6/6)" in text
               for text, _ in screen.frames[0])

    monitor.updated -= 3
    screen = MenuScreen([ord("q")])
    mod.live_choice(screen, monitor, "EXECUTE")
    assert any("DRIVER: UNKNOWN" in text for text, _ in screen.frames[0])
    assert any("CTRL: UNKNOWN" in text for text, _ in screen.frames[0])
    assert not any("RUNNING" in text or "ENABLED" in text for text, _ in screen.frames[0])


def test_missing_enable_service_explains_correct_driver_menu(monkeypatch):
    mod = load("teleop_tui")
    monkeypatch.setattr(mod, "ros_services", lambda: {"/robot/arm_left/enable_srv"})
    with pytest.raises(ValueError, match="Start Teleop submenu 2 first"):
        mod.require_enable_service("enable-teleop-left")
    mod.require_enable_service("enable-robot-left")


def test_ros_services_is_read_only_and_has_a_timeout(monkeypatch):
    mod = load("teleop_tui")
    payload = mod.xmlrpc.client.dumps(([1, "ok", [[], [], [
        ["/teleop/arm_left/enable_srv", ["/piper_teleop_left_node"]]]]],),
        methodresponse=True).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return payload

    def urlopen(request, timeout):
        assert timeout == 1
        params, method = mod.xmlrpc.client.loads(request.data)
        assert method == "getSystemState"
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    assert mod.ros_services() == {"/teleop/arm_left/enable_srv"}


def test_role_launches_never_load_the_other_roles_parameters():
    import xml.etree.ElementTree as ET
    import yaml

    package = ROOT / "src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl"
    combined = {}
    for role, prefix in (("robot", "piper_ctrl_"), ("teleop", "piper_teleop_")):
        launch = ET.parse(package / f"launch/piper_dual_{role}.launch").getroot()
        for element in launch.findall("rosparam"):
            path = package / element.attrib["file"].replace("$(find piper_ctrl)/", "")
            params = yaml.safe_load(path.read_text())
            assert params and all(name.startswith(prefix) for name in params)
            combined.update(params)
    assert combined == yaml.safe_load((package / "config/piper_dual.yaml").read_text())


@pytest.mark.parametrize("choice,action", [("1", "robot"), ("2", "teleop")])
def test_driver_stays_in_foreground_and_waits_for_ctrl_c_shutdown(monkeypatch, choice, action):
    import sys

    mod = load("teleop_tui")
    choices = iter(["2", choice, "q"])
    waits = []

    class Process:
        def wait(self):
            waits.append("wait")
            if len(waits) == 1:
                raise KeyboardInterrupt
            return 0

    def launch(command, **kwargs):
        assert command[2] == action
        # Inherit the terminal and signal group; do not detach or redirect logs.
        assert kwargs == {}
        return Process()

    monkeypatch.setattr(sys, "argv", ["teleop_tui", "--execute"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: next(choices))
    monkeypatch.setattr(mod, "hardware_status", lambda cfg: [])
    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    assert mod.main() == 0
    assert waits == ["wait", "wait"]


def test_ctrl_c_at_return_prompt_exits_cleanly(monkeypatch):
    import sys

    mod = load("teleop_tui")
    choices = iter(["2", "1"])
    monitor = SimpleNamespace(stopped=SimpleNamespace(set=lambda: None),
                              thread=SimpleNamespace(start=lambda: None, join=lambda **kw: None))
    monkeypatch.setattr(sys, "argv", ["teleop_tui", "--execute"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(mod, "StatusMonitor", lambda cfg: monitor)
    monkeypatch.setattr(mod.curses, "wrapper", lambda *args: next(choices))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: SimpleNamespace(wait=lambda: 0))

    def interrupt(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert mod.main() == 0


@pytest.mark.parametrize("execute", [False, True])
def test_removed_menu_keys_do_not_launch_actions(monkeypatch, capsys, execute):
    mod = load("teleop_tui")
    assert not {"v", "o", "g"} & mod.ACTIONS.keys()
    assert not {"camera-feeds", "observe-states"} & {action for _, action in mod.ACTIONS.values()}
    monkeypatch.setattr("sys.argv", ["tui"] + (["--execute"] if execute else []))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    answers = iter(["v", "o", "g", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(mod, "hardware_status", lambda cfg: [])
    monkeypatch.setattr(mod, "run_data_action", lambda *a: pytest.fail("Removed option dispatched inference"))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: pytest.fail("Removed option launched a process"))
    monkeypatch.setattr(mod.subprocess, "call", lambda *a, **k: pytest.fail("Removed option executed a command"))
    assert mod.main() == 0
    output = capsys.readouterr().out
    assert "Start inference cameras only" not in output
    assert "Read follower states only" not in output
    assert "GPU inference (predictions only)" not in output
