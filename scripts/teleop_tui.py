#!/usr/bin/env python3
"""Small terminal menu for the guarded Piper teleoperation commands."""
import argparse
import curses
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import xmlrpc.client

from tui_inputs import Cancelled, choice, image_size, number, path as select_path, text as ask

ROOT = Path(__file__).resolve().parents[1]
PIPER = ROOT / "scripts" / "piper.py"

ACTIONS = {
    "1": ("ROS master", "core"),
    "2": ("Teleop", "teleop-menu"),
    "i": ("Initialize CAN (4 ports)", "init-can"),
    "p": ("Preview all cameras (live)", "preview-cameras"),
    "k": ("Calibrate Top stereo fisheye", "calibrate-top"),
    "r": ("Record rosbag session", "record"),
    "b": ("Replay rosbag on robot (local ARM)", "bag-replay"),
    "c": ("Convert rosbag to RGB dataset", "convert-rgb"),
    "s": ("RGB ACT smoke test (1 step)", "act-smoke"),
    "t": ("Train RGB ACT", "act-train"),
    "f": ("Piper SoftMimic training (simulation)", "softmimic"),
    "m": ("Test trained RGB ACT", "act-test"),
}

TELEOP_ACTIONS = {
    "1": ("Start follower drivers", "robot"),
    "2": ("Start teleop (leader) + gravity compensation", "teleop"),
    "3": ("Enable left leader", "enable-teleop-left"),
    "4": ("Enable right leader", "enable-teleop-right"),
    "5": ("Enable left follower", "enable-robot-left"),
    "6": ("Enable right follower", "enable-robot-right"),
    "7": ("Route follower commands from teleop", "mux-teleop"),
    "8": ("Route follower commands from ACT", "mux-vla"),
}

MENUS = {"main": ACTIONS, "teleop": TELEOP_ACTIONS}


ARMS = ("teleop_left", "teleop_right", "robot_left", "robot_right")


def ros_services():
    request = urllib.request.Request(
        os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311"),
        data=xmlrpc.client.dumps(("/piper_tui",), methodname="getSystemState").encode(),
        headers={"Content-Type": "text/xml"})
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            result = xmlrpc.client.loads(response.read())[0][0]
        if result[0] != 1:
            raise ValueError("ROS master rejected status query")
        return {name for name, nodes in result[2][2]}
    except (OSError, ValueError, xmlrpc.client.Error) as error:
        raise ValueError("ROS master unavailable; start main-menu 1 in its own terminal first.") from error


def require_enable_service(action):
    _, role, side = action.split("-")
    service = f"/{role}/arm_{side}/enable_srv"
    if service not in ros_services():
        choice = "2" if role == "teleop" else "1"
        raise ValueError(f"{service} unavailable. Start Teleop submenu {choice} first, "
                         "in another terminal and keep it running; Ctrl-C stops that driver.")


def driver_status(state, now):
    try:
        age = now - state["published_at"]
        if not math.isfinite(age) or age < 0:
            return "UNKNOWN"
        return "RUNNING" if age <= 2 else "STALE"
    except (KeyError, TypeError, ValueError):
        return "NO DATA"


def control_status(state, now):
    if driver_status(state, now) != "RUNNING":
        return "UNKNOWN"
    enabled = state.get("driver_enabled")
    if type(enabled) is not bool:
        return "UNKNOWN"
    return "ENABLED" if enabled else "DISABLED"


def enable_status(state, now):
    """Use actual six-motor feedback; missing/stale data is never DISABLED."""
    try:
        age = now - state["published_at"]
        enabled = state["motor_enabled"]
        ages = state["enable_feedback_age_s"]
        if (not math.isfinite(age) or not 0 <= age <= 2
                or len(enabled) != 6 or any(type(v) is not bool for v in enabled)
                or len(ages) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v + age <= 2 for v in ages)):
            return "UNKNOWN (stale/invalid feedback)"
        if all(enabled):
            return "ENABLED (6/6)"
        if any(enabled):
            return f"PARTIAL ({sum(enabled)}/6)"
        return "DISABLED (0/6)"
    except (KeyError, TypeError, ValueError):
        return "UNKNOWN (no feedback)"


def gripper_status(state, now):
    try:
        age = now - state["published_at"]
        feedback_age = state["gripper_feedback_age_s"]
        enabled = state["gripper_enabled"]
        if (not math.isfinite(age) or not 0 <= age <= 2
                or type(feedback_age) not in (int, float) or not math.isfinite(feedback_age)
                or not 0 <= feedback_age <= 2 - age or type(enabled) is not bool):
            return "UNKNOWN"
        return "ENABLED" if enabled else "DISABLED"
    except (KeyError, TypeError, ValueError):
        return "UNKNOWN"


def ros_status():
    # Run in a ROS-sourced child so the menu also works outside ROS Python.
    import rospy
    from std_msgs.msg import String

    states = {}
    ready = threading.Event()
    lock = threading.Lock()

    def receive(message, arm):
        try:
            state = json.loads(message.data)
        except ValueError:
            return
        with lock:
            states[arm] = state
            if len(states) == len(ARMS):
                ready.set()

    rospy.init_node("piper_tui_status", anonymous=True, disable_signals=True)
    subscribers = [rospy.Subscriber("/" + arm.replace("_", "/arm_", 1)
                                   + "/agent_driver_state", String, receive,
                                   callback_args=arm, queue_size=1) for arm in ARMS]
    ready.wait(0.8)
    with lock:
        print(json.dumps(states), flush=True)
    for subscriber in subscribers:
        subscriber.unregister()
    rospy.signal_shutdown("status sampled")


def can_status(link):
    if link is None:
        return "MISSING"
    if link.get("link_type") != "can" and link.get("linkinfo", {}).get("info_kind") != "can":
        return "NOT CAN"
    if "UP" not in link.get("flags", []):
        return "DOWN"
    state = link.get("linkinfo", {}).get("info_data", {}).get("state", "UNKNOWN")
    return f"UP ({state})"


def hardware_status(cfg):
    links = None
    try:
        result = subprocess.run(["ip", "-details", "-json", "link", "show"],
                                capture_output=True, text=True, check=True, timeout=1)
        links = {link["ifname"]: link for link in json.loads(result.stdout)}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pass
    states = {}
    try:
        result = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null 2>&1 || exit; exec python3 "$2" --ros-status',
             "piper-status", str(ROOT / "workspaces/scripts/use_robot.sh"), str(Path(__file__).resolve())],
            capture_output=True, text=True, check=True, timeout=3)
        states = json.loads(result.stdout)
        if not isinstance(states, dict):
            states = {}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    now = time.time()
    rows = []
    for arm in ARMS:
        port = cfg.get(arm + "_can", "?")
        can = can_status(links.get(port)) if links is not None else "UNKNOWN"
        state = states.get(arm, {})
        rows += [f"{arm:13} {port:10} CAN: {can}  DRIVER: {driver_status(state, now)}",
                 f"  CTRL: {control_status(state, now)}  ARM: {enable_status(state, now)}  GRIP: {gripper_status(state, now)}"]
    return rows


class StatusMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = [f"{arm}: CAN: UNKNOWN  DRIVER: UNKNOWN  CTRL: UNKNOWN  ARM: UNKNOWN  GRIP: UNKNOWN" for arm in ARMS]
        self.updated = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stopped.is_set():
            self.rows = hardware_status(self.cfg)
            self.updated = time.monotonic()
            if self.stopped.wait(1):
                break


def ui_colors():
    colors = {"accent": curses.A_BOLD, "ok": curses.A_BOLD,
              "warn": curses.A_BOLD, "bad": curses.A_BOLD, "muted": curses.A_DIM}
    try:
        if curses.has_colors():
            curses.start_color()
            background = curses.COLOR_BLACK
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                pass
            for index, (name, foreground) in enumerate([
                    ("accent", curses.COLOR_CYAN), ("ok", curses.COLOR_GREEN),
                    ("warn", curses.COLOR_YELLOW), ("bad", curses.COLOR_RED)], 1):
                curses.init_pair(index, foreground, background)
                colors[name] = curses.color_pair(index) | curses.A_BOLD
    except curses.error:
        pass
    return colors


def status_color(value, colors):
    if any(word in value for word in ("BUS-OFF", "DOWN", "MISSING", "NOT CAN")):
        return colors["bad"]
    if any(word in value for word in ("UNKNOWN", "NO DATA", "STALE", "PARTIAL", "ERROR-PASSIVE", "ERROR-WARNING")):
        return colors["warn"]
    if "DISABLED" in value:
        return colors["muted"]
    if "UP" in value or "ENABLED" in value or "RUNNING" in value:
        return colors["ok"]
    return 0


def put(screen, row, col, text, attr=0):
    height, width = screen.getmaxyx()
    if 0 <= row < height and 0 <= col < width - 1:
        try:
            screen.addnstr(row, col, text, width - col - 1, attr)
        except curses.error:
            pass


def live_choice(screen, monitor, mode, menu="main"):
    actions = MENUS[menu]
    choices = [(key, label) for key, (label, _) in actions.items()]
    if menu == "teleop":
        choices.append(("b", "Back to main menu"))
    choices.append(("q", "Quit"))
    keys = {key for key, _ in choices}
    selected = 0
    colors = ui_colors()
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(200)
    while True:
        screen.erase()
        rows = monitor.rows
        if time.monotonic() - monitor.updated > 2:
            rows = [re.sub(r"(CAN|DRIVER|CTRL|ARM|GRIP):\s*.*?(?=\s+(?:CAN|DRIVER|CTRL|ARM|GRIP):|$)",
                           r"\1: UNKNOWN", row) for row in rows]
        height, width = screen.getmaxyx()
        if height < 10 or width < 40:
            put(screen, 0, 0, "Terminal too small (minimum 40x10)", colors["warn"])
            put(screen, 1, 0, "Resize to 80x24 for full status; q quit", colors["muted"])
            screen.refresh()
            key = screen.getch()
            if key == ord("q") or (menu == "teleop" and key == ord("b")):
                return chr(key)
            continue
        panel_width = min(width - 1, 100)
        left = max(0, (width - panel_width) // 2)
        inner = max(0, panel_width - 4)

        def divider(row, label=""):
            text = ("-- " + label + " ") if label else ""
            put(screen, row, left, "+" + text[:max(0, panel_width - 2)].ljust(
                max(0, panel_width - 2), "-") + "+", colors["accent"])

        def line(row, text, attr=0):
            put(screen, row, left, "|", colors["accent"])
            put(screen, row, left + 2, text[:inner].ljust(inner), attr)
            put(screen, row, left + panel_width - 1, "|", colors["accent"])

        title = "PIPER  /  " + ("TELEOP" if menu == "teleop" else "CONTROL CENTER")
        divider(0)
        badge = "[ " + mode + " ]"
        line(1, title[:max(0, inner - len(badge) - 2)], colors["accent"])
        put(screen, 1, left + panel_width - len(badge) - 3, badge,
            colors["bad"] if mode == "EXECUTE" else colors["ok"])
        count = len(rows) if height >= len(rows) + len(choices) + 7 else min(len(rows), max(0, height // 3))
        spinner = "|/-\\"[int(time.monotonic() * 4) % 4]
        hardware_title = f"LIVE HARDWARE {spinner}  /  READ ONLY"
        if count < len(rows):
            hardware_title += f"  ({count}/{len(rows)} rows; enlarge terminal)"
        divider(2, hardware_title)
        for index, text in enumerate(rows[:count], 3):
            line(index, text)
            for match in re.finditer(r"(?:CAN|DRIVER|CTRL|ARM|GRIP):\s*(.*?)(?=\s+(?:CAN|DRIVER|CTRL|ARM|GRIP):|$)", text):
                value = match.group(1)
                offset = match.start(1)
                put(screen, index, left + 2 + offset, value[:max(0, inner - offset)],
                    status_color(value, colors))
        divider(count + 3, "TELEOP" if menu == "teleop" else "MAIN MENU")
        menu_start = count + 4
        capacity = max(1, height - menu_start - 3)
        first = max(0, selected - capacity + 1)
        visible = choices[first:first + capacity]
        for index, (key, label) in enumerate(visible, menu_start):
            active = first + index - menu_start == selected
            line(index, f"{'>' if active else ' '} [{key}] {label}",
                 (colors["accent"] | curses.A_REVERSE) if active else 0)
        bottom = menu_start + len(visible)
        divider(bottom)
        put(screen, bottom + 1, left, "UP/DOWN select | ENTER run | Esc back | [key] shortcut | q quit", colors["muted"])
        put(screen, bottom + 2, left, "GREEN up/enabled | YELLOW unknown/partial | RED down/fault", colors["muted"])
        screen.refresh()
        key = screen.getch()
        if key == 27:
            if menu == "teleop":
                return "b"
            continue
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(choices)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(choices)
        elif key in (10, 13, curses.KEY_ENTER):
            return choices[selected][0]
        elif 0 <= key < 256 and chr(key).lower() in keys:
            return chr(key).lower()


def piper_args(action, config, execute, session=None):
    args = [sys.executable, str(PIPER), action, "--config", str(config)]
    if session:
        args += ["--session", session]
    if execute:
        args.append("--execute")
    return args


def session_name(value):
    if Path(value).name != value or value in ('.', '..') or '\x00' in value:
        raise ValueError('Use a session name without directory separators')
    return value


def checkpoint_directory(path):
    if not (path / 'config.json').is_file() or not (path / 'model.safetensors').is_file():
        raise ValueError('Choose a complete pretrained_model with config.json and model.safetensors')


def recording_directory(path):
    if not list(path.glob('episode_*/episode.bag')):
        raise ValueError('Session must contain episode_*/episode.bag files')
    if list(path.glob('episode_*/*.bag.active')):
        raise ValueError('Recording is still active; finish saving before conversion')
    if list(path.glob('episode_*/BAD_ROSBAG.txt')):
        raise ValueError('Session contains BAD_ROSBAG episodes; choose a reviewed session')
    manifest = path / 'recording_cameras.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        if not isinstance(data['image_topics'], dict) or not data['image_topics']:
            raise ValueError('Recording camera topics are missing')
        if any(type(data[key]) is not int or data[key] <= 0 for key in ('width', 'height')):
            raise ValueError('Recording image dimensions must be positive integers')


def dataset_arm_mode(path):
    features = json.loads((path / 'meta/info.json').read_text())['features']
    dims = (features['observation.state']['shape'], features['action']['shape'])
    if dims == ([14], [7]):
        return 'single'
    if dims == ([28], [14]):
        return 'bimanual'
    raise ValueError('Expected state14/action7 or state28/action14; convert the dataset first')


def json_file(path):
    json.loads(path.read_text())


def absolute_topic(value):
    if not value.startswith('/'):
        raise ValueError('Topic must be an absolute ROS topic')
    return value


def replay_duration(value):
    if value.lower() == 'all':
        return 'all'
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Use all or a finite duration greater than zero')
    return str(result)


def camera_devices():
    from camera_access import remote_devices
    devices = []
    for entry in sorted(Path('/sys/class/video4linux').glob('video*')):
        device = Path('/dev') / entry.name
        try:
            if device.is_char_device() and (entry / 'index').read_text().strip() == '0':
                devices.append(str(device))
        except OSError:
            continue
    devices = list(dict.fromkeys(devices + remote_devices()))
    if not devices:
        raise ValueError('No camera devices mapped into this environment')
    return devices


def camera_window_command(devices, preview_only=False, top_only=False):
    command = ['bash', str(ROOT / 'scripts/camera_window.sh')]
    for device in devices:
        command += ['--device', device]
    if preview_only:
        command.append('--preview-only')
    if top_only:
        command.append('--top-only')
    return command


def select_camera_roles(top_only=False):
    devices = camera_devices()
    print('Opening live camera selection window. ' +
          ('Click Top under the calibration camera, then Confirm.' if top_only else
           'Choose each view by its image, then Confirm.'), flush=True)
    command = camera_window_command(devices, top_only=top_only)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            stdout, stderr = process.communicate()
            break
        except KeyboardInterrupt:
            # The selector must release its camera processes before we return.
            continue
    if process.returncode == 2:
        raise Cancelled(stderr.strip() or 'Camera selection cancelled')
    if process.returncode:
        raise ValueError(stderr.strip() or 'Camera selection cancelled')
    chosen = json.loads(stdout)
    if (not isinstance(chosen, dict) or not chosen or not set(chosen) <= {'left', 'right', 'top'}
            or (top_only and set(chosen) != {'top'})
            or any(device not in devices for device in chosen.values())
            or len(set(chosen.values())) != len(chosen)):
        raise ValueError('Invalid camera selection returned by preview window')
    return chosen


def recording_options():
    chosen = select_camera_roles()
    cameras = [arg for role, device in chosen.items() for arg in ('--camera', role + '=' + device)]
    width, height = image_size('Recorded image size (WIDTHxHEIGHT)', '680x480')
    return cameras + ['--image-width', str(width), '--image-height', str(height)]


def data_command(action, training_python):
    if action == 'bag-replay':
        bags = sorted((ROOT / 'rosbag_data').glob('*/episode_*/*.bag'), key=lambda p: p.stat().st_mtime, reverse=True)
        bag = select_path('Finalized rosbag file', ROOT / 'rosbag_data', suffix='.bag',
                          default=bags[0] if bags else '', candidates=bags)
        source = choice('Bag values: mux=recorded commands, state=measured positions', ('mux', 'state'), 'mux')
        mode = choice('Replay mode', ('robot', 'dry-run', 'check'), 'robot')
        start = number('Start in recording (seconds)', '0', minimum=0)
        duration = ask('Recorded duration (all=whole episode, or seconds)', 'all', replay_duration)
        duration_args = [] if duration == 'all' else ['--duration', duration]
        speed = number('Replay speed', '1', minimum=0, maximum=1, exclusive_minimum=True)
        output = ROOT / 'outputs/rosbag_replays' / time.strftime('%Y%m%d_%H%M%S')
        print('Preflight checks all selected recorded targets. Local ARM moves to the start pose, then replays. '
              'Recording MIT rules apply; no ACT clipping or automatic retiming. '
              'Space pauses/resumes; Enter returns slowly to episode frame zero and holds; q/Ctrl-C exits. '
              'Keep the shared follower drivers (Teleop menu 1) running. Local ARM selects policy on the mux; stop/exit restores teleop. No cameras are required.', flush=True)
        return ['/opt/piper-inference/bin/python', str(ROOT / 'scripts/replay_rosbag.py'),
                '--bag', str(bag), '--source', source, '--mode', mode, '--start', str(start),
                *duration_args, '--speed', str(speed), '--output-dir', str(output), '--rules', 'recording']
    if action == 'act-test':
        candidates = [path for path in (ROOT / 'outputs').glob('*/checkpoints/*/pretrained_model')
                      if (path / 'config.json').is_file() and (path / 'model.safetensors').is_file()]
        latest = max(candidates, key=lambda path: (path / 'config.json').stat().st_mtime).resolve() if candidates else ''
        checkpoint = select_path('Trained checkpoint', ROOT / 'outputs', kind='directory',
                                 default=latest, candidates=sorted(candidates), validate=checkpoint_directory)
        mode = choice('Test mode', ('robot', 'dry-run', 'predictions'), 'robot')
        device = choice('Inference device', ('cuda', 'cpu'), 'cuda')
        output = ROOT / 'outputs/model_tests' / (mode + '_' + time.strftime('%Y%m%d_%H%M%S'))
        if mode == 'predictions':
            print('Predicting 10 action chunks without moving the robot. '
                  'Camera feeds start automatically or reuse existing streams. Follower states must be running. '
                  'The read-only observer starts and stops automatically.', flush=True)
        else:
            print('Local ARM first moves slowly to dataset episode 0 frame 0 measured pose, '
                  'then starts continuous live ACT with temporal ensemble. Space pauses/resumes; Enter returns to first frame and holds; q/Ctrl-C returns there before exit. '
                  'A read-only observer starts automatically. Camera feeds start automatically. Keep shared follower drivers (2 -> 1) running. ARM selects policy on the mux; stop/exit restores teleop; J2/J3 use recording MIT bounds. Other motion and feedback checks remain active.', flush=True)
        return ['/opt/piper-inference/bin/python', str(ROOT / 'scripts/test_rgb_act.py'), '--checkpoint', str(checkpoint),
                '--mode', mode, '--device', device, '--output-dir', str(output)] + (
                    ['--reset-to-first-frame'] if mode != 'predictions' else [])
    if action == "convert-rgb":
        sessions = sorted({p.parent.parent for p in (ROOT / 'rosbag_data').glob('*/episode_*/episode.bag')})
        session = select_path('Rosbag session directory', ROOT / 'rosbag_data', kind='directory',
                              candidates=sessions, validate=recording_directory)
        output = select_path('New RGB dataset directory', ROOT / 'data/lerobot', kind='directory', new=True,
                             default=ROOT / 'data/lerobot' / (session.name + '_rgb'))
        manifest_path = session / 'recording_cameras.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        topics = manifest['image_topics'] if manifest else None
        indices = ','.join(str(index) for role, index in [('top', 1), ('left', 2), ('right', 3)]
                           if role in topics) if topics else 'all'
        task = ask("Task description")
        top_format = 'mono' if manifest else choice("Top image format", ("mono", "sbs"), "sbs")
        topic = topics.get('top', '/top_camera/image_raw/compressed') if topics else ask(
            "Top compressed image topic", "/top_stereo/color/image_raw/compressed" if top_format == "sbs" else "", absolute_topic)
        width, height = image_size('RGB output size (WIDTHxHEIGHT)',
                                   f"{manifest['width']}x{manifest['height']}" if manifest else '640x320')
        command = [str(training_python), str(ROOT / "training/scripts/convert_rosbag_to_lerobot.py"),
                   "--session-dir", str(session), "--output-root", str(output.parent),
                   "--dataset-id", output.name, "--task", task, "--fps", "30",
                   "--action-source", "mux", "--convert-to-v30", "--obs-indices", indices,
                   "--state-indices", "1-14,22-35", "--action-indices", "all",
                   "--top-camera-topic", topic,
                   "--left-wrist-topic", topics.get('left', '/fisheye_left/usb_cam_node/image_raw/compressed') if topics else '/fisheye_left/usb_cam_node/image_raw/compressed',
                   "--right-wrist-topic", topics.get('right', '/fisheye_right/usb_cam_node/image_raw/compressed') if topics else '/fisheye_right/usb_cam_node/image_raw/compressed',
                   "--image-width", str(width), "--image-height", str(height)]
        if top_format == "sbs":
            command.append("--lfv-top-crop")
            for label, flag in [("Camera calibration JSON", "--actfs-calibration-json"),
                                ("Camera processing metadata JSON", "--actfs-processing-metadata")]:
                path = select_path(label, ROOT / "data/calibration", suffix=".json", validate=json_file)
                command += [flag, str(path)]
        return command
    datasets = sorted(p.parent.parent for p in (ROOT / 'data/lerobot').glob('*/meta/info.json'))
    dataset = select_path('LeRobot dataset directory', ROOT / 'data/lerobot', kind='directory',
                          candidates=datasets, validate=dataset_arm_mode)
    arm_mode = dataset_arm_mode(dataset)
    smoke = action == 'act-smoke'
    suffix = '_rgb_act_smoke' if smoke else '_rgb_act'
    output = select_path('New training output directory', ROOT / 'outputs', kind='directory', new=True,
                         default=ROOT / 'outputs' / (dataset.name + suffix))
    command = [str(training_python), str(ROOT / "scripts/train_act.py"),
               "--dataset", str(dataset), "--output", str(output), "--arm-mode", arm_mode]
    if smoke:
        command.append("--smoke")
    else:
        steps = number("Training steps", "80000", minimum=1, integer=True)
        batch = number("Batch size", "16", minimum=1, integer=True)
        command += ["--steps", str(steps), "--batch-size", str(batch)]
    return command


def run_data_action(action, training_python, execute):
    inference = action in {'act-test', 'bag-replay'}
    if inference and not Path('/.dockerenv').exists():
        raise ValueError('Model testing runs inside piper-control. Enter with: '
                         'docker exec -it -w /opt/piper piper-control bash, then run '
                         'python3 scripts/teleop_tui.py --execute')
    command = data_command(action, training_python)
    print(shlex.join(command), flush=True)
    if not execute:
        print("PREVIEW ONLY. Restart the TUI with --execute to run this action.")
        return
    # Keep ROS Python packages out of the independent training interpreter.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    runtime_python = command[0] if inference else str(training_python)
    modules = ["numpy", "pandas", "cv2", "pyarrow", "lerobot", "rosbags", "torch"] if action == "convert-rgb" else ["torch", "lerobot", "torchcodec"]
    if action == 'bag-replay':
        modules = ['numpy']
    elif inference:
        modules = ["torch", "lerobot", "cv2"]
    probe = """import importlib.util
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    sys.exit('Use a separate Python 3.10-3.12 training environment')
missing = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]
if missing:
    sys.exit('Missing training dependencies: ' + ', '.join(missing))
"""
    try:
        subprocess.run([runtime_python, "-c", probe, *modules], env=env,
                       check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        if inference:
            raise ValueError("Container inference environment is missing. See docs/09_docker_inference.md. " + str(error)) from error
        raise ValueError("Training Python unavailable or missing dependencies. Set --training-python to an installed "
                         "training environment; see docs/00_installation.md. " + str(error)) from error
    print("This action owns this terminal; Ctrl-C stops it.", flush=True)
    if action == 'act-test':
        process = subprocess.Popen(command, env=env, cwd=ROOT)
        while True:
            try:
                result = process.wait()
                break
            except KeyboardInterrupt:
                # The launcher/bridge handle Ctrl-C and must finish the slow return before cleanup.
                continue
    else:
        result = subprocess.call(command, env=env, cwd=ROOT)
    if result:
        print(f"Action exited with status {result}; review the output above.")


def softmimic_checkpoint(path):
    if not path.is_file() or path.suffix!='.pt' or not (path.parent/'manifest.json').is_file():
        raise ValueError('Choose a SoftMimic model_*.pt with its run manifest.json')


def softmimic_dataset(path):
    if not (path/'train.npz').is_file() or not (path/'manifest.json').is_file():
        raise ValueError('Choose a CMA directory containing train.npz and manifest.json')
    if (path/'failure.json').exists() or json.loads((path/'manifest.json').read_text()).get('status','complete')!='complete':
        raise ValueError('Dataset generation is incomplete or failed')


def run_softmimic_action(python,execute):
    root=ROOT/'outputs/piper_compliance/softmimic_official'
    mode=choice('Piper SoftMimic simulation',('status','resume','new','evaluate'),'status')
    runs=sorted((p for p in root.glob('*/progress.json')
                 if not p.parent.name.startswith(('smoke','preflight','tui_'))
                 and (p.parent/'manifest.json').is_file()),key=lambda p:p.stat().st_mtime,reverse=True)
    if mode=='status':
        if not runs:
            print('No SoftMimic training run found. Choose new to start one.')
            return
        progress=json.loads(runs[0].read_text())
        print(f"SoftMimic: {progress['status']} | {progress['iteration']:,}/{progress['total_iterations']:,} PPO updates",flush=True)
        print(f'Run: {runs[0].parent}')
        checkpoints=sorted(runs[0].parent.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        if checkpoints:print(f'Checkpoint: {checkpoints[-1]}')
        return
    command=[str(python),'-u','-m','sim.piper_softmimic.train','--accept-eula']
    if mode in ('resume','evaluate'):
        checkpoints=[p for run in runs for p in sorted(run.parent.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]),reverse=True)]
        checkpoint=select_path('SoftMimic PPO checkpoint',root,suffix='.pt',candidates=checkpoints,
                               default=checkpoints[0] if checkpoints else '',validate=softmimic_checkpoint)
        if mode=='resume':command+=['--resume',str(checkpoint)]
    if mode=='evaluate':
        references=[p for p in checkpoints if p!=checkpoint]
        preferred=checkpoint.parent/'model_5500.pt'
        if preferred==checkpoint or not preferred.is_file():preferred=references[0] if references else ''
        reference=select_path('Reference PPO checkpoint (e.g. model_5500)',root,suffix='.pt',
                              candidates=references,default=preferred,validate=softmimic_checkpoint)
        if checkpoint==reference:raise ValueError('Choose two different checkpoints for comparison')
        protocol=select_path('Frozen XYZ evaluation protocol',root/'evaluations',suffix='.npz',
                             default=root/'evaluations/xyz_push_v1/protocol.npz')
        noise=choice('Evaluation observation noise',('both','clean','noisy'),'both')
        output=select_path('New evaluation directory',root/'evaluations',kind='directory',new=True,
                           default=root/'evaluations'/(checkpoint.parent.name+'_'+checkpoint.stem+'_compare_'+time.strftime('%Y%m%d_%H%M%S')))
        command=[str(python),'-u','-m','sim.piper_softmimic.compare','--accept-eula',
                 '--checkpoint',str(checkpoint),'--reference-checkpoint',str(reference),
                 '--protocol',str(protocol),'--noise',noise]
        print('Independent +/-XYZ push/release: selected PPO, reference PPO and privileged analytic baseline.',flush=True)
        print('Frozen checkpoints; live training continues. Both policies use identical forces and paired physical properties.',flush=True)
    else:
        if mode=='new':
            datasets=sorted((p.parent for p in root.glob('dataset*/manifest.json')
                             if json.loads(p.read_text()).get('status','complete')=='complete'
                             and (p.parent/'train.npz').is_file() and not (p.parent/'failure.json').exists()),
                            key=lambda p:(p/'manifest.json').stat().st_mtime,reverse=True)
            dataset=select_path('CMA dataset version (rewards unchanged)',root,kind='directory',
                                candidates=datasets,default=datasets[0] if datasets else root/'dataset',validate=softmimic_dataset)
            command+=['--dataset',str(dataset)]
        output=select_path('New SoftMimic run directory',root,kind='directory',new=True,
                           default=root/('train_'+mode+'_'+time.strftime('%Y%m%d_%H%M%S')))
        print('Fixed-base Piper, fixed gripper, optimized SoftMimic PPO: 4096 environments, 100000 total updates.',flush=True)
    command+=['--output',str(output)]
    if mode=='resume':print('Restores policy and optimizer; simulation state is initialized again.',flush=True)
    print(shlex.join(command),flush=True)
    if not execute:
        print('PREVIEW ONLY. Restart the TUI with --execute to run this action.')
        return
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    probe="import importlib.util,sys; assert sys.version_info[:2]==(3,11), 'Isaac Sim 5.1 requires Python 3.11'; missing=[m for m in ['torch','isaacsim','isaaclab','rsl_rl'] if importlib.util.find_spec(m) is None]; assert not missing, 'Missing simulation modules: '+', '.join(missing)"
    try:
        subprocess.run([str(python),'-c',probe],env=env,cwd=ROOT,check=True,timeout=20)
    except (OSError,subprocess.SubprocessError) as error:
        raise ValueError('Use the GPU host TUI with --softmimic-python pointing to the installed Isaac environment; see sim/piper_softmimic/README.md. '+str(error)) from error
    if mode=='evaluate':print('Evaluation owns this terminal. Ctrl-C cancels only this evaluation. Results: '+str(output),flush=True)
    else:print('Training owns this terminal. Ctrl-C saves a checkpoint after the current PPO update, then returns.',flush=True)
    process=subprocess.Popen(command,env=env,cwd=ROOT)
    while True:
        try:
            result=process.wait();break
        except KeyboardInterrupt:
            continue
    failure=output/'failure.json'
    if failure.is_file():print('SoftMimic failed: '+str(json.loads(failure.read_text())['error']))
    elif result:print(f'SoftMimic exited with status {result}; review its output and failure.json.')
    elif mode=='evaluate':
        if not (output/'report.json').is_file():print('Evaluation did not produce a complete report; inspect its logs.')
        else:print(f'Evaluation results: {output}/xyz_positive.png, xyz_negative.png, summary.csv, report.json')


def run_calibration_action(calibration_python, execute):
    print("Top must output both fisheye views in ONE side-by-side frame.")
    device = select_camera_roles(top_only=True)['top']
    print(f"Selected Top camera: {device}", flush=True)
    width, height = image_size('Native FULL side-by-side size (WIDTHxHEIGHT)', '2560x720', even_width=True)
    fps = number('Requested camera FPS', '30', minimum=1, integer=True)
    board = choice('Board preset', ('stereo-charuco-7x5-38-a4-v1', 'existing-20mm', 'a4-30mm'),
                   'stereo-charuco-7x5-38-a4-v1')
    output = select_path('Calibration sessions directory', ROOT / 'data/calibration', kind='directory',
                         default=ROOT / 'data/calibration/top_stereo', allow_new=True)
    port = number('Local dashboard port', '8765', minimum=1, maximum=65535, integer=True)
    command = [str(calibration_python), str(ROOT / "scripts/calibrate_top_camera.py"),
               "gui", "--camera", device, "--width", str(width), "--height", str(height),
               "--fps", str(fps), "--board", board, "--output", str(output), "--port", str(port)]
    if Path("/.dockerenv").exists():
        command.append("--no-browser")
    print(shlex.join(command), flush=True)
    if not execute:
        print("PREVIEW ONLY. Add --execute to start the calibration dashboard.")
        return
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        subprocess.run([str(calibration_python), str(ROOT / "scripts/calibrate_top_camera.py"), "--check"],
                       env=env, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Run bash scripts/setup_calibration.sh in this environment first, or set "
                         "--calibration-python to its Python executable. " + str(error)) from error
    print(f"Open http://127.0.0.1:{port} in your browser. Verify the printed board dimensions in Setup.")
    print("This terminal owns the dashboard. Ctrl-C stops it and releases its camera.", flush=True)
    process = subprocess.Popen(command, env=env, cwd=ROOT)
    while True:
        try:
            result = process.wait()
            break
        except KeyboardInterrupt:
            continue
    if result:
        print(f"Calibration exited with status {result}; review the output above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/hardware.local.json")
    parser.add_argument("--execute", action="store_true", help="run the selected command instead of previewing it")
    parser.add_argument("--training-python", type=Path, default=ROOT / ".venv-training/bin/python",
                        help="Independent Python 3.10-3.12 environment for RGB conversion and ACT")
    parser.add_argument('--softmimic-python',type=Path,default=ROOT/'.venv-softmimic/bin/python',
                        help='Independent Python 3.11 Isaac Sim environment for Piper SoftMimic')
    parser.add_argument("--calibration-python", type=Path,
                        default=ROOT / "external/fisheye_calibration/camera_calib_charuco/.venv/bin/python",
                        help="Independent Python 3.12 environment for the Top fisheye dashboard")
    args = parser.parse_args()
    mode = "EXECUTE" if args.execute else "PREVIEW"

    path = args.config if args.config.exists() else ROOT / "config/hardware.example.json"
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    monitor = StatusMonitor(cfg)
    live = sys.stdin.isatty() and sys.stdout.isatty()
    if live:
        monitor.thread.start()
    menu = "main"
    try:
        while True:
            actions = MENUS[menu]
            if live:
                choice = curses.wrapper(live_choice, monitor, mode, menu)
            else:
                title = f"Piper teleop TUI ({mode})" + (" / Teleop" if menu == "teleop" else "")
                print("\n" + title)
                print("\n".join(hardware_status(cfg)))
                for key, (label, _) in actions.items():
                    print(f"  {key}. {label}")
                if menu == "teleop":
                    print("  b. Back to main menu")
                print("  q. Quit")
                try:
                    choice = input("> ").strip().lower()
                except EOFError:
                    return 0
            if choice == "\x1b":
                menu = "main"
                continue
            if choice in {"q", "quit"}:
                return 0
            if choice == "b" and menu == "teleop":
                menu = "main"
                continue
            if choice not in actions:
                print("Unknown selection.")
                continue

            label, action = actions[choice]
            if action == "teleop-menu":
                menu = "teleop"
                continue
            if action in {"convert-rgb", "act-smoke", "act-train", "act-test", "bag-replay", "calibrate-top", "softmimic"}:
                print(f"\n[{mode}] {label}", flush=True)
                try:
                    # Keep venv interpreter symlinks; resolving them loses their installed packages.
                    if action == 'softmimic':
                        run_softmimic_action(args.softmimic_python.expanduser().absolute(),args.execute)
                    elif action == "calibrate-top":
                        run_calibration_action(args.calibration_python.expanduser().absolute(), args.execute)
                    else:
                        run_data_action(action, args.training_python.expanduser().absolute(), args.execute)
                except Cancelled:
                    continue
                except EOFError:
                    return 0
                except KeyboardInterrupt:
                    print("Action interrupted.")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    print(f"Action stopped: {error}")
                if live:
                    input("Press Enter to return to the menu: ")
                continue
            session, options = None, []
            if action == 'record':
                try:
                    session = ask('Session name', validate=session_name)
                    options = recording_options()
                except Cancelled:
                    continue
                except EOFError:
                    return 0
                except (OSError, ValueError) as error:
                    print(f'Recording stopped: {error}')
                    if live:
                        input('Press Enter to return to the menu: ')
                    continue
            print(f"\n[{mode}] {label}", flush=True)
            if args.execute and action in {"core", "robot", "teleop", "record", "preview-cameras"}:
                print("This long-running command owns this terminal; Ctrl-C stops it.")
            try:
                if action == 'init-can':
                    command = [sys.executable, str(ROOT / 'scripts/setup_can.py'), '--config', str(args.config)]
                    if args.execute:
                        command += ['--execute', '--check-drivers-stopped']
                elif action == 'preview-cameras':
                    command = camera_window_command(camera_devices(), preview_only=True)
                    print(shlex.join(command), flush=True)
                    if not args.execute:
                        print('PREVIEW ONLY. Add --execute to open the live camera window.')
                        continue
                else:
                    command = piper_args(action, args.config, args.execute, session) + options
                if action in {"robot", "teleop", "record", "preview-cameras"}:
                    process = subprocess.Popen(command)
                    while True:
                        try:
                            process.wait()
                            break
                        except KeyboardInterrupt:
                            continue
                else:
                    if args.execute and action.startswith("enable-"):
                        require_enable_service(action)
                    result = subprocess.call(command)
                    if result:
                        print(f'Action exited with status {result}; review the output above.')
            except Cancelled:
                continue
            except KeyboardInterrupt:
                pass
            except (OSError, ValueError) as error:
                print(f"Action stopped: {error}")
            if live:
                try:
                    input("Press Enter to return to the menu: ")
                except (KeyboardInterrupt, EOFError):
                    return 0
    except KeyboardInterrupt:
        return 0
    finally:
        monitor.stopped.set()
        if live:
            monitor.thread.join(timeout=4)


if __name__ == "__main__":
    if sys.argv[1:] == ["--ros-status"]:
        ros_status()
    else:
        raise SystemExit(main())
