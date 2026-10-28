#!/usr/bin/env python3
"""Preview all cameras or select recording roles; never starts recording."""
import argparse
from collections import deque
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('left', 'right', 'top', None)
LABELS = ('Left wrist', 'Right wrist', 'Top', 'Skip')
WINDOW = 'Piper - Select recording cameras'


def configure_display():
    config = ROOT / 'outputs/tui/gui.json'
    if config.is_file():
        settings = json.loads(config.read_text())
        for key in ('DISPLAY', 'XAUTHORITY'):
            if settings.get(key):
                os.environ.setdefault(key, settings[key])
    if not os.environ.get('DISPLAY'):
        raise ValueError('Camera selection needs a desktop display. Configure Docker DISPLAY/XAUTHORITY; see docs/00_installation.md.')
    # Check X11 before opening a GUI backend, which may abort on a bad display.
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(os.environ['DISPLAY'].encode())
    if not display:
        raise ValueError('Cannot connect to the desktop display; check DISPLAY and XAUTHORITY.')
    x11.XCloseDisplay(display)


class CameraSelection:
    def __init__(self, devices, preview_only=False, top_only=False):
        self.devices = list(devices)
        self.preview_only = preview_only
        self.top_only = top_only
        self.roles = ('top', None) if top_only else ROLES
        self.labels = ('Top', 'Skip') if top_only else LABELS
        self.selected = {}
        self.frames = {device: None for device in devices}
        self.frame_times = {device: deque(maxlen=30) for device in devices}
        self.errors = {}
        self.buttons = []
        self.confirmed = False
        self.cancelled = False

    def assign(self, device, role):
        if device not in self.devices or role not in self.roles:
            raise ValueError('Unknown camera or role')
        self.selected = {key: value for key, value in self.selected.items()
                         if value != device and key != role}
        if role is not None:
            self.selected[role] = device

    def update(self, device, frame):
        now = time.monotonic()
        self.frames[device] = (frame, now)
        self.frame_times[device].append(now)

    def can_confirm(self):
        now = time.monotonic()
        return bool(self.selected) and all(
            self.frames[device] is not None and now - self.frames[device][1] <= 2
            and device not in self.errors for device in self.selected.values())

    def click(self, x, y):
        for (left, top, width, height), device, role in self.buttons:
            if not (left <= x < left + width and top <= y < top + height):
                continue
            if device == 'confirm':
                self.confirmed = self.can_confirm()
            elif device == 'cancel':
                self.cancelled = True
            else:
                self.assign(device, role)
            break

    def render(self):
        import cv2
        import numpy as np

        cols = min(2, len(self.devices))
        rows = (len(self.devices) + cols - 1) // cols
        row_height = 350 if self.preview_only else 380
        width, height = cols * 420 + 40, rows * row_height + (115 if self.preview_only else 150)
        canvas = np.full((height, width, 3), (28, 25, 23), dtype=np.uint8)
        self.buttons = []

        def text(value, x, y, scale=.5, color=(230, 230, 230)):
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        def button(rect, label, device, role=None, active=False, enabled=True):
            x, y, w, h = rect
            color = (83, 119, 53) if active else (68, 61, 54) if enabled else (42, 39, 36)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
            text(label, x + 9, y + h - 10, .46)
            self.buttons.append((rect, device, role))

        title = 'Select Top camera for calibration' if self.top_only else 'Select recording cameras'
        text('Live camera preview' if self.preview_only else title, 20, 29, .8)
        text('All connected cameras. Q / ESC / Close returns to the TUI.' if self.preview_only else
             'Choose roles by looking at the live views. ENTER confirms; ESC cancels.', 20, 55, .48)
        for index, device in enumerate(self.devices):
            x, y = 20 + (index % cols) * 420, 75 + (index // cols) * row_height
            role = next((key for key, value in self.selected.items() if value == device), None)
            sample = self.frames[device]
            fresh = sample is not None and time.monotonic() - sample[1] <= 2 and device not in self.errors
            times = self.frame_times[device]
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0
            label = (f'{fps:.1f} FPS' if fresh else 'NO LIVE IMAGE') if self.preview_only else role or 'not selected'
            text(device + '  -  ' + label, x, y + 22, .58)
            if fresh:
                canvas[y + 35:y + 335, x:x + 400] = cv2.resize(sample[0], (400, 300))
            else:
                cv2.rectangle(canvas, (x, y + 35), (x + 400, y + 335), (42, 39, 36), -1)
                status = self.errors.get(device, 'Waiting for live image...' if sample is None else 'Preview stale')
                for line in range(3):
                    text(status[line * 48:(line + 1) * 48], x + 10, y + 155 + line * 24, .44)
            if not self.preview_only:
                for n, (value, label) in enumerate(zip(self.roles, self.labels)):
                    button((x + n * 102, y + 343, 94, 28), label, device, value, active=role == value)
        if not self.preview_only:
            text(f'{len(self.selected)} view(s) selected', 20, height - 30, .52)
            button((width - 280, height - 55, 130, 36), 'Confirm', 'confirm', enabled=self.can_confirm())
        button((width - 140, height - 55, 120, 36), 'Close' if self.preview_only else 'Cancel', 'cancel')
        return canvas


def active_camera_topics(devices, publishers, get_param):
    topics = {}
    for topic, nodes in publishers:
        if not topic.endswith('/image_raw/compressed'):
            continue
        for node in nodes:
            device = get_param(node + '/video_device', None)
            if device in devices:
                topics[device] = topic
    return topics


def select_cameras(devices, preview_only=False, top_only=False):
    from record_session import camera_command, ensure_camera_free, stop_cameras
    import cv2
    import numpy as np
    import rospy
    import rosgraph
    from sensor_msgs.msg import CompressedImage

    configure_display()
    rospy.init_node('piper_camera_selector', anonymous=True, disable_signals=True)
    selection = CameraSelection(devices, preview_only, top_only)
    window = ('Piper - Select Top calibration camera' if top_only else
              'Piper - Live camera preview' if preview_only else WINDOW)
    processes, subscribers = [], []
    logs = ROOT / 'outputs/tui'
    logs.mkdir(parents=True, exist_ok=True)

    def receive(message, device):
        frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            selection.update(device, frame)

    try:
        shared = active_camera_topics(devices, rosgraph.Master('/piper_camera_preview').getSystemState()[0],
                                      rospy.get_param) if preview_only else {}
        with (logs / 'camera_selection.log').open('ab') as log:
            for index, device in enumerate(devices):
                try:
                    if device in shared:
                        subscribers.append(rospy.Subscriber(shared[device], CompressedImage, receive,
                                                            callback_args=device, queue_size=1))
                        continue
                    ensure_camera_free(device)
                    namespace = f'/piper_camera_selector_{os.getpid()}/camera_{index}'
                    command = camera_command('top', device, 400, 300)
                    if top_only:
                        command = ['_auto_exposure:=false' if arg == '_auto_exposure:=true' else arg
                                   for arg in command]
                    command = [arg for arg in command if not arg.startswith(('__name:=', '__ns:='))]
                    command += [f'__name:=camera_{index}', '__ns:=' + namespace]
                    subscribers.append(rospy.Subscriber(namespace + '/image_raw/compressed', CompressedImage,
                                                        receive, callback_args=device, queue_size=1))
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append((device, process))
                except (OSError, ValueError) as error:
                    selection.errors[device] = str(error)
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, lambda event, x, y, flags, param:
                                 selection.click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)
            while not selection.cancelled and not rospy.is_shutdown():
                for device, process in processes:
                    if process.poll() is not None:
                        selection.errors[device] = 'Preview stopped; see outputs/tui/camera_selection.log'
                cv2.imshow(window, selection.render())
                key = cv2.waitKey(30) & 0xff
                if preview_only and key in (ord('q'), ord('Q')):
                    break
                if key in (10, 13) and selection.can_confirm():
                    selection.confirmed = True
                if selection.confirmed:
                    return dict(selection.selected)
                if key == 27:
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_AUTOSIZE) < 0:
                        break
                except cv2.error:
                    break
        return None
    finally:
        for sub in subscribers:
            sub.unregister()
        stop_cameras([process for device, process in processes])
        cv2.destroyAllWindows()
        rospy.signal_shutdown('camera selection finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', action='append', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview-only', action='store_true', help='Show all live views without selecting roles')
    mode.add_argument('--top-only', action='store_true', help='Select exactly one Top camera for calibration')
    args = parser.parse_args()
    if (not args.device or len(set(args.device)) != len(args.device)
            or any(not re.fullmatch(r'/dev/video\d+', device) for device in args.device)):
        parser.error('Choose distinct available camera devices')
    selected = select_cameras(args.device, args.preview_only, args.top_only)
    if args.preview_only:
        return 0
    if selected is None:
        print('Camera selection cancelled.', file=sys.stderr)
        return 2
    print(json.dumps(selected), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Camera selection cancelled.', file=sys.stderr)
        raise SystemExit(2)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
