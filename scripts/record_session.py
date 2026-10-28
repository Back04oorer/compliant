#!/usr/bin/env python3
"""Start selected USB camera views and reuse the keyboard rosbag recorder."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CAMERA_NODE = ROOT / 'src/zeno-wholebody-intervation/zeno-wholebody-robot/robot_side/robot_setup/scripts/top_mjpg_camera_node.py'
CAMERA_BASES = {
    'left': '/fisheye_left/usb_cam_node/image_raw',
    'right': '/fisheye_right/usb_cam_node/image_raw',
    'top': '/top_camera/image_raw',
}


def parse_cameras(values):
    cameras = {}
    for value in values:
        role, sep, device = value.partition('=')
        if not sep or role not in CAMERA_BASES or not re.fullmatch(r'/dev/video\d+', device):
            raise ValueError('Use --camera left=/dev/video0, right=/dev/video2, or top=/dev/videoN')
        if role in cameras or device in cameras.values():
            raise ValueError('Each role and physical camera may only be selected once')
        cameras[role] = device
    if not cameras:
        raise ValueError('Select at least one camera')
    return cameras


def recording_profile(base, cameras):
    profile = dict(base)
    profile['profile_name'] = 'selected_rgb_' + '_'.join(cameras)
    profile['camera_transport'] = 'compressed'
    profile['camera_streams'] = [
        {'image_base': CAMERA_BASES[role],
         'camera_info': CAMERA_BASES[role].rsplit('/', 1)[0] + '/camera_info',
         'camera_info_required': False} for role in cameras]
    profile['hz_monitor_topics'] = list(profile['required_topics']) + [CAMERA_BASES[role] for role in cameras]
    return profile


def camera_command(role, device, width, height):
    namespace = CAMERA_BASES[role].rsplit('/', 1)[0]
    return [sys.executable, str(CAMERA_NODE), '__name:=record_camera_' + role, '__ns:=' + namespace,
            '_video_device:=' + device, '_image_width:=640', '_image_height:=480',
            '_pixel_format:=mjpeg', '_framerate:=30', '_output_width:=' + str(width),
            '_output_height:=' + str(height), '_enforce_actual_resolution:=true',
            '_enforce_actual_pixel_format:=true', '_auto_exposure:=true',
            '_camera_frame_id:=record_' + role + '_optical_frame']


def wait_for_recording(process):
    # The child handles Ctrl-C and must finish saving before its cameras stop.
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def stop_cameras(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def ensure_camera_free(device, proc_root=Path('/proc')):
    from camera_access import remote_devices
    if not Path(device).is_char_device() and device not in remote_devices():
        raise ValueError('Camera device unavailable: ' + device)
    for fd in proc_root.glob('[0-9]*/fd/*'):
        try:
            if fd.resolve() == Path(device):
                raise ValueError('Camera already in use: ' + device)
        except OSError:
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--camera', action='append', required=True)
    parser.add_argument('--image-width', type=int, default=680)
    parser.add_argument('--image-height', type=int, default=480)
    args = parser.parse_args()
    cameras = parse_cameras(args.camera)
    if not args.session or Path(args.session).name != args.session or args.session in ('.', '..'):
        parser.error('Session must be a simple name, not a path')
    if args.image_width <= 0 or args.image_height <= 0:
        parser.error('Image dimensions must be positive')
    import rosgraph
    import yaml
    master = rosgraph.Master('/piper_record_session')
    master.getPid()
    published = {topic for topic, _ in master.getPublishedTopics('/')}
    for role, device in cameras.items():
        ensure_camera_free(device)
        if CAMERA_BASES[role] in published or CAMERA_BASES[role] + '/compressed' in published:
            parser.error('Camera topic already has a publisher: ' + CAMERA_BASES[role])
    base = yaml.safe_load((ROOT / 'workspaces/config/rosbag_profiles/lfv_teleop_top_stereo_raw_profile.yaml').read_text())
    profile = recording_profile(base, cameras)
    session = ROOT / 'rosbag_data' / args.session
    manifest = {'cameras': cameras, 'width': args.image_width, 'height': args.image_height,
                'fps': 30, 'capture_width': 640, 'capture_height': 480,
                'image_topics': {role: CAMERA_BASES[role] + '/compressed' for role in cameras}}
    manifest_path = session / 'recording_cameras.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error('This session uses different cameras or dimensions; choose a new session name')
    elif session.exists() and any(session.iterdir()):
        parser.error('Existing session has no camera manifest; choose a new session name')
    session.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    profile_path = session / 'recording_profile.json'
    profile_path.write_text(json.dumps(profile, indent=2) + '\n')
    processes = []
    try:
        for role, device in cameras.items():
            processes.append(subprocess.Popen(camera_command(role, device, args.image_width, args.image_height),
                                               start_new_session=True))
        recorder = [sys.executable, str(ROOT / 'workspaces/scripts/record_act_keyboard.py'),
                    '--profile', str(profile_path), '--session-root', str(ROOT / 'rosbag_data'),
                    '--session-name', args.session, '--expected-camera-fps', '30', '--save-workers', '1',
                    '--wait-for-save', '--save-progress-mode', 'verbose',
                    '--dropframe-warn-ratio', '0.90']
        return wait_for_recording(subprocess.Popen(recorder))
    finally:
        stop_cameras(processes)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
