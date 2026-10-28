#!/usr/bin/env python3
"""Start missing RGB ACT feeds and stop only cameras owned by this process."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from record_session import CAMERA_BASES, ROOT, camera_command, ensure_camera_free, parse_cameras, stop_cameras


def live_camera_settings(master):
    """Remember the already selected live camera roles without guessing device order."""
    publishers = dict(master.getSystemState()[0])
    cameras, sizes = {}, set()
    for role, base in CAMERA_BASES.items():
        nodes = publishers.get(base + '/compressed', [])
        if len(nodes) != 1:
            raise ValueError('No saved inference cameras. Configure all three roles with scripts/start_rgb_cameras.py --camera left=DEVICE --camera right=DEVICE --camera top=DEVICE.')
        node = nodes[0]
        cameras[role] = master.getParam(node + '/video_device')
        sizes.add((master.getParam(node + '/output_width'), master.getParam(node + '/output_height')))
    if len(sizes) != 1:
        raise ValueError('Live RGB camera dimensions differ; configure matching feeds with scripts/start_rgb_cameras.py')
    width, height = sizes.pop()
    return {'cameras': cameras, 'width': width, 'height': height}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', action='append', help='left=/dev/videoN, right=/dev/videoN, top=/dev/videoN')
    parser.add_argument('--config', type=Path, help='Saved role/device mapping; default outputs/tui/inference_cameras.json')
    parser.add_argument('--reuse-existing', action='store_true', help='Keep existing camera publishers; start only missing feeds')
    parser.add_argument('--session-file', type=Path, help='New file describing started/reused cameras for the managed inference launcher')
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--image-height', type=int)
    args = parser.parse_args()
    if args.camera:
        cameras = parse_cameras(args.camera)
        if set(cameras) != set(CAMERA_BASES):
            parser.error('RGB ACT needs all three roles: left, right, top')
    else:
        cameras = None
    if any(v is not None and v <= 0 for v in (args.image_width, args.image_height)):
        parser.error('Image dimensions must be positive')
    import rosgraph

    master = rosgraph.Master('/piper_rgb_camera_start')
    master.getPid()
    published = dict(master.getPublishedTopics('/'))
    config_path = args.config or ROOT / 'outputs/tui/inference_cameras.json'
    settings = ({'cameras': cameras, 'width': 680, 'height': 480} if cameras is not None else
                json.loads(config_path.read_text()) if config_path.is_file() else live_camera_settings(master))
    cameras = parse_cameras([role + '=' + device for role, device in settings['cameras'].items()])
    if set(cameras) != set(CAMERA_BASES):
        raise ValueError('Saved inference cameras require left, right and top roles; configure them with scripts/start_rgb_cameras.py --camera')
    width = args.image_width if args.image_width is not None else settings['width']
    height = args.image_height if args.image_height is not None else settings['height']
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('Saved camera dimensions must be positive integers')
    started, reused = [], []
    for role, device in cameras.items():
        base = CAMERA_BASES[role]
        if base + '/compressed' in published:
            if not args.reuse_existing:
                raise ValueError('Camera topic already has a publisher: ' + base)
            if published[base + '/compressed'] != 'sensor_msgs/CompressedImage':
                raise ValueError('Unexpected compressed camera message type: ' + base)
            reused.append(role)
        else:
            if base in published:
                raise ValueError('Camera raw topic exists without its required compressed feed: ' + base)
            ensure_camera_free(device)
            started.append(role)
    processes = []
    try:
        for role in started:
            processes.append(subprocess.Popen(camera_command(role, cameras[role], width, height),
                                               start_new_session=True))
        settings = {'cameras': cameras, 'width': width, 'height': height}
        if args.camera or not config_path.is_file():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(settings, indent=2) + '\n')
        session = dict(settings, started=started, reused=reused)
        if args.session_file is not None:
            with args.session_file.open('x') as output:
                json.dump(session, output, indent=2)
        print('RGB cameras: started %s; reused %s; %dx%d at 30 FPS.' %
              (', '.join(started) or 'none', ', '.join(reused) or 'none', width, height), flush=True)
        while True:
            for role, process in zip(started, processes):
                if process.poll() is not None:
                    raise ValueError('Camera exited: ' + role + '; review its output above')
            time.sleep(.2)
    finally:
        stop_cameras(processes)


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Owned RGB camera feeds stopped; reused feeds remain running.')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        raise SystemExit('Camera feeds stopped: ' + str(error))
