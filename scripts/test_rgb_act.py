#!/usr/bin/env python3
"""Container-local entry for RGB ACT predictions, dry runs and locally armed trials."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from execute_rgb_act import request, run_client, validate_checkpoint_contract
from predict_rgb_act import NoRedirect, RGBActPredictor, read_ros_observation, checkpoint_dataset_root
from rgb_act_execution import ExecutionGuard, finite_vector
from live_act import LiveACTGuard

ROOT = Path(__file__).resolve().parents[1]
IN_CONTAINER = Path('/.dockerenv').exists()


def ros_command(script, *arguments):
    # Rebuild the ROS environment with system Python, isolated from model packages.
    command = ['/usr/bin/python3', str(ROOT / 'scripts' / script), *map(str, arguments)]
    shell = ('unset PYTHONPATH PYTHONHOME VIRTUAL_ENV; export PATH=/usr/bin:/bin:$PATH; '
             'source ' + shlex.quote(str(ROOT / 'workspaces/scripts/use_robot.sh')) +
             ' >/dev/null && exec ' + shlex.join(command))
    return ['bash', '-c', shell]


@contextmanager
def managed_cameras(output, enabled=True):
    if not enabled:
        yield
        return
    session_path = output / 'camera_session.json'
    process = None
    with (output / 'cameras.log').open('x') as log:
        try:
            print('Starting missing inference cameras; existing feeds are reused.', flush=True)
            command = ros_command('start_rgb_cameras.py', '--reuse-existing', '--session-file', session_path)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
            deadline = time.monotonic() + 10
            session = None
            while session is None:
                if process.poll() is not None:
                    raise ValueError('Automatic camera setup failed: ' + log.name +
                                     '; configure all three camera roles with scripts/start_rgb_cameras.py --camera if no settings are saved')
                if session_path.is_file():
                    try:
                        session = json.loads(session_path.read_text())
                    except json.JSONDecodeError:
                        pass
                if time.monotonic() >= deadline:
                    raise ValueError('Timed out starting inference cameras; see ' + log.name)
                if session is None:
                    time.sleep(.05)
            print('Cameras started: %s; reused: %s.' %
                  (', '.join(session['started']) or 'none', ', '.join(session['reused']) or 'none'), flush=True)
            yield
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError('Camera cleanup did not finish; see ' + log.name) from error


@contextmanager
def managed_observer(output, external_url=None, states_only=False):
    if external_url is not None:
        yield external_url
        return
    session_path = output / 'observer_session.json'
    process, session = None, None
    with (output / 'observer.log').open('x') as log:
        try:
            command = ros_command('rgb_act_ros_observer.py', '--port', '0', '--session-file', session_path)
            if states_only:
                command[-1] += ' --states-only'
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
            deadline, last_error = time.monotonic() + 10, 'waiting for the observer to start'
            while True:
                if process.poll() is not None:
                    raise ValueError('Read-only observer exited; see ' + str(output / 'observer.log'))
                if session is None and session_path.is_file():
                    try:
                        candidate = json.loads(session_path.read_text())
                    except json.JSONDecodeError:
                        candidate = None
                    if candidate is not None:
                        url = urllib.parse.urlparse(candidate.get('url', ''))
                        if session_path.stat().st_mode & 0o077 or candidate.get('schema_version') != 1 or \
                                candidate.get('prediction_only') is not True or url.scheme != 'http' or \
                                url.hostname != '127.0.0.1' or url.path != '/observation' or \
                                url.query or url.fragment or url.username or not url.port or \
                                type(candidate.get('pid')) is not int or candidate['pid'] <= 1 or \
                                not isinstance(candidate.get('shutdown_token'), str) or len(candidate['shutdown_token']) != 64:
                            raise ValueError('Invalid private read-only observer session')
                        session = candidate
                if session is not None:
                    try:
                        read_ros_observation(session['url'], timeout=.5, states_only=states_only)
                        break
                    except urllib.error.HTTPError as error:
                        if error.code != 503:
                            raise
                        last_error = json.loads(error.read(4096)).get('error', str(error))
                    except OSError as error:
                        last_error = str(error)
                if time.monotonic() >= deadline:
                    raise ValueError('Live inputs unavailable: ' + last_error + '; see ' + str(output / 'observer.log'))
                time.sleep(.05)
            print('Live observations connected. Read-only observer started for this test.', flush=True)
            yield session['url']
        finally:
            if session is not None:
                try:
                    request_quit = urllib.request.Request(session['url'].removesuffix('/observation') + '/quit', data=b'',
                        headers={'Authorization': 'Bearer ' + session['shutdown_token']})
                    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({})).open(request_quit, timeout=1):
                        pass
                except (OSError, ValueError):
                    # This child only subscribes; terminate the owned process, never a PID from a file.
                    if process is not None and process.poll() is None:
                        process.terminate()
            if process is not None and process.poll() is None:
                if session is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def stop_bridge(process, session, immediate=False):
    if session is not None:
        try:
            request(session, '/quit', {'immediate': True} if immediate else {})
        except KeyboardInterrupt:
            # A repeated Ctrl-C also reaches the bridge; wait for its immediate stop.
            pass
        except (OSError, ValueError, KeyError):
            print('Could not reach the bridge for shutdown. Check its terminal; motor power may remain on.', flush=True)
    if process is not None and process.poll() is None:
        if session is None:
            process.send_signal(signal.SIGINT)
        timeout = 65 if not immediate and session and session.get('live_act') and session.get('reset_positions') is not None else 5
        while True:
            try:
                process.wait(timeout=timeout)
                break
            except KeyboardInterrupt:
                # The bridge handles Ctrl-C; keep its feedback/cameras alive until it exits.
                continue
            except subprocess.TimeoutExpired:
                # Do not kill a possibly active bridge and silently remove its watchdog.
                raise RuntimeError('Bridge did not exit. Use its terminal/hardware stop; the process was left running.')


def run_trial(args, predictor, config, client=None):
    output = args.output_dir
    session_path = output / 'bridge/session.json'
    bridge, session = None, None
    config_path = output / 'execution_config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    failed = False
    try:
        command = ros_command('rgb_act_ros_executor.py', '--config', config_path,
                              '--session-dir', output / 'bridge')
        states_only = getattr(args, 'states_only', False)
        if states_only:
            command[-1] += ' --states-only'
        if getattr(args, 'live_act', False):
            command[-1] += ' --live-act'
        if getattr(args, 'recording_plan', None) is not None:
            command[-1] += ' --recording-plan ' + shlex.quote(str(args.recording_plan))
        reset_pose = getattr(args, 'reset_pose', None)
        if reset_pose is not None:
            command[-1] += ' --reset-pose ' + shlex.quote(str(output / 'reset_pose.json'))
        if args.mode == 'robot':
            command[-1] += ' --execute'
        bridge = subprocess.Popen(command)
        deadline = time.monotonic() + 10
        while True:
            if bridge.poll() is not None:
                raise ValueError('ROS test bridge exited during startup; review its output above')
            if session_path.is_file():
                try:
                    candidate = json.loads(session_path.read_text())
                except json.JSONDecodeError:
                    candidate = None  # The bridge is still writing its private session file.
                if candidate is not None:
                    if session_path.stat().st_mode & 0o077 or candidate.get('schema_version') != 1 or \
                            candidate.get('execute') is not (args.mode == 'robot') or candidate.get('routing') != 'mux' or candidate.get('config') != config or \
                            candidate.get('reset_positions') != (None if reset_pose is None else reset_pose['positions']) or \
                            candidate.get('states_only', False) is not states_only or \
                            candidate.get('live_act', False) is not getattr(args, 'live_act', False) or \
                            candidate.get('recording_rules') != getattr(args, 'recording_rules', None):
                        raise ValueError('Unexpected ROS bridge session permissions, mode, or limits')
                    session = candidate
                    break
            if time.monotonic() >= deadline:
                raise ValueError('Timed out waiting for the ROS test bridge')
            time.sleep(.05)
        # The model is loaded before this bridge starts, so ARM never waits for model loading.
        return (client or run_client)(session, predictor, args.observer_url, config)
    except Exception:
        failed = True
        raise
    finally:
        stop_bridge(bridge, session, immediate=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=('predictions', 'dry-run', 'robot'), default='dry-run')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--observer-url', help='Reuse an existing observer; default starts a private read-only observer')
    parser.add_argument('--max-predictions', type=int, default=10)
    parser.add_argument('--reset-to-first-frame', action='store_true',
                        help='After local ARM, move slowly to episode 0 frame 0 measured pose before live ACT')
    args = parser.parse_args()
    if not IN_CONTAINER:
        parser.error('Run inside piper-control with /opt/piper-inference/bin/python; host inference was removed')
    if args.mode == 'predictions' and args.reset_to_first_frame:
        parser.error('First-frame positioning requires robot or dry-run mode; predictions never commands movement')
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() or args.max_predictions <= 0:
        parser.error('Choose a new output directory and a positive prediction count')
    if args.observer_url is None or args.mode != 'predictions':
        try:
            args.output_dir.relative_to(ROOT / 'outputs')
        except ValueError:
            parser.error('Managed observer/trial output must be inside this project\'s outputs directory')
    if args.mode != 'predictions' and not sys.stdin.isatty():
        parser.error('A local interactive terminal is required for ARM and Enter-to-stop')
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    with managed_cameras(args.output_dir, enabled=args.observer_url is None), \
            managed_observer(args.output_dir, args.observer_url) as url:
        args.observer_url = url
        if args.mode == 'predictions':
            return subprocess.call([sys.executable, str(ROOT / 'scripts/predict_rgb_act.py'),
                '--checkpoint', str(args.checkpoint), '--observer-url', url,
                '--output-dir', str(args.output_dir / 'predictions'), '--device', args.device,
                '--max-predictions', str(args.max_predictions), '--rate', '1'])
        return preflight_trial(args)


def preflight_trial(args):
    config = json.loads((ROOT / 'config/rgb_act_execution.json').read_text())
    config['trial_seconds'] = 5.0
    args.live_act = True
    guard = LiveACTGuard(config)
    print('LIVE ACT: live RGB/state -> policy mux input -> shared MIT follower drivers. J2/J3 use ±12.5 rad protocol bounds. '
          'Other position, speed, tracking and watchdog checks remain active. Keep Teleop menu 1 drivers running; local ARM selects policy, stop/exit restores teleop.', flush=True)
    print('Finite gripper model targets are clipped to configured bounds (0–0.08 m); NaN/Inf still stop ACT.', flush=True)
    validate_checkpoint_contract(args.checkpoint, config)
    if getattr(args, 'reset_to_first_frame', False):
        args.reset_pose = dataset_first_frame(args.checkpoint)
        (args.output_dir / 'reset_pose.json').write_text(json.dumps(args.reset_pose, indent=2) + '\n')
        # A measured dataset pose is a position target, not an ACT action or an effort command.
        try:
            guard.position(args.reset_pose['positions'])
        except ValueError as error:
            raise ValueError('Dataset episode 0 frame 0 cannot be used as a reset pose: ' + str(error)) from error
        print('Start pose: dataset episode 0, frame 0 measured joint/gripper positions. '
              'Local ARM will move there before starting live ACT.', flush=True)
    import torch
    torch.set_num_threads(1)
    print('Live ACT uses fresh observations, one action per inference and temporal ensemble (0.01). '
          'q/Ctrl-C returns to first frame before exit; Space pauses; Enter returns to first frame and holds.', flush=True)
    print('Loading trained model: ' + str(args.checkpoint), flush=True)
    predictor = RGBActPredictor(args.checkpoint, args.device, True)
    if predictor.contract['action_shape'] != [50, 14]:
        raise ValueError('Expected a bimanual ACT checkpoint with 50x14 actions')
    observation, metadata = read_ros_observation(args.observer_url)
    if getattr(args, 'reset_pose', None) is not None:
        measured = observation['observation.state'][:7].tolist() + observation['observation.state'][14:21].tolist()
        guard.position(measured)
        (args.output_dir / 'contract.json').write_text(json.dumps(predictor.contract, indent=2) + '\n')
        (args.output_dir / 'preflight.json').write_text(json.dumps({
            'mode': args.mode, 'live_act': True, 'continuous': True, 'j2_j3_bounds_rad': [-12.5, 12.5],
            'measured_positions': measured, 'reset_pose': args.reset_pose,
            'action_preflight_passed': None, 'action_preflight': 'Deferred until fresh observations after positioning',
        }, indent=2) + '\n')
        print('Reset pose and current pose passed position limits. ACT targets will be checked after positioning.', flush=True)
        return run_trial(args, predictor, config)
    actions = predictor.select_action(observation)
    predictor.reset()
    measured = observation['observation.state'][:7].tolist() + observation['observation.state'][14:21].tolist()
    (args.output_dir / 'contract.json').write_text(json.dumps(predictor.contract, indent=2) + '\n')
    import numpy as np
    np.save(args.output_dir / 'preflight_prediction.npy', actions, allow_pickle=False)
    report = {'mode': args.mode, 'live_act': True, 'continuous': True, 'j2_j3_bounds_rad': [-12.5, 12.5], 'trial_seconds': None, 'source_stamps': metadata['source_stamps'],
              'measured_positions': measured, 'action_preflight_passed': False, 'error': None}
    try:
        guard.validate_actions([actions.tolist()], measured)
        report['action_preflight_passed'] = True
    except ValueError as error:
        report['error'] = str(error)
    (args.output_dir / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Test output: ' + str(args.output_dir), flush=True)
    if not report['action_preflight_passed']:
        raise ValueError('Model test blocked before starting the execution bridge: ' + report['error'])
    print('Action preflight passed. ' + ('ROBOT TRIAL' if args.mode == 'robot' else 'DRY RUN') +
          ': wait for the bridge, then type ARM + Enter. Space pauses; Enter returns to first frame; q/Ctrl-C returns there before exit.', flush=True)
    return run_trial(args, predictor, config)


def dataset_first_frame(checkpoint):
    """Read only numeric columns; never decode videos or replay recorded commands."""
    import pyarrow.parquet as pq

    dataset = checkpoint_dataset_root(checkpoint)
    if dataset is None:
        raise ValueError('First-frame positioning requires the checkpoint dataset')
    matches = []
    for path in sorted((dataset / 'data').glob('chunk-*/*.parquet')):
        table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state'],
                              filters=[('episode_index', '=', 0), ('frame_index', '=', 0)])
        matches.extend(table.to_pylist())
    if len(matches) != 1:
        raise ValueError('Dataset must contain exactly one episode 0 frame 0')
    state = finite_vector(matches[0]['observation.state'], 28, 'dataset first-frame state')
    return {'dataset': str(dataset), 'episode_index': 0, 'frame_index': 0,
            'source': 'observation.state', 'positions': state[:7] + state[14:21]}


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Model test stopped. Motor power is unchanged.')
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        raise SystemExit('Model test stopped: ' + str(error))
