#!/usr/bin/env python3
"""Replay selected rosbag joint targets through the shared follower command mux, with local ARM."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from execute_rgb_act import request
from predict_rgb_act import read_ros_observation
from rgb_act_execution import ExecutionGuard, MAX_REPLAY_SECONDS
from rgb_act_ros_observer import ACTION_NAMES, JOINT_NAMES
from recorded_replay import RecordedReplayGuard, load_recording_rules, recorded_positions, validate_recording_rules
from test_rgb_act import IN_CONTAINER, managed_observer, ros_command, run_trial

ROOT = Path(__file__).resolve().parents[1]


def retime(actions, config):
    """Stretch fast path segments; retain the path and check the final 30 Hz targets."""
    import numpy as np

    guard = ExecutionGuard(config, states_only=True)
    rate = config['rate_hz']
    vmax = np.asarray(guard.limits(config['joint_velocity_rad_s'], config['gripper_velocity_m_s'])) * .95
    amax = np.asarray(guard.limits(config['joint_acceleration_rad_s2'], config['gripper_acceleration_m_s2'])) * .95
    delta = np.diff(actions, axis=0)
    periods = np.maximum(1 / rate, (np.abs(delta) / vmax).max(axis=1))
    for _ in range(40):
        velocity = delta / periods[:, None]
        acceleration = np.abs(np.diff(np.vstack([np.zeros(14), velocity, np.zeros(14)]), axis=0))
        intervals = np.r_[periods[0] / 2, (periods[:-1] + periods[1:]) / 2, periods[-1] / 2]
        ratio = (acceleration / intervals[:, None] / amax).max(axis=1)
        scale = np.sqrt(np.maximum(1, np.maximum(ratio[:-1], ratio[1:])))
        if scale.max() <= 1.000001:
            break
        periods *= scale
    clock = np.r_[0., np.cumsum(periods)]
    for _ in range(12):
        if clock[-1] + 8 > MAX_REPLAY_SECONDS:
            raise ValueError('Retimed replay exceeds %ds; select a shorter segment' % MAX_REPLAY_SECONDS)
        ticks = np.minimum(np.arange(math.ceil(clock[-1] * rate) + 1) / rate, clock[-1])
        result = np.column_stack([np.interp(ticks, clock, actions[:, i]) for i in range(14)])
        velocity = np.diff(result, axis=0) * rate
        acceleration = np.diff(np.vstack([np.zeros(14), velocity, np.zeros(14)]), axis=0) * rate
        scale = max(1., float((np.abs(velocity) / vmax).max()),
                    math.sqrt(float((np.abs(acceleration) / amax).max())))
        if scale <= 1.000001:
            # Give the shaped controller time to reach the final target before stopping.
            return np.vstack([result, np.repeat(result[-1:], 2 * rate, axis=0)])
        clock *= scale * 1.02
    raise ValueError('Could not retime trajectory within velocity/acceleration limits')


def trajectory_streams(recording):
    import numpy as np

    if recording.get('schema_version') != 1 or recording.get('joint_names') != JOINT_NAMES:
        raise ValueError('Unknown recorded trajectory contract')
    streams = {}
    for side in ('left', 'right'):
        stream = recording['samples'][side]
        stamps = np.asarray(stream['timestamps'], dtype=float)
        positions = np.asarray(stream['positions'], dtype=float)
        if (stamps.ndim != 1 or len(stamps) < 2 or positions.shape != (len(stamps), 7)
                or not np.isfinite(stamps).all() or not np.isfinite(positions).all()
                or not (np.diff(stamps) > 0).all()):
            raise ValueError('Invalid/non-increasing recorded samples: ' + side)
        streams[side] = (stamps, positions)
    return streams


def feedback_pose(streams, stamp):
    import numpy as np

    pose = []
    for side, (stamps, positions) in streams.items():
        if not stamps[0] <= stamp <= stamps[-1]:
            raise ValueError('Recorded feedback does not cover the reset frame: ' + side)
        index = min(max(1, int(np.searchsorted(stamps, stamp))), len(stamps) - 1)
        if stamps[index] - stamps[index - 1] > .1:
            raise ValueError('Recorded feedback gap at the reset frame exceeds 0.1s: ' + side)
        pose.extend(float(np.interp(stamp, stamps, positions[:, i])) for i in range(7))
    return recorded_positions(pose)


def build_plan(recording, config, start, duration, speed, constrain=False, rules=None, feedback=None):
    import numpy as np

    if rules is not None:
        validate_recording_rules(rules)
        if constrain or config['rate_hz'] != rules['rate_hz']:
            raise ValueError('Recording rules require their original rate and unmodified targets')
    guard = ExecutionGuard(dict(config, rate_hz=30) if rules else config, states_only=True)
    maximum = MAX_REPLAY_SECONDS - 5 if constrain or rules else 55
    if (not all(math.isfinite(v) for v in (start, speed)) or start < 0 or not 0 < speed <= 1
            or (duration is not None and (not math.isfinite(duration) or duration <= 0
                                         or duration / speed > maximum))):
        raise ValueError('Use start >= 0, duration > 0, speed in (0, 1], and duration/speed <= %d seconds' % maximum)
    streams = trajectory_streams(recording)
    origin = max(stamps[0] for stamps, _ in streams.values())
    available = float(min(stamps[-1] for stamps, _ in streams.values()) - origin)
    if duration is None:
        duration = available - start
    if duration <= 0 or duration / speed > maximum:
        raise ValueError('Replay duration must be positive and duration/speed <= %d seconds' % maximum)
    if start + duration > available:
        raise ValueError('Requested segment exceeds the shared left/right duration %.3fs' % available)
    begin, end = origin + start, origin + start + duration
    ticks = np.arange(math.ceil(duration / speed * config['rate_hz']) + 1)
    query = np.minimum(begin + ticks * speed / config['rate_hz'], end)
    targets, first_frame, violations, errors = [], [], [], []
    for side, (stamps, positions) in streams.items():
        # Include interpolation neighbours, so resampling cannot hide a raw violation or dropout.
        first = max(0, int(np.searchsorted(stamps, begin, side='right')) - 1)
        last = min(len(stamps) - 1, int(np.searchsorted(stamps, end, side='left')))
        raw = positions[first:last + 1]
        if np.diff(stamps[first:last + 1]).max() > 0.1:
            errors.append(side + ': recorded feedback/command gap exceeds 0.1s')
        if rules:
            for row in raw:
                recorded_positions(row.tolist() * 2)
        elif (np.abs(np.diff(raw, axis=0)) > np.asarray(guard.limits(
                config['raw_joint_step_rad'], config['raw_gripper_step_m'])[:7])).any():
            errors.append(side + ': raw recording contains an excessive joint/gripper jump')
        for i, name in enumerate(JOINT_NAMES):
            low, high = config['arm_position_min'][i], config['arm_position_max'][i]
            bad = np.flatnonzero((raw[:, i] < low) | (raw[:, i] > high))
            if len(bad):
                index = first + int(bad[0])
                violations.append({'joint': side + '.' + name, 'count': len(bad),
                    'min': float(raw[:, i].min()), 'max': float(raw[:, i].max()),
                    'limit': [low, high], 'first_recorded_time_s': float(stamps[index] - origin),
                    'first_value': float(positions[index, i])})
        if constrain:
            positions = np.clip(positions, config['arm_position_min'], config['arm_position_max'])
        first_frame.extend(float(np.interp(origin, stamps, positions[:, i])) for i in range(7))
        targets.append(np.column_stack([np.interp(query, stamps, positions[:, i]) for i in range(7)]))
    if rules:
        recorded_positions(first_frame)
    actions = np.concatenate(targets, axis=1)
    if violations and not constrain and rules is None:
        errors.append('Recorded positions exceed configured execution limits')
    if constrain and not errors:
        actions = retime(actions, config)
    steps = np.abs(np.diff(actions, axis=0))
    jump = np.asarray(guard.limits(config['raw_joint_step_rad'], config['raw_gripper_step_m']))
    if rules is None and (steps > jump).any():
        errors.append('Recorded trajectory contains an excessive joint/gripper jump')
    peak_speed = (steps * config['rate_hz']).max(axis=0)
    velocity = np.diff(actions, axis=0) * config['rate_hz']
    peak_acceleration = np.abs(np.diff(np.vstack([np.zeros(14), velocity, np.zeros(14)]), axis=0)
                               * config['rate_hz']).max(axis=0)
    vmax = np.asarray(guard.limits(config['joint_velocity_rad_s'], config['gripper_velocity_m_s']))
    if rules is None and (peak_speed > vmax + 1e-8).any():
        errors.append('Trajectory is too fast for current limits; reduce replay speed or select a shorter segment')
    feedback_fields = {}
    if feedback is not None:
        if feedback['source'] != 'state' or feedback['bag'] != recording['bag']:
            raise ValueError('Reset feedback must come from the same bag measured state streams')
        measured = trajectory_streams(feedback)
        feedback_fields = {'reset_feedback_positions': feedback_pose(measured, begin),
                           'first_frame_feedback_positions': feedback_pose(measured, origin)}
    return {'bag': recording['bag'], 'source': recording['source'], 'topics': recording['topics'],
            'start_s': start, 'duration_s': duration, 'speed': speed, 'available_s': float(available),
            'rules': 'recording' if rules else 'act',
            'constrained': constrain, 'execution_seconds': duration / speed if rules else len(actions) / config['rate_hz'],
            'rate_hz': config['rate_hz'], 'action_names': ACTION_NAMES,
            'sample_count': len(actions), 'peak_target_speed': peak_speed.tolist(),
            'peak_target_acceleration': peak_acceleration.tolist(),
            'position_violations': violations, 'errors': errors, 'passed': not errors,
            'first_frame_positions': first_frame, 'actions': actions.tolist(), **feedback_fields}


def run_replay_client(session, actions, observer_url, config):
    started, cursor, previous = False, 0, None
    last_progress = 0.
    try:
        while True:
            status = request(session, '/status')
            if status.get('local_exit_requested'):
                break
            if not status['armed']:
                if started:
                    raise ValueError('Replay stopped before completion: ' + status['reason'])
                readiness = (status['ready'], status['error'])
                if readiness != previous:
                    print('Replay ready. Type ARM + Enter to position and replay.' if status['ready'] else
                          'Replay blocked: ' + str(status['error']), flush=True)
                    previous = readiness
                time.sleep(.1)
                continue
            started = True
            if not status['ready']:
                raise ValueError('Replay lost readiness: ' + str(status['error']))
            if session.get('recording_rules') is not None:
                phase = status.get('phase')
                progress = status.get('reset_progress')
                if phase == 'reset' and progress and time.monotonic() - last_progress >= 1.:
                    last_progress = time.monotonic()
                    print('Positioning %.1fs/60s: max joint error %.3f rad; grippers %.1f / %.1f mm '
                          '(recorded feedback %.1f / %.1f mm).' %
                          (progress['elapsed_s'], progress['joint_error_rad'],
                           *(v * 1000 for v in progress['gripper_measured_m']),
                           *(v * 1000 for v in progress['gripper_expected_m'])), flush=True)
                if phase != previous:
                    if phase == 'finished':
                        print('Replay finished; holding last frame. Enter returns to first frame; q exits.', flush=True)
                    elif phase == 'paused':
                        print('Replay paused. Space resumes; Enter returns to first frame.', flush=True)
                    elif phase == 'policy':
                        print('First frame reached; replay is running. Space pauses; Enter returns to first frame.', flush=True)
                    previous = phase
                time.sleep(.01)
                continue
            if status.get('phase') == 'reset' or status['remaining']:
                time.sleep(.03)
                continue
            if cursor >= len(actions):
                print('Rosbag segment replay finished.', flush=True)
                break
            _, metadata = read_ros_observation(observer_url, timeout=.5,
                                               max_age=config['observation_timeout_s'], states_only=True)
            if any(stamp <= status.get('policy_after', 0) for stamp in metadata['source_stamps'].values()):
                time.sleep(.03)
                continue
            chunk = actions[cursor:cursor + config['chunk_size']]
            chunk = chunk + [chunk[-1]] * (config['chunk_size'] - len(chunk))
            request(session, '/chunk', {'sequence': status['sequence'] + 1,
                'rate_hz': config['rate_hz'], 'action_names': ACTION_NAMES,
                'source_stamps': metadata['source_stamps'], 'observation_stamp': metadata['observation_stamp'],
                'actions': chunk})
            cursor = min(len(actions), cursor + config['chunk_size'])
            print('Replay targets queued: %d/%d' % (cursor, len(actions)), flush=True)
    finally:
        try:
            request(session, '/stop', {})
        except (OSError, ValueError) as error:
            print('Could not request replay stop; check bridge/hardware stop: ' + str(error), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--source', choices=('mux', 'state'), default='mux')
    parser.add_argument('--mode', choices=('check', 'dry-run', 'robot'), default='check')
    parser.add_argument('--start', type=float, default=0., help='Seconds from shared left/right stream start')
    parser.add_argument('--duration', type=float, help='Seconds of recorded data; omitted plays the entire remaining episode')
    parser.add_argument('--speed', type=float, default=1., help='Recorded-time playback multiplier')
    parser.add_argument('--rules', choices=('recording', 'act'), default='recording',
                        help='Default matches the episode recording snapshot; act uses the separate ACT trial limits')
    parser.add_argument('--constrain', action='store_true',
                        help='Report and clip out-of-limit targets, then retime to speed/acceleration limits')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.constrain and args.rules != 'act':
        parser.error('--constrain changes the trajectory; use --rules act explicitly for that workflow')
    if not IN_CONTAINER:
        parser.error('Run inside piper-control with /opt/piper-inference/bin/python')
    args.bag = args.bag.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.bag.is_file() or args.bag.suffix != '.bag':
        parser.error('Select one finalized .bag file')
    if args.output_dir.exists() or ROOT / 'outputs' not in args.output_dir.parents:
        parser.error('Choose a new output directory under project outputs/')
    config = json.loads((ROOT / 'config/rgb_act_execution.json').read_text())
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / 'request.json').write_text(json.dumps({
        'bag': str(args.bag), 'source': args.source, 'mode': args.mode, 'start': args.start,
        'duration': args.duration, 'speed': args.speed, 'constrain': args.constrain,
        'rules': args.rules}, indent=2) + '\n')
    extracted = args.output_dir / 'recorded_trajectory.json'
    subprocess.run(ros_command('rosbag_trajectory.py', '--bag', args.bag, '--source', args.source,
                               '--output', extracted), check=True)
    try:
        rules = load_recording_rules(args.bag) if args.rules == 'recording' else None
        if rules:
            config['rate_hz'] = rules['rate_hz']
        feedback = None
        if rules and args.source == 'mux':
            measured_path = args.output_dir / 'recorded_feedback.json'
            subprocess.run(ros_command('rosbag_trajectory.py', '--bag', args.bag, '--source', 'state',
                                       '--output', measured_path), check=True)
            feedback = json.loads(measured_path.read_text())
        plan = build_plan(json.loads(extracted.read_text()), config, args.start, args.duration, args.speed,
                          constrain=args.constrain, rules=rules, feedback=feedback)
    except ValueError as error:
        (args.output_dir / 'preflight.json').write_text(json.dumps({'passed': False, 'errors': [str(error)]}) + '\n')
        raise
    actions = plan.pop('actions')
    (args.output_dir / 'preflight.json').write_text(json.dumps(plan, indent=2) + '\n')
    (args.output_dir / 'replay_targets.json').write_text(json.dumps(actions) + '\n')
    print('Replay preflight: ' + str(args.output_dir / 'preflight.json'), flush=True)
    if rules:
        print('RECORDING RULES: original MIT target range and %.0f Hz command rate; no ACT clipping, '
              'retiming, or trajectory smoothing. Requested segment executes in %.2fs.' %
              (config['rate_hz'], plan['execution_seconds']), flush=True)
    for violation in plan['position_violations']:
        print('%s: recorded [%.6f, %.6f], %s %s; %d samples' %
              (violation['joint'], violation['min'], violation['max'],
               'ACT reference only' if rules else 'allowed', violation['limit'], violation['count']))
    if args.constrain:
        print('CONSTRAINED REPLAY: targets clipped to the listed limits; motion retimed to %.2fs. '
              'This changes the recorded trajectory. Raw bag is unchanged.' % plan['execution_seconds'], flush=True)
    if not plan['passed']:
        raise ValueError('Replay blocked before starting any execution bridge: ' + '; '.join(plan['errors']))
    if rules:
        args.recording_rules = rules
        args.recording_plan = args.output_dir / 'recording_plan.json'
        args.recording_plan.write_text(json.dumps({'rules': rules, 'actions': actions,
            'execution_seconds': plan['execution_seconds'],
            'first_frame_positions': plan['first_frame_positions'],
            'reset_feedback_positions': plan.get('reset_feedback_positions', actions[0]),
            'first_frame_feedback_positions': plan.get('first_frame_feedback_positions', plan['first_frame_positions'])}) + '\n')
    if args.mode == 'check':
        print('Trajectory check passed; no robot processes started.', flush=True)
        return 0
    if not sys.stdin.isatty():
        raise ValueError('Robot/dry-run replay requires a local terminal for ARM and playback controls')
    config['trial_seconds'] = (plan['execution_seconds'] if rules else
        math.ceil(len(actions) / config['chunk_size']) * config['chunk_size'] / config['rate_hz']) + 3
    args.reset_pose = {'source': str(args.bag), 'trajectory_source': args.source,
                       'start_s': args.start, 'positions': actions[0]}
    (args.output_dir / 'reset_pose.json').write_text(json.dumps(args.reset_pose, indent=2) + '\n')
    if rules:
        RecordedReplayGuard(config, actions[0], json.loads(args.recording_plan.read_text()))
    else:
        ExecutionGuard(config, states_only=True)
    print('Local ARM will first move slowly to the selected segment start, then replay %.2fs of data '
          '(requested %.2fx; execution %.2fs).' %
          (plan['duration_s'], args.speed, plan['execution_seconds']), flush=True)
    print(('Space pauses/resumes; Enter returns slowly to episode frame zero and holds; Space restarts. q/Ctrl-C exits. '
           if rules else 'Enter stops; Ctrl-C exits. ') +
          'Mux replay: follower feedback, driver and gravity checks apply; keep Teleop menu 1 drivers running. ARM selects policy; stop/exit restores teleop. No cameras needed.', flush=True)
    args.states_only = True
    with managed_observer(args.output_dir, states_only=True) as url:
        args.observer_url = url
        observation, _ = read_ros_observation(url, states_only=True)
        measured = observation['observation.state'][:7].tolist() + observation['observation.state'][14:21].tolist()
        if rules:
            recorded_positions(measured)
        else:
            ExecutionGuard(config, states_only=True).measured_position(measured)
        return run_trial(args, actions, config, client=run_replay_client)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Rosbag replay stopped. Motor power is unchanged.')
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        raise SystemExit('Rosbag replay stopped: ' + str(error))
