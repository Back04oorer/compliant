"""Recording-compatible MIT replay. No ROS/CAN IO or hardware parameter writes."""
import hashlib
import json
import math
from pathlib import Path

from rgb_act_execution import ExecutionGuard, finite_vector

DRIVER_PARAMETERS = (
    'can_port', 'ctrl_mode', 'control_rate', 'publish_rate', 'mit', 'filter', 'follower_gravity',
    'gripper_exist', 'enable_gripper', 'enable_gripper_haptic', 'gripper_reverse',
    'gripper_val_mutiple', 'gripper_range', 'gripper_cmd_limit_enable',
    'gripper_cmd_min_raw', 'gripper_cmd_max_raw', 'gripper_cmd_deadband_raw',
)


def load_recording_rules(bag):
    path = Path(bag).parent / 'repro_snapshot.json'
    if not path.is_file():
        raise ValueError('Recording rules require the episode repro_snapshot.json: ' + str(path))
    raw = path.read_bytes()
    namespaces = json.loads(raw)['ros']['key_param_namespaces']
    drivers = {}
    for side in ('left', 'right'):
        item = namespaces['/piper_ctrl_%s_node' % side]
        if item.get('status') != 'ok':
            raise ValueError('Recording driver parameters are unavailable: ' + side)
        drivers[side] = {key: item['value'][key] for key in DRIVER_PARAMETERS}
    return validate_recording_rules({'schema_version': 1, 'snapshot': str(path),
        'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
        'rate_hz': drivers['left']['publish_rate'], 'drivers': drivers})


def validate_recording_rules(rules):
    if rules.get('schema_version') != 1 or set(rules.get('drivers', {})) != {'left', 'right'}:
        raise ValueError('Unknown recording rules contract')
    rate = rules.get('rate_hz')
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 1 <= rate <= 200:
        raise ValueError('Invalid recorded publish rate')
    for side, params in rules['drivers'].items():
        if set(params) != set(DRIVER_PARAMETERS):
            raise ValueError('Incomplete recorded driver parameters: ' + side)
        mit = params['mit']
        if params['ctrl_mode'] != 'mit' or mit.get('enable_pos') is not True or \
                mit.get('enable_vel') is not False or mit.get('enable_tor') is not False or \
                params['enable_gripper_haptic'] is not False:
            raise ValueError('Recording replay requires position-only MIT drivers: ' + side)
        if params['publish_rate'] != rate or not 1 <= params['control_rate'] <= 200:
            raise ValueError('Recorded driver rates are incompatible: ' + side)
        if params['gripper_reverse'] is not False or params['gripper_val_mutiple'] != 1 or \
                params['gripper_range'] != .08 or params['gripper_cmd_limit_enable'] is not True or \
                params['gripper_cmd_min_raw'] != 0 or params['gripper_cmd_max_raw'] != 80000:
            raise ValueError('Unsupported recorded gripper mapping: ' + side)
        for key, upper in (('kp', 500), ('kd', 5)):
            value = mit[key]
            values = value if isinstance(value, list) else [value] * 6
            if any(not 0 <= v <= upper for v in finite_vector(values, 6, side + '.' + key)):
                raise ValueError('Recorded MIT gain is outside the protocol range')
    return rules


def recorded_positions(values):
    values = finite_vector(values, 14, 'recorded MIT positions')
    # Match the deployed MIT driver, rather than imposing JointCtrl/URDF bounds on this path.
    if any(abs(value) > (0.08 + 1e-8 if i % 7 == 6 else 12.5) for i, value in enumerate(values)):
        raise ValueError('Recorded target exceeds the deployed MIT/gripper input range')
    return values


class RecordedReplayGuard(ExecutionGuard):
    """Use the recorded controller and a preloaded, elapsed-time trajectory without ACT shaping."""

    def __init__(self, config, reset_positions, plan):
        self.recording_rules = validate_recording_rules(plan['rules'])
        if config['rate_hz'] != self.recording_rules['rate_hz']:
            raise ValueError('Replay rate differs from the recording')
        # Base validation still owns watchdogs, reset speed, and the bounded session budget.
        super().__init__(dict(config, rate_hz=30), reset_positions, states_only=True)
        self.config = dict(config)
        self.replay = False
        self.reset_joint_tolerance = .05
        self.actions = [recorded_positions(row) for row in plan['actions']]
        self.duration = plan['execution_seconds']
        if len(self.actions) < 2 or self.reset_positions != self.actions[0] or \
                not math.isfinite(self.duration) or self.duration <= 0 or \
                len(self.actions) != math.ceil(self.duration * config['rate_hz']) + 1:
            raise ValueError('Invalid preloaded recording trajectory')
        self.playback_started = None
        self.cursor = 0
        self.first_frame = recorded_positions(plan.get('first_frame_positions', self.actions[0]))
        self.segment_feedback = recorded_positions(plan.get('reset_feedback_positions', self.actions[0]))
        self.first_frame_feedback = recorded_positions(plan.get('first_frame_feedback_positions', self.first_frame))
        self.reset_feedback_positions = self.segment_feedback[:]
        self.pause_started = None
        self.resume_phase = None
        self.hold_after_reset = False

    def position(self, values):
        return recorded_positions(values)

    def measured_position(self, values):
        return recorded_positions(values)

    def arm(self, measured, diagnostics, now):
        self.reset_positions = self.actions[0][:]
        self.reset_feedback_positions = self.segment_feedback[:]
        super().arm(measured, diagnostics, now)
        self.queue.extend(self.actions)
        self.playback_started, self.cursor = None, 0
        self.pause_started, self.resume_phase = None, None
        self.hold_after_reset = False

    def toggle_pause(self, now):
        if not self.active:
            raise ValueError('Type ARM + Enter before using replay controls')
        if self.phase == 'finished':
            return
        if self.phase == 'paused':
            self.phase = self.resume_phase
            if self.playback_started is not None:
                self.playback_started += now - self.pause_started
            self.reset_settled_since = None
            self.reason = 'Replay resumed'
        else:
            self.resume_phase, self.phase = self.phase, 'paused'
            self.pause_started = now
            self.velocity = [0.] * 14
            self.reason = 'Replay paused; holding current target'

    def rewind(self, measured, now):
        if not self.active:
            raise ValueError('Type ARM + Enter before using replay controls')
        self.reset_positions = self.first_frame[:]
        self.reset_feedback_positions = self.first_frame_feedback[:]
        self.last_command = self.measured_position(measured)
        self.velocity = [0.] * 14
        self.queue.clear()
        self.queue.extend(self.actions)
        self.playback_started, self.cursor = None, 0
        self.phase, self.reason = 'reset', 'Returning slowly to dataset first frame'
        self.started = self.last_tick = self.last_plan = now
        self.reset_settled_since = None
        self.hold_after_reset = True

    def submit(self, *_args):
        raise ValueError('Recording replay is preloaded; action chunks cannot replace it')

    def step(self, measured, now, stationary=True):
        if not self.active:
            return None
        if self.phase == 'reset':
            command = super().step(measured, now, stationary=stationary)
            if self.phase == 'policy' and self.hold_after_reset:
                self.hold_after_reset = False
                self.phase, self.resume_phase = 'paused', 'reset'
                self.pause_started = now
                self.reason = 'At dataset first frame; Space starts replay'
                # A selected segment can start later than frame zero; approach it slowly on resume.
                self.reset_positions = self.actions[0][:]
                self.reset_feedback_positions = self.segment_feedback[:]
            return command
        if self.phase in ('paused', 'finished'):
            # Holding does not consume the motion budget; heartbeat and tick watchdogs still apply.
            started = self.started
            self.started = now
            self.last_plan = now
            self.check_step(measured, now, check_tracking=False)
            self.started = started + now - self.last_tick
            self.last_tick = now
            self.velocity = [0.] * 14
            return self.last_command[:]
        # Recording used the driver's MIT response; ACT's tracking cutoff also rejects real bag feedback.
        dt = self.check_step(measured, now, check_tracking=False)
        if self.playback_started is None:
            self.playback_started = now
        elapsed = now - self.playback_started
        index = min(int(elapsed * self.config['rate_hz']), len(self.actions) - 1)
        if elapsed >= self.duration:
            index = len(self.actions) - 1
            self.phase, self.reason = 'finished', 'Recording replay finished'
        while self.cursor <= index:
            self.queue.popleft()
            self.cursor += 1
        command = self.actions[index][:]
        self.velocity = [(v - old) / dt for v, old in zip(command, self.last_command)]
        self.last_command, self.last_tick, self.last_plan = command, now, now
        return command
