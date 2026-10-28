"""Shared, ROS-free guards for bounded bimanual RGB ACT trials."""
from collections import deque
import math

from rgb_act_ros_observer import ACTION_NAMES, IMAGE_TOPICS, STATE_TOPICS

COMMAND_TOPICS = {side: "/robot/arm_%s/vla_joint_cmd" % side for side in STATE_TOPICS}
MUX_TOPICS = {side: "/robot/arm_%s/joint_cmd_mux" % side for side in STATE_TOPICS}
MAX_REPLAY_SECONDS = 1800


def finite_vector(values, size, label):
    if not isinstance(values, (list, tuple)) or len(values) != size or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Expected %d finite numbers for %s" % (size, label))
    return list(values)


def validate_config(config, replay=False):
    keys = {"rate_hz", "chunk_size", "arm_position_min", "arm_position_max", "joint_velocity_rad_s",
            "joint_acceleration_rad_s2", "gripper_velocity_m_s", "gripper_acceleration_m_s2",
            "initial_joint_delta_rad", "initial_gripper_delta_m", "tracking_joint_delta_rad",
            "tracking_gripper_delta_m", "raw_joint_step_rad", "raw_gripper_step_m", "feedback_timeout_s",
            "observation_timeout_s", "plan_timeout_s", "client_timeout_s", "max_tick_delay_s", "trial_seconds"}
    if set(config) != keys:
        raise ValueError("Execution config has missing or unknown fields")
    lo = finite_vector(config["arm_position_min"], 7, "lower limits")
    hi = finite_vector(config["arm_position_max"], 7, "upper limits")
    for key in keys - {"arm_position_min", "arm_position_max"}:
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Config must be finite and positive: " + key)
    if config["rate_hz"] != 30 or type(config["chunk_size"]) is not int or config["chunk_size"] != 50:
        raise ValueError("This executor requires the recorded 30 Hz / 50-action contract")
    # Intersection of repository URDF and both followers' configured limits, queried 2026-10-21.
    hard_lo = [-math.radians(150), 0, -math.radians(170), -1.57, -math.radians(70), -3.14, 0]
    hard_hi = [math.radians(150), 3.14, 0, 1.57, math.radians(70), 3.14, 0.08]
    if any(not hard_lo[i] <= lo[i] < hi[i] <= hard_hi[i] for i in range(7)):
        raise ValueError("Position limits must stay within the Piper joint/gripper bounds")
    maximum = MAX_REPLAY_SECONDS if replay else 60
    if config["trial_seconds"] > maximum:
        raise ValueError("Supervised trials are limited to at most %d seconds" % maximum)
    if config["joint_velocity_rad_s"] > 0.2:
        raise ValueError("Initial trial speed is conservatively capped at 0.2 rad/s")
    return dict(config)


class ExecutionGuard:
    """No IO: every output must pass fresh feedback, routing and motion guards."""

    def __init__(self, config, reset_positions=None, states_only=False, mit_j2_j3=False):
        self.config = validate_config(config, replay=states_only)
        self.continuous = False
        self.mit_j2_j3 = mit_j2_j3
        self.replay = states_only
        self.driver_topics = MUX_TOPICS
        self.source_topics = set(STATE_TOPICS) if states_only else {*IMAGE_TOPICS, *STATE_TOPICS}
        if mit_j2_j3:
            self.reset_joint_tolerance = .05
        self.reset_positions = None if reset_positions is None else self.position(reset_positions)
        self.phase, self.policy_after = "disarmed", 0.0
        self.reset_settled_since = None
        self.active, self.reason = False, "DISARMED: type ARM locally for a bounded trial"
        self.queue = deque()
        self.last_command = self.velocity = None
        self.sequence = 0
        self.last_observation = 0.0

    def limits(self, joint, gripper):
        return [joint] * 6 + [gripper] + [joint] * 6 + [gripper]

    def position(self, values):
        values = finite_vector(values, 14, "bimanual positions")
        lower = self.config["arm_position_min"][:]
        upper = self.config["arm_position_max"][:]
        if self.mit_j2_j3:
            # The recorded MIT path encodes J2/J3 in radians within the protocol range.
            for index in (1, 2):
                lower[index], upper[index] = -12.5, 12.5
        # Float32 ACT postprocessing can put an endpoint one ULP past 0.08 m.
        for index in (6, 13):
            low, high = lower[6], upper[6]
            if low - 1e-8 <= values[index] <= high + 1e-8:
                values[index] = max(low, min(high, values[index]))
        invalid = ["%s=%.12g outside [%.12g, %.12g] (excess %.12g)" %
                   (name, value, low, high, max(low - value, value - high))
                   for name, value, low, high in zip(ACTION_NAMES, values,
                       lower * 2, upper * 2)
                   if not low <= value <= high]
        if invalid:
            raise ValueError("Position exceeds configured joint/gripper limits: " + "; ".join(invalid))
        return values

    def feedback(self, observation, diagnostics, selected, gravity, ros_now, now, require_control=True, require_routing=True):
        cfg = self.config
        self.measured_position(observation["observation.state"][:7] + observation["observation.state"][14:21])
        measured = []
        for side in STATE_TOPICS:
            if side not in diagnostics:
                raise ValueError('Missing follower driver diagnostics: ' + side)
            if require_control and require_routing and selected.get(side) != COMMAND_TOPICS[side]:
                raise ValueError("Mux is not routed to policy: " + side)
            item, received = diagnostics[side]
            age = now - received
            wall_age = ros_now - item["published_at"]
            if not 0 <= age <= cfg["feedback_timeout_s"] or not 0 <= wall_age <= cfg["feedback_timeout_s"]:
                raise ValueError("Stale driver diagnostics: " + side)
            if item.get("schema_version") != 1 or item.get("position_only") is not True or \
                    item.get("motor_faults") != [0] * 6 or item.get("err_code") != 0 or item.get("arm_status") != 0:
                raise ValueError("Driver is faulted or incompatible: " + side)
            # SDK V2: 0 = teaching off, 2 = recording stopped; other teaching states can command motion.
            if require_control and (item.get("driver_enabled") is not True or item.get("blocked") is not False or \
                    item.get("motor_enabled") != [True] * 6 or item.get("gripper_enabled") is not True or \
                    item.get("teach_status") not in (0, 2) or item.get("ctrl_mode") != 1 or \
                    item.get("expected_mode_feedback") not in (1, 4) or \
                    item.get("mode_feedback") != item["expected_mode_feedback"] or \
                    item.get("command_topic") != self.driver_topics[side] or item.get("gripper_topic") != self.driver_topics[side]):
                raise ValueError("Driver is disabled, faulted, teaching, or incompatible: " + side)
            for key, size in (("joint_feedback_age_s", 6), ("speed_feedback_age_s", 6),
                              ("enable_feedback_age_s", 6), ("gripper_feedback_age_s", 1), ("status_age_s", 1)):
                raw = item[key] if size == 6 else [item[key]]
                ages = finite_vector(raw, size, side + "." + key)
                if any(v < 0 or v + age > cfg["feedback_timeout_s"] for v in ages):
                    raise ValueError("Stale CAN feedback: " + side + "." + key)
            finite_vector(item["velocities_rad_s"], 6, side + ".velocity")
            # Guards and stopping use the latest CAN-backed pose, rather than camera-aligned historical state.
            measured.extend(finite_vector(item["positions_rad"], 6, side + ".position"))
            measured.extend(finite_vector([item["gripper_width_m"]], 1, side + ".gripper"))
            if type(item.get("gravity_required")) is not bool:
                raise ValueError("Unknown gravity configuration: " + side)
            if require_control and item["gravity_required"]:
                enabled, received = gravity[side]
                if enabled is not True or not 0 <= now - received <= cfg["feedback_timeout_s"]:
                    raise ValueError("Follower gravity is unavailable: " + side)
        return self.measured_position(measured)

    def measured_position(self, values):
        if not self.replay:
            return self.position(values)
        values = finite_vector(values, 14, 'measured bimanual positions')
        # The measured near-zero J2/J3 excursions are present in experiment1 and raw CAN feedback.
        margin = min(.05, self.config['tracking_joint_delta_rad'])
        lower_margin = [0., margin, 0., 0., 0., 0., 0.] * 2
        upper_margin = [0., 0., margin, 0., 0., 0., 0.] * 2
        if any(not low - down - 1e-8 <= value <= high + up + 1e-8 for value, low, high, down, up in zip(
                values, self.config['arm_position_min'] * 2, self.config['arm_position_max'] * 2,
                lower_margin, upper_margin)):
            raise ValueError('Measured pose exceeds the replay recovery envelope')
        return values

    def arm(self, measured, diagnostics, now):
        if self.active:
            raise ValueError("Trial already armed")
        if any(abs(v) > 0.03 for item, _ in diagnostics.values() for v in item["velocities_rad_s"]):
            raise ValueError("Arming requires stationary arms (<=0.03 rad/s)")
        self.last_command = self.measured_position(measured)
        if self.replay and self.reset_positions is None:
            self.position(measured)  # Recovery needs an explicit in-limit destination.
        self.velocity = [0.0] * 14
        self.queue.clear()
        self.sequence, self.last_observation = 0, 0.0
        self.started = self.last_plan = self.last_tick = self.last_client = now
        self.reset_settled_since, self.policy_after = None, 0.0
        self.phase = "reset" if self.reset_positions is not None else "policy"
        self.active, self.reason = True, ("ARMED: moving to dataset first frame" if self.phase == "reset" else
                                          "ARMED: waiting for a fresh ACT chunk")

    def stop(self, reason):
        self.active, self.reason = False, "STOPPED: " + str(reason)
        self.phase = "stopped"
        self.queue.clear()

    def submit(self, payload, measured, ros_now, now):
        if not self.active:
            raise ValueError("Trial is not armed locally")
        if self.phase == "reset":
            raise ValueError("Dataset first-frame positioning is not complete")
        if self.queue:
            raise ValueError("Previous chunk is still executing; chunks cannot overwrite it")
        if payload.get("action_names") != ACTION_NAMES or payload.get("rate_hz") != self.config["rate_hz"]:
            raise ValueError("Action order/rate differs from the execution contract")
        if type(payload.get("sequence")) is not int or payload["sequence"] != self.sequence + 1:
            raise ValueError("Duplicate/out-of-order action chunk")
        stamps = payload["source_stamps"]
        if set(stamps) != self.source_topics:
            raise ValueError("A chunk requires source timestamps for: " + ', '.join(sorted(self.source_topics)))
        values = finite_vector(list(stamps.values()), len(self.source_topics), "source timestamps")
        if any(stamp <= self.policy_after for stamp in values):
            raise ValueError("Policy requires fresh observations after first-frame positioning")
        if any(not 0 <= ros_now - stamp <= self.config["observation_timeout_s"] for stamp in values) or \
                max(values) - min(values) > 0.05:
            raise ValueError("Chunk observations are stale, in the future, or unsynchronized")
        stamp = payload["observation_stamp"]
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or \
                stamp not in values or stamp <= self.last_observation:
            raise ValueError("Chunk must use a new synchronized observation")
        actions = self.validate_actions(payload.get("actions"), measured)
        self.queue.extend(actions)
        self.sequence, self.last_observation, self.last_plan = payload["sequence"], stamp, now
        self.reason = "ARMED: executing chunk %d" % self.sequence

    def validate_actions(self, actions, measured):
        """Shared model preflight; live feedback/routing/timing checks remain in the bridge."""
        measured = self.measured_position(measured)
        if not isinstance(actions, list) or len(actions) != self.config["chunk_size"]:
            raise ValueError("Expected exactly 50 bimanual actions")
        actions = [self.position(row) for row in actions]
        cfg = self.config
        initial = self.limits(cfg["initial_joint_delta_rad"], cfg["initial_gripper_delta_m"])
        if any(abs(a - b) > limit for a, b, limit in zip(actions[0], measured, initial)):
            raise ValueError("First target is too far from the measured pose")
        jump = self.limits(cfg["raw_joint_step_rad"], cfg["raw_gripper_step_m"])
        if any(abs(a - b) > limit for previous, row in zip(actions, actions[1:])
               for a, b, limit in zip(previous, row, jump)):
            raise ValueError("Predicted chunk contains an excessive joint/gripper jump")
        return actions

    def check_step(self, measured, now, check_tracking=True):
        cfg = self.config
        dt = now - self.last_tick
        if not 0 < dt <= cfg["max_tick_delay_s"]:
            raise ValueError("Control loop missed its deadline")
        if (self.phase == "reset" or not self.continuous) and now - self.started >= (60.0 if self.phase == "reset" else cfg["trial_seconds"]):
            if self.phase == 'reset':
                progress = self.reset_progress(measured, now)
                raise ValueError('First-frame positioning timed out: joint error %.4f rad; gripper errors %.1f / %.1f mm' %
                                 (progress['joint_error_rad'], *(v * 1000 for v in progress['gripper_error_m'])))
            raise ValueError('Bounded trial time elapsed')
        if now - self.last_client > cfg["client_timeout_s"]:
            raise ValueError("ACT client heartbeat expired")
        tolerance = self.limits(cfg["tracking_joint_delta_rad"], cfg["tracking_gripper_delta_m"])
        if check_tracking and any(abs(a - b) > limit for a, b, limit in zip(self.last_command, measured, tolerance)):
            raise ValueError("Measured arm is not tracking the commanded pose")
        if self.phase not in ("reset", "paused") and not self.queue and now - self.last_plan > cfg["plan_timeout_s"]:
            raise ValueError("ACT client/plan watchdog expired")
        return min(dt, 1.0 / cfg['rate_hz'])

    def reset_progress(self, measured, now):
        expected = getattr(self, 'reset_feedback_positions', self.reset_positions)
        return {'elapsed_s': now - self.started,
                'joint_error_rad': max(abs(a - b) for i, (a, b) in enumerate(zip(expected, measured)) if i % 7 != 6),
                'gripper_error_m': [abs(expected[i] - measured[i]) for i in (6, 13)],
                'gripper_measured_m': [measured[i] for i in (6, 13)],
                'gripper_expected_m': [expected[i] for i in (6, 13)],
                'gripper_command_m': [self.reset_positions[i] for i in (6, 13)]}

    def step(self, measured, now, stationary=True):
        if not self.active:
            return None
        cfg = self.config
        dt = self.check_step(measured, now)
        target = self.reset_positions if self.phase == "reset" else (self.queue.popleft() if self.queue else self.last_command)
        vmax = self.limits(cfg["joint_velocity_rad_s"], cfg["gripper_velocity_m_s"])
        amax = self.limits(cfg["joint_acceleration_rad_s2"], cfg["gripper_acceleration_m_s2"])
        # Fixed maximum period prevents a late loop from making a catch-up jump.
        dt = min(dt, 1.0 / cfg["rate_hz"])
        velocity = [max(old - acc * dt, min(old + acc * dt, max(-speed, min(speed, 3.0 * (goal - pos)))))
                    for old, acc, speed, goal, pos in zip(self.velocity, amax, vmax, target, self.last_command)]
        command = [pos + vel * dt for pos, vel in zip(self.last_command, velocity)]
        if self.replay:
            # An already out-of-limit measured start may only move inward, with the same slew limits.
            command = [max(min(low, previous), min(max(high, previous), value))
                       for value, previous, low, high in zip(command, self.last_command,
                           cfg['arm_position_min'] * 2, cfg['arm_position_max'] * 2)]
            command = self.measured_position(command)
            velocity = [(value - previous) / dt for value, previous in zip(command, self.last_command)]
        else:
            command = self.position(command)
        self.last_command, self.velocity, self.last_tick = command, velocity, now
        if self.phase == "reset":
            tolerance = self.limits(getattr(self, 'reset_joint_tolerance', 0.05 if self.replay else 0.01), 0.002)
            stopped = self.limits(0.01, 0.002)
            expected = getattr(self, 'reset_feedback_positions', target)
            settled = stationary and all(abs(reference - actual) <= limit and abs(goal - sent) <= limit and abs(speed) <= slow
                          for goal, reference, actual, sent, speed, limit, slow in
                          zip(target, expected, measured, command, velocity, tolerance, stopped))
            if not settled:
                self.reset_settled_since = None
            elif self.reset_settled_since is None:
                self.reset_settled_since = now
            if self.reset_settled_since is not None and now - self.reset_settled_since >= 0.5:
                self.phase, self.reason = "policy", "ARMED: first-frame pose reached; waiting for fresh ACT observations"
                self.started = self.last_plan = now
        # Watchdog starts when the last queued action is consumed, not at chunk upload.
        if self.queue:
            self.last_plan = now
        return command


def command_messages(command, message_type, stamp):
    messages = []
    for index in (0, 7):
        msg = message_type()
        msg.header.stamp = stamp
        msg.name = ["joint%d" % i for i in range(1, 7)] + ["gripper"]
        msg.position = command[index:index + 7]
        # No velocity or effort channels; gravity remains owned by the existing driver.
        messages.append(msg)
    return messages
