#!/usr/bin/env python3
"""Intervention-only policy selector layered on the unchanged Piper baseline.

The normal HUMAN/SYNC path remains the baseline node's direct self-feedback
loop.  Only the controlled leader switches its MIT position/gripper reference
to the arbiter output while /intervention/mode is POLICY.
"""

import math
import threading

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_ctrl_node import C_PiperCtrlNode, check_ros_master


class C_PiperInterventionCtrlNode(C_PiperCtrlNode):
    def __init__(self):
        super().__init__()

        # RLocks let the thin selector temporarily substitute a reference while
        # reusing the baseline _mit_control implementation without copying it.
        self.control_data_lock = threading.RLock()
        self.gripper_cmd_lock = threading.RLock()
        self._intervention_lock = threading.Lock()
        self._intervention_mode = "SYNC"
        self._intervention_policy_positions = None
        self._intervention_policy_stamp = rospy.Time(0)
        self._intervention_policy_enter_stamp = rospy.Time(0)

        # HUMAN/SYNC deliberately retain the low-gain gravity-compensated
        # leader settings loaded by the unchanged Piper baseline.  Those gains
        # make the leader pleasant to hand-guide, but they are far too soft to
        # track a moving policy reference.  POLICY therefore uses the same MIT
        # gains as the follower controllers while preserving the exact same
        # raw position reference from the arbiter.
        self._human_mit_kp = list(self.mit_kp)
        self._human_mit_kd = list(self.mit_kd)
        self._human_mit_enable_gravity = bool(self.mit_enable_gravity)
        self._policy_mit_kp = self._read_six_gains(
            "~intervention_switch/policy_kp", 7.0
        )
        self._policy_mit_kd = self._read_six_gains(
            "~intervention_switch/policy_kd", 0.8
        )
        self._policy_enable_gravity = bool(
            rospy.get_param("~intervention_switch/policy_enable_gravity", False)
        )
        self._policy_gain_ramp_sec = max(
            0.0,
            float(
                rospy.get_param(
                    "~intervention_switch/policy_gain_ramp_sec", 0.25
                )
            ),
        )

        self.intervention_switch_enable = bool(
            rospy.get_param("~intervention_switch/enable", False)
        )
        self.intervention_mode_topic = str(
            rospy.get_param("~intervention_switch/mode_topic", "/intervention/mode")
        )
        self.intervention_policy_topic = str(
            rospy.get_param(
                "~intervention_switch/policy_topic", "/robot/arm_left/robot_cmd"
            )
        )
        self.intervention_policy_timeout_sec = max(
            0.05,
            float(rospy.get_param("~intervention_switch/policy_timeout_sec", 0.30)),
        )

        if self.intervention_switch_enable:
            rospy.Subscriber(
                self.intervention_mode_topic,
                String,
                self._intervention_mode_callback,
                queue_size=5,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.intervention_policy_topic,
                JointState,
                self._intervention_policy_callback,
                queue_size=5,
                tcp_nodelay=True,
            )
            rospy.loginfo(
                "Intervention selector enabled: HUMAN/SYNC=direct self feedback, "
                "POLICY=%s, mode=%s, timeout=%.3fs",
                self.intervention_policy_topic,
                self.intervention_mode_topic,
                self.intervention_policy_timeout_sec,
            )
            rospy.loginfo(
                "Intervention POLICY MIT tracking: kp=%s kd=%s gravity=%s "
                "gain_ramp=%.3fs; HUMAN gains remain kp=%s kd=%s gravity=%s",
                self._policy_mit_kp,
                self._policy_mit_kd,
                self._policy_enable_gravity,
                self._policy_gain_ramp_sec,
                self._human_mit_kp,
                self._human_mit_kd,
                self._human_mit_enable_gravity,
            )
        else:
            rospy.loginfo("Intervention selector disabled; using unchanged baseline behavior")

    @staticmethod
    def _read_six_gains(param_name, default):
        raw = rospy.get_param(param_name, default)
        if isinstance(raw, (list, tuple)):
            if len(raw) != 6:
                raise ValueError(f"{param_name} must contain exactly 6 values")
            values = [float(value) for value in raw]
        else:
            values = [float(raw)] * 6
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"{param_name} values must be finite and >= 0")
        return values

    def _intervention_mode_callback(self, msg: String):
        mode = str(msg.data).strip().upper()
        if mode not in ("SYNC", "HUMAN", "POLICY"):
            return
        with self._intervention_lock:
            previous = self._intervention_mode
            self._intervention_mode = mode
            if mode == "POLICY" and previous != "POLICY":
                self._intervention_policy_enter_stamp = rospy.Time.now()
        if mode != previous:
            rospy.loginfo(
                "Intervention leader selector mode: %s -> %s", previous, mode
            )

    def _intervention_policy_callback(self, msg: JointState):
        if len(msg.position) < 6:
            return
        with self._intervention_lock:
            self._intervention_policy_positions = [float(v) for v in msg.position[:7]]
            self._intervention_policy_stamp = rospy.Time.now()

    def _policy_reference(self):
        with self._intervention_lock:
            if self._intervention_mode != "POLICY":
                return None
            positions = self._intervention_policy_positions
            now = rospy.Time.now()
            age = (now - self._intervention_policy_stamp).to_sec()
            policy_elapsed = (
                now - self._intervention_policy_enter_stamp
            ).to_sec()
            if positions is None or age < 0.0 or age > self.intervention_policy_timeout_sec:
                rospy.logwarn_throttle(
                    1.0,
                    "POLICY requested but intervention policy reference is missing/stale; "
                    "holding direct self-feedback path",
                )
                return None
            if self._policy_gain_ramp_sec <= 0.0:
                gain_alpha = 1.0
            else:
                gain_alpha = max(
                    0.0, min(1.0, policy_elapsed / self._policy_gain_ramp_sec)
                )
            return list(positions), gain_alpha

    def _mit_control(self):
        policy_reference = self._policy_reference()
        if policy_reference is None:
            # This is byte-for-byte the baseline HUMAN/SYNC behavior: the
            # position callback subscribes directly to this node's own state.
            return super()._mit_control()

        policy_positions, gain_alpha = policy_reference
        policy_kp = [
            human + gain_alpha * (policy - human)
            for human, policy in zip(self._human_mit_kp, self._policy_mit_kp)
        ]
        policy_kd = [
            human + gain_alpha * (policy - human)
            for human, policy in zip(self._human_mit_kd, self._policy_mit_kd)
        ]

        with self.control_data_lock, self.gripper_cmd_lock:
            saved_positions = self.joint_positions_cmd[:]
            saved_gripper = self.gripper_cmd_position
            saved_kp = self.mit_kp
            saved_kd = self.mit_kd
            saved_enable_gravity = self.mit_enable_gravity
            try:
                self.joint_positions_cmd[:6] = policy_positions[:6]
                if len(policy_positions) >= 7:
                    self.joint_positions_cmd[6] = policy_positions[6]
                    self.gripper_cmd_position = policy_positions[6]
                self.mit_kp = policy_kp
                self.mit_kd = policy_kd
                self.mit_enable_gravity = self._policy_enable_gravity
                rospy.loginfo_throttle(
                    1.0,
                    "Intervention POLICY tracking active: source=%s "
                    "gain_alpha=%.2f kp=%s kd=%s gravity=%s",
                    self.intervention_policy_topic,
                    gain_alpha,
                    [round(value, 3) for value in policy_kp],
                    [round(value, 3) for value in policy_kd],
                    self.mit_enable_gravity,
                )
                return super()._mit_control()
            finally:
                self.joint_positions_cmd[:] = saved_positions
                self.gripper_cmd_position = saved_gripper
                self.mit_kp = saved_kp
                self.mit_kd = saved_kd
                self.mit_enable_gravity = saved_enable_gravity


if __name__ == "__main__":
    try:
        check_ros_master()
        node = C_PiperInterventionCtrlNode()
        node.Run()
    except rospy.ROSInterruptException:
        pass
