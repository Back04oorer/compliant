#!/usr/bin/env python3
"""Per-arm action arbiter with a guarded follower-isolation state.

Architecture:
- Persistent policy + human inputs for both arms.
- ``~controlled_arm`` selects ``left``, ``right`` or ``both``. An arm outside
  the control domain is always a HUMAN passthrough and never follows policy.
- Normal runtime state machine:
    POLICY <-> HUMAN
- Explicit startup-only state:
    SYNC (followers hold a captured pose; leader behavior is profile-specific)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String


MODE_POLICY = "POLICY"
MODE_HUMAN = "HUMAN"
MODE_SYNC = "SYNC"


@dataclass
class TimedJointState:
    msg: JointState
    stamp_sec: float


class DualArmActionArbiter:
    def __init__(self) -> None:
        self.rate_hz = float(rospy.get_param("~rate", 200.0))
        self.blend_sec = max(0.0, float(rospy.get_param("~blend_sec", 0.8)))
        # Manual takeover must stay immediate, while releasing control back to
        # policy should be continuous.  Keep this directional parameter
        # separate from blend_sec so HUMAN -> POLICY can be softened without
        # delaying the safety-critical POLICY -> HUMAN transition.
        self.human_to_policy_blend_sec = max(
            0.0,
            float(rospy.get_param("~human_to_policy_blend_sec", self.blend_sec)),
        )
        self.sync_to_human_blend_sec = max(
            0.0,
            float(rospy.get_param("~sync_to_human_blend_sec", self.blend_sec)),
        )
        self.source_timeout_sec = float(rospy.get_param("~source_timeout_sec", 0.30))
        self.allow_fallback = bool(rospy.get_param("~allow_fallback", False))
        self.hold_last_on_timeout = bool(rospy.get_param("~hold_last_on_timeout", True))

        controlled_arm_raw = str(rospy.get_param("~controlled_arm", "both")).strip().lower()
        if controlled_arm_raw not in ("left", "right", "both"):
            raise ValueError("~controlled_arm must be left, right, or both")
        self.controlled_arm = controlled_arm_raw

        self.publish_follow_flag = bool(rospy.get_param("~publish_follow_flag", True))
        self.follow_flag_policy_value = bool(rospy.get_param("~follow_flag_policy_value", True))
        self.follow_flag_human_value = bool(rospy.get_param("~follow_flag_human_value", False))
        self.follow_flag_sync_value = bool(rospy.get_param("~follow_flag_sync_value", True))

        self.left_policy_topic = rospy.get_param("~left_policy_topic", "/robot/arm_left/vla_joint_cmd")
        self.right_policy_topic = rospy.get_param("~right_policy_topic", "/robot/arm_right/vla_joint_cmd")
        self.left_human_topic = rospy.get_param("~left_human_topic", "/teleop/arm_left/joint_states_single")
        self.right_human_topic = rospy.get_param("~right_human_topic", "/teleop/arm_right/joint_states_single")

        self.left_output_topic = rospy.get_param("~left_output_topic", "/robot/arm_left/robot_cmd")
        self.right_output_topic = rospy.get_param("~right_output_topic", "/robot/arm_right/robot_cmd")
        self.left_leader_output_topic = rospy.get_param(
            "~left_leader_output_topic",
            "/intervention/arm_left/leader_joint_cmd",
        )
        self.right_leader_output_topic = rospy.get_param(
            "~right_leader_output_topic",
            "/intervention/arm_right/leader_joint_cmd",
        )

        self.left_hold_topic = rospy.get_param("~left_hold_topic", "/intervention/arm_left/hold_joint_cmd")
        self.right_hold_topic = rospy.get_param("~right_hold_topic", "/intervention/arm_right/hold_joint_cmd")

        self.mode_cmd_topic = rospy.get_param("~mode_cmd_topic", "/intervention/mode_cmd")
        self.mode_topic = rospy.get_param("~mode_topic", "/intervention/mode")
        self.phase_topic = rospy.get_param("~phase_topic", "/intervention/phase")
        self.active_source_topic = rospy.get_param("~active_source_topic", "/intervention/active_source")
        self.left_active_source_topic = rospy.get_param(
            "~left_active_source_topic", "/intervention/arm_left/active_source"
        )
        self.right_active_source_topic = rospy.get_param(
            "~right_active_source_topic", "/intervention/arm_right/active_source"
        )
        self.follow_flag_topic = rospy.get_param("~follow_flag_topic", "/conrft_robot/slave_follow_flag")
        self.left_follow_flag_topic = rospy.get_param(
            "~left_follow_flag_topic", "/intervention/arm_left/slave_follow_flag"
        )
        self.right_follow_flag_topic = rospy.get_param(
            "~right_follow_flag_topic", "/intervention/arm_right/slave_follow_flag"
        )

        self.prepare_human_hold_reference = bool(rospy.get_param("~prepare_human_hold_reference", True))
        self.left_robot_state_topic = rospy.get_param(
            "~left_robot_state_topic",
            "/robot/arm_left/joint_states_single",
        )
        self.right_robot_state_topic = rospy.get_param(
            "~right_robot_state_topic",
            "/robot/arm_right/joint_states_single",
        )
        prepare_human_reference_source_raw = str(
            rospy.get_param("~prepare_human_reference_source", "robot_state")
        ).strip().lower()
        if prepare_human_reference_source_raw not in ("robot_state", "last_output"):
            rospy.logwarn(
                "[arbiter] unsupported prepare_human_reference_source=%s, fallback to robot_state",
                prepare_human_reference_source_raw,
            )
            prepare_human_reference_source_raw = "robot_state"
        self.prepare_human_reference_source = prepare_human_reference_source_raw

        self.prepare_policy_gripper_handover_sec = max(
            0.0, float(rospy.get_param("~prepare_policy_gripper_handover_sec", 0.25))
        )

        self.gripper_handover_sec = max(0.0, float(rospy.get_param("~gripper_handover_sec", 0.20)))

        initial_mode_raw = str(rospy.get_param("~initial_mode", MODE_POLICY))
        self.mode = self._normalize_mode(initial_mode_raw)
        self.phase = self.mode
        self.active_source = "init"
        self.active_source_left = "init"
        self.active_source_right = "init"

        self._lock = threading.Lock()

        self.policy_left: Optional[TimedJointState] = None
        self.policy_right: Optional[TimedJointState] = None
        self.human_left: Optional[TimedJointState] = None
        self.human_right: Optional[TimedJointState] = None
        self.robot_state_left: Optional[TimedJointState] = None
        self.robot_state_right: Optional[TimedJointState] = None

        # SYNC is entered explicitly by the startup coordinator.  These are
        # fixed snapshots, not live feedback: the followers remain still until
        # the startup coordinator connects the selected command source.
        self.sync_hold_left: Optional[JointState] = None
        self.sync_hold_right: Optional[JointState] = None

        self.last_output_left: Optional[JointState] = None
        self.last_output_right: Optional[JointState] = None

        self.blend_start_sec = 0.0
        self.blend_end_sec = 0.0
        self.blend_from_left: Optional[JointState] = None
        self.blend_from_right: Optional[JointState] = None

        self.phase_start_sec = rospy.Time.now().to_sec()

        self.gripper_handover_start_sec = 0.0
        self.gripper_handover_end_sec = 0.0
        self.gripper_handover_from_left: Optional[float] = None
        self.gripper_handover_from_right: Optional[float] = None

        self.last_follow_flag: Optional[bool] = None
        self.last_left_follow_flag: Optional[bool] = None
        self.last_right_follow_flag: Optional[bool] = None

        self.left_pub = rospy.Publisher(self.left_output_topic, JointState, queue_size=10)
        self.right_pub = rospy.Publisher(self.right_output_topic, JointState, queue_size=10)
        self.left_leader_pub = rospy.Publisher(
            self.left_leader_output_topic, JointState, queue_size=10
        )
        self.right_leader_pub = rospy.Publisher(
            self.right_leader_output_topic, JointState, queue_size=10
        )
        self.left_hold_pub = rospy.Publisher(self.left_hold_topic, JointState, queue_size=10)
        self.right_hold_pub = rospy.Publisher(self.right_hold_topic, JointState, queue_size=10)

        self.mode_pub = rospy.Publisher(self.mode_topic, String, queue_size=20, latch=True)
        self.phase_pub = rospy.Publisher(self.phase_topic, String, queue_size=20, latch=True)
        self.active_source_pub = rospy.Publisher(self.active_source_topic, String, queue_size=20, latch=True)
        self.left_active_source_pub = rospy.Publisher(
            self.left_active_source_topic, String, queue_size=20, latch=True
        )
        self.right_active_source_pub = rospy.Publisher(
            self.right_active_source_topic, String, queue_size=20, latch=True
        )
        self.follow_flag_pub = rospy.Publisher(self.follow_flag_topic, Bool, queue_size=5, latch=True)
        self.left_follow_flag_pub = rospy.Publisher(
            self.left_follow_flag_topic, Bool, queue_size=5, latch=True
        )
        self.right_follow_flag_pub = rospy.Publisher(
            self.right_follow_flag_topic, Bool, queue_size=5, latch=True
        )

        rospy.Subscriber(self.left_policy_topic, JointState, self._left_policy_cb, queue_size=20, tcp_nodelay=True)
        rospy.Subscriber(self.right_policy_topic, JointState, self._right_policy_cb, queue_size=20, tcp_nodelay=True)
        rospy.Subscriber(self.left_human_topic, JointState, self._left_human_cb, queue_size=20, tcp_nodelay=True)
        rospy.Subscriber(self.right_human_topic, JointState, self._right_human_cb, queue_size=20, tcp_nodelay=True)
        rospy.Subscriber(
            self.left_robot_state_topic,
            JointState,
            self._left_robot_state_cb,
            queue_size=20,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.right_robot_state_topic,
            JointState,
            self._right_robot_state_cb,
            queue_size=20,
            tcp_nodelay=True,
        )
        rospy.Subscriber(self.mode_cmd_topic, String, self._mode_cmd_cb, queue_size=10, tcp_nodelay=True)

        rospy.loginfo(
            "[arbiter] started controlled_arm=%s mode=%s phase=%s rate=%.1f "
            "blend=%.3f human_to_policy_blend=%.3f sync_to_human_blend=%.3f "
            "timeout=%.3f fallback=%s hold_last=%s",
            self.controlled_arm,
            self.mode,
            self.phase,
            self.rate_hz,
            self.blend_sec,
            self.human_to_policy_blend_sec,
            self.sync_to_human_blend_sec,
            self.source_timeout_sec,
            str(self.allow_fallback),
            str(self.hold_last_on_timeout),
        )
        rospy.loginfo(
            "[arbiter] human_switch reference=%s hold_reference=%s robot_state_topics=(%s,%s)",
            self.prepare_human_reference_source,
            str(self.prepare_human_hold_reference),
            self.left_robot_state_topic,
            self.right_robot_state_topic,
        )
        rospy.loginfo(
            "[arbiter] policy_switch gripper_handover_sec=%.3f",
            self.prepare_policy_gripper_handover_sec,
        )
        rospy.loginfo(
            "[arbiter] modes enabled: POLICY <-> HUMAN; explicit startup SYNC isolates followers"
        )
        rospy.loginfo(
            "[arbiter] leader command outputs=(%s,%s); HUMAN/SYNC use raw leader state, "
            "controlled POLICY uses the exact follower arbiter output",
            self.left_leader_output_topic,
            self.right_leader_output_topic,
        )

        self._publish_mode_phase_and_source(force=True)
        self._publish_follow_flag(force=True)

    @staticmethod
    def _normalize_mode(text: str) -> str:
        value = text.strip().upper()
        if value in ("POLICY", "AUTO", "AUTONOMOUS"):
            return MODE_POLICY
        if value in ("HUMAN", "TELEOP", "INTERVENING"):
            return MODE_HUMAN
        if value in ("SYNC", "ALIGN", "STARTUP_SYNC"):
            return MODE_SYNC
        return MODE_POLICY

    def _arm_is_controlled(self, arm_name: str) -> bool:
        return self.controlled_arm == "both" or self.controlled_arm == arm_name

    @staticmethod
    def _clone_joint_state(msg: JointState, stamp: Optional[rospy.Time] = None) -> JointState:
        out = JointState()
        out.header.seq = msg.header.seq
        out.header.stamp = stamp if stamp is not None else msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        return out

    @staticmethod
    def _blend_array(a: Tuple[float, ...], b: Tuple[float, ...], alpha: float) -> Tuple[float, ...]:
        if len(a) != len(b):
            return b
        return tuple((1.0 - alpha) * av + alpha * bv for av, bv in zip(a, b))

    def _blend_joint_state(self, src: JointState, target: JointState, alpha: float, stamp: rospy.Time) -> JointState:
        out = JointState()
        out.header.stamp = stamp
        out.header.frame_id = target.header.frame_id if target.header.frame_id else src.header.frame_id
        out.name = list(target.name) if target.name else list(src.name)

        src_pos = tuple(src.position)
        tgt_pos = tuple(target.position)
        src_vel = tuple(src.velocity)
        tgt_vel = tuple(target.velocity)
        src_eff = tuple(src.effort)
        tgt_eff = tuple(target.effort)

        out.position = list(self._blend_array(src_pos, tgt_pos, alpha)) if tgt_pos else list(src_pos)
        out.velocity = list(self._blend_array(src_vel, tgt_vel, alpha)) if tgt_vel else list(src_vel)
        out.effort = list(self._blend_array(src_eff, tgt_eff, alpha)) if tgt_eff else list(src_eff)
        return out

    def _left_policy_cb(self, msg: JointState) -> None:
        with self._lock:
            self.policy_left = TimedJointState(msg=self._clone_joint_state(msg), stamp_sec=rospy.Time.now().to_sec())

    def _right_policy_cb(self, msg: JointState) -> None:
        with self._lock:
            self.policy_right = TimedJointState(msg=self._clone_joint_state(msg), stamp_sec=rospy.Time.now().to_sec())

    def _left_human_cb(self, msg: JointState) -> None:
        with self._lock:
            self.human_left = TimedJointState(msg=self._clone_joint_state(msg), stamp_sec=rospy.Time.now().to_sec())

    def _right_human_cb(self, msg: JointState) -> None:
        with self._lock:
            self.human_right = TimedJointState(msg=self._clone_joint_state(msg), stamp_sec=rospy.Time.now().to_sec())

    def _left_robot_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self.robot_state_left = TimedJointState(
                msg=self._clone_joint_state(msg),
                stamp_sec=rospy.Time.now().to_sec(),
            )

    def _right_robot_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self.robot_state_right = TimedJointState(
                msg=self._clone_joint_state(msg),
                stamp_sec=rospy.Time.now().to_sec(),
            )

    def _mode_cmd_cb(self, msg: String) -> None:
        cmd = str(msg.data).strip().lower()
        with self._lock:
            if cmd == "toggle":
                next_mode = MODE_HUMAN if self.mode == MODE_POLICY else MODE_POLICY
                self._request_mode_locked(next_mode, reason="toggle")
                return
            if cmd in ("policy", "auto", "autonomous"):
                self._request_mode_locked(MODE_POLICY, reason=cmd)
                return
            if cmd in ("human", "teleop", "intervening"):
                self._request_mode_locked(MODE_HUMAN, reason=cmd)
                return
            if cmd in ("sync", "align", "startup_sync"):
                self._request_mode_locked(MODE_SYNC, reason=cmd)
                return

        rospy.logwarn_throttle(2.0, "[arbiter] unsupported mode_cmd=%s", cmd)

    def _request_mode_locked(self, desired_mode: str, reason: str) -> None:
        if desired_mode == self.mode:
            return

        now = rospy.Time.now().to_sec()
        if desired_mode == MODE_SYNC:
            robot_pair = self._get_robot_state_pair_locked(now)
            if robot_pair is None:
                rospy.logwarn(
                    "[arbiter] refusing SYNC: fresh left/right follower feedback is required"
                )
                return
            self.sync_hold_left = robot_pair[0]
            self.sync_hold_right = robot_pair[1]
        # SYNC captures the measured pose and must hold it immediately.  A
        # normal source blend would defeat the startup isolation guarantee.
        self._enter_mode_locked(
            desired_mode,
            now,
            reason=f"cmd={reason}",
            use_blend=desired_mode != MODE_SYNC,
        )

    def _set_blend_from_reference_locked(
        self,
        now_sec: float,
        target_mode: str,
        previous_mode: str,
    ) -> Optional[Tuple[JointState, JointState]]:
        blend_from_pair: Optional[Tuple[JointState, JointState]] = None
        if target_mode == MODE_HUMAN and self.prepare_human_hold_reference:
            blend_from_pair, _ = self._get_human_switch_reference_pair_locked(now_sec)
        if blend_from_pair is None and self.last_output_left is not None and self.last_output_right is not None:
            blend_from_pair = (
                self._clone_joint_state(self.last_output_left),
                self._clone_joint_state(self.last_output_right),
            )

        if previous_mode == MODE_HUMAN and target_mode == MODE_POLICY:
            blend_sec = self.human_to_policy_blend_sec
        elif previous_mode == MODE_SYNC and target_mode == MODE_HUMAN:
            blend_sec = self.sync_to_human_blend_sec
        else:
            blend_sec = self.blend_sec
        if blend_sec > 0.0 and blend_from_pair is not None:
            self.blend_from_left = blend_from_pair[0]
            self.blend_from_right = blend_from_pair[1]
            self.blend_start_sec = now_sec
            self.blend_end_sec = now_sec + blend_sec
            return blend_from_pair

        self.blend_start_sec = 0.0
        self.blend_end_sec = 0.0
        self.blend_from_left = None
        self.blend_from_right = None
        return blend_from_pair

    def _enter_mode_locked(self, mode: str, now_sec: float, reason: str, use_blend: bool) -> None:
        old_mode = self.mode
        self.mode = mode
        self.phase = mode
        self.phase_start_sec = now_sec

        blend_from_pair: Optional[Tuple[JointState, JointState]] = None
        if use_blend:
            blend_from_pair = self._set_blend_from_reference_locked(
                now_sec,
                target_mode=mode,
                previous_mode=old_mode,
            )
        else:
            self.blend_start_sec = 0.0
            self.blend_end_sec = 0.0
            self.blend_from_left = None
            self.blend_from_right = None

        gripper_ref_left: Optional[JointState] = None
        gripper_ref_right: Optional[JointState] = None
        if mode == MODE_HUMAN:
            if blend_from_pair is None:
                blend_from_pair, _ = self._get_human_switch_reference_pair_locked(now_sec)
            if blend_from_pair is not None:
                gripper_ref_left = blend_from_pair[0]
                gripper_ref_right = blend_from_pair[1]
            handover_sec = self.gripper_handover_sec
        elif mode == MODE_POLICY:
            handover_sec = self.prepare_policy_gripper_handover_sec
        else:
            handover_sec = 0.0

        if gripper_ref_left is None:
            gripper_ref_left = self.last_output_left
        if gripper_ref_right is None:
            gripper_ref_right = self.last_output_right

        if handover_sec > 0.0:
            self.gripper_handover_start_sec = now_sec
            self.gripper_handover_end_sec = now_sec + handover_sec
            self.gripper_handover_from_left = (
                gripper_ref_left.position[6]
                if gripper_ref_left is not None and len(gripper_ref_left.position) >= 7
                else None
            )
            self.gripper_handover_from_right = (
                gripper_ref_right.position[6]
                if gripper_ref_right is not None and len(gripper_ref_right.position) >= 7
                else None
            )
        else:
            self.gripper_handover_start_sec = 0.0
            self.gripper_handover_end_sec = 0.0
            self.gripper_handover_from_left = None
            self.gripper_handover_from_right = None

        rospy.loginfo("[arbiter] mode %s -> %s by %s", old_mode, mode, reason)

    def _source_is_fresh(self, stamp_sec: float, now_sec: float) -> bool:
        if self.source_timeout_sec <= 0.0:
            return True
        return (now_sec - stamp_sec) <= self.source_timeout_sec

    def _timed_source_for_arm_locked(self, arm_name: str, mode: str) -> Optional[TimedJointState]:
        if arm_name == "left":
            return self.policy_left if mode == MODE_POLICY else self.human_left
        return self.policy_right if mode == MODE_POLICY else self.human_right

    def _last_output_for_arm_locked(self, arm_name: str) -> Optional[JointState]:
        return self.last_output_left if arm_name == "left" else self.last_output_right

    def _select_arm_target_locked(
        self,
        arm_name: str,
        requested_mode: str,
        now_sec: float,
    ) -> Tuple[Optional[JointState], str]:
        if requested_mode == MODE_SYNC:
            hold = self.sync_hold_left if arm_name == "left" else self.sync_hold_right
            if hold is not None:
                return self._clone_joint_state(hold), "sync_hold"
            return None, "sync_no_hold"

        # The uncontrolled arm is a hard HUMAN passthrough. It is deliberately
        # independent of the controlled arm's policy freshness and mode.
        desired_mode = requested_mode if self._arm_is_controlled(arm_name) else MODE_HUMAN
        primary = self._timed_source_for_arm_locked(arm_name, desired_mode)
        if primary is not None and self._source_is_fresh(primary.stamp_sec, now_sec):
            return self._clone_joint_state(primary.msg), desired_mode.lower()

        if self.allow_fallback and self._arm_is_controlled(arm_name):
            fallback_mode = MODE_HUMAN if desired_mode == MODE_POLICY else MODE_POLICY
            fallback = self._timed_source_for_arm_locked(arm_name, fallback_mode)
            if fallback is not None and self._source_is_fresh(fallback.stamp_sec, now_sec):
                return self._clone_joint_state(fallback.msg), f"{fallback_mode.lower()}_fallback"

        last_output = self._last_output_for_arm_locked(arm_name)
        if self.hold_last_on_timeout and last_output is not None:
            return self._clone_joint_state(last_output), "hold_last"
        return None, "no_cmd"

    def _select_target_for_mode_locked(
        self, mode: str, now_sec: float
    ) -> Tuple[Optional[JointState], Optional[JointState], str]:
        left, left_source = self._select_arm_target_locked("left", mode, now_sec)
        right, right_source = self._select_arm_target_locked("right", mode, now_sec)
        self.active_source_left = left_source
        self.active_source_right = right_source
        if self.controlled_arm == "left":
            selected = left_source
        elif self.controlled_arm == "right":
            selected = right_source
        elif left_source == right_source:
            selected = left_source
        else:
            selected = f"left={left_source},right={right_source}"
        return left, right, selected

    def _get_robot_state_pair_locked(self, now_sec: float) -> Optional[Tuple[JointState, JointState]]:
        left = self.robot_state_left
        right = self.robot_state_right
        if left is None or right is None:
            return None
        if not self._source_is_fresh(left.stamp_sec, now_sec):
            return None
        if not self._source_is_fresh(right.stamp_sec, now_sec):
            return None
        return self._clone_joint_state(left.msg), self._clone_joint_state(right.msg)

    def _get_human_switch_reference_pair_locked(
        self,
        now_sec: float,
    ) -> Tuple[Optional[Tuple[JointState, JointState]], str]:
        if self.prepare_human_reference_source == "robot_state":
            robot_pair = self._get_robot_state_pair_locked(now_sec)
            if robot_pair is not None:
                return robot_pair, "robot_state"
        if self.last_output_left is not None and self.last_output_right is not None:
            return (
                self._clone_joint_state(self.last_output_left),
                self._clone_joint_state(self.last_output_right),
            ), "last_output"
        return None, "none"

    def _leader_target_for_arm_locked(
        self,
        arm_name: str,
        follower_output: Optional[JointState],
        now_sec: float,
        now: rospy.Time,
    ) -> Optional[JointState]:
        """Select a Piper leader command without using Piper's master/slave path.

        In HUMAN and startup SYNC, a leader receives its own fresh measured
        state, matching the established teleop position-reference loop. In
        POLICY, a controlled leader receives the exact post-blend command sent
        to its follower. An uncontrolled arm always remains on its own state.
        """
        if self.mode == MODE_POLICY and self._arm_is_controlled(arm_name):
            if follower_output is None:
                return None
            return self._clone_joint_state(follower_output, stamp=now)

        human = self.human_left if arm_name == "left" else self.human_right
        if human is None or not self._source_is_fresh(human.stamp_sec, now_sec):
            return None
        return self._clone_joint_state(human.msg, stamp=now)

    def _phase_follow_flag_value(self) -> bool:
        if self.mode == MODE_SYNC:
            return self.follow_flag_sync_value
        if self.mode == MODE_HUMAN:
            return self.follow_flag_human_value
        return self.follow_flag_policy_value

    def _arm_follow_flag_value(self, arm_name: str) -> bool:
        # SYNC always isolates the followers at their captured pose. Whether
        # leaders follow that pose is launch-profile specific. Manual
        # intervention disables it to preserve the established teleop control
        # direction: leader command -> follower.
        if self.mode == MODE_SYNC:
            return self.follow_flag_sync_value
        if not self._arm_is_controlled(arm_name):
            return self.follow_flag_human_value
        return self._phase_follow_flag_value()

    def _publish_mode_phase_and_source(self, force: bool = False) -> None:
        self.mode_pub.publish(String(data=self.mode))
        self.phase_pub.publish(String(data=self.phase))
        if force or self.active_source:
            self.active_source_pub.publish(String(data=self.active_source))
            self.left_active_source_pub.publish(String(data=self.active_source_left))
            self.right_active_source_pub.publish(String(data=self.active_source_right))

    def _publish_follow_flag(self, force: bool = False) -> None:
        if not self.publish_follow_flag:
            return

        left_flag = self._arm_follow_flag_value("left")
        right_flag = self._arm_follow_flag_value("right")
        # Retain the aggregate topic for legacy diagnostics. Intervention
        # leader nodes subscribe to the per-arm topics below.
        flag = left_flag if self.controlled_arm != "right" else right_flag
        if force or self.last_follow_flag is None or flag != self.last_follow_flag:
            self.follow_flag_pub.publish(Bool(data=flag))
            self.last_follow_flag = flag
        if force or self.last_left_follow_flag is None or left_flag != self.last_left_follow_flag:
            self.left_follow_flag_pub.publish(Bool(data=left_flag))
            self.last_left_follow_flag = left_flag
        if force or self.last_right_follow_flag is None or right_flag != self.last_right_follow_flag:
            self.right_follow_flag_pub.publish(Bool(data=right_flag))
            self.last_right_follow_flag = right_flag

    def _maybe_apply_gripper_handover_locked(
        self,
        left_out: Optional[JointState],
        right_out: Optional[JointState],
        now_sec: float,
    ) -> Tuple[Optional[JointState], Optional[JointState]]:
        if self.gripper_handover_end_sec <= self.gripper_handover_start_sec:
            return left_out, right_out
        if now_sec >= self.gripper_handover_end_sec:
            return left_out, right_out

        span = max(1e-6, self.gripper_handover_end_sec - self.gripper_handover_start_sec)
        alpha = min(max((now_sec - self.gripper_handover_start_sec) / span, 0.0), 1.0)

        if left_out is not None and self.gripper_handover_from_left is not None and len(left_out.position) >= 7:
            left_out.position[6] = (1.0 - alpha) * self.gripper_handover_from_left + alpha * left_out.position[6]
        if right_out is not None and self.gripper_handover_from_right is not None and len(right_out.position) >= 7:
            right_out.position[6] = (1.0 - alpha) * self.gripper_handover_from_right + alpha * right_out.position[6]
        return left_out, right_out


    def _step_steady_locked(self, now_sec: float, now: rospy.Time) -> Tuple[Optional[JointState], Optional[JointState], str]:
        left_target, right_target, selected = self._select_target_for_mode_locked(self.mode, now_sec)
        in_blend = (
            self.blend_end_sec > self.blend_start_sec
            and now_sec < self.blend_end_sec
        )
        if in_blend:
            alpha = (now_sec - self.blend_start_sec) / max(
                self.blend_end_sec - self.blend_start_sec,
                1e-6,
            )
            alpha = min(max(alpha, 0.0), 1.0)
            # Quintic smootherstep has zero velocity and acceleration at both
            # ends.  A linear blend is position-continuous but still commands
            # an instantaneous non-zero velocity at handoff, which is felt as
            # a kick when I releases HUMAN control back to POLICY.
            alpha = alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
            left_out = (
                self._blend_joint_state(self.blend_from_left, left_target, alpha, now)
                if self.blend_from_left is not None and left_target is not None
                else self._clone_joint_state(left_target, stamp=now) if left_target is not None else None
            )
            right_out = (
                self._blend_joint_state(self.blend_from_right, right_target, alpha, now)
                if self.blend_from_right is not None and right_target is not None
                else self._clone_joint_state(right_target, stamp=now) if right_target is not None else None
            )
            return left_out, right_out, f"{selected}_blend"

        if now_sec >= self.blend_end_sec:
            self.blend_start_sec = 0.0
            self.blend_end_sec = 0.0
            self.blend_from_left = None
            self.blend_from_right = None

        left_out = self._clone_joint_state(left_target, stamp=now) if left_target is not None else None
        right_out = self._clone_joint_state(right_target, stamp=now) if right_target is not None else None
        return left_out, right_out, selected

    def run(self) -> None:
        if self.rate_hz <= 0.0:
            raise RuntimeError("~rate must be > 0")

        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            now_sec = now.to_sec()

            with self._lock:
                left_out, right_out, selected = self._step_steady_locked(now_sec, now)

                self.active_source = selected
                left_out, right_out = self._maybe_apply_gripper_handover_locked(left_out, right_out, now_sec)
                left_leader_out = self._leader_target_for_arm_locked(
                    "left", left_out, now_sec, now
                )
                right_leader_out = self._leader_target_for_arm_locked(
                    "right", right_out, now_sec, now
                )

                if left_out is not None:
                    self.last_output_left = self._clone_joint_state(left_out)
                if right_out is not None:
                    self.last_output_right = self._clone_joint_state(right_out)

                # Keep the arbiter authoritative even if a leader node starts
                # later and emits its own latched startup=false on the topic.
                self._publish_follow_flag(force=True)
                mode_snapshot = self.mode
                phase_snapshot = self.phase
                source_snapshot = self.active_source
                left_source_snapshot = self.active_source_left
                right_source_snapshot = self.active_source_right

            if left_out is not None:
                self.left_pub.publish(left_out)
                self.left_hold_pub.publish(left_out)
            if right_out is not None:
                self.right_pub.publish(right_out)
                self.right_hold_pub.publish(right_out)
            if left_leader_out is not None:
                self.left_leader_pub.publish(left_leader_out)
            if right_leader_out is not None:
                self.right_leader_pub.publish(right_leader_out)
            if left_out is None or right_out is None:
                rospy.logwarn_throttle(
                    1.0,
                    "[arbiter] missing command (mode=%s phase=%s left=%s right=%s)",
                    mode_snapshot,
                    phase_snapshot,
                    left_source_snapshot,
                    right_source_snapshot,
                )

            self.mode_pub.publish(String(data=mode_snapshot))
            self.phase_pub.publish(String(data=phase_snapshot))
            self.active_source_pub.publish(String(data=source_snapshot))
            self.left_active_source_pub.publish(String(data=left_source_snapshot))
            self.right_active_source_pub.publish(String(data=right_source_snapshot))
            rate.sleep()


def main() -> None:
    rospy.init_node("dual_arm_action_arbiter")
    node = DualArmActionArbiter()
    node.run()


if __name__ == "__main__":
    main()
