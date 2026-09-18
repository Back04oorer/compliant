#!/usr/bin/env python3
"""Smoothly drive both robot arms to zero via command topics without mode switching."""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import JointState


PER_ARM_DIM = 7
DEFAULT_JOINT_NAMES = "joint1,joint2,joint3,joint4,joint5,joint6,gripper"


class StartupZeroViaTopic:
    def __init__(self) -> None:
        self.left_state_topic = rospy.get_param("~left_state_topic", "/robot/arm_left/joint_states_single")
        self.right_state_topic = rospy.get_param("~right_state_topic", "/robot/arm_right/joint_states_single")
        self.left_cmd_topic = rospy.get_param("~left_cmd_topic", "/robot/arm_left/joint_cmd_mux")
        self.right_cmd_topic = rospy.get_param("~right_cmd_topic", "/robot/arm_right/joint_cmd_mux")
        self.duration_sec = float(rospy.get_param("~duration_sec", 3.0))
        self.rate_hz = float(rospy.get_param("~rate", 30.0))
        self.wait_timeout_sec = float(rospy.get_param("~wait_timeout_sec", 10.0))
        self.startup_grace_sec = float(rospy.get_param("~startup_grace_sec", 0.0))
        self.joint_names = self._parse_joint_names(rospy.get_param("~joint_names", DEFAULT_JOINT_NAMES))
        self.settle_enable = bool(rospy.get_param("~settle_enable", True))
        self.settle_iters = max(0, int(rospy.get_param("~settle_iters", 12)))
        self.settle_tol_rad = max(0.0, float(rospy.get_param("~settle_tol_rad", 0.01)))
        self.settle_gain = float(rospy.get_param("~settle_gain", 0.9))
        self.settle_max_step_rad = max(0.0, float(rospy.get_param("~settle_max_step_rad", 0.03)))
        self.settle_sample_sec = max(0.02, float(rospy.get_param("~settle_sample_sec", 0.15)))
        self.settle_hold_sec = max(0.0, float(rospy.get_param("~settle_hold_sec", 0.8)))
        self.settle_include_gripper = bool(rospy.get_param("~settle_include_gripper", False))

        self.lock = threading.Lock()
        self.left_pos: Optional[np.ndarray] = None
        self.right_pos: Optional[np.ndarray] = None

        self.left_pub = rospy.Publisher(self.left_cmd_topic, JointState, queue_size=1)
        self.right_pub = rospy.Publisher(self.right_cmd_topic, JointState, queue_size=1)

        rospy.Subscriber(self.left_state_topic, JointState, self._cb_left, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.right_state_topic, JointState, self._cb_right, queue_size=1, tcp_nodelay=True)

        rospy.loginfo(
            "startup_zero_via_topic configured: left_state=%s right_state=%s left_cmd=%s right_cmd=%s duration=%.2fs rate=%.2fHz settle_enable=%s settle_iters=%d settle_tol=%.4frad",
            self.left_state_topic,
            self.right_state_topic,
            self.left_cmd_topic,
            self.right_cmd_topic,
            self.duration_sec,
            self.rate_hz,
            self.settle_enable,
            self.settle_iters,
            self.settle_tol_rad,
        )

    def _parse_joint_names(self, raw: str):
        names = [x.strip() for x in str(raw).split(",") if x.strip()]
        if len(names) != PER_ARM_DIM:
            raise RuntimeError(f"~joint_names must contain exactly {PER_ARM_DIM} names, got {len(names)}")
        return names

    def _cb_left(self, msg: JointState) -> None:
        if len(msg.position) < PER_ARM_DIM:
            return
        with self.lock:
            self.left_pos = np.asarray(msg.position[:PER_ARM_DIM], dtype=np.float32)

    def _cb_right(self, msg: JointState) -> None:
        if len(msg.position) < PER_ARM_DIM:
            return
        with self.lock:
            self.right_pos = np.asarray(msg.position[:PER_ARM_DIM], dtype=np.float32)

    def _snapshot(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            if self.left_pos is None or self.right_pos is None:
                return None
            return self.left_pos.copy(), self.right_pos.copy()

    def _publish_pair(self, left_pos: np.ndarray, right_pos: np.ndarray) -> None:
        stamp = rospy.Time.now()

        left_msg = JointState()
        left_msg.header.stamp = stamp
        left_msg.name = list(self.joint_names)
        left_msg.position = [float(v) for v in left_pos]
        left_msg.velocity = [0.0] * PER_ARM_DIM
        left_msg.effort = [0.0] * PER_ARM_DIM

        right_msg = JointState()
        right_msg.header.stamp = stamp
        right_msg.name = list(self.joint_names)
        right_msg.position = [float(v) for v in right_pos]
        right_msg.velocity = [0.0] * PER_ARM_DIM
        right_msg.effort = [0.0] * PER_ARM_DIM

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def _measure_mean_state_while_holding(
        self,
        left_cmd: np.ndarray,
        right_cmd: np.ndarray,
        sample_sec: float,
        rate: rospy.Rate,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        deadline = rospy.Time.now().to_sec() + max(0.02, float(sample_sec))
        left_sum = np.zeros(PER_ARM_DIM, dtype=np.float64)
        right_sum = np.zeros(PER_ARM_DIM, dtype=np.float64)
        count = 0
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            snap = self._snapshot()
            if snap is not None:
                left_now, right_now = snap
                left_sum += left_now.astype(np.float64)
                right_sum += right_now.astype(np.float64)
                count += 1
            self._publish_pair(left_cmd, right_cmd)
            rate.sleep()
        if count <= 0:
            return None
        left_mean = (left_sum / float(count)).astype(np.float32)
        right_mean = (right_sum / float(count)).astype(np.float32)
        return left_mean, right_mean

    def run(self) -> None:
        if self.startup_grace_sec > 0.0:
            rospy.loginfo("startup_zero_via_topic grace sleep: %.2fs", self.startup_grace_sec)
            rospy.sleep(self.startup_grace_sec)

        rate_hz = max(1.0, self.rate_hz)
        rate = rospy.Rate(rate_hz)
        deadline = rospy.Time.now().to_sec() + max(0.1, self.wait_timeout_sec)
        start_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            start_pair = self._snapshot()
            if start_pair is not None:
                break
            rate.sleep()

        if rospy.is_shutdown():
            return
        if start_pair is None:
            rospy.logwarn("startup_zero_via_topic timeout waiting for joint states, skip auto zero.")
            return

        start_left, start_right = start_pair
        target_left = np.zeros(PER_ARM_DIM, dtype=np.float32)
        target_right = np.zeros(PER_ARM_DIM, dtype=np.float32)
        duration = max(0.001, float(self.duration_sec))
        t0 = rospy.Time.now().to_sec()

        rospy.loginfo("startup_zero_via_topic begin ramp to zero: duration=%.2fs", duration)
        while not rospy.is_shutdown():
            alpha = (rospy.Time.now().to_sec() - t0) / duration
            alpha = min(1.0, max(0.0, alpha))
            left_cmd = start_left + alpha * (target_left - start_left)
            right_cmd = start_right + alpha * (target_right - start_right)
            self._publish_pair(left_cmd, right_cmd)
            if alpha >= 1.0:
                break
            rate.sleep()

        # Publish final zero a few extra frames to improve latching at startup.
        cmd_left = target_left.copy()
        cmd_right = target_right.copy()
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self._publish_pair(cmd_left, cmd_right)
            rate.sleep()

        # Closed-loop settle: MIT mode can leave non-trivial static bias near gravity-loaded joints.
        if self.settle_enable and self.settle_iters > 0 and not rospy.is_shutdown():
            eval_end = PER_ARM_DIM if self.settle_include_gripper else (PER_ARM_DIM - 1)
            rospy.loginfo(
                "startup_zero_via_topic settle begin: iters=%d tol=%.4f gain=%.2f step_max=%.4f sample=%.2fs",
                self.settle_iters,
                self.settle_tol_rad,
                self.settle_gain,
                self.settle_max_step_rad,
                self.settle_sample_sec,
            )
            for i in range(self.settle_iters):
                mean_pair = self._measure_mean_state_while_holding(
                    cmd_left,
                    cmd_right,
                    self.settle_sample_sec,
                    rate,
                )
                if mean_pair is None:
                    rospy.logwarn("startup_zero_via_topic settle: no state samples, stop settle loop.")
                    break
                left_mean, right_mean = mean_pair
                left_err = target_left - left_mean
                right_err = target_right - right_mean
                left_max_abs = float(np.max(np.abs(left_err[:eval_end])))
                right_max_abs = float(np.max(np.abs(right_err[:eval_end])))
                rospy.loginfo(
                    "startup_zero_via_topic settle iter=%d left_max_abs=%.5f right_max_abs=%.5f",
                    i + 1,
                    left_max_abs,
                    right_max_abs,
                )
                if left_max_abs <= self.settle_tol_rad and right_max_abs <= self.settle_tol_rad:
                    rospy.loginfo("startup_zero_via_topic settle converged.")
                    break

                left_step = np.clip(self.settle_gain * left_err, -self.settle_max_step_rad, self.settle_max_step_rad)
                right_step = np.clip(
                    self.settle_gain * right_err,
                    -self.settle_max_step_rad,
                    self.settle_max_step_rad,
                )
                if not self.settle_include_gripper:
                    left_step[6] = 0.0
                    right_step[6] = 0.0
                cmd_left = (cmd_left + left_step).astype(np.float32)
                cmd_right = (cmd_right + right_step).astype(np.float32)

        if self.settle_hold_sec > 0.0 and not rospy.is_shutdown():
            hold_deadline = rospy.Time.now().to_sec() + self.settle_hold_sec
            while not rospy.is_shutdown() and rospy.Time.now().to_sec() < hold_deadline:
                self._publish_pair(cmd_left, cmd_right)
                rate.sleep()

        rospy.loginfo("startup_zero_via_topic completed and exiting.")


def main() -> int:
    rospy.init_node("startup_zero_via_topic")
    node = StartupZeroViaTopic()
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
