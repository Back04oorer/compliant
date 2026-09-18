#!/usr/bin/env python3
"""Republish sparse opp-arm commands as a continuous teleop topic."""

from __future__ import annotations

import argparse
import sys
import threading
from typing import List, Optional

import rospy
from sensor_msgs.msg import JointState


DEFAULT_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def _normalize_names(raw: str) -> List[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    return names if len(names) == 7 else list(DEFAULT_JOINT_NAMES)


class CommandCache:
    def __init__(self, joint_names: List[str]) -> None:
        self.joint_names = joint_names
        self._lock = threading.Lock()
        self._hold_target: Optional[List[float]] = None
        self._seeded_from_robot_state = False

    def _to_vec(self, msg: JointState) -> Optional[List[float]]:
        if len(msg.position) < 6:
            return None
        pos = [float(v) for v in msg.position[:7]]
        if len(pos) < 7:
            pos += [0.0] * (7 - len(pos))
        return pos

    def cmd_callback(self, msg: JointState) -> None:
        pos = self._to_vec(msg)
        if pos is None:
            return
        with self._lock:
            self._hold_target = pos

    def robot_state_callback(self, msg: JointState) -> None:
        pos = self._to_vec(msg)
        if pos is None:
            return
        with self._lock:
            if self._hold_target is None:
                self._hold_target = pos
                self._seeded_from_robot_state = True

    def snapshot(self) -> Optional[List[float]]:
        with self._lock:
            return None if self._hold_target is None else list(self._hold_target)

    def consume_seeded_from_robot_state(self) -> bool:
        with self._lock:
            seeded = self._seeded_from_robot_state
            self._seeded_from_robot_state = False
            return seeded


def build_msg(joint_names: List[str], position: List[float]) -> JointState:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = joint_names
    msg.position = [float(v) for v in position[:7]]
    if len(msg.position) < 7:
        msg.position += [0.0] * (7 - len(msg.position))
    msg.velocity = [0.0] * len(joint_names)
    msg.effort = [0.0] * len(joint_names)
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-name", default="opp_command_state_republisher")
    parser.add_argument("--input-topic", default="/teleop/arm_opp/joint_states_opp")
    parser.add_argument("--robot-state-topic", default="/robot/arm_opp/joint_states_single")
    parser.add_argument("--output-topic", default="/teleop/arm_opp/joint_states_single")
    parser.add_argument("--joint-names", default=",".join(DEFAULT_JOINT_NAMES))
    parser.add_argument("--publish-rate", type=float, default=30.0)
    argv = [arg for arg in sys.argv[1:] if ":=" not in arg]
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if float(args.publish_rate) <= 0.0:
        raise SystemExit("--publish-rate must be > 0")

    joint_names = _normalize_names(args.joint_names)
    rospy.init_node(str(args.node_name), anonymous=False)

    cache = CommandCache(joint_names)
    rospy.Subscriber(str(args.input_topic), JointState, cache.cmd_callback, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber(
        str(args.robot_state_topic), JointState, cache.robot_state_callback, queue_size=1, tcp_nodelay=True
    )
    pub = rospy.Publisher(str(args.output_topic), JointState, queue_size=1)

    rospy.loginfo(
        "opp command republisher: input=%s robot_state=%s output=%s rate=%.1f",
        args.input_topic,
        args.robot_state_topic,
        args.output_topic,
        float(args.publish_rate),
    )

    rate = rospy.Rate(float(args.publish_rate))
    while not rospy.is_shutdown():
        if cache.consume_seeded_from_robot_state():
            rospy.loginfo("opp hold target seeded from current robot state")
        latest = cache.snapshot()
        if latest is not None:
            pub.publish(build_msg(joint_names, latest))
        rate.sleep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
