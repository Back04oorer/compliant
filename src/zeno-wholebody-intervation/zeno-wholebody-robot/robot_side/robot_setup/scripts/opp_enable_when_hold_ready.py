#!/usr/bin/env python3
"""Enable the opp arm only after its hold stream is ready."""

from __future__ import annotations

import argparse
import sys
import time

import rospy
from sensor_msgs.msg import JointState

from piper_msgs.srv import Enable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-name", default="opp_enable_when_hold_ready")
    parser.add_argument("--robot-state-topic", default="/robot/arm_opp/joint_states_single")
    parser.add_argument("--teleop-topic", default="/teleop/arm_opp/joint_states_single")
    parser.add_argument("--enable-service", default="/robot/arm_opp/enable_srv")
    parser.add_argument("--wait-timeout-sec", type=float, default=8.0)
    parser.add_argument("--settle-sec", type=float, default=0.3)
    argv = [arg for arg in sys.argv[1:] if ":=" not in arg]
    return parser.parse_args(argv)


def wait_for_joint_state(topic: str, timeout_sec: float, label: str) -> JointState:
    deadline = time.time() + timeout_sec
    while not rospy.is_shutdown():
        remaining = deadline - time.time()
        if remaining <= 0.0:
            raise TimeoutError(f"Timed out waiting for {label}: {topic}")
        try:
            msg = rospy.wait_for_message(topic, JointState, timeout=min(remaining, 1.0))
        except rospy.ROSException:
            continue
        if len(msg.position) >= 6:
            return msg
    raise RuntimeError(f"ROS shutdown while waiting for {label}: {topic}")


def main() -> int:
    args = parse_args()
    if float(args.wait_timeout_sec) <= 0.0:
        raise SystemExit("--wait-timeout-sec must be > 0")
    if float(args.settle_sec) < 0.0:
        raise SystemExit("--settle-sec must be >= 0")

    rospy.init_node(str(args.node_name), anonymous=False)

    rospy.loginfo("Waiting for opp robot state: %s", args.robot_state_topic)
    wait_for_joint_state(str(args.robot_state_topic), float(args.wait_timeout_sec), "opp robot state")

    rospy.loginfo("Waiting for opp hold topic: %s", args.teleop_topic)
    wait_for_joint_state(str(args.teleop_topic), float(args.wait_timeout_sec), "opp hold topic")

    if float(args.settle_sec) > 0.0:
        rospy.sleep(float(args.settle_sec))

    rospy.loginfo("Waiting for opp enable service: %s", args.enable_service)
    rospy.wait_for_service(str(args.enable_service), timeout=float(args.wait_timeout_sec))
    enable_srv = rospy.ServiceProxy(str(args.enable_service), Enable)

    rospy.loginfo("Enabling opp arm after hold topic is ready")
    response = enable_srv(True)
    if not bool(response.enable_response):
        rospy.logerr("opp enable service returned failure")
        return 1

    rospy.loginfo("opp arm enabled successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
