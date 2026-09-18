#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call Piper right-arm go_zero once, then refresh policy hold with feedback."""

from __future__ import annotations

import os
from typing import Optional

import rospy
from sensor_msgs.msg import JointState
from piper_msgs.srv import GoZero, GoZeroRequest


def _msg_to_q7(msg: JointState) -> Optional[list[float]]:
    if len(msg.position) < 6:
        return None
    q = [float(v) for v in msg.position[:6]]
    q.append(float(msg.position[6]) if len(msg.position) >= 7 else 0.0)
    return q


def _make_joint_state(q7: list[float]) -> JointState:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    msg.position = [float(v) for v in q7[:7]]
    msg.velocity = [0.0] * 7
    msg.effort = [0.0] * 7
    return msg


def _publish_feedback_hold(
    *,
    state_topic: str,
    cmd_topic: str,
    timeout_sec: float,
    duration_sec: float,
    rate_hz: float,
) -> None:
    if duration_sec <= 0.0:
        return
    try:
        msg = rospy.wait_for_message(state_topic, JointState, timeout=timeout_sec)
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logwarn("post-hold skipped: could not read %s: %s", state_topic, exc)
        return
    q7 = _msg_to_q7(msg)
    if q7 is None:
        rospy.logwarn("post-hold skipped: %s has fewer than 6 positions", state_topic)
        return

    pub = rospy.Publisher(cmd_topic, JointState, queue_size=1)
    rospy.sleep(0.25)
    rate = rospy.Rate(max(1.0, rate_hz))
    end_time = rospy.Time.now().to_sec() + duration_sec
    count = 0
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() < end_time:
        pub.publish(_make_joint_state(q7))
        count += 1
        rate.sleep()
    rospy.loginfo("post-hold published %d samples to %s: %s", count, cmd_topic, q7)


def main() -> int:
    rospy.init_node("right_arm_go_zero_once", anonymous=False)

    allow_motion = bool(rospy.get_param("~allow_motion", False))
    require_env = bool(rospy.get_param("~require_lfv_allow_env", True))
    if not allow_motion:
        rospy.logerr("Refusing go-zero: set launch arg allow_motion:=true")
        return 2
    if require_env and os.environ.get("LFV_ALLOW_ROBOT_MOTION", "0") != "1":
        rospy.logerr("Refusing go-zero: export LFV_ALLOW_ROBOT_MOTION=1 first")
        return 2

    service_name = str(rospy.get_param("~go_zero_service", "/robot/arm_right/go_zero_srv"))
    is_mit_mode = bool(rospy.get_param("~is_mit_mode", True))
    wait_timeout_sec = float(rospy.get_param("~wait_timeout_sec", 10.0))
    post_hold_sec = float(rospy.get_param("~post_hold_sec", 1.0))
    hold_rate_hz = float(rospy.get_param("~hold_rate_hz", 50.0))
    state_topic = str(rospy.get_param("~state_topic", "/robot/arm_right/joint_states_single"))
    cmd_topic = str(rospy.get_param("~cmd_topic", "/robot/arm_right/vla_joint_cmd"))

    rospy.loginfo("Right arm go-zero requested.")
    rospy.loginfo("service=%s is_mit_mode=%s", service_name, is_mit_mode)
    rospy.loginfo("state_topic=%s cmd_topic=%s", state_topic, cmd_topic)

    try:
        rospy.wait_for_service(service_name, timeout=wait_timeout_sec)
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logerr("Service not available: %s: %s", service_name, exc)
        return 1

    try:
        proxy = rospy.ServiceProxy(service_name, GoZero)
        response = proxy(GoZeroRequest(is_mit_mode=is_mit_mode))
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logerr("go_zero service call failed: %s", exc)
        return 1

    rospy.loginfo("go_zero response: status=%s code=%s", response.status, response.code)
    if not bool(response.status):
        return 1

    _publish_feedback_hold(
        state_topic=state_topic,
        cmd_topic=cmd_topic,
        timeout_sec=wait_timeout_sec,
        duration_sec=post_hold_sec,
        rate_hz=hold_rate_hz,
    )
    rospy.loginfo("Right arm go-zero completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
