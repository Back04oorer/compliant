#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from sensor_msgs.msg import JointState

def joint_state_callback(msg: JointState) -> None:
    msg.name = msg.name[:-1]
    msg.position = msg.position[:-1]
    msg.velocity = msg.velocity[:-1]
    msg.effort = msg.effort[:-1]
    joint_state_pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("joint_states_remap", anonymous=True)

    joint_state_sub_topic = rospy.get_param("~joint_state_sub_topic", "/robot/arm_left/joint_states_single")
    joint_state_pub_topic = rospy.get_param("~joint_state_pub_topic", "/robot/arm_left/joint_states_single_no_gripper")

    joint_state_pub = rospy.Publisher(joint_state_pub_topic, JointState, queue_size=10)
    joint_state_sub = rospy.Subscriber(joint_state_sub_topic, JointState, joint_state_callback)

    rospy.spin()