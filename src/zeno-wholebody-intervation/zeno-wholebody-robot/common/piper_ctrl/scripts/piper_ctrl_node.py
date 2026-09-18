#!/usr/bin/env python3
# -*-coding:utf8-*-
from typing import Optional
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from piper_agent_diagnostics import DriverDiagnostics  # harness joint diagnostics v1
from std_srvs.srv import SetBool, SetBoolResponse
import time
import threading
import math
import numpy as np
from piper_sdk import *
from piper_sdk import C_PiperInterface
from std_srvs.srv import Trigger, TriggerResponse
from piper_msgs.msg import PiperStatusMsg, PiperEulerPose
from piper_msgs.srv import Enable, EnableResponse
from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.srv import GoZero, GoZeroResponse
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def check_ros_master():
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


class C_PiperCtrlNode:
    """Unified robot arm control node supporting position and MIT control modes"""

    def __init__(self) -> None:
        check_ros_master()
        if not rospy.get_node_uri():
            rospy.init_node("piper_ctrl_node", anonymous=True)

        # Basic parameters
        self.can_port = rospy.get_param("~can_port", "can0")
        rospy.loginfo("CAN port: %s", self.can_port)

        self.topic_prefix = rospy.get_param("~topic_prefix", "/")
        if not self.topic_prefix.endswith("/"):
            self.topic_prefix += "/"
        rospy.loginfo("Topic prefix: %s", self.topic_prefix)

        self.auto_enable = rospy.get_param("~auto_enable", False)
        rospy.loginfo("Auto enable: %s", self.auto_enable)

        self.gripper_exist = rospy.get_param("~gripper_exist", True)
        rospy.loginfo("Gripper exist: %s", self.gripper_exist)

        self.gripper_val_mutiple = rospy.get_param("~gripper_val_mutiple", 1.0)
        if self.gripper_val_mutiple <= 0:
            rospy.logwarn("Invalid gripper_val_mutiple value: must be positive. Using default value of 1.")
            self.gripper_val_mutiple = 1.0
        rospy.loginfo("Gripper value multiple: %.2f", self.gripper_val_mutiple)

        self.enable_gripper = rospy.get_param("~enable_gripper", True)
        rospy.loginfo("Enable gripper: %s", self.enable_gripper)

        self.enable_gripper_haptic = rospy.get_param("~enable_gripper_haptic", False)
        rospy.loginfo("Enable gripper haptic: %s", self.enable_gripper_haptic)

        self.gripper_range = rospy.get_param("~gripper_range", 0.1)
        rospy.loginfo("Gripper range: %.2f", self.gripper_range)

        self.gripper_reverse = rospy.get_param("~gripper_reverse", False)
        rospy.loginfo("Gripper reverse: %s", self.gripper_reverse)

        # Gripper command shaping/limit parameters (raw unit, 1e-6 m after scaling).
        self.gripper_cmd_limit_enable = bool(rospy.get_param("~gripper_cmd_limit_enable", True))
        self.gripper_cmd_min_raw = int(rospy.get_param("~gripper_cmd_min_raw", 0))
        self.gripper_cmd_max_raw = int(rospy.get_param("~gripper_cmd_max_raw", 80000))
        if self.gripper_cmd_max_raw < self.gripper_cmd_min_raw:
            rospy.logwarn(
                "Invalid gripper raw range [%d, %d], swapping.",
                self.gripper_cmd_min_raw,
                self.gripper_cmd_max_raw,
            )
            self.gripper_cmd_min_raw, self.gripper_cmd_max_raw = (
                self.gripper_cmd_max_raw,
                self.gripper_cmd_min_raw,
            )
        self.gripper_cmd_deadband_raw = max(0, int(rospy.get_param("~gripper_cmd_deadband_raw", 200)))
        rospy.loginfo(
            "Gripper cmd shaping: limit_enable=%s raw_range=[%d,%d] deadband_raw=%d",
            self.gripper_cmd_limit_enable,
            self.gripper_cmd_min_raw,
            self.gripper_cmd_max_raw,
            self.gripper_cmd_deadband_raw,
        )

        # Control mode parameters
        self.ctrl_mode = rospy.get_param("~ctrl_mode", "p").lower()
        if self.ctrl_mode not in ["p", "mit"]:
            rospy.logwarn("Invalid ctrl_mode: %s. Using default 'p'", self.ctrl_mode)
            self.ctrl_mode = "p"
        rospy.loginfo("Control mode: %s", self.ctrl_mode)

        # Position control parameters
        self.p_speed = rospy.get_param("~p/speed", 50)
        self.p_speed = max(0, min(100, self.p_speed))
        rospy.loginfo("Position control speed: %d", self.p_speed)

        # MIT control parameters
        self.mit_speed = rospy.get_param("~mit/speed", 50)
        self.mit_speed = max(0, min(100, self.mit_speed))

        # kp parameter: can be a single value or a list of 6 values
        kp_param = rospy.get_param("~mit/kp", 10.0)
        if isinstance(kp_param, list):
            if len(kp_param) == 6:
                self.mit_kp = list(kp_param)
            else:
                rospy.logwarn("kp list length should be 6. Using first value for all joints.")
                self.mit_kp = [kp_param[0] if len(kp_param) > 0 else 10.0] * 6
        else:
            self.mit_kp = [float(kp_param)] * 6

        # kd parameter: can be a single value or a list of 6 values
        kd_param = rospy.get_param("~mit/kd", 0.8)
        if isinstance(kd_param, list):
            if len(kd_param) == 6:
                self.mit_kd = list(kd_param)
            else:
                rospy.logwarn("kd list length should be 6. Using first value for all joints.")
                self.mit_kd = [kd_param[0] if len(kd_param) > 0 else 0.8] * 6
        else:
            self.mit_kd = [float(kd_param)] * 6

        self.mit_enable_pos = rospy.get_param("~mit/enable_pos", True)
        self.mit_enable_vel = rospy.get_param("~mit/enable_vel", False)
        self.mit_enable_tor = rospy.get_param("~mit/enable_tor", False)
        self.mit_enable_gravity = rospy.get_param("~mit/enable_gravity", False)
        self.mit_gravity_mix_mode = rospy.get_param("~mit/gravity_mix_mode", "additive")
        if self.mit_gravity_mix_mode not in ["additive", "legacy_diff", "gravity_only"]:
            rospy.logwarn(
                "Invalid ~mit/gravity_mix_mode=%s, fallback to 'additive'",
                self.mit_gravity_mix_mode,
            )
            self.mit_gravity_mix_mode = "additive"
        torque_scale_param = rospy.get_param("~mit/torque_scale", 1.0)
        if isinstance(torque_scale_param, list):
            if len(torque_scale_param) == 6:
                self.mit_torque_scale = list(torque_scale_param)
            else:
                rospy.logwarn("torque_scale list length should be 6. Using first value for all joints.")
                self.mit_torque_scale = [torque_scale_param[0] if len(torque_scale_param) > 0 else 1.0] * 6
        else:
            self.mit_torque_scale = [float(torque_scale_param)] * 6
        self.mit_max_torque_abs = float(rospy.get_param("~mit/max_torque_abs", 18.0))
        if self.mit_max_torque_abs <= 0.0:
            rospy.logwarn("~mit/max_torque_abs must be > 0. Fallback to 18.0")
            self.mit_max_torque_abs = 18.0
        rospy.loginfo("MIT control parameters: speed=%d", self.mit_speed)
        rospy.loginfo("MIT kp: %s", self.mit_kp)
        rospy.loginfo("MIT kd: %s", self.mit_kd)
        rospy.loginfo(
            "MIT enable flags: pos=%s, vel=%s, tor=%s, gravity=%s",
            self.mit_enable_pos,
            self.mit_enable_vel,
            self.mit_enable_tor,
            self.mit_enable_gravity,
        )
        rospy.loginfo("MIT torque scale: %s", self.mit_torque_scale)
        rospy.loginfo("MIT gravity mix mode: %s", self.mit_gravity_mix_mode)
        rospy.loginfo("MIT max torque abs: %.2f", self.mit_max_torque_abs)

        # Topic name parameters
        def ensure_topic_prefix(param_name, default_suffix):
            """Ensure topic name uses topic_prefix as prefix"""
            if rospy.has_param(param_name):
                topic = rospy.get_param(param_name)
                topic = topic.lstrip("/")
                return self.topic_prefix + topic
            else:
                return self.topic_prefix + default_suffix

        def get_remap_topic(remap_param_name, default_topic):
            """Get remapped topic name from parameter, or use default if not set.
            
            If remap parameter is set, use it directly as the full topic path.
            Otherwise, use the default topic (typically topic_prefix + suffix).
            """
            if rospy.has_param(remap_param_name):
                return rospy.get_param(remap_param_name)
            return default_topic

        # Build default topic names (topic_prefix + suffix)
        default_p_joint_pos_cmd_topic = ensure_topic_prefix("~p/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_pos_cmd_topic = ensure_topic_prefix("~mit/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_vel_cmd_topic = ensure_topic_prefix("~mit/joint_vel_cmd_topic", "joint_vel_cmd")
        default_mit_joint_tor_cmd_topic = ensure_topic_prefix("~mit/joint_tor_cmd_topic", "joint_tor_cmd")
        default_gripper_pos_cmd_topic = ensure_topic_prefix("~gripper_pos_cmd_topic", "gripper_pos_cmd")
        default_gripper_effort_cmd_topic = ensure_topic_prefix("~gripper_effort_cmd_topic", "gripper_effort_cmd")

        # Apply remap if configured (remap/xxx_to parameters override default topics)
        self.p_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_p_joint_pos_cmd_topic)
        self.mit_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_mit_joint_pos_cmd_topic)
        self.mit_joint_vel_cmd_topic = get_remap_topic("~remap/joint_vel_cmd_to", default_mit_joint_vel_cmd_topic)
        self.mit_joint_tor_cmd_topic = get_remap_topic("~remap/joint_tor_cmd_to", default_mit_joint_tor_cmd_topic)
        self.gripper_pos_cmd_topic = get_remap_topic("~remap/gripper_pos_cmd_to", default_gripper_pos_cmd_topic)
        self.gripper_effort_cmd_topic = get_remap_topic("~remap/gripper_effort_cmd_to", default_gripper_effort_cmd_topic)

        if self.ctrl_mode == "p":
            rospy.loginfo("Position control command topic: %s", self.p_joint_pos_cmd_topic)
        else:
            rospy.loginfo("MIT control command topics:")
            if self.mit_enable_pos:
                rospy.loginfo("  Position: %s", self.mit_joint_pos_cmd_topic)
            if self.mit_enable_vel:
                rospy.loginfo("  Velocity: %s", self.mit_joint_vel_cmd_topic)
            if self.mit_enable_tor:
                rospy.loginfo("  Torque: %s", self.mit_joint_tor_cmd_topic)

        if self.mit_enable_gravity:
            default_joint_states_compensated_topic = ensure_topic_prefix(
                "~mit/joint_states_compensated_topic", "joint_states_compensated"
            )
            self.mit_joint_states_compensated_topic = get_remap_topic(
                "~remap/joint_states_compensated_to", default_joint_states_compensated_topic
            )
            rospy.loginfo("Gravity compensation topic: %s", self.mit_joint_states_compensated_topic)

        # Log gripper topics
        if self.enable_gripper:
            rospy.loginfo("Gripper position command topic: %s", self.gripper_pos_cmd_topic)
            rospy.loginfo("Gripper effort command topic: %s", self.gripper_effort_cmd_topic)

        # Thread rate parameters
        self.publish_rate = rospy.get_param("~publish_rate", 200.0)
        self.control_rate = rospy.get_param("~control_rate", 200.0)
        self.subscribe_rate = rospy.get_param("~subscribe_rate", 100.0)
        rospy.loginfo(
            "Thread rates: publish=%.1f Hz, control=%.1f Hz, subscribe=%.1f Hz",
            self.publish_rate,
            self.control_rate,
            self.subscribe_rate,
        )

        # Filter parameters
        self.filter_enable = rospy.get_param("~filter/enable", True)
        self.filter_alpha_pos = rospy.get_param("~filter/alpha_position", 0.7)
        self.filter_alpha_vel = rospy.get_param("~filter/alpha_velocity", 0.5)
        self.filter_alpha_effort = rospy.get_param("~filter/alpha_effort", 0.5)
        # Clamp alpha values to [0, 1]
        self.filter_alpha_pos = max(0.0, min(1.0, self.filter_alpha_pos))
        self.filter_alpha_vel = max(0.0, min(1.0, self.filter_alpha_vel))
        self.filter_alpha_effort = max(0.0, min(1.0, self.filter_alpha_effort))
        rospy.loginfo(
            "Filter enabled: %s, alpha: pos=%.2f, vel=%.2f, effort=%.2f",
            self.filter_enable,
            self.filter_alpha_pos,
            self.filter_alpha_vel,
            self.filter_alpha_effort,
        )

        # Create piper interface
        self._agent_diagnostics = DriverDiagnostics(self.topic_prefix)
        self.piper = self._agent_diagnostics.interface(C_PiperInterface)(can_name=self.can_port)
        self.piper.ConnectPort()

        if self.ctrl_mode == "mit":
            self.piper.MotionCtrl_2(0x01, 0x04, self.mit_speed, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)

        self.block_ctrl_flag = False
        self.__enable_flag = False

        # Publishers
        self.joint_pub = rospy.Publisher(self.topic_prefix + "joint_states_single", JointState, queue_size=1)
        self.arm_status_pub = rospy.Publisher(self.topic_prefix + "arm_status", PiperStatusMsg, queue_size=1)
        self.end_pose_pub = rospy.Publisher(self.topic_prefix + "end_pose", PoseStamped, queue_size=1)
        self.end_pose_euler_pub = rospy.Publisher(self.topic_prefix + "end_pose_euler", PiperEulerPose, queue_size=1)

        # Services
        self.enable_service = rospy.Service(self.topic_prefix + "enable_srv", Enable, self.handle_enable_service)
        self.gripper_service = rospy.Service(self.topic_prefix + "gripper_srv", Gripper, self.handle_gripper_service)
        self.stop_service = rospy.Service(self.topic_prefix + "stop_srv", Trigger, self.handle_stop_service)
        self.reset_service = rospy.Service(self.topic_prefix + "reset_srv", Trigger, self.handle_reset_service)
        self.go_zero_service = rospy.Service(self.topic_prefix + "go_zero_srv", GoZero, self.handle_go_zero_service)
        self.block_arm_service = rospy.Service(self.topic_prefix + "block_arm", SetBool, self.handle_block_arm_service)
        self.set_ctrl_mode_service = rospy.Service(
            self.topic_prefix + "set_ctrl_mode_srv", SetBool, self.handle_set_ctrl_mode_service
        )

        # Joint state message
        self.joint_states = JointState()
        self.joint_states.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        self.joint_states.position = [0.0] * 7
        self.joint_states.velocity = [0.0] * 7
        self.joint_states.effort = [0.0] * 7

        # Control data storage (thread-safe)
        self.control_data_lock = threading.Lock()
        self.joint_positions_cmd = [0.0] * 7
        self.joint_velocities_cmd = [0.0] * 6
        self.joint_torques_cmd = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        self.current_joint_efforts = [0.0] * 6
        self.gravity_torques_lock = threading.Lock()
        self.gravity_torques = [0.0] * 6

        # Gripper command data storage (thread-safe)
        self.gripper_cmd_lock = threading.Lock()
        self.gripper_cmd_position = 0.0
        self.gripper_cmd_effort = 0.0

        # Filter state storage (thread-safe)
        self.filter_lock = threading.Lock()
        self.filtered_positions = [0.0] * 7
        self.filtered_velocities = [0.0] * 7
        self.filtered_efforts = [0.0] * 7
        self.filter_initialized = False

        # Subscribers
        if self.ctrl_mode == "p":
            rospy.Subscriber(
                self.p_joint_pos_cmd_topic, JointState, self.joint_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
            rospy.loginfo("Subscribing to position command: %s", self.p_joint_pos_cmd_topic)
        else:
            if self.mit_enable_pos:
                rospy.Subscriber(
                    self.mit_joint_pos_cmd_topic,
                    JointState,
                    self.joint_pos_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to position command: %s", self.mit_joint_pos_cmd_topic)

            if self.mit_enable_vel:
                rospy.Subscriber(
                    self.mit_joint_vel_cmd_topic,
                    JointState,
                    self.joint_vel_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to velocity command: %s", self.mit_joint_vel_cmd_topic)

            if self.mit_enable_tor:
                rospy.Subscriber(
                    self.mit_joint_tor_cmd_topic,
                    JointState,
                    self.joint_tor_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to torque command: %s", self.mit_joint_tor_cmd_topic)

            if self.mit_enable_gravity:
                rospy.Subscriber(
                    self.mit_joint_states_compensated_topic,
                    JointState,
                    self.gravity_torque_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to gravity compensation: %s", self.mit_joint_states_compensated_topic)

        rospy.Subscriber(self.topic_prefix + "enable_flag", Bool, self.enable_callback, queue_size=1, tcp_nodelay=True)

        # Gripper command subscriber
        if self.enable_gripper:
            rospy.Subscriber(
                self.gripper_pos_cmd_topic, JointState, self.gripper_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
            rospy.loginfo("Subscribing to gripper position command: %s", self.gripper_pos_cmd_topic)

            rospy.Subscriber(
                self.gripper_effort_cmd_topic,
                JointState,
                self.gripper_effort_cmd_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo("Subscribing to gripper effort command: %s", self.gripper_effort_cmd_topic)

        # Thread control flags
        self.publish_thread_running = False
        self.control_thread_running = False
        self.publish_thread = None
        self.control_thread = None

    def GetEnableFlag(self):
        return self.__enable_flag

    # Callback functions
    def joint_pos_cmd_callback(self, msg: JointState):
        """Position command callback"""
        if not self.block_ctrl_flag and len(msg.position) >= 6:
            with self.control_data_lock:
                self.joint_positions_cmd[:6] = list(msg.position[:6])
                if len(msg.position) >= 7:
                    self.joint_positions_cmd[6] = msg.position[6]

    def joint_vel_cmd_callback(self, msg: JointState):
        """Velocity command callback"""
        if not self.block_ctrl_flag and len(msg.velocity) >= 6:
            with self.control_data_lock:
                self.joint_velocities_cmd = list(msg.velocity[:6])

    def joint_tor_cmd_callback(self, msg: JointState):
        """Torque command callback"""
        if not self.block_ctrl_flag and len(msg.effort) >= 6:
            with self.control_data_lock:
                self.joint_torques_cmd = list(msg.effort[:6])

    def gravity_torque_callback(self, msg: JointState):
        """Gravity compensation torque callback"""
        if len(msg.effort) >= 6:
            with self.gravity_torques_lock:
                self.gravity_torques = list(msg.effort[:6])

    def gripper_pos_cmd_callback(self, msg: JointState):
        """Gripper command callback"""
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.position) >= 7:
                    self.gripper_cmd_position = msg.position[6]

    def gripper_effort_cmd_callback(self, msg: JointState):
        """Gripper effort command callback"""
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.effort) >= 7:
                    self.gripper_cmd_effort = msg.effort[6]

    def enable_callback(self, enable_flag: Bool):
        """Enable callback"""
        rospy.loginfo("Received enable flag: %s", enable_flag.data)
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x00, 0)

    # Control thread
    def ControlThread(self):
        """Control thread"""
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown() and self.control_thread_running:
            if not self.__enable_flag:
                rate.sleep()
                continue

            try:
                if self.ctrl_mode == "p":
                    self._position_control()
                else:
                    self._mit_control()
            except Exception as e:
                rospy.logerr("Error in control thread: %s", str(e))

            rate.sleep()

    def _sanitize_gripper_raw_cmd(self, joint_6_raw: int) -> int:
        joint_6 = int(joint_6_raw)
        if self.gripper_cmd_limit_enable:
            joint_6 = max(self.gripper_cmd_min_raw, min(self.gripper_cmd_max_raw, joint_6))
        if self.gripper_cmd_deadband_raw > 0 and abs(joint_6) < self.gripper_cmd_deadband_raw:
            joint_6 = 0
        return joint_6

    def _position_control(self):
        """Position control implementation"""
        with self.control_data_lock:
            positions = self.joint_positions_cmd[:]
            has_gripper = len(self.joint_positions_cmd) >= 7
            gripper_pos = self.joint_positions_cmd[6] if has_gripper else 0.0

        for i in range(6):
            positions[i] = max(-12.5, min(12.5, positions[i]))

        factor = 1000 * 180 / np.pi
        joint_0 = round(positions[0] * factor)
        joint_1 = round(positions[1] * factor)
        joint_2 = round(positions[2] * factor)
        joint_3 = round(positions[3] * factor)
        joint_4 = round(positions[4] * factor)
        joint_5 = round(positions[5] * factor)

        # self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        if self.enable_gripper and self.gripper_exist:
            if has_gripper:
                joint_6 = round(gripper_pos * 1000 * 1000 * self.gripper_val_mutiple)
            else:
                # Use gripper command if available
                with self.gripper_cmd_lock:
                    joint_6 = round(self.gripper_cmd_position * 1000 * 1000 * self.gripper_val_mutiple)

            joint_6 = self._sanitize_gripper_raw_cmd(joint_6)

            # Get effort from gripper command if haptic is enabled
            if self.enable_gripper_haptic:
                with self.gripper_cmd_lock:
                    gripper_effort = abs(self.gripper_cmd_effort)
                    gripper_effort = max(0, min(gripper_effort, 3))
                    gripper_effort = round(gripper_effort * 1000)
            else:
                gripper_effort = round(1000 * 1.0)

            self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)

    def _mit_control(self):
        """MIT control implementation"""
        with self.control_data_lock:
            if self.mit_enable_pos:
                positions = self.joint_positions_cmd[:]
            else:
                positions = self.current_joint_positions[:]

            if self.mit_enable_vel:
                velocities = self.joint_velocities_cmd[:]
            else:
                velocities = [0.0] * 6

            if self.mit_enable_tor:
                torques = self.joint_torques_cmd[:]
            else:
                torques = [0.0] * 6

        gravity_torques = [0.0] * 6
        if self.mit_enable_gravity:
            with self.gravity_torques_lock:
                gravity_torques = self.gravity_torques[:]

            for i in range(6):
                if self.mit_gravity_mix_mode == "legacy_diff":
                    torques[i] = -(torques[i] - gravity_torques[i]) * self.mit_torque_scale[i] + gravity_torques[i]
                elif self.mit_gravity_mix_mode == "gravity_only":
                    torques[i] = gravity_torques[i]
                else:
                    torques[i] = torques[i] * self.mit_torque_scale[i] + gravity_torques[i]
        else:
            for i in range(6):
                torques[i] = torques[i] * self.mit_torque_scale[i]

        torques_pre_clamp = torques[:]
        for i in range(6):
            positions[i] = max(-12.5, min(12.5, positions[i]))
            velocities[i] = max(-45.0, min(45.0, velocities[i]))
            torques[i] = max(-self.mit_max_torque_abs, min(self.mit_max_torque_abs, torques[i]))

        clipped_indices = [i for i in range(6) if abs(torques_pre_clamp[i] - torques[i]) > 1e-6]
        if clipped_indices:
            max_idx = max(clipped_indices, key=lambda j: abs(torques_pre_clamp[j] - torques[j]))
            rospy.logwarn_throttle(
                0.5,
                "MIT torque clamped on joint%d: raw=%.3f clamped=%.3f gravity=%.3f pos_ref=%.3f",
                max_idx + 1,
                float(torques_pre_clamp[max_idx]),
                float(torques[max_idx]),
                float(gravity_torques[max_idx]),
                float(positions[max_idx]),
            )

        try:
            for motor_num in range(1, 7):
                joint_idx = motor_num - 1
                self.piper.JointMitCtrl(
                    motor_num=motor_num,
                    pos_ref=positions[joint_idx],
                    vel_ref=velocities[joint_idx],
                    kp=self.mit_kp[joint_idx],
                    kd=self.mit_kd[joint_idx],
                    t_ref=torques[joint_idx],
                )
        except Exception as e:
            rospy.logerr("Error in JointMitCtrl: %s", str(e))

        # Gripper control for MIT mode
        if self.enable_gripper and self.gripper_exist:
            # Get gripper position from command
            with self.gripper_cmd_lock:
                gripper_pos = self.gripper_cmd_position

            joint_6 = round(gripper_pos * 1000 * 1000 * self.gripper_val_mutiple)
            joint_6 = self._sanitize_gripper_raw_cmd(joint_6)

            # Get effort from gripper command if haptic is enabled
            if self.enable_gripper_haptic:
                with self.gripper_cmd_lock:
                    gripper_effort = -self.gripper_cmd_effort
                    gripper_effort = max(0, min(gripper_effort, 0.5))
                    gripper_effort = round(gripper_effort * 1000)
            else:
                gripper_effort = round(3 * 1000)

            if gripper_effort < 50:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x00, 0)
            else:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)

    # Publish thread
    def PublishThread(self):
        """Publish thread"""
        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown() and self.publish_thread_running:
            self.PublishArmState()
            self.PublishArmEndPose()
            self.PublishArmJointAndGripper()
            self._agent_diagnostics.publish(self)
            rate.sleep()

    def PublishArmState(self):
        """Publish arm status"""
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        )
        arm_status.communication_status_joint_2 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        )
        arm_status.communication_status_joint_3 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        )
        arm_status.communication_status_joint_4 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        )
        arm_status.communication_status_joint_5 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        )
        arm_status.communication_status_joint_6 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        )
        self.arm_status_pub.publish(arm_status)

    def PublishArmJointAndGripper(self):
        """Publish joint states and gripper state"""
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.effort / 1000
        effort_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.effort / 1000
        effort_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.effort / 1000
        effort_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.effort / 1000
        effort_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.effort / 1000
        effort_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.effort / 1000

        with self.control_data_lock:
            self.current_joint_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            self.current_joint_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            self.current_joint_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        # Get gripper data
        if self.enable_gripper and self.gripper_exist:
            joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
            if self.gripper_reverse:
                joint_6 = self.gripper_range - joint_6
            effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]
        else:
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        # Apply filter if enabled
        if self.filter_enable:
            with self.filter_lock:
                if not self.filter_initialized:
                    # Initialize filter with current values
                    self.filtered_positions = raw_positions[:]
                    self.filtered_velocities = raw_velocities[:]
                    self.filtered_efforts = raw_efforts[:]
                    self.filter_initialized = True
                else:
                    # Apply low-pass filter (exponential moving average)
                    # filtered = alpha * new + (1 - alpha) * previous
                    for i in range(len(raw_positions)):
                        self.filtered_positions[i] = (
                            self.filter_alpha_pos * raw_positions[i]
                            + (1.0 - self.filter_alpha_pos) * self.filtered_positions[i]
                        )
                    for i in range(len(raw_velocities)):
                        self.filtered_velocities[i] = (
                            self.filter_alpha_vel * raw_velocities[i]
                            + (1.0 - self.filter_alpha_vel) * self.filtered_velocities[i]
                        )
                    for i in range(len(raw_efforts)):
                        self.filtered_efforts[i] = (
                            self.filter_alpha_effort * raw_efforts[i]
                            + (1.0 - self.filter_alpha_effort) * self.filtered_efforts[i]
                        )
                # Use filtered values
                filtered_positions = self.filtered_positions[:]
                filtered_velocities = self.filtered_velocities[:]
                filtered_efforts = self.filtered_efforts[:]
        else:
            # Use raw values without filtering
            filtered_positions = raw_positions[:]
            filtered_velocities = raw_velocities[:]
            filtered_efforts = raw_efforts[:]

        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_states.position = filtered_positions
        self.joint_states.velocity = filtered_velocities
        self.joint_states.effort = filtered_efforts

        self.joint_pub.publish(self.joint_states)

    def PublishArmEndPose(self):
        """Publish end effector pose"""
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)

        end_pose_euler = PiperEulerPose()
        end_pose_euler.header.stamp = rospy.Time.now()
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        self.end_pose_euler_pub.publish(end_pose_euler)

    # Service handlers
    def handle_gripper_service(self, req):
        response = GripperResponse()
        response.code = 15999
        response.status = False
        if self.gripper_exist:
            rospy.loginfo("-----------------------Gripper---------------------------")
            rospy.loginfo("Received request:")
            rospy.loginfo("PS: Piper should be enable. Please ensure piper is enable")
            rospy.loginfo("gripper_angle:%f, range is [0m, 0.07m]", req.gripper_angle)
            rospy.loginfo("gripper_effort:%f, range is [0.5N/m, 2N/m]", req.gripper_effort)
            rospy.loginfo(
                "gripper_code:%d, range is [0, 1, 2, 3]\n \
                0x00: Disable\n \
                0x01: Enable\n \
                0x03/0x02: Enable and clear error / Disable and clear error",
                req.gripper_code,
            )
            rospy.loginfo(
                "set_zero:%d, range is [0, 0xAE] \n \
                0x00: Invalid value \n \
                0xAE: Set zero point",
                req.set_zero,
            )
            rospy.loginfo("-----------------------Gripper---------------------------")
            gripper_angle = round(max(0, min(req.gripper_angle, 0.07)) * 1e6)
            gripper_effort = round(max(0.5, min(req.gripper_effort, 2)) * 1e3)
            if req.gripper_code not in [0x00, 0x01, 0x02, 0x03]:
                rospy.logwarn("gripper_code should be in [0, 1, 2, 3], default val is 1")
                gripper_code = 1
                response.code = 15901
            else:
                gripper_code = req.gripper_code
            if req.set_zero not in [0x00, 0xAE]:
                rospy.logwarn("set_zero should be in [0, 0xAE], default val is 0")
                set_zero = 0
                response.code = 15902
            else:
                set_zero = req.set_zero
            response.code = 15900
            self.piper.GripperCtrl(abs(gripper_angle), gripper_effort, gripper_code, set_zero)
            response.status = True
        else:
            rospy.logwarn("gripper_exist param is False.")
            response.code = 15903
            response.status = False
        rospy.loginfo("Returning GripperResponse: %d, %s", response.code, response.status)
        return response

    def handle_enable_service(self, req):
        rospy.loginfo("Received request: %s", req.enable_request)
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not loop_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if req.enable_request:
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
            rospy.loginfo("Enable status: %s", enable_flag)
            self.__enable_flag = enable_flag
            rospy.loginfo("--------------------")
            if enable_flag == req.enable_request:
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                rospy.logwarn("Timeout...")
                elapsed_time_flag = True
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        response = enable_flag
        rospy.loginfo("Returning response: %s", response)
        return EnableResponse(response)

    def handle_stop_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "stop piper failed"
        rospy.loginfo("-----------------------STOP---------------------------")
        rospy.loginfo("Stop piper.")
        rospy.loginfo("-----------------------STOP---------------------------")
        self.piper.MotionCtrl_1(0x01, 0, 0)
        response.success = True
        response.message = "stop piper success"
        rospy.loginfo("Returning StopResponse: %s, %s", response.success, response.message)
        return response

    def handle_reset_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "reset piper failed"
        rospy.loginfo("-----------------------RESET---------------------------")
        rospy.loginfo("reset piper.")
        rospy.loginfo("-----------------------RESET---------------------------")
        self.piper.MotionCtrl_1(0x02, 0, 0)
        response.success = True
        response.message = "reset piper success"
        rospy.loginfo("Returning resetResponse: %s, %s", response.success, response.message)
        return response

    def handle_go_zero_service(self, req):
        response = GoZeroResponse()
        response.status = False
        response.code = 151000
        rospy.loginfo("-----------------------GOZERO---------------------------")
        rospy.loginfo("piper go zero.")
        rospy.loginfo("-----------------------GOZERO---------------------------")
        if req.is_mit_mode:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        if req.is_mit_mode:
            # Keep behavior explicit: go_zero uses JointCtrl path first, then switch back to MIT.
            rospy.sleep(0.2)
            self.piper.MotionCtrl_2(0x01, 0x04, self.mit_speed, 0xAD)
            rospy.loginfo("Go-zero completed, switched back to MIT mode.")
        response.status = True
        response.code = 151001
        rospy.loginfo("Returning GoZeroResponse: %s, %d", response.status, response.code)
        return response

    def _apply_ctrl_mode(self, mode: str):
        target_mode = str(mode).strip().lower()
        if target_mode == "mit":
            self.piper.MotionCtrl_2(0x01, 0x04, self.mit_speed, 0xAD)
        elif target_mode == "p":
            self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)
        else:
            raise ValueError(f"Unsupported ctrl_mode: {mode}")
        self.ctrl_mode = target_mode
        rospy.set_param("~ctrl_mode", self.ctrl_mode)
        rospy.loginfo("Control mode switched to: %s", self.ctrl_mode)

    def handle_set_ctrl_mode_service(self, req):
        target_mode = "mit" if req.data else "p"
        prev_mode = self.ctrl_mode
        try:
            self._apply_ctrl_mode(target_mode)
            return SetBoolResponse(True, f"ctrl_mode switched: {prev_mode} -> {target_mode}")
        except Exception as exc:
            rospy.logerr("Failed to switch ctrl_mode from %s to %s: %s", prev_mode, target_mode, str(exc))
            return SetBoolResponse(False, f"switch failed: {str(exc)}")

    def handle_block_arm_service(self, req):
        response = SetBoolResponse()
        rospy.loginfo("-----------------------BLOCK_ARM---------------------------")
        if req.data:
            response.success = req.data
            response.message = "You will block arm ctrl msg send"
        else:
            response.success = req.data
            response.message = "You will unblock arm ctrl msg send"
        self.block_ctrl_flag = req.data
        rospy.loginfo("piper block arm.")
        rospy.loginfo("Returning BlockArmResponse: %s, %s", response.success, response.message)
        rospy.loginfo("-----------------------BLOCK_ARM---------------------------")
        return response

    # Enable function
    def EnableArm(self):
        """Enable arm"""
        enable_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------[Arm Enable]--------------------")
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            rospy.loginfo("Enable status: %s", enable_flag)
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.__enable_flag = True
            rospy.loginfo("--------------------[Arm Enable]--------------------")

            if elapsed_time > timeout:
                rospy.logwarn("Enable timeout")
                break
            time.sleep(1)

    # Run function
    def Run(self):
        """Run node"""
        if self.auto_enable:
            enable_thread = threading.Thread(target=self.EnableArm, daemon=True)
            enable_thread.start()

        self.publish_thread_running = True
        self.publish_thread = threading.Thread(target=self.PublishThread, daemon=True)
        self.publish_thread.start()
        rospy.loginfo("Publish thread started")

        self.control_thread_running = True
        self.control_thread = threading.Thread(target=self.ControlThread, daemon=True)
        self.control_thread.start()
        rospy.loginfo("Control thread started")

        rospy.spin()


if __name__ == "__main__":
    try:
        check_ros_master()
        node = C_PiperCtrlNode()
        node.Run()
    except rospy.ROSInterruptException:
        pass
