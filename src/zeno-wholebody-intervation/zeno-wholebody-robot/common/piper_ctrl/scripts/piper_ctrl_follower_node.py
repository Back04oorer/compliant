#!/usr/bin/env python3
"""Follower-only MIT feedforward adapter; baseline/leader driver is unchanged.

The original driver remains the only CAN command writer. Its position, Kp/Kd,
velocity, gripper, mux and service semantics are preserved. This adapter only
guards the gravity buffer immediately before the original MIT send path.
"""
import math
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from follower_gravity_guard import FeedforwardGuard, NAMES
from piper_ctrl_node import C_PiperCtrlNode


class C_PiperFollowerCtrlNode(C_PiperCtrlNode):
    def __init__(self):
        # Fail configuration checks BEFORE the base constructor connects CAN.
        if rospy.get_param('~ctrl_mode', 'mit') != 'mit':
            raise ValueError('follower gravity adapter must start in MIT mode')
        if not rospy.get_param('~mit/enable_gravity', False):
            raise ValueError('follower adapter requires mit/enable_gravity=true')
        if rospy.get_param('~mit/gravity_mix_mode', 'additive') != 'additive':
            raise ValueError('follower gravity supports additive mixing only')
        if rospy.get_param('~mit/enable_tor', False):
            raise ValueError('follower gravity must not mix external effort commands')
        prefix = rospy.get_param('~topic_prefix', '').rstrip('/') + '/'
        if prefix not in ('/robot/arm_left/', '/robot/arm_right/'):
            raise ValueError('follower adapter is restricted to the two robot arms')
        topic = rospy.get_param('~remap/joint_states_compensated_to', '')
        if topic != prefix + 'follower_gravity_cmd':
            raise ValueError('follower gravity requires its dedicated effort topic')
        self._gravity_guard = FeedforwardGuard(
            rospy.get_param('~follower_gravity/max_abs_t_ref'),
            rospy.get_param('~follower_gravity/max_t_ref_rate'),
            rospy.get_param('~follower_gravity/state_timeout_sec', 0.10),
            rospy.get_param('~follower_gravity/ramp_sec', 0.5))
        self._can_stamp = None
        self._can_advanced_at = None
        self._gravity_reason = None
        # The guarded callback may fire during super().__init__.
        super().__init__()
        self._gravity_applied = rospy.Publisher(prefix + 'follower_gravity_applied', JointState, queue_size=1)
        self._gravity_active = rospy.Publisher(prefix + 'follower_gravity_active', Bool, queue_size=1)
        rospy.logwarn(
            'Follower gravity ENABLED: kp=%s kd=%s; additive t_ref limits=%s, '
            'slew=%s/s, source watchdog=%.3fs, ramp=%.2fs. '
            'Model/scales are initial estimates: validate unloaded before policy use.',
            self.mit_kp, self.mit_kd, self._gravity_guard.limits, self._gravity_guard.slew,
            self._gravity_guard.timeout, self._gravity_guard.ramp_sec)

    def gravity_torque_callback(self, msg):
        if msg.position or msg.velocity:
            self._gravity_guard.invalidate()
            rospy.logwarn_throttle(1., 'Follower gravity rejected: expected effort-only message')
            return
        ok, reason = self._gravity_guard.receive(
            msg.name, msg.effort, msg.header.stamp.to_sec(), rospy.Time.now().to_sec(), time.monotonic())
        if not ok:
            rospy.logwarn_throttle(1., 'Follower gravity rejected: %s', reason)

    def _can_feedback_fresh(self, now):
        # joint_states_single is stamped at publication by the base driver,
        # so ROS freshness alone cannot detect a completely stalled CAN feed.
        # Public SDK aggregate timestamp proves traffic, not each joint's age.
        try:
            stamp = float(self.piper.GetArmJointMsgs().time_stamp)
            if not math.isfinite(stamp) or stamp <= 0:
                raise ValueError('invalid SDK timestamp')
            if self._can_stamp is not None and stamp > self._can_stamp:
                self._can_advanced_at = now
            elif self._can_stamp is not None and stamp < self._can_stamp:
                self._can_advanced_at = None
            self._can_stamp = stamp
            return (self._can_advanced_at is not None
                    and 0 <= now - self._can_advanced_at <= self._gravity_guard.timeout)
        except Exception:
            self._can_advanced_at = None
            return False

    def _update_gravity(self, active):
        now = time.monotonic()
        effort, ready, reason = self._gravity_guard.step(
            now, rospy.Time.now().to_sec(), active=active,
            feedback_ok=self._can_feedback_fresh(now) if active else False)
        with self.gravity_torques_lock:
            self.gravity_torques = effort
        if reason != self._gravity_reason:
            rospy.loginfo('Follower gravity %s: %s; applied t_ref=%s', self.topic_prefix, reason, effort)
            self._gravity_reason = reason
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = NAMES
        msg.effort = effort
        # Diagnostic transport must not interrupt the sole CAN send loop.
        try:
            self._gravity_applied.publish(msg)
            self._gravity_active.publish(Bool(data=ready))
        except rospy.ROSException as exc:
            rospy.logwarn_throttle(1., 'Follower gravity diagnostics unavailable: %s', exc)

    def _mit_control(self):
        self._update_gravity(active=True)
        return super()._mit_control()

    def ControlThread(self):
        # Same cadence/routing as the baseline; observe inactive cycles too so
        # disable -> enable and P -> MIT restart the feedforward ramp at zero.
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown() and self.control_thread_running:
            try:
                if not self.GetEnableFlag():
                    self._update_gravity(active=False)
                elif self.ctrl_mode == 'p':
                    self._update_gravity(active=False)
                    self._position_control()
                else:
                    self._mit_control()
            except Exception as exc:
                rospy.logerr('Error in follower control thread: %s', str(exc))
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('piper_ctrl_follower_node')
    C_PiperFollowerCtrlNode().Run()
