#!/usr/bin/env python3
"""Follower-only gravity feedforward; does not publish position/CAN commands.

Run alongside the guarded follower MIT adapter. Output effort is in the
driver's t_ref units after per-joint scaling, not necessarily SI joint torque.
The output timestamp preserves source-state age, including during fade-out.
"""
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

# Match the Pinocchio installation used by the existing gravity node.
sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')
import pinocchio as pin

NAMES = ['joint%d' % i for i in range(1, 7)]


def vector6(value, label, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError('%s must contain six finite values' % label)
    if positive and np.any(result <= 0):
        raise ValueError('%s must be positive' % label)
    return result


def read_positions(msg):
    if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
        raise ValueError('state needs unique names matching positions')
    values = dict(zip(msg.name, msg.position))
    q = np.array([values[n] for n in NAMES], dtype=float)
    width = float(values.get('gripper', 0.0))
    if not np.all(np.isfinite(q)) or not math.isfinite(width):
        raise ValueError('nonfinite state')
    if np.any(np.abs(q) > 2 * np.pi) or not -0.001 <= width <= 0.081:
        raise ValueError('state outside expected radian/metre domain')
    return q, max(0.0, width)


class GravityModel:
    def __init__(self, path, gravity):
        self.model = pin.buildModelFromUrdf(path)
        self.data = self.model.createData()
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError('gravity_vector must contain three finite values')
        self.model.gravity.linear = gravity
        self.joints = []
        for name in NAMES + ['joint7', 'joint8']:
            if not self.model.existJointName(name):
                raise ValueError('URDF missing ' + name)
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError('expected 1-DOF joint: ' + name)
            self.joints.append(joint)

    def compute(self, angles, width):
        q = pin.neutral(self.model)
        for joint, angle in zip(self.joints[:6], angles):
            q[joint.idx_q] = angle
        # The state reports total jaw opening; URDF has two opposing fingers.
        for joint, pos in zip(self.joints[6:], [width / 2, -width / 2]):
            idx = joint.idx_q
            q[idx] = np.clip(pos, self.model.lowerPositionLimit[idx], self.model.upperPositionLimit[idx])
        tau = pin.computeGeneralizedGravity(self.model, self.data, q)
        return np.array([tau[j.idx_v] for j in self.joints[:6]])


class FollowerArm:
    def __init__(self, side, path):
        self.side = side
        self.model = GravityModel(path, rospy.get_param('~gravity_vector_' + side, [0., 0., -9.81]))
        self.scale = vector6(rospy.get_param('~joint_scale_' + side), 'joint_scale_' + side)
        self.limits = vector6(rospy.get_param('~max_abs_t_ref'), 'max_abs_t_ref', True)
        self.slew = vector6(rospy.get_param('~max_t_ref_rate'), 'max_t_ref_rate', True)
        self.timeout = float(rospy.get_param('~state_timeout_sec', 0.10))
        self.ramp_sec = float(rospy.get_param('~ramp_sec', 0.5))
        if not all(math.isfinite(x) and x > 0 for x in [self.timeout, self.ramp_sec]):
            raise ValueError('timeout and ramp must be finite and positive')
        self.lock = threading.Lock()
        self.sample = None
        self.last_stamp = None
        self.output = np.zeros(6)
        self.valid_since = None
        self.last_tick = time.monotonic()
        prefix = '/robot/arm_%s/' % side
        self.pub = rospy.Publisher(prefix + 'follower_gravity_cmd', JointState, queue_size=1)
        self.ready = rospy.Publisher(prefix + 'follower_gravity_ready', Bool, queue_size=1)
        self.sub = rospy.Subscriber(prefix + 'joint_states_single', JointState, self.receive,
                                    queue_size=1, tcp_nodelay=True)

    def receive(self, msg):
        try:
            angles, width = read_positions(msg)
            stamp = msg.header.stamp.to_sec()
            age = rospy.Time.now().to_sec() - stamp
            if stamp <= 0 or not -0.02 <= age <= self.timeout:
                raise ValueError('stale/future state timestamp')
            with self.lock:
                if self.last_stamp is not None and stamp <= self.last_stamp:
                    raise ValueError('non-increasing state timestamp')
                self.last_stamp = stamp
                self.sample = (angles, width, stamp, time.monotonic())
        except (ValueError, KeyError) as exc:
            with self.lock:
                self.sample = None
            rospy.logwarn_throttle(1.0, '[follower gravity %s] %s', self.side, str(exc))

    def tick(self):
        now = time.monotonic()
        dt = min(max(now - self.last_tick, 0.0), 0.05)
        self.last_tick = now
        with self.lock:
            sample = self.sample
        valid = (sample is not None and now - sample[3] <= self.timeout
                 and -0.02 <= rospy.Time.now().to_sec() - sample[2] <= self.timeout)
        target = np.zeros(6)
        if valid:
            try:
                target = self.model.compute(sample[0], sample[1]) * self.scale
                if not np.all(np.isfinite(target)):
                    raise ValueError('nonfinite gravity torque')
                if np.any(np.abs(target) > self.limits):
                    rospy.logwarn_throttle(1.0, '[follower gravity %s] t_ref clipped: %s', self.side, target)
                target = np.clip(target, -self.limits, self.limits)
                if self.valid_since is None:
                    self.valid_since = now
                target *= min((now - self.valid_since) / self.ramp_sec, 1.0)
            except Exception as exc:
                valid = False
                rospy.logerr_throttle(1.0, '[follower gravity %s] %s', self.side, str(exc))
        if not valid:
            self.valid_since = None
            target = np.zeros(6)
            rospy.logwarn_throttle(2.0, '[follower gravity %s] no fresh state; fading feedforward to zero', self.side)
        self.output += np.clip(target - self.output, -self.slew * dt, self.slew * dt)
        msg = JointState()
        # Never refresh an old state's timestamp: the consumer has its own
        # watchdog even if this calculator keeps publishing (or crashes).
        msg.header.stamp = rospy.Time.from_sec(sample[2]) if valid else rospy.Time(0)
        msg.name = NAMES
        msg.effort = self.output.tolist()
        self.pub.publish(msg)
        self.ready.publish(Bool(data=bool(valid and now - self.valid_since >= self.ramp_sec)))


def main():
    rospy.init_node('follower_gravity_compensation')
    path = os.path.join(rospkg.RosPack().get_path(rospy.get_param('~urdf_package', 'piper_description')),
                        rospy.get_param('~urdf_relpath', 'urdf/piper_description.urdf'))
    arms = [FollowerArm(side, path) for side in ('left', 'right')]
    rate = float(rospy.get_param('~rate_hz', 100.0))
    if not math.isfinite(rate) or not 20 <= rate <= 500:
        raise ValueError('rate_hz must be in [20,500]')
    rospy.loginfo('Follower gravity active: dedicated effort topics; MIT position gains remain unchanged')
    # Wall-time loop also detects stalled ROS time / missing feedback.
    while not rospy.is_shutdown():
        for arm in arms:
            arm.tick()
        time.sleep(1.0 / rate)


if __name__ == '__main__':
    main()
