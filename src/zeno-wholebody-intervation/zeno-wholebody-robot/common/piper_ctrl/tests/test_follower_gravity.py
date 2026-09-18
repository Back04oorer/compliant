#!/usr/bin/env python3
"""Offline tests. No node init, ROS master, driver construction, or real CAN.

Run in the sourced robot container so ROS/Pinocchio integration is tested too.
"""
import copy
import math
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from follower_gravity_guard import FeedforwardGuard, NAMES


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.g = FeedforwardGuard([3, 3, 3, 2, 2, 2], [4] * 6)

    def feed(self, t, effort=None):
        self.assertTrue(self.g.receive(NAMES, effort or [1.] * 6, 100 + t, 100 + t, t)[0])
        return self.g.step(t, 100 + t)

    def warm(self):
        for i in range(61):
            result = self.feed(i * .02)
        return result

    def test_missing_stays_zero(self):
        for t in (0., 1., 2.):
            out, ready, _ = self.g.step(t, 100 + t)
            self.assertEqual(out, [0.] * 6)
            self.assertFalse(ready)

    def test_start_ramp(self):
        self.assertEqual(self.feed(0)[0], [0.] * 6)
        self.assertAlmostEqual(self.feed(.02)[0][0], .04)
        for i in range(2, 61):
            result = self.feed(i * .02)
        self.assertEqual(result[:2], ([1.] * 6, True))

    def test_name_mapping(self):
        for i in range(80):
            t = i * .02
            self.g.receive(NAMES[::-1], [1, -1, 1, -1, 1, -1], 100+t, 100+t, t)
            out, _, _ = self.g.step(t, 100+t)
        self.assertEqual(out, [-1, 1, -1, 1, -1, 1])

    def test_saturation_and_slew(self):
        previous = [0.] * 6
        for i in range(101):
            out, _, _ = self.feed(i * .02, [100, -100, 100, -100, 100, -100])
            for a, b, limit in zip(out, previous, self.g.limits):
                self.assertLessEqual(abs(a-b), .08000001)
                self.assertLessEqual(abs(a), limit)
            previous = out
        self.assertEqual(out, [3, -3, 3, -2, 2, -2])

    def test_process_death_fades_out(self):
        self.warm()
        previous = 1.
        for i in range(61, 110):
            out, ready, _ = self.g.step(i*.02, 100+i*.02)
            self.assertLessEqual(abs(out[0]-previous), .08000001)
            previous = out[0]
        self.assertEqual(out, [0.] * 6)
        self.assertFalse(ready)

    def test_duplicate_stamp_cannot_refresh_with_frozen_ros_time(self):
        self.warm()
        for i in range(61, 110):
            self.g.receive(NAMES, [1.]*6, 101.2, 101.2, i*.02)
            out, ready, reason = self.g.step(i*.02, 101.2)
        self.assertEqual(out, [0.] * 6)
        self.assertFalse(ready)
        self.assertIn('wall clock', reason)

    def test_duplicate_fresh_stamp_does_not_interrupt_ramp(self):
        self.feed(0)
        self.g.receive(NAMES, [1.]*6, 100, 100.02, .02)
        self.assertGreater(self.g.step(.02, 100.02)[0][0], 0)

    def test_invalid_messages(self):
        cases = [
            (NAMES, [float('nan')]*6, 100),
            (NAMES, [float('inf')]*6, 100),
            (NAMES, [1]*5, 100),
            (['joint1']*6, [1]*6, 100),
            (NAMES, [1]*6, 99),
            (NAMES, [1]*6, 101),
            (NAMES, [1]*6, 0),
            (NAMES, [1]*6, float('nan')),
        ]
        for names, effort, stamp in cases:
            with self.subTest(stamp=stamp, effort=effort, names=names):
                self.assertFalse(self.g.receive(names, effort, stamp, 100, 0)[0])
                self.assertFalse(self.g.step(0, 100)[1])

    def test_out_of_order_and_recovery(self):
        self.warm()
        self.assertFalse(self.g.receive(NAMES, [1]*6, 101.19, 101.2, 1.2)[0])
        self.assertFalse(self.g.step(1.22, 101.22)[1])
        out, ready, _ = self.feed(1.24)
        self.assertFalse(ready)
        self.assertTrue(all(math.isfinite(x) for x in out))

    def test_disable_enable_reramps(self):
        self.warm()
        self.assertEqual(self.g.step(1.22, 101.22, active=False)[0], [0.] * 6)
        self.assertEqual(self.feed(1.24)[0], [0.] * 6)
        self.assertAlmostEqual(self.feed(1.26)[0][0], .04)

    def test_can_loss_fades_out(self):
        self.warm()
        for i in range(61, 100):
            t = i * .02
            self.g.receive(NAMES, [1]*6, 100+t, 100+t, t)
            out, ready, reason = self.g.step(t, 100+t, feedback_ok=False)
        self.assertEqual(out, [0.] * 6)
        self.assertIn('CAN', reason)

    def test_scheduling_gap_does_not_jump_torque(self):
        self.warm()
        out, ready, _ = self.g.step(10., 110.)
        self.assertLessEqual(1. - out[0], .20000001)
        self.assertFalse(ready)

    def test_bad_config(self):
        for limits, slew, timeout, ramp in [
            ([19]*6, [4]*6, .1, .5), ([3]*5, [4]*6, .1, .5),
            ([3]*6, [0]*6, .1, .5), ([3]*6, [4]*6, 0, .5),
            ([3]*6, [4]*6, .1, float('nan')),
        ]:
            with self.assertRaises(ValueError):
                FeedforwardGuard(limits, slew, timeout, ramp)


try:
    import rospy
    import roslaunch
    import rospkg
    from sensor_msgs.msg import JointState
    import follower_gravity_compensation as calculator
    import piper_ctrl_follower_node as driver
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class FakePublisher:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def publish(self, msg):
        self.messages.append(copy.deepcopy(msg))


class FakeCAN:
    def __init__(self):
        self.stamp = 100.
        self.commands = []
        self.gripper = []

    def GetArmJointMsgs(self):
        return SimpleNamespace(time_stamp=self.stamp)

    def JointMitCtrl(self, **kwargs):
        self.commands.append(kwargs)

    def GripperCtrl(self, *args):
        self.gripper.append(args)


@unittest.skipUnless(ROS_AVAILABLE, 'run inside the sourced ROS robot container')
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.path = rospkg.RosPack().get_path('piper_description') + '/urdf/piper_description.urdf'
        self.pkg = Path(rospkg.RosPack().get_path('piper_ctrl'))

    def node(self):
        # Never call the constructor (which connects CAN).
        n = object.__new__(driver.C_PiperFollowerCtrlNode)
        n._gravity_guard = FeedforwardGuard([3,3,3,2,2,2], [4]*6)
        n._can_stamp = n._can_advanced_at = n._gravity_reason = None
        n._gravity_applied = FakePublisher()
        n._gravity_active = FakePublisher()
        n.gravity_torques_lock = threading.Lock()
        n.control_data_lock = threading.Lock()
        n.gripper_cmd_lock = threading.Lock()
        n.gravity_torques = [0.]*6
        n.joint_positions_cmd = [.1, .2, -.3, .4, -.5, .6, .03]
        n.joint_velocities_cmd = [0.]*6
        n.joint_torques_cmd = [0.]*6
        n.mit_enable_pos = n.mit_enable_gravity = True
        n.mit_enable_vel = n.mit_enable_tor = False
        n.mit_gravity_mix_mode = 'additive'
        n.mit_torque_scale = [1.]*6
        n.mit_max_torque_abs = 18.
        n.mit_kp, n.mit_kd = [7.]*6, [.8]*6
        n.topic_prefix = '/robot/arm_left/'
        n.gripper_cmd_position = .03
        n.enable_gripper = n.gripper_exist = n.gripper_cmd_limit_enable = True
        n.enable_gripper_haptic = False
        n.gripper_val_mutiple = 1.
        n.gripper_cmd_min_raw, n.gripper_cmd_max_raw, n.gripper_cmd_deadband_raw = 0, 80000, 200
        n.piper = FakeCAN()
        return n

    def test_real_send_path_keeps_refs_gains_and_gripper(self):
        baseline, new = self.node(), self.node()
        baseline.mit_enable_gravity = False
        driver.C_PiperCtrlNode._mit_control(baseline)
        for i in range(101):
            t = i*.02
            new.piper.stamp = 100+t
            msg = JointState()
            msg.name, msg.effort = NAMES, [1.]*6
            msg.header.stamp = rospy.Time.from_sec(100+t)
            with patch.object(driver.time, 'monotonic', return_value=t), patch.object(
                    rospy.Time, 'now', return_value=rospy.Time.from_sec(100+t)):
                new.gravity_torque_callback(msg)
                new._mit_control()
        for original, updated in zip(baseline.piper.commands, new.piper.commands[-6:]):
            self.assertEqual(updated['t_ref'], 1.)
            self.assertEqual(original['t_ref'], 0.)
            self.assertEqual({k:v for k,v in original.items() if k!='t_ref'},
                             {k:v for k,v in updated.items() if k!='t_ref'})
        self.assertEqual(new.piper.gripper[-1], baseline.piper.gripper[-1])

    def test_can_stamp_watchdog(self):
        n = self.node()
        self.assertFalse(n._can_feedback_fresh(0))
        n.piper.stamp += .02
        self.assertTrue(n._can_feedback_fresh(.02))
        self.assertFalse(n._can_feedback_fresh(.2))
        n.piper.stamp = float('nan')
        self.assertFalse(n._can_feedback_fresh(.21))

    def test_rejects_position_bearing_gravity_message(self):
        n = self.node()
        msg = JointState(name=NAMES, effort=[1.]*6, position=[0.]*6)
        with patch.object(rospy, 'logwarn_throttle'):
            n.gravity_torque_callback(msg)
        self.assertIsNone(n._gravity_guard.sample)

    def test_bad_adapter_config_fails_before_can_constructor(self):
        with patch.object(rospy, 'get_param', return_value='p'), patch.object(
                driver.C_PiperCtrlNode, '__init__') as construct:
            with self.assertRaises(ValueError):
                driver.C_PiperFollowerCtrlNode()
            construct.assert_not_called()

    def test_named_state_and_model(self):
        msg = JointState(name=NAMES[::-1]+['gripper'], position=[0.]*6+[.04])
        q, width = calculator.read_positions(msg)
        model = calculator.GravityModel(self.path, [0,0,-9.81])
        tau = model.compute(q, width)
        self.assertTrue(all(math.isfinite(x) for x in tau))
        self.assertGreater(tau[1], 0)
        self.assertLess(tau[2], 0)
        msg.position[1] = float('nan')
        with self.assertRaises(ValueError):
            calculator.read_positions(msg)

    def test_calculator_preserves_source_stamp_and_invalidates_stale_output(self):
        import yaml
        cfg = yaml.safe_load((self.pkg / 'config/follower_gravity.yaml').read_text())
        def param(key, default=None):
            return cfg.get(key.lstrip('~'), default)
        with patch.object(rospy, 'get_param', side_effect=param), patch.object(
                rospy, 'Publisher', FakePublisher), patch.object(rospy, 'Subscriber'), patch.object(
                calculator.time, 'monotonic', return_value=0):
            arm = calculator.FollowerArm('left', self.path)
        msg = JointState(name=NAMES+['gripper'], position=[0.]*7)
        msg.header.stamp = rospy.Time.from_sec(100)
        with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(100.01)), patch.object(
                calculator.time, 'monotonic', return_value=.01):
            arm.receive(msg)
            arm.tick()
        self.assertEqual(arm.pub.messages[-1].header.stamp.to_sec(), 100)
        with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(100.3)), patch.object(
                calculator.time, 'monotonic', return_value=.3):
            arm.tick()
        self.assertEqual(arm.pub.messages[-1].header.stamp.to_sec(), 0)
        self.assertFalse(arm.ready.messages[-1].data)

    def load_launch(self, path, args):
        config = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(str(path), config, argv=args, verbose=False)
        return config, {k: v.value for k,v in config.params.items()}

    def test_launch_on_off_and_position_mode(self):
        import roslib.packages
        for args, enabled in [([], True), (['enable_follower_gravity:=true'], True),
                              (['enable_follower_gravity:=false'], False),
                              (['piper_ctrl_mode:=p','enable_follower_gravity:=true'], False)]:
            with self.subTest(args=args):
                cfg, params = self.load_launch(self.pkg / 'launch/piper_dual_robot.launch', args)
                nodes = {n.name: n for n in cfg.nodes}
                self.assertEqual('follower_gravity_compensation' in nodes, enabled)
                for side in ('left','right'):
                    name = 'piper_ctrl_%s_node' % side
                    key = '/' + name
                    self.assertEqual(params[key+'/mit/enable_gravity'], enabled)
                    self.assertEqual(params[key+'/mit/kp'], 7.)
                    self.assertEqual(params[key+'/mit/kd'], .8)
                    self.assertEqual(nodes[name].type, 'piper_ctrl_follower_node.py' if enabled else 'piper_ctrl_node.py')
                    self.assertTrue(roslib.packages.find_node('piper_ctrl', nodes[name].type))
                    self.assertEqual(params[key+'/remap/joint_pos_cmd_to'], '/robot/arm_'+side+'/joint_cmd_mux')
                    if enabled:
                        self.assertEqual(params[key+'/remap/joint_states_compensated_to'],
                                         '/robot/arm_'+side+'/follower_gravity_cmd')
                        self.assertEqual(params[key+'/follower_gravity/max_abs_t_ref'], [3,3,3,2,2,2])
                if enabled:
                    self.assertTrue(roslib.packages.find_node('piper_ctrl','follower_gravity_compensation.py'))

    def test_robot_root_forwarding(self):
        root = Path(rospkg.RosPack().get_path('robot_setup')) / 'launch'
        for file in ('start_robot_all.launch', 'start_robot_all_intervention.launch'):
            default_cfg, default_params = self.load_launch(root / file, [])
            self.assertIn('follower_gravity_compensation', [n.name for n in default_cfg.nodes])
            for side in ('left', 'right'):
                self.assertTrue(default_params['/piper_ctrl_'+side+'_node/mit/enable_gravity'])
            for on in ('true','false'):
                cfg, params = self.load_launch(root / file, ['enable_follower_gravity:='+on])
                self.assertEqual(params['/piper_ctrl_left_node/mit/enable_gravity'], on=='true')
        cfg, _ = self.load_launch(root/'start_robot_all.launch', ['enable_dual_arm:=false'])
        self.assertNotIn('follower_gravity_compensation', [n.name for n in cfg.nodes])


if __name__ == '__main__':
    unittest.main(verbosity=2)
