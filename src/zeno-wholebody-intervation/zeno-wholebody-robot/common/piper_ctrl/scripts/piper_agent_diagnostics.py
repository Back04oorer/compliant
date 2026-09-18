"""Read-only instrumentation for the EXISTING driver-owned SDK connection.

Installed as piper_agent_diagnostics.py alongside piper_ctrl_node.py. It wraps
the public ParseCANFrame callback, observes RX only, and never sends a CAN frame.
CAN IDs follow the deployed piper_sdk msg_v2 protocol. Unknown ages remain null.
"""

import json
import threading
import time


class DriverDiagnostics:
    def __init__(self, prefix, clock=time.monotonic):
        self.prefix, self.clock = prefix, clock
        self.lock = threading.Lock()
        self.stamps = {}
        self.publisher = None
        self.next_publish = 0.0

    def interface(self, base):
        monitor = self

        class MonitoredInterface(base):
            # Preserve a single instance per bus within this driver process.
            def ParseCANFrame(self, message):
                result = super().ParseCANFrame(message)
                monitor.receive(message)
                return result

        return MonitoredInterface

    def receive(self, message):
        if (
            message is None
            or len(message.data) != 8
            or getattr(message, "is_error_frame", False)
            or getattr(message, "is_extended_id", False)
            or getattr(message, "is_remote_frame", False)
            or getattr(message, "is_rx", True) is False
        ):
            return
        can_id = message.arbitration_id
        if (
            can_id in [0x2A1, 0x2A5, 0x2A6, 0x2A7]
            or 0x251 <= can_id <= 0x256
            or 0x261 <= can_id <= 0x266
        ):
            with self.lock:
                self.stamps[can_id] = self.clock()

    def collect(self, driver):
        now = self.clock()
        with self.lock:
            stamps = dict(self.stamps)

        def ages(ids):
            return [max(0.0, now - stamps[key]) if key in stamps else None for key in ids]

        with driver.control_data_lock:
            q = list(driver.current_joint_positions)
            dq = list(driver.current_joint_velocities)
            command = list(driver.joint_positions_cmd)
        status = driver.piper.GetArmStatus().arm_status
        motors = driver.piper.GetArmLowSpdInfoMsgs()
        enabled, faults = [], []
        for index in range(1, 7):
            motor = getattr(motors, "motor_%d" % index)
            enabled.append(bool(motor.foc_status.driver_enable_status))
            # bit 6 = enabled; the other bits are documented fault flags.
            faults.append(int(motor.foc_status_code) & 0xBF)
        position_only = driver.ctrl_mode == "p" or (
            driver.mit_enable_pos and not driver.mit_enable_vel and not driver.mit_enable_tor
        )
        position_only = position_only and not driver.enable_gripper_haptic
        topic = (
            driver.p_joint_pos_cmd_topic
            if driver.ctrl_mode == "p"
            else driver.mit_joint_pos_cmd_topic
        )
        return {
            "schema_version": 1,
            "published_at": time.time(),
            "positions_rad": q,
            "velocities_rad_s": dq,
            "driver_command": command,
            "gripper_width_m": float(driver.joint_states.position[6]),
            "joint_feedback_age_s": ages([0x2A5, 0x2A5, 0x2A6, 0x2A6, 0x2A7, 0x2A7]),
            "speed_feedback_age_s": ages(range(0x251, 0x257)),
            "enable_feedback_age_s": ages(range(0x261, 0x267)),
            "status_age_s": ages([0x2A1])[0],
            "motor_enabled": enabled,
            "motor_faults": faults,
            "driver_enabled": bool(driver.GetEnableFlag()),
            "blocked": bool(driver.block_ctrl_flag),
            "position_only": bool(position_only),
            "command_topic": topic,
            "gripper_topic": driver.gripper_pos_cmd_topic,
            "ctrl_mode": int(status.ctrl_mode),
            "teach_status": int(status.teach_status),
            "mode_feedback": int(status.mode_feed),
            "expected_mode_feedback": 4 if driver.ctrl_mode == "mit" else 1,
            "err_code": int(status.err_code),
            "gravity_required": bool(driver.ctrl_mode == "mit" and driver.mit_enable_gravity),
        }

    def publish(self, driver):
        # Diagnostics must never disrupt the driver's state or CAN threads.
        import rospy
        from std_msgs.msg import String

        if self.clock() < self.next_publish:
            return
        self.next_publish = self.clock() + 0.04
        try:
            if self.publisher is None:
                self.publisher = rospy.Publisher(
                    self.prefix + "agent_driver_state", String, queue_size=1
                )
            self.publisher.publish(String(data=json.dumps(self.collect(driver), allow_nan=False)))
        except Exception as error:
            rospy.logwarn_throttle(5.0, "Agent diagnostics unavailable: %s", str(error))
