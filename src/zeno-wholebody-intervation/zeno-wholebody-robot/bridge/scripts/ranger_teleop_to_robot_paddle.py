#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from dm_hw.msg import MotorState


class RangerTeleopToRobot:
    def __init__(self):
        rospy.init_node("ranger_teleop_to_robot_paddle_node", anonymous=True)

        # Load paddle configuration parameters
        self.x_config = rospy.get_param("~x", {})
        self.y_config = rospy.get_param("~y", {})
        self.z_config = rospy.get_param("~z", {})

        # Extract motor names from config
        self.x_motor_name = self.x_config.get("name", "joint2_motor")
        self.y_motor_name = self.y_config.get("name", "joint1_motor")
        self.z_motor_name = self.z_config.get("name", "joint0_motor")

        rospy.Subscriber("/paddle/state", MotorState, self.paddle_state_callback, queue_size=1)

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("Ranger Teleop to Robot Paddle node started")
        rospy.loginfo("Subscribing to: /paddle/state")
        rospy.loginfo("Publishing to: /cmd_vel")
        rospy.loginfo("X motor: %s, Y motor: %s, Z motor: %s", self.x_motor_name, self.y_motor_name, self.z_motor_name)

    def map_motor_value(self, value, config):
        """Map motor value according to configuration parameters"""
        # Check if output is locked
        if config.get("lock_output", False):
            return config.get("lock_output_value", 0.0)

        # Clamp to min/max range first
        min_val = config.get("min", -1.0)
        max_val = config.get("max", 1.0)
        value = max(min_val, min(max_val, value))

        # Reverse input if needed
        if config.get("reverse_input", False):
            value = -value

        # Apply deadzone: subtract deadzone value, then map from [deadzone, max/min] to [0, max/min]
        deadzone = config.get("deadzone", 0.0)
        
        # If within deadzone, return 0
        if abs(value) <= deadzone:
            return 0.0

        # Map to velocity range
        max_vel = config.get("max_vel", 1.0)

        # Subtract deadzone and map from [deadzone, max_val] to [0, max_val] for positive values
        # or from [-max_val, -deadzone] to [-max_val, 0] for negative values
        if value > 0:
            value_after_deadzone = value - deadzone
            if max_val - deadzone > 0:
                normalized = value_after_deadzone / (max_val - deadzone)
            else:
                normalized = 1.0
            velocity = normalized * max_vel
        else:
            value_after_deadzone = value + deadzone
            if abs(min_val) - deadzone > 0:
                normalized = value_after_deadzone / (abs(min_val) - deadzone)
            else:
                normalized = -1.0
            velocity = normalized * max_vel

        return velocity

    def paddle_state_callback(self, msg):
        """Callback for /paddle/state topic"""
        # Find motor values by matching names
        x_cmd = None
        y_cmd = None
        z_cmd = None

        # Match motors by name
        for i, name in enumerate(msg.names):
            if name == self.x_motor_name:
                if i < len(msg.position):
                    x_cmd = msg.position[i]
            elif name == self.y_motor_name:
                if i < len(msg.position):
                    y_cmd = msg.position[i]
            elif name == self.z_motor_name:
                if i < len(msg.position):
                    z_cmd = msg.position[i]

        # Check if all motors were found
        if x_cmd is None or y_cmd is None or z_cmd is None:
            missing = []
            if x_cmd is None:
                missing.append(self.x_motor_name)
            if y_cmd is None:
                missing.append(self.y_motor_name)
            if z_cmd is None:
                missing.append(self.z_motor_name)
            rospy.logwarn_throttle(1.0, "Motors not found in message: %s. Available motors: %s", missing, msg.names)
            return

        # Map motor values to cmd_vel
        cmd_vel = Twist()
        cmd_vel.linear.x = self.map_motor_value(x_cmd, self.x_config)
        cmd_vel.linear.y = self.map_motor_value(y_cmd, self.y_config)
        cmd_vel.angular.z = self.map_motor_value(z_cmd, self.z_config)

        # Priority logic: y cmd and z cmd are conflicting
        # If z cmd exists (non-zero), prioritize z cmd and set y cmd to 0
        if abs(cmd_vel.angular.z) > 1e-6:  # Check if z cmd is non-zero (with small epsilon for floating point)
            cmd_vel.linear.y = 0.0

        rospy.loginfo(
            "Publishing cmd_vel: linear.x=%.3f, linear.y=%.3f, angular.z=%.3f",
            cmd_vel.linear.x,
            cmd_vel.linear.y,
            cmd_vel.angular.z,
        )

        self.cmd_vel_pub.publish(cmd_vel)


def main():
    try:
        node = RangerTeleopToRobot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
