#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import struct
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Vector3Stamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def weight(r: float, r_min: float = 0.3, r_far: float = 1.2, weight_max: float = 1.0, delta: float = 0.05) -> float:
    """Calculate weight based on distance. weight_max controls the maximum value.
    
    - [r_min, r_far]: Quadratic function, monotonically decreasing from weight_max to 0
    - [r_min-delta, r_min): Smoothstep function, monotonically increasing from 0 to weight_max
      with zero derivatives at both endpoints
    - Outside these intervals: returns 0
    """
    if r < r_min - delta or r > r_far:
        return 0.0

    if r < r_min:
        # Interval [r_min - delta, r_min): sigmoid-like smooth transition
        # t goes from 0 to 1 as r goes from (r_min - delta) to r_min
        t = (r - (r_min - delta)) / delta
        # Smoothstep: 3t^2 - 2t^3, has zero derivative at t=0 and t=1
        return weight_max * (3.0 * t * t - 2.0 * t * t * t)
    else:
        # Interval [r_min, r_far]: quadratic function
        # Normalized distance from r_min to r_far (0 to 1)
        normalized = (r - r_min) / (r_far - r_min)
        # Quadratic decrease: (1 - normalized)^2 * weight_max
        return weight_max * (1.0 - normalized) * (1.0 - normalized)


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


class LowPassFilter:
    """First-order low-pass filter for smoothing signals."""

    def __init__(self, alpha: float = 0.3):
        """Initialize low-pass filter. alpha: 0-1, smaller = more smoothing."""
        self.alpha = max(0.0, min(1.0, alpha))
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.initialized = False

    def filter(self, x: float, y: float) -> tuple:
        """Apply low-pass filter to input vector."""
        if not self.initialized:
            self.filtered_x = x
            self.filtered_y = y
            self.initialized = True
        else:
            self.filtered_x = self.alpha * x + (1.0 - self.alpha) * self.filtered_x
            self.filtered_y = self.alpha * y + (1.0 - self.alpha) * self.filtered_y
        return (self.filtered_x, self.filtered_y)

    def reset(self):
        """Reset filter state."""
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.initialized = False


def create_circle_marker(
    radius: float, frame_id: str, marker_id: int, color: tuple, ns: str = "range_circles", num_points: int = 64
) -> Marker:
    """Create a Marker message representing a circle in the xy plane."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.02
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    marker.points = []
    for i in range(num_points + 1):
        angle = 2.0 * math.pi * i / num_points
        point = Point()
        point.x = radius * math.cos(angle)
        point.y = radius * math.sin(angle)
        point.z = 0.0
        marker.points.append(point)

    return marker


def weight_to_color(weight: float, max_weight: float = 1.0) -> tuple:
    """Convert weight to RGB color: low->green, medium->yellow, high->red."""
    if max_weight <= 0:
        return (0.0, 1.0, 0.0)

    normalized = min(weight / max_weight, 1.0)
    if normalized < 0.5:
        r = normalized * 2.0
        g = 1.0
        b = 0.0
    else:
        r = 1.0
        g = (1.0 - normalized) * 2.0
        b = 0.0
    return (r, g, b)


def create_axis_marker(frame_id: str, axis: str, length: float = 0.3) -> Marker:
    """Create a marker for coordinate axis (x, y, or z)."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "laser_frame"
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0

    if axis == "x":
        marker.id = 0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
    elif axis == "y":
        marker.id = 1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        angle = math.pi / 2.0
        marker.pose.orientation.z = math.sin(angle / 2.0)
        marker.pose.orientation.w = math.cos(angle / 2.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
    elif axis == "z":
        marker.id = 2
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        angle = -math.pi / 2.0
        marker.pose.orientation.y = math.sin(angle / 2.0)
        marker.pose.orientation.w = math.cos(angle / 2.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.z = 0.0

    marker.scale.x = length
    marker.scale.y = 0.02
    marker.scale.z = 0.02
    marker.color.a = 1.0
    return marker


def scan_callback(
    scan: LaserScan, pub_vector, pub_marker, pub_pointcloud, pub_circles, pub_frame, force_filter, r_min, r_far, weight_max, delta
):
    """Callback function for processing laser scan messages."""
    count = len(scan.ranges)
    force_x_sum = 0.0
    force_y_sum = 0.0
    valid_count = 0
    points = []
    weights = []
    max_weight = 0.0

    for i in range(count):
        r = scan.ranges[i]
        if math.isnan(r) or math.isinf(r) or r <= 0:
            continue

        w = weight(r, r_min=r_min, r_far=r_far, weight_max=weight_max, delta=delta)
        max_weight = max(max_weight, w)

        angle = scan.angle_min + scan.angle_increment * i
        angle += math.pi
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y, 0.0))
        weights.append(w)

        if w > 0.0:
            repulsive_force = -w
            force_x_sum += repulsive_force * math.cos(angle)
            force_y_sum += repulsive_force * math.sin(angle)
            valid_count += 1

    if valid_count > 0:
        force_x = force_x_sum / valid_count
        force_y = force_y_sum / valid_count
    else:
        force_x = 0.0
        force_y = 0.0

    force_x_filtered, force_y_filtered = force_filter.filter(force_x, force_y)

    vector_msg = Vector3Stamped()
    vector_msg.header.stamp = rospy.Time.now()
    vector_msg.header.frame_id = scan.header.frame_id
    vector_msg.vector.x = force_x_filtered
    vector_msg.vector.y = force_y_filtered
    vector_msg.vector.z = 0.0
    pub_vector.publish(vector_msg)

    marker_msg = Marker()
    marker_msg.header.frame_id = scan.header.frame_id
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "repulsive_force"
    marker_msg.id = 0
    marker_msg.type = Marker.ARROW
    marker_msg.action = Marker.ADD
    marker_msg.pose.position.x = 0.0
    marker_msg.pose.position.y = 0.0
    marker_msg.pose.position.z = 0.0
    marker_msg.pose.orientation.w = 1.0

    force_magnitude = math.sqrt(force_x_filtered * force_x_filtered + force_y_filtered * force_y_filtered)
    scale_factor = 1.0
    marker_msg.scale.x = force_magnitude * scale_factor
    marker_msg.scale.y = 0.01
    marker_msg.scale.z = 0.01
    marker_msg.color.r = 1.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 0.0
    marker_msg.color.a = 1.0

    if force_magnitude > 0.001:
        angle_force = math.atan2(force_y_filtered, force_x_filtered)
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = math.sin(angle_force / 2.0)
        marker_msg.pose.orientation.w = math.cos(angle_force / 2.0)
    else:
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

    pub_marker.publish(marker_msg)

    if len(points) > 0:
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = rospy.Time.now()
        cloud_msg.header.frame_id = scan.header.frame_id
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = False

        points_data = []
        for (x, y, z), w in zip(points, weights):
            r, g, b = weight_to_color(w, max_weight)
            rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            point_data = struct.pack("<fffI", x, y, z, rgb)
            points_data.append(point_data)

        cloud_msg.data = b"".join(points_data)
        pub_pointcloud.publish(cloud_msg)

    circle_min = create_circle_marker(
        radius=r_min, frame_id=scan.header.frame_id, marker_id=0, color=(0.0, 0.0, 1.0, 0.8), ns="range_circles"
    )
    pub_circles.publish(circle_min)

    circle_far = create_circle_marker(
        radius=r_far, frame_id=scan.header.frame_id, marker_id=1, color=(0.0, 1.0, 0.0, 0.8), ns="range_circles"
    )
    pub_circles.publish(circle_far)

    # Publish coordinate frame markers
    axis_length = 0.3
    x_axis = create_axis_marker(scan.header.frame_id, "x", axis_length)
    y_axis = create_axis_marker(scan.header.frame_id, "y", axis_length)
    z_axis = create_axis_marker(scan.header.frame_id, "z", axis_length)
    pub_frame.publish(x_axis)
    pub_frame.publish(y_axis)
    pub_frame.publish(z_axis)


def main():
    """Main function to initialize ROS node and subscriber."""
    rospy.init_node("rplidar_node_client", anonymous=True)

    scan_topic = rospy.get_param("~scan_topic", "/scan")
    repulsive_force_vector_topic = rospy.get_param("~repulsive_force_vector_topic", "/repulsive_force_vector")
    repulsive_force_marker_topic = rospy.get_param("~repulsive_force_marker_topic", "/repulsive_force_marker")
    weighted_pointcloud_topic = rospy.get_param("~weighted_pointcloud_topic", "/weighted_pointcloud")
    range_circles_topic = rospy.get_param("~range_circles_topic", "/range_circles")
    frame_marker_topic = rospy.get_param("~frame_marker_topic", "/laser_frame_marker")

    filter_alpha = rospy.get_param("~filter_alpha", 0.3)
    r_min = rospy.get_param("~r_min", 0.3)
    r_far = rospy.get_param("~r_far", 1.2)
    weight_max = rospy.get_param("~weight_max", 1.0)
    delta = rospy.get_param("~delta", 0.05)

    rospy.loginfo(f"Topics: scan={scan_topic}, vector={repulsive_force_vector_topic}")
    rospy.loginfo(f"Params: r_min={r_min}, r_far={r_far}, weight_max={weight_max}, delta={delta}, filter_alpha={filter_alpha}")

    force_filter = LowPassFilter(alpha=filter_alpha)

    pub_vector = rospy.Publisher(repulsive_force_vector_topic, Vector3Stamped, queue_size=10)
    pub_marker = rospy.Publisher(repulsive_force_marker_topic, Marker, queue_size=10)
    pub_pointcloud = rospy.Publisher(weighted_pointcloud_topic, PointCloud2, queue_size=10)
    pub_circles = rospy.Publisher(range_circles_topic, Marker, queue_size=10)
    pub_frame = rospy.Publisher(frame_marker_topic, Marker, queue_size=10)

    rospy.Subscriber(
        scan_topic,
        LaserScan,
        lambda msg: scan_callback(
            msg, pub_vector, pub_marker, pub_pointcloud, pub_circles, pub_frame, force_filter, r_min, r_far, weight_max, delta
        ),
        queue_size=1000,
    )

    rospy.loginfo("rplidar_node_client started")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
