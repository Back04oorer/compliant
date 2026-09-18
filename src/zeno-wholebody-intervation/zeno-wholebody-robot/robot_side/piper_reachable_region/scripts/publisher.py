#!/usr/bin/env python3
"""
ROS1 Node for Visualizing Reachability Maps from HDF5 Files in RViz

This node reads a reachability map stored in an HDF5 file and publishes it as a
PointCloud2 message for visualization in RViz.

USAGE:
    rosrun zeno-wholebody-teleop publisher.py [options]

ROS Parameters:
    ~hdf5_path (str): Path to the HDF5 file
    ~frame_id (str): TF frame for the point cloud (default: "base_footprint")
    ~publish_rate (float): Publishing frequency in Hz (default: 1.0)
    ~min_score (float): Minimum score filter
    ~max_score (float): Maximum score clamp
    ~downsample_factor (int): Keep every N-th point
"""

import sys
import numpy as np
import h5py
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


class ReachabilityVisualizer:
    """
    ROS node class for visualizing reachability maps from HDF5 files.
    """

    def __init__(self, hdf5_path, frame_id="base_footprint", reachability_pub_topic="/reachability_cloud", min_score=None, max_score=None, downsample_factor=1):
        self.frame_id = frame_id
        self.downsample_factor = downsample_factor
        self.reachability_pub_topic = reachability_pub_topic

        # Load data from HDF5 file
        try:
            with h5py.File(hdf5_path, "r") as f:
                # Load sphere data: [x, y, z, score]
                if "/Spheres/sphere_dataset" not in f:
                    raise KeyError("/Spheres/sphere_dataset not found in HDF5 file")
                
                sphere_data = f["/Spheres/sphere_dataset"][:]
                rospy.loginfo(f"Loaded {len(sphere_data)} reachability points from {hdf5_path}")

                # Apply downsampling
                if downsample_factor > 1:
                    sphere_data = sphere_data[::downsample_factor]
                    rospy.loginfo(f"Downsampled to {len(sphere_data)} points (factor: {downsample_factor})")

                # Extract coordinates and scores
                self.points = sphere_data[:, :3]  # [x, y, z]
                scores = sphere_data[:, 3]        # reachability scores

                # import matplotlib.pyplot as plt
                # plt.scatter(list(range(len(scores))), scores)
                # plt.show()

                # Apply score filtering
                if min_score is not None:
                    valid_mask = scores >= min_score
                    self.points = self.points[valid_mask]
                    scores = scores[valid_mask]
                    rospy.loginfo(f"Filtered {np.sum(~valid_mask)} points below min_score {min_score}")

                # Apply score clamping
                if max_score is not None:
                    scores = np.clip(scores, None, max_score)

                # Normalize scores to 0-255 range for intensity
                if len(scores) > 0:
                    score_min = np.min(scores)
                    score_max = np.max(scores)
                    if score_max > score_min:
                        # Normalize to 0-255
                        self.intensities = ((scores - score_min) / (score_max - score_min) * 255).astype(np.uint8)
                    else:
                        self.intensities = np.full(len(scores), 128, dtype=np.uint8)
                else:
                    self.intensities = np.array([], dtype=np.uint8)

                if len(self.points) == 0:
                    rospy.logwarn("No valid points after filtering!")

        except Exception as e:
            rospy.logerr(f"Error loading HDF5 file {hdf5_path}: {e}")
            sys.exit(1)

        # Initialize publisher
        self.cloud_pub = rospy.Publisher(self.reachability_pub_topic, PointCloud2, queue_size=1)
        self.seq = 0

    def create_point_cloud_msg(self):
        """
        Create a PointCloud2 message from the loaded reachability data.
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        header.seq = self.seq
        self.seq += 1

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
        ]

        if len(self.points) == 0:
            return point_cloud2.create_cloud(header, fields, [])

        # Create a list of points with correct types.
        # We explicitly cast intensity to int to avoid struct.pack errors with float types.
        # np.column_stack would convert everything to float, causing "required argument is not an integer".
        # create_cloud needs len() to allocate buffer, so we use a list instead of generator.
        points_list = [
            (float(x), float(y), float(z), int(i))
            for x, y, z, i in zip(self.points[:, 0], self.points[:, 1], self.points[:, 2], self.intensities)
        ]

        # Create the PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, points_list)

        return cloud_msg

    def publish_cloud(self):
        """
        Publish the reachability data.
        """
        cloud_msg = self.create_point_cloud_msg()
        self.cloud_pub.publish(cloud_msg)
        if len(self.points) > 0 and self.seq % 10 == 0: # Log every 10th publish to avoid spam
            rospy.logdebug(f"Published {len(self.points)} reachability points")


def main():
    rospy.init_node("reachability_visualizer", anonymous=True)

    hdf5_path = rospy.get_param("~hdf5_path", "")
    if not hdf5_path:
        rospy.logerr("Required parameter ~hdf5_path not set!")
        # hdf5_path = "/home/jeong/zeno/wholebody-teleop/teleop/src/zeno-wholebody-teleop/robot_side/piper_reachable_region/maps/3D_reach_map_gripper_base_0.05_2025-12-11-17-29-20.h5"
        sys.exit(1)

    frame_id = rospy.get_param("~frame_id", "base_footprint")
    publish_rate = rospy.get_param("~publish_rate", 1.0)
    min_score = rospy.get_param("~min_score", None)
    max_score = rospy.get_param("~max_score", None)
    downsample_factor = int(rospy.get_param("~downsample_factor", 1))
    reachability_pub_topic = rospy.get_param("~reachability_pub_topic", "/reachability_cloud")

    rospy.loginfo(f"Settings: file={hdf5_path}, frame={frame_id}, rate={publish_rate}")

    visualizer = ReachabilityVisualizer(
        hdf5_path=hdf5_path,
        frame_id=frame_id,
        reachability_pub_topic=reachability_pub_topic,
        min_score=min_score,
        max_score=max_score,
        downsample_factor=downsample_factor
    )

    rate = rospy.Rate(publish_rate)

    while not rospy.is_shutdown():
        visualizer.publish_cloud()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        sys.exit(1)
