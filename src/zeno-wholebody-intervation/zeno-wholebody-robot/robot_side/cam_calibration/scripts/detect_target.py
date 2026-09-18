#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArUco GridBoard 位姿检测节点，发布 target 相对 camera 的位姿"""

import sys
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

# cv2.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
cv2.setLogLevel(2)


class TargetDetector:
    ARUCO_DICT_MAP = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self) -> None:
        # ROS topics
        self.image_topic = rospy.get_param("~image_topic", "/realsense_left/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense_left/color/camera_info")
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/handeye/target_pose")

        # ArUco dictionary
        aruco_dict_name = rospy.get_param("~aruco_dict", "DICT_5X5_250")
        if aruco_dict_name not in self.ARUCO_DICT_MAP:
            rospy.logwarn(f"Unknown ArUco dictionary '{aruco_dict_name}', using DICT_5X5_250")
            aruco_dict_name = "DICT_5X5_250"
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT_MAP[aruco_dict_name])

        # GridBoard parameters
        self.markers_x = rospy.get_param("~markers_x", 3)
        self.markers_y = rospy.get_param("~markers_y", 4)
        self.marker_size = rospy.get_param("~marker_size", 0.0564)
        self.marker_separation = rospy.get_param("~marker_separation", 0.0057)

        # ArUco detector and GridBoard (origin at bottom-left corner)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        self.grid_board = cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_size,
            self.marker_separation,
            self.aruco_dict,
        )

        # Visualization
        self.show_visualization = rospy.get_param("~show_visualization", True)
        self.window_name = "Target Detection"
        if self.show_visualization:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Camera intrinsics (populated from CameraInfo)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_frame_id: str = ""

        self.bridge = CvBridge()
        self.last_pose: Optional[PoseStamped] = None

        # Publisher & Subscribers
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo(
            f"TargetDetector: {aruco_dict_name}, {self.markers_x}x{self.markers_y} board, "
            f"size={self.marker_size}m, sep={self.marker_separation}m"
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.camera_frame_id = msg.header.frame_id
            rospy.loginfo(f"Camera intrinsics received from {self.camera_frame_id}")

    def _image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"CvBridge error: {e}")
            return

        if image is None:
            return

        pose_result = self._detect_board(image, msg.header.stamp)
        self.pose_pub.publish(pose_result)

        # Only update last_pose for valid detections (not inf poses)
        if not (pose_result.pose.position.x == float('inf')):
            self.last_pose = pose_result

    def _detect_board(self, image: np.ndarray, stamp: rospy.Time) -> PoseStamped:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self._draw_visualization(image, None, None)
            return self._create_inf_pose(stamp)

        obj_points, img_points = self.grid_board.matchImagePoints(corners, ids)
        if obj_points is None or len(obj_points) == 0:
            self._draw_visualization(image, corners, ids)
            return self._create_inf_pose(stamp)

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
        if not success:
            self._draw_visualization(image, corners, ids)
            return self._create_inf_pose(stamp)

        self._draw_visualization(image, corners, ids, rvec, tvec)
        return self._to_pose_stamped(rvec, tvec, stamp)

    def _to_pose_stamped(self, rvec: np.ndarray, tvec: np.ndarray, stamp: rospy.Time) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.camera_frame_id

        t = tvec.flatten()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = t[0], t[1], t[2]

        quat = Rotation.from_rotvec(rvec.flatten()).as_quat()
        pose.pose.orientation.x, pose.pose.orientation.y = quat[0], quat[1]
        pose.pose.orientation.z, pose.pose.orientation.w = quat[2], quat[3]

        return pose

    def _create_inf_pose(self, stamp: rospy.Time) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.camera_frame_id

        # Fill position with infinity
        pose.pose.position.x = float('inf')
        pose.pose.position.y = float('inf')
        pose.pose.position.z = float('inf')

        # Fill orientation with infinity (quaternion)
        pose.pose.orientation.x = float('inf')
        pose.pose.orientation.y = float('inf')
        pose.pose.orientation.z = float('inf')
        pose.pose.orientation.w = float('inf')

        return pose

    def _draw_visualization(
        self,
        image: np.ndarray,
        corners,
        ids,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
    ) -> None:
        if not self.show_visualization:
            return

        vis = image.copy()

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 1.5)

        status = "Detected" if rvec is not None else "Searching..."
        color = (0, 255, 0) if rvec is not None else (0, 0, 255)
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if self.last_pose is not None:
            p = self.last_pose.pose.position
            cv2.putText(
                vis,
                f"Pos: [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        self.last_vis_image = vis

    def spin(self) -> None:
        rospy.loginfo("Target detector running. Press 'q' to exit.")
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.show_visualization and hasattr(self, "last_vis_image"):
                cv2.imshow(self.window_name, self.last_vis_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            rate.sleep()

        if self.show_visualization:
            cv2.destroyAllWindows()


def main() -> int:
    rospy.init_node("detect_target", anonymous=False)
    try:
        TargetDetector().spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
