#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class DepthViewer:
    """Subscribe to 16UC1 depth image and render in an OpenCV window."""

    def __init__(self, topic: str, window_name: str = "Depth View") -> None:
        self.topic = topic
        self.window_name = window_name
        self.colormap_window = window_name + " (Colormap)"
        self.last_frame: Optional[np.ndarray] = None

        self.bridge = CvBridge()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.colormap_window, cv2.WINDOW_NORMAL)

        rospy.Subscriber(self.topic, Image, self._callback, queue_size=1)
        rospy.loginfo("Subscribed to %s", self.topic)

    def _callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "CvBridge error: %s", str(e))
            return

        if depth is None:
            rospy.logwarn_throttle(5.0, "Received empty depth image")
            return

        rospy.loginfo_once(
            "Depth image encoding: %s, dtype: %s, shape: %s", msg.encoding, str(depth.dtype), str(depth.shape)
        )

        valid_mask = depth > 0
        num_valid = int(np.count_nonzero(valid_mask))
        if num_valid == 0:
            rospy.logwarn_throttle(2.0, "Depth frame has NO valid (>0) pixels")
        else:
            dmin = int(depth[valid_mask].min())
            dmax = int(depth[valid_mask].max())
            # rospy.loginfo_throttle(
            #     2.0, "Depth stats: dtype=%s, valid_pixels=%d, min=%d, max=%d", str(depth.dtype), num_valid, dmin, dmax
            # )

        if depth.dtype != np.uint16:
            rospy.logwarn_throttle(
                5.0,
                "Expected 16UC1 depth image, but got dtype %s (still trying to display)",
                str(depth.dtype),
            )

        self.last_frame = depth

    @staticmethod
    def depth_to_vis(depth: np.ndarray) -> np.ndarray:
        """
        将深度图转换为 8UC1 可视化灰度图。
        对 >0 的像素用 1%–99% 分位数做归一化。
        """
        valid_mask = depth > 0
        if not np.any(valid_mask):
            return np.zeros(depth.shape, dtype=np.uint8)

        depth_valid = depth[valid_mask].astype(np.float32)

        vmin = np.percentile(depth_valid, 1)
        vmax = np.percentile(depth_valid, 99)

        if vmax <= vmin:
            return np.zeros(depth.shape, dtype=np.uint8)

        depth_f = depth.astype(np.float32)
        depth_f = np.clip(depth_f, vmin, vmax)
        depth_norm = (depth_f - vmin) / (vmax - vmin + 1e-6)
        depth_vis = (depth_norm * 255.0).astype(np.uint8)

        depth_vis[~valid_mask] = 0

        return depth_vis

    def spin(self) -> None:
        rospy.loginfo("Rendering depth stream. Press 'q' or Ctrl+C to exit.")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.last_frame is not None:
                depth_vis = self.depth_to_vis(self.last_frame)

                cv2.imshow(self.window_name, depth_vis)

                colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow(self.colormap_window, colormap)
            else:
                blank = np.zeros((240, 320), dtype=np.uint8)
                cv2.putText(blank, "No frame yet", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
                cv2.imshow(self.window_name, blank)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            rate.sleep()

        cv2.destroyAllWindows()


def main() -> int:
    rospy.init_node("depth_test", anonymous=True)

    topic = "/realsense_top/aligned_depth_to_color/image_raw"
    viewer = DepthViewer(topic=topic, window_name="Depth Top")
    viewer.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
