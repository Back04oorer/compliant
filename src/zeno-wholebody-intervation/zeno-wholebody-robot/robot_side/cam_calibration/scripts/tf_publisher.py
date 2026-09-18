#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布标定结果的 TF。

当前读取 handeye_data_*.json 中的 R_gripper2left / t_gripper2left
以及 R_base2top / t_base2top，并分别发布 TF：
- gripper -> left camera
- base   -> top camera

逻辑：
1. 自动查找 samples 目录下最新的标定文件（文件名含时间戳 handeye_data_YYYYMMDD_HHMMSS.json）。
2. 读取矩阵并转换为平移 + 四元数。
3. 持续发布 TF。
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import rospy
import rospkg
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_matrix


def _extract_timestamp(filename: str) -> datetime:
    """从文件名 handeye_data_YYYYMMDD_HHMMSS.json 提取时间戳，解析失败返回最小时间。"""
    try:
        if filename.startswith("handeye_data_") and filename.endswith(".json"):
            ts_str = filename[len("handeye_data_") : -len(".json")]
            if "_" in ts_str:
                date_part, time_part = ts_str.split("_", 1)
                if len(date_part) == 8 and len(time_part) == 6:
                    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        pass
    return datetime.min


class GripperCamTFPublisher:
    def __init__(self) -> None:
        # 参数
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("cam_calibration")
        self.samples_dir = rospy.get_param("~samples_dir", os.path.join(pkg_path, "data", "samples", "left"))
        # 相机（gripper->cam）帧
        self.parent_frame = rospy.get_param("~parent_frame", "gripper_base")
        self.child_frame = rospy.get_param("~child_frame", "realsense_left_color_optical_frame")

        # 顶部相机（base->top）帧
        self.top_parent_frame = rospy.get_param("~top_parent_frame", "base_link")
        self.top_child_frame = rospy.get_param("~top_child_frame", "realsense_top_color_optical_frame")

        # 可开关
        self.publish_wrist = rospy.get_param("~publish_wrist", True)
        self.publish_top = rospy.get_param("~publish_top", True)
        self.publish_rate = rospy.get_param("~publish_rate", 30.0)
        self.specific_file = rospy.get_param("~trajectory_file", "")

        # 反向变换参数
        self.wrist_reverse = rospy.get_param("~wrist_reverse", False)
        self.top_reverse = rospy.get_param("~top_reverse", False)

        os.makedirs(self.samples_dir, exist_ok=True)

        # 载入变换列表
        self.transforms = self._load_transforms()
        self.br = tf2_ros.TransformBroadcaster()

    def _find_latest_file(self) -> Optional[str]:
        """根据文件名中的时间戳选择最新的 handeye_data_*.json。"""
        try:
            candidates = [
                f
                for f in os.listdir(self.samples_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(self.samples_dir, f))
            ]
            if not candidates:
                return None
            candidates.sort(key=_extract_timestamp, reverse=True)
            latest = candidates[0]
            return os.path.join(self.samples_dir, latest)
        except Exception as e:
            rospy.logerr(f"Failed to list samples in {self.samples_dir}: {e}")
            return None

    def _build_transform(
        self,
        data: Dict,
        R_key: str,
        t_key: str,
        parent_frame: str,
        child_frame: str,
        reverse: bool = False,
    ) -> Dict[str, np.ndarray]:
        """从 data 中取出旋转和平移并转换为 TF 所需格式。"""
        if R_key not in data or t_key not in data:
            raise KeyError(f"Missing {R_key} or {t_key}")

        R = np.array(data[R_key], dtype=float)
        t = np.array(data[t_key], dtype=float)

        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"{R_key} or {t_key} has invalid shape")

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        if reverse:
            # 反向变换：求逆矩阵，并交换父子帧
            T = np.linalg.inv(T)
            parent_frame, child_frame = child_frame, parent_frame

        quat = quaternion_from_matrix(T)
        t_new = T[:3, 3]

        return {
            "parent": parent_frame,
            "child": child_frame,
            "translation": t_new,
            "quaternion": quat,
        }

    def _load_transforms(self) -> List[Dict[str, np.ndarray]]:
        """加载需要发布的所有变换。"""
        # 优先使用参数指定的文件
        filepath = self.specific_file
        if filepath and not os.path.isabs(filepath):
            filepath = os.path.join(self.samples_dir, filepath)

        if filepath and not os.path.exists(filepath):
            rospy.logwarn(f"Specified trajectory_file not found: {filepath}, fallback to latest.")
            filepath = ""

        if not filepath:
            filepath = self._find_latest_file()

        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("No valid calibration file found in samples directory.")

        rospy.loginfo(f"Using calibration file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        transforms: List[Dict[str, np.ndarray]] = []

        if self.publish_wrist:
            try:
                transforms.append(
                    self._build_transform(
                        data,
                        "R_gripper2cam",
                        "t_gripper2cam",
                        self.parent_frame,
                        self.child_frame,
                        self.wrist_reverse,
                    )
                )
            except Exception as e:
                rospy.logwarn(f"Failed to load gripper->cam transform: {e}")

        if self.publish_top:
            try:
                transforms.append(
                    self._build_transform(
                        data,
                        "R_base2top",
                        "t_base2top",
                        self.top_parent_frame,
                        self.top_child_frame,
                        self.top_reverse,
                    )
                )
            except Exception as e:
                rospy.logwarn(f"Failed to load base->top transform: {e}")

        if not transforms:
            raise ValueError("No valid transforms loaded from calibration file.")

        return transforms

    def publish(self) -> None:
        rate = rospy.Rate(self.publish_rate)

        if not self.transforms:
            rospy.logerr("No transform loaded, aborting TF publishing.")
            return

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            for tf_data in self.transforms:
                msg = TransformStamped()
                msg.header.stamp = now
                msg.header.frame_id = tf_data["parent"]
                msg.child_frame_id = tf_data["child"]
                msg.transform.translation.x = float(tf_data["translation"][0])
                msg.transform.translation.y = float(tf_data["translation"][1])
                msg.transform.translation.z = float(tf_data["translation"][2])
                msg.transform.rotation.x = float(tf_data["quaternion"][0])
                msg.transform.rotation.y = float(tf_data["quaternion"][1])
                msg.transform.rotation.z = float(tf_data["quaternion"][2])
                msg.transform.rotation.w = float(tf_data["quaternion"][3])
                self.br.sendTransform(msg)
            rate.sleep()


def main() -> int:
    rospy.init_node("handeye_tf_publisher")
    try:
        node = GripperCamTFPublisher()
        node.publish()
    except Exception as e:
        rospy.logerr(f"TF publisher failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
