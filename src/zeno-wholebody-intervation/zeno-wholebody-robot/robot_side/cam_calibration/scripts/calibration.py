#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手眼标定节点，使用 AX=YB 求解器同时估计 gripper2left 与 base2top 变换"""

import json
import os
import sys
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from pymlg import SE3

from solver import AXYSolver


class HandEyeCalibrator:
    def __init__(self) -> None:
        # ROS topics 配置
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/robot/arm_left/joint_states_single")
        self.end_pose_topic = rospy.get_param("~end_pose_topic", "/robot/arm_left/end_pose")

        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/handeye/left_cam_from_target")
        self.top_target_pose_topic = rospy.get_param("~top_target_pose_topic", "/handeye/top_cam_from_target")
        self.joint_cmd_topic = rospy.get_param("~joint_cmd_topic", "/robot/arm_left/joint_pos_cmd")

        # 最小采样数量
        self.min_samples = rospy.get_param("~min_samples", 5)

        # 轨迹执行参数
        self.settle_time = rospy.get_param("~settle_time", 30.0)  # 到达点后等待时间(秒)
        self.interpolation_steps = rospy.get_param("~interpolation_steps", 50)  # 插值步数
        self.interpolation_duration = rospy.get_param("~interpolation_duration", 2.0)  # 插值持续时间(秒)
        self.trajectory_file = rospy.get_param("~trajectory_file", "")  # 轨迹文件路径

        # 获取package路径
        direction = rospy.get_param("~direction", "left")
        if direction not in ["left", "right"]:
            raise ValueError(f"Invalid direction: {direction}")
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path("cam_calibration")
        self.samples_dir = os.path.join(self.package_path, "data", "samples", direction)
        os.makedirs(self.samples_dir, exist_ok=True)

        # 数据存储
        self.lock = threading.Lock()
        self.joint_states: Optional[JointState] = None
        self.end_pose: Optional[PoseStamped] = None
        self.target_pose: Optional[PoseStamped] = None
        self.top_target_pose: Optional[PoseStamped] = None

        # 采集的样本数据（统一 xxx2xxx 命名，AfromB 等价于 A2B）
        self.R_base2gripper_samples: List[np.ndarray] = []  # end_pose 提供 base2gripper
        self.t_base2gripper_samples: List[np.ndarray] = []
        self.R_cam2target_samples: List[np.ndarray] = []   # /handeye/left_cam_from_target
        self.t_cam2target_samples: List[np.ndarray] = []
        self.R_top2target_samples: List[np.ndarray] = []    # /handeye/top_cam_from_target
        self.t_top2target_samples: List[np.ndarray] = []

        # 记录的关节角列表
        self.recorded_joint_poses: List[np.ndarray] = []

        # 标定结果（gripper2left 与 base2top）
        self.calib_result: Optional[PoseStamped] = None
        self.R_gripper2left: Optional[np.ndarray] = None
        self.t_gripper2left: Optional[np.ndarray] = None
        self.R_base2top: Optional[np.ndarray] = None
        self.t_base2top: Optional[np.ndarray] = None
        
        # 当前使用的轨迹文件路径
        self.current_trajectory_file: Optional[str] = None

        # Publisher
        self.joint_cmd_pub = rospy.Publisher(self.joint_cmd_topic, JointState, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.joint_state_topic, JointState, self._joint_state_cb, queue_size=1)
        rospy.Subscriber(self.end_pose_topic, PoseStamped, self._end_pose_cb, queue_size=1)
        rospy.Subscriber(self.target_pose_topic, PoseStamped, self._target_pose_cb, queue_size=1)
        rospy.Subscriber(self.top_target_pose_topic, PoseStamped, self._top_target_pose_cb, queue_size=1)

        # Services
        rospy.Service("~capture_sample", Trigger, self._capture_sample_srv)
        rospy.Service("~compute_calibration", Trigger, self._compute_calibration_srv)
        rospy.Service("~clear_samples", Trigger, self._clear_samples_srv)
        rospy.Service("~save_result", Trigger, self._save_result_srv)

        # 新增的服务
        rospy.Service("~record_joint_pose", Trigger, self._record_joint_pose_srv)
        rospy.Service("~save_joint_poses", Trigger, self._save_joint_poses_srv)
        rospy.Service("~clear_joint_poses", Trigger, self._clear_joint_poses_srv)
        rospy.Service("~execute_trajectory", Trigger, self._execute_trajectory_srv)

        rospy.loginfo("HandEyeCalibrator initialized")
        rospy.loginfo(f"  Joint state topic: {self.joint_state_topic}")
        rospy.loginfo(f"  End pose topic: {self.end_pose_topic}")
        rospy.loginfo(f"  Target pose topic: {self.target_pose_topic}")
        rospy.loginfo(f"  Top target pose topic: {self.top_target_pose_topic}")
        rospy.loginfo(f"  Joint cmd topic: {self.joint_cmd_topic}")
        rospy.loginfo(f"  Minimum samples required: {self.min_samples}")
        rospy.loginfo(f"  Samples directory: {self.samples_dir}")
        rospy.loginfo(f"  Settle time: {self.settle_time}s")
        rospy.loginfo("Services available:")
        rospy.loginfo("  ~capture_sample - Capture current pose pair")
        rospy.loginfo("  ~compute_calibration - Compute hand-eye calibration")
        rospy.loginfo("  ~clear_samples - Clear all captured samples")
        rospy.loginfo("  ~save_result - Save calibration result to file")
        rospy.loginfo("  ~record_joint_pose - Record current joint pose")
        rospy.loginfo("  ~save_joint_poses - Save recorded joint poses to file")
        rospy.loginfo("  ~clear_joint_poses - Clear recorded joint poses")
        rospy.loginfo("  ~execute_trajectory - Execute trajectory from file")

    def _joint_state_cb(self, msg: JointState) -> None:
        """关节状态回调"""
        with self.lock:
            self.joint_states = msg

    def _end_pose_cb(self, msg: PoseStamped) -> None:
        """末端位姿回调 (base2gripper，AfromB 等价 A2B)"""
        with self.lock:
            self.end_pose = msg

    def _target_pose_cb(self, msg: PoseStamped) -> None:
        """目标位姿回调 (left2target，对应 left_cam_from_target)"""
        with self.lock:
            # Check if pose contains infinity values (indicating no target detected)
            p = msg.pose.position
            o = msg.pose.orientation
            if (p.x == float('inf') or p.y == float('inf') or p.z == float('inf') or
                o.x == float('inf') or o.y == float('inf') or o.z == float('inf') or o.w == float('inf')):
                self.target_pose = None
            else:
                self.target_pose = msg

    def _top_target_pose_cb(self, msg: PoseStamped) -> None:
        """顶摄相机位姿回调 (top2target，对应 top_cam_from_target)"""
        with self.lock:
            # Check if pose contains infinity values (indicating no target detected)
            p = msg.pose.position
            o = msg.pose.orientation
            if (p.x == float('inf') or p.y == float('inf') or p.z == float('inf') or
                o.x == float('inf') or o.y == float('inf') or o.z == float('inf') or o.w == float('inf')):
                self.top_target_pose = None
            else:
                self.top_target_pose = msg

    def _quat_to_matrix(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵 (兼容旧版 scipy)"""
        rot = Rotation.from_quat(quat)
        # 兼容旧版 scipy: as_matrix() 在 scipy < 1.4 中是 as_dcm()
        if hasattr(rot, "as_matrix"):
            return rot.as_matrix()
        else:
            return rot.as_dcm()

    def _matrix_to_quat(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转四元数 (兼容旧版 scipy)"""
        # 兼容旧版 scipy: from_matrix() 在 scipy < 1.4 中是 from_dcm()
        if hasattr(Rotation, "from_matrix"):
            rot = Rotation.from_matrix(R)
        else:
            rot = Rotation.from_dcm(R)
        return rot.as_quat()

    def _matrix_to_euler(self, R: np.ndarray, seq: str = "xyz", degrees: bool = True) -> np.ndarray:
        """旋转矩阵转欧拉角 (兼容旧版 scipy)"""
        if hasattr(Rotation, "from_matrix"):
            rot = Rotation.from_matrix(R)
        else:
            rot = Rotation.from_dcm(R)
        return rot.as_euler(seq, degrees=degrees)

    def _pose_to_dict(self, pose: Optional[PoseStamped]) -> Optional[dict]:
        """将 PoseStamped 转为字典便于序列化"""
        if pose is None:
            return None
        return {
            "frame_id": pose.header.frame_id,
            "stamp": pose.header.stamp.to_sec(),
            "position": {
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
                "z": pose.pose.position.z,
            },
            "orientation": {
                "x": pose.pose.orientation.x,
                "y": pose.pose.orientation.y,
                "z": pose.pose.orientation.z,
                "w": pose.pose.orientation.w,
            },
        }

    def _state_as_dict(self) -> dict:
        """将当前数据打包为可写入 JSON 的字典"""
        with self.lock:
            return {
                "joint_poses": [pose.tolist() for pose in self.recorded_joint_poses],
                "R_base2gripper_samples": [mat.tolist() for mat in self.R_base2gripper_samples],
                "t_base2gripper_samples": [t.flatten().tolist() for t in self.t_base2gripper_samples],
                "R_cam2target_samples": [mat.tolist() for mat in self.R_cam2target_samples],
                "t_cam2target_samples": [t.flatten().tolist() for t in self.t_cam2target_samples],
                "R_top2target_samples": [mat.tolist() for mat in self.R_top2target_samples],
                "t_top2target_samples": [t.flatten().tolist() for t in self.t_top2target_samples],
                "calib_result": self._pose_to_dict(self.calib_result),
                "R_gripper2left": self.R_gripper2left.tolist() if self.R_gripper2left is not None else None,
                "t_gripper2left": (
                    self.t_gripper2left.flatten().tolist() if self.t_gripper2left is not None else None
                ),
                "R_base2top": self.R_base2top.tolist() if self.R_base2top is not None else None,
                "t_base2top": (
                    self.t_base2top.flatten().tolist() if self.t_base2top is not None else None
                ),
            }

    def _save_state_to_json(self, save_path: str) -> str:
        """将当前数据写入 JSON 文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = self._state_as_dict()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        rospy.loginfo(f"All data saved to: {save_path}")
        return save_path

    def _pose_to_rt(self, pose: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
        """将 PoseStamped 转换为旋转矩阵和平移向量"""
        p = pose.pose.position
        o = pose.pose.orientation
        t = np.array([[p.x], [p.y], [p.z]])
        R = self._quat_to_matrix(np.array([o.x, o.y, o.z, o.w]))
        return R, t

    def _rt_to_pose_stamped(self, R: np.ndarray, t: np.ndarray, frame_id: str = "end_effector") -> PoseStamped:
        """将旋转矩阵和平移向量转换为 PoseStamped"""
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id

        t_flat = t.flatten()
        pose.pose.position.x = t_flat[0]
        pose.pose.position.y = t_flat[1]
        pose.pose.position.z = t_flat[2]

        quat = self._matrix_to_quat(R)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose

    def _rt_to_se3(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """将 R, t 转为 4x4 SE(3) 矩阵"""
        return SE3.from_components(R, t.flatten())

    def _capture_sample_srv(self, req) -> TriggerResponse:
        """服务: 采集当前位姿样本"""
        with self.lock:
            if self.end_pose is None:
                return TriggerResponse(success=False, message="No end pose received yet")
            if self.target_pose is None:
                return TriggerResponse(success=False, message="No target pose received yet")
            if self.top_target_pose is None:
                return TriggerResponse(success=False, message="No top camera pose received yet")

            # 转换位姿为 R, t（统一 xxx2xxx 命名）
            R_base2gripper, t_base2gripper = self._pose_to_rt(self.end_pose)          # base2gripper
            R_left2target, t_left2target = self._pose_to_rt(self.target_pose)         # left2target
            R_top2target, t_top2target = self._pose_to_rt(self.top_target_pose)       # top2target

            # 存储样本
            self.R_base2gripper_samples.append(R_base2gripper)
            self.t_base2gripper_samples.append(t_base2gripper)
            self.R_cam2target_samples.append(R_left2target)
            self.t_cam2target_samples.append(t_left2target)
            self.R_top2target_samples.append(R_top2target)
            self.t_top2target_samples.append(t_top2target)

            num_samples = len(self.R_base2gripper_samples)
            rospy.loginfo(f"Sample captured. Total samples: {num_samples}")

            return TriggerResponse(
                success=True,
                message=f"Sample captured. Total samples: {num_samples}/{self.min_samples}",
            )

    def _compute_calibration_srv(self, req) -> TriggerResponse:
        """服务: 计算手眼标定"""
        with self.lock:
            num_samples = len(self.R_base2gripper_samples)
            if num_samples < self.min_samples:
                return TriggerResponse(
                    success=False,
                    message=f"Not enough samples. Current: {num_samples}, Required: {self.min_samples}",
                )

            try:
                # 使用 AX=YB 求解器，联合求解 X(gripper2left) 与 Y(base2top)
                solver = AXYSolver()
                A_list: List[np.ndarray] = []  # base2gripper
                B_list: List[np.ndarray] = []  # top->target * target->left = top->left

                for R_b2g, t_b2g, R_left2t, t_left2t, R_top2t, t_top2t in zip(
                    self.R_base2gripper_samples,
                    self.t_base2gripper_samples,
                    self.R_cam2target_samples,
                    self.t_cam2target_samples,
                    self.R_top2target_samples,
                    self.t_top2target_samples,
                ):
                    T_b2g = self._rt_to_se3(R_b2g, t_b2g)           # base2gripper

                    T_left2target = self._rt_to_se3(R_left2t, t_left2t)  # target->left
                    T_target2left = SE3.inverse(T_left2target)
                    T_top2target = self._rt_to_se3(R_top2t, t_top2t)     # top->target

                    A_list.append(T_b2g)
                    B_list.append(T_top2target @ T_target2left)

                sol, X_hat, Y_hat = solver.solve(A_list, B_list)
                if not sol.success:
                    rospy.logwarn(f"AX=YB solver did not fully converge: {sol.message}")

                R_gripper2left, t_gripper2left = SE3.to_components(X_hat)
                R_base2top, t_base2top = SE3.to_components(Y_hat)

                self.R_gripper2left = R_gripper2left
                self.t_gripper2left = t_gripper2left.reshape((3, 1))
                self.R_base2top = R_base2top
                self.t_base2top = t_base2top.reshape((3, 1))
                self.calib_result = self._rt_to_pose_stamped(R_gripper2left, self.t_gripper2left, "end_effector")

                # 计算旋转角度和平移距离用于日志
                euler = self._matrix_to_euler(R_gripper2left, "xyz", degrees=True)
                t_flat = self.t_gripper2left.flatten()

                rospy.loginfo("=" * 50)
                rospy.loginfo("Hand-eye calibration completed! (gripper2left & base2top)")
                rospy.loginfo(f"Translation (x, y, z): [{t_flat[0]:.4f}, {t_flat[1]:.4f}, {t_flat[2]:.4f}] m")
                rospy.loginfo(f"Rotation (rx, ry, rz): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")
                if self.R_base2top is not None and self.t_base2top is not None:
                    y_euler = self._matrix_to_euler(self.R_base2top, "xyz", degrees=True)
                    y_t_flat = self.t_base2top.flatten()
                    rospy.loginfo("Base -> Top camera transform:")
                    rospy.loginfo(
                        f"  Translation (x, y, z): [{y_t_flat[0]:.4f}, {y_t_flat[1]:.4f}, {y_t_flat[2]:.4f}] m"
                    )
                    rospy.loginfo(
                        f"  Rotation (rx, ry, rz): [{y_euler[0]:.2f}, {y_euler[1]:.2f}, {y_euler[2]:.2f}] deg"
                    )
                rospy.loginfo("=" * 50)

                return TriggerResponse(
                    success=True,
                    message=f"Calibration completed. T=[{t_flat[0]:.4f}, {t_flat[1]:.4f}, {t_flat[2]:.4f}]",
                )

            except Exception as e:
                rospy.logerr(f"Calibration failed: {e}")
                return TriggerResponse(success=False, message=f"Calibration failed: {e}")

    def _clear_samples_srv(self, req) -> TriggerResponse:
        """服务: 清除所有采集的样本"""
        with self.lock:
            self.R_base2gripper_samples.clear()
            self.t_base2gripper_samples.clear()
            self.R_cam2target_samples.clear()
            self.t_cam2target_samples.clear()
            self.R_top2target_samples.clear()
            self.t_top2target_samples.clear()
            rospy.loginfo("All samples cleared")
            return TriggerResponse(success=True, message="All samples cleared")

    def _save_result_srv(self, req) -> TriggerResponse:
        """服务: 保存标定结果到文件"""
        with self.lock:
            if self.R_gripper2left is None or self.t_gripper2left is None:
                return TriggerResponse(success=False, message="No calibration result available")

        try:
            # 如果存在当前轨迹文件，更新该文件中的对应字段
            if self.current_trajectory_file and os.path.exists(self.current_trajectory_file):
                # 读取现有文件
                with open(self.current_trajectory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新标定结果字段
                with self.lock:
                    data['calib_result'] = self._pose_to_dict(self.calib_result)
                    data['R_gripper2left'] = self.R_gripper2left.tolist() if self.R_gripper2left is not None else None
                    data['t_gripper2left'] = (
                        self.t_gripper2left.flatten().tolist() if self.t_gripper2left is not None else None
                    )
                    # 同时更新样本数据
                    data['R_base2gripper_samples'] = [mat.tolist() for mat in self.R_base2gripper_samples]
                    data['t_base2gripper_samples'] = [t.flatten().tolist() for t in self.t_base2gripper_samples]
                    data['R_cam2target_samples'] = [mat.tolist() for mat in self.R_cam2target_samples]
                    data['t_cam2target_samples'] = [t.flatten().tolist() for t in self.t_cam2target_samples]
                    data['R_top2target_samples'] = [mat.tolist() for mat in self.R_top2target_samples]
                    data['t_top2target_samples'] = [t.flatten().tolist() for t in self.t_top2target_samples]
                    data['R_base2top'] = (
                        self.R_base2top.tolist() if self.R_base2top is not None else None
                    )
                    data['t_base2top'] = (
                        self.t_base2top.flatten().tolist() if self.t_base2top is not None else None
                    )
                
                # 保存更新后的文件
                with open(self.current_trajectory_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                rospy.loginfo(f"Calibration result updated in trajectory file: {self.current_trajectory_file}")
                return TriggerResponse(success=True, message=f"Result saved to {self.current_trajectory_file}")
            else:
                # 如果没有轨迹文件，使用默认路径保存
                default_path = os.path.join(self.samples_dir, "handeye_calibration.json")
                save_path = rospy.get_param("~save_path", default_path)
                saved_path = self._save_state_to_json(save_path)
                return TriggerResponse(success=True, message=f"Result saved to {saved_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save result: {e}")
            return TriggerResponse(success=False, message=f"Failed to save: {e}")

    def _record_joint_pose_srv(self, req) -> TriggerResponse:
        """服务: 记录当前关节角到列表"""
        with self.lock:
            if self.joint_states is None:
                return TriggerResponse(success=False, message="No joint state received yet")

            joint_pos = np.array(self.joint_states.position)
            self.recorded_joint_poses.append(joint_pos)

            num_poses = len(self.recorded_joint_poses)
            rospy.loginfo(f"Joint pose recorded. Total poses: {num_poses}")
            rospy.loginfo(f"  Position: {joint_pos}")

            return TriggerResponse(
                success=True,
                message=f"Joint pose recorded. Total poses: {num_poses}",
            )

    def _save_joint_poses_srv(self, req) -> TriggerResponse:
        """服务: 保存记录的关节角列表到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"handeye_data_{timestamp}.json"
            filepath = os.path.join(self.samples_dir, filename)
            saved_path = self._save_state_to_json(filepath)
            with self.lock:
                num = len(self.recorded_joint_poses)
            return TriggerResponse(
                success=True,
                message=f"Saved {num} poses (and related data) to {saved_path}",
            )
        except Exception as e:
            rospy.logerr(f"Failed to save joint poses: {e}")
            return TriggerResponse(success=False, message=f"Failed to save: {e}")

    def _clear_joint_poses_srv(self, req) -> TriggerResponse:
        """服务: 清除记录的关节角列表"""
        with self.lock:
            self.recorded_joint_poses.clear()
            rospy.loginfo("All recorded joint poses cleared")
            return TriggerResponse(success=True, message="All recorded joint poses cleared")

    def _interpolate_joint_positions(self, start: np.ndarray, end: np.ndarray, steps: int) -> List[np.ndarray]:
        """线性插值两个关节位置之间的点"""
        return [start + (end - start) * t / (steps - 1) for t in range(steps)]

    def _send_joint_command(self, joint_positions: np.ndarray) -> None:
        """发送关节位置命令"""
        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.position = joint_positions.tolist()
        self.joint_cmd_pub.publish(cmd)

    def _execute_trajectory_srv(self, req) -> TriggerResponse:
        """服务: 读取并执行轨迹文件，每个点自动采样"""
        # 获取轨迹文件路径
        trajectory_file = rospy.get_param("~trajectory_file", "")
        
        # 如果未指定轨迹文件，自动获取samples文件夹中最新的文件（根据文件名中的时间戳）
        if not trajectory_file:
            try:
                # 获取samples目录下所有JSON文件
                json_files = [
                    f for f in os.listdir(self.samples_dir)
                    if f.endswith('.json') and os.path.isfile(os.path.join(self.samples_dir, f))
                ]
                if not json_files:
                    return TriggerResponse(
                        success=False, 
                        message="No trajectory file specified and no JSON files found in samples directory."
                    )
                
                # 从文件名中提取时间戳并排序（格式：handeye_data_YYYYMMDD_HHMMSS.json）
                def extract_timestamp(filename):
                    try:
                        # 提取文件名中的时间戳部分
                        if filename.startswith('handeye_data_') and filename.endswith('.json'):
                            timestamp_str = filename[13:-5]  # 去掉前缀和后缀
                            # 解析时间戳：YYYYMMDD_HHMMSS
                            if '_' in timestamp_str:
                                date_part, time_part = timestamp_str.split('_', 1)
                                if len(date_part) == 8 and len(time_part) == 6:
                                    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except (ValueError, IndexError):
                        pass
                    # 如果无法解析，返回最小时间
                    return datetime.min
                
                # 按时间戳排序，获取最新的文件
                json_files.sort(key=extract_timestamp, reverse=True)
                trajectory_file = json_files[0]
                rospy.loginfo(f"Auto-selected latest file by timestamp: {trajectory_file}")
            except Exception as e:
                return TriggerResponse(
                    success=False,
                    message=f"Failed to auto-select trajectory file: {e}"
                )

        # 如果是相对路径，则在samples目录下查找
        if not os.path.isabs(trajectory_file):
            trajectory_file = os.path.join(self.samples_dir, trajectory_file)

        if not os.path.exists(trajectory_file):
            return TriggerResponse(success=False, message=f"Trajectory file not found: {trajectory_file}")

        # 保存当前轨迹文件路径，供_save_result_srv使用
        self.current_trajectory_file = trajectory_file

        try:
            # 加载轨迹点（从JSON文件加载）
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从JSON数据中提取joint_poses
            if 'joint_poses' not in data:
                return TriggerResponse(success=False, message="Trajectory file does not contain 'joint_poses' field")
            
            joint_poses = np.array(data['joint_poses'])
            if len(joint_poses) == 0:
                return TriggerResponse(success=False, message="Trajectory file is empty")

            rospy.loginfo(f"Loaded {len(joint_poses)} waypoints from {trajectory_file}")

            # 清除之前的样本
            with self.lock:
                self.R_base2gripper_samples.clear()
                self.t_base2gripper_samples.clear()
                self.R_cam2target_samples.clear()
                self.t_cam2target_samples.clear()
                self.R_top2target_samples.clear()
                self.t_top2target_samples.clear()

            # 获取当前关节位置作为起点
            with self.lock:
                if self.joint_states is None:
                    return TriggerResponse(success=False, message="No joint state available")
                current_pos = np.array(self.joint_states.position)

            rate = rospy.Rate(50)  # 50 Hz 控制频率

            # 依次运动到每个路径点
            for i, target_pos in enumerate(joint_poses):
                rospy.loginfo(f"Moving to waypoint {i + 1}/{len(joint_poses)}")

                # 获取当前位置
                with self.lock:
                    current_pos = np.array(self.joint_states.position)

                # 直线插值
                interpolated = self._interpolate_joint_positions(
                    current_pos, target_pos, self.interpolation_steps
                )

                # 执行插值轨迹
                step_duration = self.interpolation_duration / self.interpolation_steps
                for interp_pos in interpolated:
                    if rospy.is_shutdown():
                        return TriggerResponse(success=False, message="Node shutdown during execution")
                    self._send_joint_command(interp_pos)
                    rospy.sleep(step_duration)
                    rate.sleep()

                # 等待机械臂稳定
                rospy.loginfo(f"Waiting {self.settle_time}s for arm to settle...")
                rospy.sleep(self.settle_time)

                # 采集样本
                rospy.loginfo(f"Capturing sample at waypoint {i + 1}")
                result = self._capture_sample_srv(None)
                if not result.success:
                    rospy.logwarn(f"Failed to capture sample at waypoint {i + 1}: {result.message}")

            # 所有点采样完成，检查采样数量
            with self.lock:
                num_samples = len(self.R_base2gripper_samples)

            if num_samples >= self.min_samples:
                # 计算标定
                compute_result = self._compute_calibration_srv(None)
                if compute_result.success:
                    # 保存结果
                    save_result = self._save_result_srv(None)
                    return TriggerResponse(
                        success=True,
                        message=f"Trajectory executed. Collected {num_samples} samples. Calibration completed and saved.",
                    )
                else:
                    return TriggerResponse(
                        success=True,
                        message=f"Trajectory executed. Collected {num_samples} samples. Calibration failed: {compute_result.message}",
                    )
            else:
                return TriggerResponse(
                    success=True,
                    message=f"Trajectory executed. Only collected {num_samples}/{self.min_samples} valid samples.",
                )

        except Exception as e:
            rospy.logerr(f"Trajectory execution failed: {e}")
            return TriggerResponse(success=False, message=f"Execution failed: {e}")

    def spin(self) -> None:
        """主循环"""
        rospy.loginfo("HandEyeCalibrator running...")
        rospy.loginfo("Use 'rosservice call /handeye_calibrator/capture_sample' to capture samples")
        rospy.loginfo("Use 'rosservice call /handeye_calibrator/compute_calibration' to compute calibration")

        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # 打印状态
            with self.lock:
                has_joint = self.joint_states is not None
                has_end = self.end_pose is not None
                has_target = self.target_pose is not None
                has_top_target = self.top_target_pose is not None
                num_samples = len(self.R_base2gripper_samples)

            rospy.loginfo_throttle(
                10.0,
                f"Status - Joint: {'OK' if has_joint else 'NO'}, "
                f"EndPose: {'OK' if has_end else 'NO'}, "
                f"Target: {'OK' if has_target else 'NO'}, "
                f"TopTarget: {'OK' if has_top_target else 'NO'}, "
                f"Samples: {num_samples}/{self.min_samples}",
            )

            rate.sleep()


def main() -> int:
    rospy.init_node("handeye_calibrator", anonymous=False)
    try:
        HandEyeCalibrator().spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
