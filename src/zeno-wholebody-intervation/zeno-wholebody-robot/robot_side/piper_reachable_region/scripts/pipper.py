from typing import Optional

import numpy as np
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel

robot_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class Piper:
    def __init__(self, robot_model_path, semantics_path=None):
        self.robot = pp.load_model(robot_model_path)
        self.robot_joint_ids = [pp.joint_from_name(self.robot, name) for name in robot_joint_names]

        if semantics_path is not None:
            robot_model = RobotModel.from_urdf_file(robot_model_path)
            semantics = RobotSemantics.from_srdf_file(semantics_path, robot_model)
            self.disabled_collisions = pp.get_disabled_collisions(self.robot, semantics.disabled_collisions)
        else:
            self.disabled_collisions = {}

        self.joint_pos_sampler = None
        self.collision_fn = None

    def set_joint_positions(self, joint_positions: list[float]) -> None:
        pp.set_joint_positions(self.robot, self.robot_joint_ids, joint_positions)

    def get_joint_positions(self) -> list[float]:
        return list(pp.get_joint_positions(self.robot, self.robot_joint_ids))

    def get_collision_fn(self, obs: list[float] = []) -> callable:
        self.collision_fn = pp.get_collision_fn(
            self.robot, self.robot_joint_ids, obstacles=obs, disabled_collisions=self.disabled_collisions
        )
        return self.collision_fn

    def sample_joint_positions(self, dim: int = 1) -> np.ndarray:
        if self.joint_pos_sampler is None:
            self.joint_pos_sampler = pp.get_sample_fn(self.robot, self.robot_joint_ids)

        if dim == 1:
            return np.array(self.joint_pos_sampler())

        samples = []
        for _ in range(dim):
            joint_positions = self.joint_pos_sampler()
            samples.append(list(joint_positions))
        return np.array(samples)

    def check_collision(self, joint_positions: list[float]) -> bool:
        if self.collision_fn is None:
            self.get_collision_fn()
        return self.collision_fn(joint_positions)

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) joint limits using pybullet planning."""
        lower, upper = pp.get_custom_limits(self.robot, self.robot_joint_ids, circular_limits=pp.CIRCULAR_LIMITS)
        return np.array(lower), np.array(upper)
