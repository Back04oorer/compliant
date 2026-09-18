import math
import os
import pickle
import subprocess
from datetime import datetime
from typing import Optional, Tuple

import h5py
import numpy as np
import pytorch_kinematics as pk
import torch

class TorchKinMapper:
    """
    Forward-kinematics based reachability map generator.
    """

    def __init__(
        self,
        robot_urdf: Optional[str] = None,
        name_end_effector: str = "gripper_base",
        name_base_link: str = "base_link",
        n_dof: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        angular_res: float = math.pi / 8,
        cartesian_res: float = 0.05,
        x_lim=None,
        y_lim=None,
        z_lim=None,
        r_lim=None,
        p_lim=None,
        yaw_lim=None,
        joint_pos_min=None,
        joint_pos_max=None,
        N_fk: int = 1280000000,
        num_loops: int = 500,
        save_freq: int = 100,
        log_progress: bool = True,
        reach_map_dir: Optional[str] = None
    ):
        """
        Configure the reachability map sampler.

        Args:
            robot_urdf: Path to the URDF file; defaults to the repository asset.
            name_end_effector: End-effector link name in the URDF.
            name_base_link: Base link name in the URDF.
            n_dof: Number of joints; inferred from limits when omitted.
            device: Torch device; defaults to CUDA when available.
            dtype: Torch dtype for kinematics and map buffers.
            angular_res: Resolution (rad) for roll/pitch/yaw bins.
            cartesian_res: Resolution (m) for x/y/z bins.
            x_lim: [min, max] x bounds in meters.
            y_lim: [min, max] y bounds in meters.
            z_lim: [min, max] z bounds in meters.
            r_lim: Roll bounds in radians.
            p_lim: Pitch bounds in radians.
            yaw_lim: Yaw bounds in radians.
            joint_pos_min: Lower joint limits; optional override.
            joint_pos_max: Upper joint limits; optional override.
            N_fk: Total forward-kinematics samples to draw.
            num_loops: Number of sampling batches.
            log_progress: Whether to print progress logs.
            reach_map_dir: Output directory for map artifacts.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.dtype = dtype
        self.name_end_effector = name_end_effector
        self.name_base_link = name_base_link
        self.n_dof = n_dof
        self.N_fk = N_fk
        self.num_loops = num_loops
        self.save_freq = save_freq
        self.N_fk_loop = max(int(N_fk / num_loops), 1)
        self.log_progress = log_progress

        # Paths
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # zeno-wholebody-teleop
        default_urdf = self._resolve_default_urdf(repo_root)
        self.robot_urdf = os.path.abspath(robot_urdf) if robot_urdf else default_urdf
        default_map_dir = os.path.abspath(os.path.join(repo_root, "robot_side", "piper_reachable_region", "maps"))
        self.reach_map_dir = os.path.abspath(reach_map_dir) if reach_map_dir else default_map_dir
        os.makedirs(self.reach_map_dir, exist_ok=True)
        self.reach_map_file_name = (
            f"reach_map_{name_end_effector}_{cartesian_res}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )

        # Limits
        self.angular_res = angular_res
        self.cartesian_res = cartesian_res
        self.r_lim = r_lim if r_lim is not None else [-math.pi, math.pi]
        self.p_lim = p_lim if p_lim is not None else [-math.pi / 2, math.pi / 2]
        self.yaw_lim = yaw_lim if yaw_lim is not None else [-math.pi, math.pi]
        self.roll_bins = math.ceil((2 * math.pi) / angular_res)
        self.pitch_bins = math.ceil((math.pi) / angular_res)
        self.yaw_bins = math.ceil((2 * math.pi) / angular_res)
        self.x_lim = x_lim if x_lim is not None else [-1.5, 1.5]
        self.y_lim = y_lim if y_lim is not None else [-1.5, 1.5]
        self.z_lim = z_lim if z_lim is not None else [-2.0, 2.0]
        self.x_bins = math.ceil((self.x_lim[1] - self.x_lim[0]) / cartesian_res)
        self.y_bins = math.ceil((self.y_lim[1] - self.y_lim[0]) / cartesian_res)
        self.z_bins = math.ceil((self.z_lim[1] - self.z_lim[0]) / cartesian_res)

        # Joint limits
        default_joint_pos_min, default_joint_pos_max = self._get_limits_from_pybullet()

        self.joint_pos_min = joint_pos_min if joint_pos_min is not None else default_joint_pos_min
        self.joint_pos_max = joint_pos_max if joint_pos_max is not None else default_joint_pos_max
        if self.n_dof is None:
            self.n_dof = len(self.joint_pos_min)

        self.joint_pos_centers = self.joint_pos_min + (self.joint_pos_max - self.joint_pos_min) / 2
        self.joint_pos_range_sq = (self.joint_pos_max - self.joint_pos_min).pow(2) / 4

        # Map offsets
        self.num_voxels = self.x_bins * self.y_bins * self.z_bins * self.roll_bins * self.pitch_bins * self.yaw_bins
        num_values = 6 + 2  # 6D pose + visitation freq + manipulability
        self.reach_map = torch.zeros((self.num_voxels, num_values), dtype=self.dtype, device="cpu")
        self.x_ind_offset = self.y_bins * self.z_bins * self.roll_bins * self.pitch_bins * self.yaw_bins
        self.y_ind_offset = self.z_bins * self.roll_bins * self.pitch_bins * self.yaw_bins
        self.z_ind_offset = self.roll_bins * self.pitch_bins * self.yaw_bins
        self.roll_ind_offset = self.pitch_bins * self.yaw_bins
        self.pitch_ind_offset = self.yaw_bins
        self.yaw_ind_offset = 1
        self.offsets = torch.tensor(
            [
                self.x_ind_offset,
                self.y_ind_offset,
                self.z_ind_offset,
                self.roll_ind_offset,
                self.pitch_ind_offset,
                self.yaw_ind_offset,
            ],
            dtype=torch.long,
            device=self.device,
        )

        # Sampling distribution
        self.sampling_distr = torch.distributions.uniform.Uniform(self.joint_pos_min, self.joint_pos_max)

    @staticmethod
    def _resolve_default_urdf(repo_root: str) -> str:
        """Resolve the default URDF path for Piper, with Piper X as fallback."""
        try:
            package_path = (
                subprocess.check_output(["rospack", "find", "piper_description"], stderr=subprocess.DEVNULL)
                .strip()
                .decode("utf-8")
            )
            return os.path.abspath(os.path.join(package_path, "urdf", "piper_description.urdf"))
        except Exception:
            pass

        local_piper = os.path.abspath(
            os.path.join(repo_root, "..", "piper_ros", "src", "piper_description", "urdf", "piper_description.urdf")
        )
        if os.path.exists(local_piper):
            return local_piper

        try:
            package_path = (
                subprocess.check_output(["rospack", "find", "piper_x_description"], stderr=subprocess.DEVNULL)
                .strip()
                .decode("utf-8")
            )
            return os.path.abspath(os.path.join(package_path, "urdf", "piper_x_description.urdf"))
        except Exception:
            pass

        local_piper_x = os.path.abspath(
            os.path.join(repo_root, "..", "piper_ros", "src", "piper_x_description", "urdf", "piper_x_description.urdf")
        )
        if os.path.exists(local_piper_x):
            return local_piper_x

        # Final fallback for older layouts.
        return os.path.abspath(
            os.path.join(repo_root, "..", "piper_ros", "src", "piper_description", "urdf", "piper_description.urdf")
        )

    def _build_chain(self) -> pk.SerialChain:
        """
        Build the serial chain from the URDF and move it to the target device.
        """
        if self.log_progress:
            print("[Building kinematic chain from URDF...]")
        with open(self.robot_urdf, "rb") as f:
            urdf_bytes = f.read()
        chain = pk.build_serial_chain_from_urdf(urdf_bytes, self.name_end_effector)
        chain = chain.to(dtype=self.dtype, device=self.device)
        assert len(chain.get_joint_parameter_names()) == self.n_dof, "Incorrect number of DOFs set"
        if self.log_progress:
            print("...chain ready")
        return chain

    def _compute_indices(self, poses_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous 6D poses into bin indices.
        """
        indices_6d = poses_6d - torch.tensor(
            [self.x_lim[0], self.y_lim[0], self.z_lim[0], self.r_lim[0], self.p_lim[0], self.yaw_lim[0]],
            dtype=self.dtype,
            device=self.device,
        )
        indices_6d /= torch.tensor(
            [
                self.cartesian_res,
                self.cartesian_res,
                self.cartesian_res,
                self.angular_res,
                self.angular_res,
                self.angular_res,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        indices_6d = torch.floor(indices_6d)
        indices_6d[indices_6d[:, 3] >= self.roll_bins, 3] = self.roll_bins - 1
        indices_6d[indices_6d[:, 4] >= self.pitch_bins, 4] = self.pitch_bins - 1
        indices_6d[indices_6d[:, 5] >= self.yaw_bins, 5] = self.yaw_bins - 1
        return indices_6d

    def _discretize_indices(self, indices_6d: torch.Tensor) -> torch.Tensor:
        """
        Flatten 6D indices into a 1D voxel index.
        """
        indices_6d = indices_6d.to(dtype=torch.long)
        indices_6d = (
            indices_6d[:, 5] * self.offsets[5]
            + indices_6d[:, 4] * self.offsets[4]
            + indices_6d[:, 3] * self.offsets[3]
            + indices_6d[:, 2] * self.offsets[2]
            + indices_6d[:, 1] * self.offsets[1]
            + indices_6d[:, 0] * self.offsets[0]
        )
        return indices_6d

    def _get_limits_from_pybullet(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Query joint limits from pybullet if the dependencies are present.
        """
        try:
            import pybullet as p
            import pybullet_planning as pp
            from pipper import robot_joint_names
        except ImportError:
            if self.log_progress:
                print("[WARN] pybullet/pybullet_planning/pipper not available; falling back to default limits")
            return None

        cid = pp.connect(use_gui=True)
        try:
            robot = pp.load_model(os.path.abspath(self.robot_urdf))
            joint_ids = [pp.joint_from_name(robot, name) for name in robot_joint_names]
            lower, upper = pp.get_custom_limits(robot, joint_ids)
            if self.n_dof is None:
                self.n_dof = len(joint_ids)
            lower_t = torch.as_tensor(lower, dtype=self.dtype, device=self.device)
            upper_t = torch.as_tensor(upper, dtype=self.dtype, device=self.device)
            return lower_t, upper_t
        finally:
            pp.disconnect()

    def _save_map(self, reach_map_nonzero: np.ndarray, prefix: str, manip_scaling: float) -> str:
        """
        Persist reachability spheres and poses to an H5 file.

        Returns:
            Path to the written H5 file.
        """
        indx = 0
        first = True
        z_ind_offset = self.z_ind_offset

        while indx < reach_map_nonzero.shape[0]:
            sphere_3d = reach_map_nonzero[indx][:3]
            end_idx = min(indx + z_ind_offset, reach_map_nonzero.shape[0])
            num_repetitions = int((reach_map_nonzero[indx:end_idx, :3] == sphere_3d).all(axis=1).sum())
            num_repetitions = max(num_repetitions, 1)
            manip_avg = reach_map_nonzero[indx : indx + num_repetitions, 7].mean() * manip_scaling
            if first:
                first = False
                sphere_array = np.append(reach_map_nonzero[indx][:3], manip_avg)
                pose_array = np.append(reach_map_nonzero[0, :6], np.array([0.0, 0.0, 0.0, 1.0])).astype(np.single)
            else:
                sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], manip_avg)))
                pose_array = np.vstack(
                    (
                        pose_array,
                        np.append(reach_map_nonzero[indx, :6], np.array([0.0, 0.0, 0.0, 1.0])).astype(np.single),
                    )
                )
            indx += num_repetitions

        h5_path = os.path.join(self.reach_map_dir, f"{prefix}_{self.reach_map_file_name}.h5")
        with h5py.File(h5_path, "w") as f:
            sphere_group = f.create_group("/Spheres")
            sphere_dat = sphere_group.create_dataset("sphere_dataset", data=sphere_array)
            sphere_dat.attrs.create("Resolution", data=self.cartesian_res)
            pose_group = f.create_group("/Poses")
            pose_group.create_dataset("poses_dataset", dtype=float, data=pose_array)
        return h5_path

    def generate_maps(self) -> None:
        """
        Sample joint configurations, run FK/Jacobian, and accumulate maps.
        """
        if self.log_progress:
            print("[Starting FK and Jacobian calculations...]")
            print(f"[Number of loops is: {self.num_loops}]")
            print(f"[Map save frequency is {self.save_freq} loops]")
            print(f"[Will save file named: '{self.reach_map_file_name}' at path: '{self.reach_map_dir}']")

        chain = self._build_chain()
        offsets = torch.tensor(
            [
                self.cartesian_res,
                self.cartesian_res,
                self.cartesian_res,
                self.angular_res,
                self.angular_res,
                self.angular_res,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        centers = torch.tensor(
            [
                (self.cartesian_res / 2) + self.x_lim[0],
                (self.cartesian_res / 2) + self.y_lim[0],
                (self.cartesian_res / 2) + self.z_lim[0],
                (self.angular_res / 2) + self.r_lim[0],
                (self.angular_res / 2) + self.p_lim[0],
                (self.angular_res / 2) + self.yaw_lim[0],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(self.num_loops):
            loop_t0 = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            if loop_t0:
                loop_t0.record()

            th_batch = self.sampling_distr.sample([self.N_fk_loop])

            ee_transf_batch = chain.forward_kinematics(th_batch).get_matrix()
            torch.cuda.empty_cache()
            poses_6d = torch.hstack(
                (
                    ee_transf_batch[:, :3, 3],
                    pk.transforms.matrix_to_euler_angles(ee_transf_batch[:, :3, :3], "XYZ"),
                )
            )
            indices_6d = self._compute_indices(poses_6d)
            poses_6d = indices_6d * offsets
            poses_6d += centers
            indices_6d = self._discretize_indices(indices_6d)
            indices_6d = indices_6d.cpu()
            poses_6d = poses_6d.cpu()
            del ee_transf_batch
            torch.cuda.empty_cache()

            J = chain.jacobian(th_batch)
            torch.cuda.empty_cache()
            M = torch.det(J @ torch.transpose(J, 1, 2)).cpu()
            del J
            torch.cuda.empty_cache()

            self.reach_map[indices_6d, :6] = poses_6d
            self.reach_map[indices_6d, -2] += 1
            self.reach_map[indices_6d, -1] = np.maximum(self.reach_map[indices_6d, -1], M)

            if i % self.save_freq == 0:
                if loop_t0 and self.device == "cuda":
                    loop_t1 = torch.cuda.Event(enable_timing=True)
                    loop_t1.record()
                    torch.cuda.synchronize()
                    t_comp = loop_t0.elapsed_time(loop_t1) / 1000.0
                else:
                    t_comp = None

                nonzero_rows = torch.abs(self.reach_map).sum(dim=1) > 0
                reach_map_nonzero = self.reach_map[nonzero_rows].numpy()
                pkl_path = os.path.join(self.reach_map_dir, f"{self.reach_map_file_name}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(reach_map_nonzero, f)

                manip_scaling = 500
                h5_path = self._save_map(reach_map_nonzero, "3D", manip_scaling)
                if self.log_progress:
                    msg = f"Loop: {i}."
                    if t_comp is not None:
                        msg += f" Comp Time = {t_comp:.3f}s"
                    msg += f" Saved: {os.path.basename(pkl_path)}, {os.path.basename(h5_path)}"
                    print(msg)

    def visualize_h5(self, h5_path: str, downsample: int = 1, save_path: Optional[str] = None) -> None:
        """
        Visualize the reachability spheres using matplotlib.
        """
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"h5 file not found: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            spheres = np.array(f["/Spheres/sphere_dataset"])

        if downsample > 1:
            spheres = spheres[::downsample]

        coords = spheres[:, :3]
        manip = spheres[:, 3] if spheres.shape[1] > 3 else np.zeros(coords.shape[0])

        import matplotlib.pyplot as plt

        # Approximate marker area so screen-space diameter scales with cartesian_res
        # marker_size = max(4.0, (self.cartesian_res * 1200.0) ** 2)
        marker_size = 500

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=manip,
            cmap="viridis",
            s=marker_size,
            alpha=0.1,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("manipulability")
        ax.set_title(f"Reachability map: {os.path.basename(h5_path)}")
        ax.view_init(elev=25, azim=45)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()

    def visualize_h5_pybullet(
        self,
        h5_path: str,
        slice_thickness: Optional[float] = None,
        downsample: int = 1,
        max_points: int = 20000,
    ) -> None:
        """
        Interactive pybullet visualization of a reachability map slice.
        """
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"h5 file not found: {h5_path}")

        try:
            import pybullet as p
        except ImportError as exc:
            raise ImportError("pybullet is required for visualize_h5_pybullet") from exc

        import time

        with h5py.File(h5_path, "r") as f:
            spheres = np.array(f["/Spheres/sphere_dataset"])

        if downsample > 1:
            spheres = spheres[::downsample]

        coords = spheres[:, :3]
        manip = spheres[:, 3] if spheres.shape[1] > 3 else np.zeros(coords.shape[0])

        slice_thickness = slice_thickness or self.cartesian_res
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        manip_min, manip_max = manip.min(), manip.max()
        manip_range = manip_max - manip_min if manip_max != manip_min else 1.0

        if p.isConnected():
            _ = p.getConnectionInfo().get("connectionMethod")
        else:
            _ = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        # p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0, 0.5, 0.8])

        slider = p.addUserDebugParameter("z_slice", z_min, z_max, (z_min + z_max) / 2.0)
        point_id = None

        while True:
            z_slice = p.readUserDebugParameter(slider)
            mask = np.abs(coords[:, 2] - z_slice) <= (slice_thickness / 2.0)
            idxs = np.nonzero(mask)[0]

            if idxs.size == 0:
                time.sleep(0.05)
                continue

            if idxs.size > max_points:
                step = max(1, idxs.size // max_points)
                idxs = idxs[::step]

            pts = coords[idxs]
            mvals = manip[idxs]
            norm = np.clip((mvals - manip_min) / manip_range, 0.0, 1.0)

            stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            r = np.interp(norm, stops, [1.0, 1.0, 0.0, 0.0, 0.0])  # red -> yellow -> green -> cyan -> blue
            g = np.interp(norm, stops, [0.0, 1.0, 1.0, 1.0, 0.0])
            b = np.interp(norm, stops, [0.0, 0.0, 0.0, 1.0, 1.0])
            colors = np.stack([r, g, b], axis=1)

            if point_id is not None:
                p.removeUserDebugItem(point_id)
            point_id = p.addUserDebugPoints(pointPositions=pts.tolist(), pointColorsRGB=colors.tolist(), pointSize=25)

            # p.stepSimulation()
            time.sleep(0.001)


def _default_sampler_kwargs() -> dict:
    """
    Provide a readable set of defaults for TorchKinMapper construction.
    """
    return {
        # "robot_urdf": "YOUR_ROBOT_URDF_PATH",
        # "name_end_effector": "YOUR_END_EFFECTOR_LINK_NAME",
        # "name_base_link": "YOUR_BASE_LINK_NAME",
        # "n_dof": YOUR_NUMBER_OF_JOINTS,
        "device": "cuda" ,
        "dtype": torch.float32,
        # "angular_res": RESOLUTION_IN_RADIANS,
        "cartesian_res": 0.05,
        "x_lim": [-1, 1],
        "y_lim": [-1, 1],
        "z_lim": [-0.5, 1],
        # "z_lim": [MIN_Z, MAX_Z],
        # "r_lim": [MIN_ROLL, MAX_ROLL],
        # "p_lim": [MIN_PITCH, MAX_PITCH],
        # "yaw_lim": [MIN_YAW, MAX_YAW],
        # "joint_pos_min": [MIN_JOINT_1, ..., MIN_JOINT_N],
        # "joint_pos_max": [MAX_JOINT_1, ..., MAX_JOINT_N],
        "N_fk": 1280000000, # 1280000000
        "num_loops": 500,
        "save_freq": 10,
        # "log_progress": LOG_PROGRESS_FLAG,
        # "reach_map_dir": OUTPUT_DIRECTORY_FOR_MAP_ARTIFACTS
    }


def main() -> None:
    """
    Entry point for generating a reachability map with sensible defaults.
    """
    sampler = TorchKinMapper(**_default_sampler_kwargs())
    sampler.generate_maps()
    # sampler.visualize_h5(os.path.join(sampler.reach_map_dir, f"3D_{sampler.reach_map_file_name}.h5"))
    # sampler.visualize_h5_pybullet(os.path.join(sampler.reach_map_dir, f"3D_{sampler.reach_map_file_name}.h5"))


if __name__ == "__main__":
    main()
