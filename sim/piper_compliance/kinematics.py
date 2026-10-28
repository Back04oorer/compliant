"""MuJoCo model and fixed-pose IK used by SoftMimic augmentation and evaluation."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .model import ASSET, CONTROL_URDF, check_model_consistency, synchronize_joints, urdf_reference

Q_REF = np.array([0.0, 2.0, -2.0, 0.2, -0.8, 0.0])
KP = np.array([60., 80., 60., 20., 15., 10.])
TORQUE_LIMIT = np.array([12., 12., 12., 5., 5., 5.])


def load_model(urdf_path=CONTROL_URDF):
    """Retain vendor meshes/inertia and synchronize joints from the control URDF."""
    root = ET.parse(ASSET).getroot()
    reference = urdf_reference(urdf_path)
    synchronize_joints(root, reference)
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((ASSET.parent / mesh.get("file")).resolve()))
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            if joint.get("name") in ("joint7", "joint8"):
                body.remove(joint)  # Both fingers are rigidly fixed at model zero.
            else:
                joint.set("damping", "0.1")
                joint.set("armature", "0.01")
    # shortcut: no collision contacts in this force-only test; add calibrated
    # collision geometry before training on obstacles or grasping.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    actuators = root.find("actuator")
    actuators.clear()
    for i, limit in enumerate(TORQUE_LIMIT, 1):
        ET.SubElement(actuators, "motor", name=f"motor{i}", joint=f"joint{i}",
                      ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "geom", type="plane", size="2 2 0.1", pos="0 0 -0.01",
                  rgba="0.22 0.25 0.28 1", contype="0", conaffinity="0")
    wrist = next(b for b in root.iter("body") if b.get("name") == "link6")
    ET.SubElement(wrist, "site", name="tcp", pos="0 0 0.16", size="0.008",
                  rgba="1 0.4 0.1 1")
    # Preserve finger bodies (and their masses) for inspection and visualization.
    root.find("compiler").set("fusestatic", "false")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    check_model_consistency(model, reference)
    return model


class PiperKinematics:
    def __init__(self):
        self.model = load_model()
        self.ik_data = mujoco.MjData(self.model)
        self.tcp_id = self.model.site("tcp").id
        self.ik_data.qpos[:] = Q_REF
        mujoco.mj_forward(self.model, self.ik_data)
        self.reference_pos = self.ik_data.site_xpos[self.tcp_id].copy()
        self.reference_rot = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def augmented_pose(self, force, stiffness, *, initial=None):
        """Damped IK: translate the TCP by F/K while retaining its orientation."""
        self.ik_data.qpos[:] = Q_REF if initial is None else initial
        self.ik_data.qvel[:] = 0
        target = self.reference_pos + force / stiffness
        jp, jr = np.zeros((3, 6)), np.zeros((3, 6))
        for _ in range(60):
            mujoco.mj_forward(self.model, self.ik_data)
            rotation = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3)
            rot_error = 0.5 * sum(np.cross(rotation[:, i], self.reference_rot[:, i])
                                  for i in range(3))
            error = np.r_[target - self.ik_data.site_xpos[self.tcp_id], 0.2 * rot_error]
            if np.linalg.norm(error) < 1e-6:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.tcp_id)
            jac = np.vstack([jp, 0.2 * jr])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
            self.ik_data.qpos[:] = np.clip(self.ik_data.qpos + np.clip(delta, -.1, .1),
                                          self.model.jnt_range[:, 0] + .02,
                                          self.model.jnt_range[:, 1] - .02)
        mujoco.mj_forward(self.model, self.ik_data)
        if np.linalg.norm(self.ik_data.site_xpos[self.tcp_id] - target) > .002:
            raise ValueError("Requested force/stiffness produces an unreachable TCP target")
        return self.ik_data.qpos.copy()
