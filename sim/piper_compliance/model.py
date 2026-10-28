"""Piper meshes/inertia from MJCF; joint definitions from the control URDF."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


DESCRIPTION = (Path(__file__).resolve().parents[2]
               / "src/piper_ros/src/piper_description")
ASSET = DESCRIPTION / "mujoco_model/piper_description.xml"
CONTROL_URDF = DESCRIPTION / "urdf/piper_description.urdf"


def urdf_reference(path):
    root = ET.parse(path).getroot()
    # Only the kinematic/dynamic reference is needed; mesh loading stays in MJCF.
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                link.remove(element)
    reference = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    names = [reference.joint(i).name for i in range(reference.njnt)]
    if names != [f"joint{i}" for i in range(1, 9)]:
        raise ValueError("Control URDF must describe six arm joints and two finger joints")
    return reference


def synchronize_joints(root, reference):
    for parent in root.iter():
        for body in parent.findall("body"):
            joints = body.findall("joint")
            if len(joints) != 1:
                raise ValueError("Expected one URDF joint per Piper MJCF body")
            joint = joints[0]
            ref_joint = reference.joint(joint.get("name"))
            ref_body = reference.body(int(reference.jnt_bodyid[ref_joint.id]))
            ref_parent = reference.body(int(reference.body_parentid[ref_body.id])).name
            parent_name = "world" if parent.tag == "worldbody" else parent.get("name")
            if ref_body.name != body.get("name") or ref_parent != parent_name:
                raise ValueError(f"MJCF/URDF body hierarchy differs at {body.get('name')}")
            if not reference.jnt_limited[ref_joint.id]:
                raise ValueError(f"Control URDF requires finite position limits: {ref_joint.name}")
            joint_type = {mujoco.mjtJoint.mjJNT_HINGE: "hinge",
                          mujoco.mjtJoint.mjJNT_SLIDE: "slide"}.get(ref_joint.type[0])
            if joint_type is None:
                raise ValueError(f"Unsupported joint type: {ref_joint.name}")
            body.set("pos", " ".join(map(str, reference.body_pos[ref_body.id])))
            body.set("quat", " ".join(map(str, reference.body_quat[ref_body.id])))
            joint.set("pos", " ".join(map(str, reference.jnt_pos[ref_joint.id])))
            joint.set("axis", " ".join(map(str, reference.jnt_axis[ref_joint.id])))
            joint.set("range", " ".join(map(str, reference.jnt_range[ref_joint.id])))
            joint.set("type", joint_type)
            joint.set("limited", "true")


def inertia_tensor(model, body_id):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T


def check_model_consistency(model, reference):
    """Fail before simulation if joint, fixed-finger, or inertial assumptions drift."""
    names = [f"joint{i}" for i in range(1, 7)]
    if ((model.nq, model.nv, model.nu) != (6, 6, 6)
            or [model.joint(i).name for i in range(model.njnt)] != names):
        raise ValueError("Expected exactly six movable arm joints and locked fingers")
    for name in names:
        actual, expected = model.joint(name).id, reference.joint(name).id
        if (model.jnt_type[actual] != reference.jnt_type[expected]
                or not model.jnt_limited[actual]
                or not np.allclose(model.jnt_range[actual], reference.jnt_range[expected], atol=1e-12, rtol=0)
                or not np.allclose(model.jnt_axis[actual], reference.jnt_axis[expected], atol=1e-12, rtol=0)):
            raise ValueError(f"Joint definition differs from control URDF: {name}")
    for i in range(1, 9):
        name = f"link{i}"
        actual, expected = model.body(name).id, reference.body(name).id
        # MJCF stores rounded principal inertias, including merged gripper_base.
        expected_tensor = inertia_tensor(reference, expected)
        if (not np.isclose(model.body_mass[actual], reference.body_mass[expected], rtol=3e-5, atol=1e-9)
                or np.linalg.norm(model.body_ipos[actual] - reference.body_ipos[expected]) > 1e-6
                or np.linalg.norm(inertia_tensor(model, actual) - expected_tensor)
                > 3e-5 * np.linalg.norm(expected_tensor)):
            raise ValueError(f"Retained MJCF inertia differs from control URDF: {name}")
        if i >= 7 and model.body_jntnum[actual] != 0:
            raise ValueError("Finger joints must remain locked")
    actual_data, ref_data = mujoco.MjData(model), mujoco.MjData(reference)
    for fraction in (None, .25, .5, .75):
        q = np.zeros(6) if fraction is None else (
            model.jnt_range[:, 0] * (1. - fraction) + model.jnt_range[:, 1] * fraction)
        actual_data.qpos[:] = q
        ref_data.qpos[:6] = q
        mujoco.mj_forward(model, actual_data)
        mujoco.mj_forward(reference, ref_data)
        for i in range(1, 9):
            name = f"link{i}"
            actual, expected = model.body(name).id, reference.body(name).id
            if (not np.allclose(actual_data.xpos[actual], ref_data.xpos[expected], atol=1e-9, rtol=0)
                    or not np.allclose(actual_data.xmat[actual], ref_data.xmat[expected], atol=1e-9, rtol=0)):
                raise ValueError(f"Forward kinematics differs from control URDF: {name}")
