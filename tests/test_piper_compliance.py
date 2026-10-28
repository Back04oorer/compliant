"""Shared model and IK checks for SoftMimic target generation."""
import numpy as np
import pytest
import xml.etree.ElementTree as ET

mujoco = pytest.importorskip("mujoco")

from sim.piper_compliance.kinematics import Q_REF, PiperKinematics, load_model
from sim.piper_compliance.model import ASSET, CONTROL_URDF, check_model_consistency, urdf_reference


def test_fk_matches_control_urdf_across_workspace():
    model, reference = load_model(), urdf_reference(CONTROL_URDF)
    data, ref_data = mujoco.MjData(model), mujoco.MjData(reference)
    rng = np.random.default_rng(0)
    poses = [np.zeros(6), Q_REF]
    poses.extend(rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1], size=(50, 6)))
    for q in poses:
        data.qpos[:], ref_data.qpos[:6] = q, q
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(reference, ref_data)
        for i in range(1, 9):
            actual, expected = model.body(f"link{i}").id, reference.body(f"link{i}").id
            np.testing.assert_allclose(data.xpos[actual], ref_data.xpos[expected], atol=1e-10, rtol=0)
            np.testing.assert_allclose(data.xmat[actual], ref_data.xmat[expected], atol=1e-10, rtol=0)
    np.testing.assert_allclose(model.jnt_range, reference.jnt_range[:6], atol=1e-12, rtol=0)



def test_joint_definitions_follow_updated_urdf(tmp_path):
    root = ET.parse(CONTROL_URDF).getroot()
    root.find("joint[@name='joint1']/limit").set("upper", "2.4")
    root.find("joint[@name='joint2']/origin").set("rpy", "1.5708 -0.17 -3.1416")
    root.find("joint[@name='joint4']/axis").set("xyz", "0 0.3 1")
    path = tmp_path / "updated.urdf"
    ET.ElementTree(root).write(path)
    model, reference = load_model(path), urdf_reference(path)
    assert model.jnt_range[model.joint("joint1").id, 1] == 2.4
    np.testing.assert_allclose(model.body_quat[model.body("link2").id],
                               reference.body_quat[reference.body("link2").id], atol=1e-12)
    np.testing.assert_allclose(model.jnt_axis[model.joint("joint4").id],
                               np.array([0., .3, 1.]) / np.linalg.norm([0., .3, 1.]), atol=1e-12)



def test_vendor_meshes_and_inertias_are_preserved():
    original = mujoco.MjModel.from_xml_path(str(ASSET))
    synchronized = load_model()
    for array in ("mesh_vert", "mesh_face", "body_mass", "body_ipos", "body_inertia", "body_iquat"):
        np.testing.assert_array_equal(getattr(synchronized, array), getattr(original, array))



@pytest.mark.parametrize("change", ["origin", "range", "inertia"])
def test_automatic_consistency_check_detects_drift(change):
    model = load_model()
    if change == "origin":
        old = ET.parse(ASSET).getroot().find(".//body[@name='link2']")
        model.body_quat[model.body("link2").id] = np.fromstring(old.get("quat"), sep=" ")
    elif change == "range":
        model.jnt_range[model.joint("joint1").id, 1] = 2.168
    else:
        model.body_mass[model.body("link2").id] *= 1.1
    with pytest.raises(ValueError, match="differs"):
        check_model_consistency(model, urdf_reference(CONTROL_URDF))



@pytest.mark.parametrize("force", [(2., 0., 0.), (0., -2., 0.), (0., 0., 2.)])
def test_ik_targets_preserve_fingers_orientation_and_force_displacement(force):
    env = PiperKinematics()
    assert (env.model.nq, env.model.nv, env.model.nu) == (6, 6, 6)
    assert env.model.body("link7").jntnum == env.model.body("link8").jntnum == 0
    q = env.augmented_pose(np.array(force), 100.)
    assert np.all(q >= env.model.jnt_range[:, 0])
    assert np.all(q <= env.model.jnt_range[:, 1])
    np.testing.assert_allclose(env.ik_data.site_xpos[env.tcp_id],
                               env.reference_pos + np.array(force) / 100., atol=1e-5)
    np.testing.assert_allclose(env.ik_data.site_xmat[env.tcp_id].reshape(3, 3),
                               env.reference_rot, atol=1e-4)


def test_ik_rejects_unreachable_evaluation_targets():
    with pytest.raises(ValueError, match="unreachable"):
        PiperKinematics().augmented_pose(np.array([1000., 0., 0.]), 100.)
