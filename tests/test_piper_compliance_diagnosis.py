import numpy as np
import pytest

from sim.piper_softmimic.diagnose_compliance import audit_dataset, external_force, split_push_error


def test_dataset_audit_excludes_padding_and_measures_ik_fidelity(tmp_path):
    force=np.zeros((2,4,3));force[0,1,2]=2;force[0,2:,2]=1000
    positions=np.zeros((2,4,1,3));positions[0,1,0,2]=.02
    path=tmp_path/'dataset.npz'
    np.savez(path,force=force,lengths=np.array([2,4]),keypoints=positions,k=np.full((2,4),100.),metadata=np.zeros((2,4,12)))
    report=audit_dataset(path)
    assert report['frames']==6 and report['active_frames']==1
    assert report['active_fraction']==pytest.approx(1/6)
    assert report['force_N']['max']==2
    assert report['ik_target_error_mm']['max']==0
    assert report['directional_frames']['+Z']['near_2N_K100_frames']==1


def test_initial_bias_can_hide_bad_positive_z_compliance():
    time=np.arange(601)*.02;force=np.zeros((601,3));force[(time>=3)&(time<=6),2]=2
    steady=(time>=3)&(time<=6)
    old=np.zeros_like(force);old[:,2]=.0075;old[steady,2]+=.0046
    new=np.zeros_like(force);new[:,2]=.0023;new[steady,2]+=.0045
    a=split_push_error(time,old,np.zeros(3),force,100)
    b=split_push_error(time,new,np.zeros(3),force,100)
    assert a['debiased_push_rmse_m']==pytest.approx(.0154)
    assert b['debiased_push_rmse_m']==pytest.approx(.0155)
    # Absolute error increases mostly because the upward initial bias shrank.
    assert .02-new[200,2]>.02-old[200,2]


def test_reactive_field_has_same_desired_equilibrium_but_stronger_force_when_resisted():
    peak=np.array([0.,0.,2.]);reference=np.zeros(3)
    for field in (30.,100.,500.):
        at_goal=external_force('field',1.,peak,100.,field,reference,peak/100)
        holding=external_force('field',1.,peak,100.,field,reference,reference)
        np.testing.assert_allclose(at_goal,peak)
        np.testing.assert_allclose(holding,peak*(1+field/100))
    np.testing.assert_array_equal(external_force('field',0.,peak,100.,500.,reference,peak),np.zeros(3))
    np.testing.assert_array_equal(external_force('prescribed',1.,peak,100.,500.,reference,reference),peak)
    np.testing.assert_array_equal(external_force('frozen_zero',1.,peak,100.,500.,reference,reference),np.zeros(3))
