import json

import numpy as np
import pytest

from sim.piper_softmimic.balanced_augmentation import DT, audit, plans, schedule


def test_plan_covers_each_small_force_stiffness_direction_and_field_equally():
    cases=plans()
    assert len(cases)==252
    assert cases==plans()
    assert cases!=plans(seed=7)
    for axis in 'XYZ':
        for sign in (-1,1):
            selected=[c for c in cases if c['axis']==axis and c['sign']==sign]
            assert len(selected)==42
            for force in (.5,1,2,3,5):
                for k in (60,100,160):
                    group=[c for c in selected if c['force_N']==force and c['stiffness_N_per_m']==k]
                    if force==5 and k==60:assert not group
                    else:
                        assert {c['profile'] for c in group}=={'steady','two_peaks','modulated'}
                        assert {c['field_stiffness_N_per_m'] for c in group}=={10.,100.,500.}


@pytest.mark.parametrize('profile',['steady','two_peaks','modulated'])
def test_each_clip_contains_hold_push_and_unforced_recovery(profile):
    time,stage,fraction=schedule(profile)
    np.testing.assert_array_equal(np.bincount(stage),[180,360,180])
    assert np.all(fraction[time<6]==0) and np.all(fraction[time>=19]==0)
    assert np.all(fraction[(time>6)&(time<18)]>0)
    assert fraction.max()==pytest.approx(1)
    assert np.abs(np.diff(fraction)).max()<.22


def test_audit_rejects_metadata_origin_jump_at_force_onset():
    clips=[dict(axis=a,sign=s,force_N=2.,stiffness_N_per_m=100.) for a in 'XYZ' for s in (-1,1)]
    time,stage,fraction=schedule('steady');reference=np.array([.5,.1,.6])
    force=np.stack([fraction[:,None]*np.eye(3)['XYZ'.index(c['axis'])]*c['sign']*2 for c in clips])
    metadata=np.zeros((6,len(time),12));metadata[:,:,0]=100;metadata[:,:,2:5]=reference+force*.02
    data=dict(force=force,metadata=metadata,stage=np.tile(stage,(6,1)),lengths=np.full(6,len(time)),
              force_fraction=np.tile(fraction,(6,1)),keypoints=(reference+force/100)[:,:,None])
    result=audit(data,clips)
    assert result['phase_fractions']=={'hold':.25,'push':.5,'release_recovery':.25}
    assert result['interpolated_50hz_forcefield_equilibrium_verified']
    # A zero inactive world origin produces a large artificial force during interpolation.
    data['metadata'][0,fraction==0,2:5]=0
    with pytest.raises(AssertionError):audit(data,clips)


def test_resume_selects_recorded_dataset_and_rejects_changed_distribution(tmp_path,monkeypatch):
    from sim.piper_softmimic import train
    old=tmp_path/'old';old.mkdir();(old/'manifest.json').write_text('{"sha256":"old"}')
    new=tmp_path/'balanced';new.mkdir();(new/'manifest.json').write_text('{"sha256":"new"}')
    checkpoint=tmp_path/'model_1.pt';checkpoint.touch()
    monkeypatch.setattr(train,'DATASET',old)
    manifest=dict(upstream_commit=train.UPSTREAM_COMMIT,num_envs=4096,physics_dt=.005,control_dt=.02,
                  actor_observation_dim=117,critic_observation_dim=793,dataset={'sha256':'old'},agent=train.AGENT)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert train.training_dataset(checkpoint)==old
    with pytest.raises(ValueError,match='dataset does not match'):train.validate_resume(checkpoint,4096,new)
    manifest.update(dataset_path=str(new),dataset={'sha256':'new'})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert train.training_dataset(checkpoint)==new
    train.validate_resume(checkpoint,4096,new)
