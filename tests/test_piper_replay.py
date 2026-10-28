import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sim.piper_softmimic.export_replay import extract, robot_description


def test_replay_controls_and_three_kinematics_with_exported_artifacts():
    root=Path(__file__).resolve().parents[1]
    if not shutil.which('node') or not (root/'outputs/piper_compliance/softmimic_official/replay/data.json').exists():
        pytest.skip('Requires Node and exported evaluation replay')
    subprocess.run(['node','--experimental-vm-modules',str(root/'tests/piper_replay_ui.cjs')],
                   cwd=root,check=True,capture_output=True,text=True)


@pytest.mark.parametrize('amplitude',[.5,1,2,3,5])
def test_export_keeps_only_matched_nominal_cases_and_preserves_time(tmp_path,amplitude):
    rows=[dict(controller='policy',variant='nominal',force_N=amplitude,stiffness_N_per_m=100,
               sign=sign,axis=axis) for axis in 'XYZ' for sign in (-1,1)]
    rows += [dict(rows[0],controller='analytic'),dict(rows[0],variant='heavy')]
    (tmp_path/'report.json').write_text(json.dumps(dict(status='complete',cases=rows,
        checkpoint_name='model_19999.pt',checkpoint_sha256='weights',protocol_sha256='protocol')))
    q=np.arange(3*8*6).reshape(3,8,6)*.001
    tcp=np.arange(3*8*3).reshape(3,8,3)*.001
    np.savez(tmp_path/'traces.npz',q=q,tcp=tcp,force=np.zeros_like(tcp),time=[0,2.6,12],reference_pos=[0,0,.3])
    result=extract(tmp_path,'policy',amplitude)
    assert set(result['cases'])=={'-X','+X','-Y','+Y','-Z','+Z'}
    assert result['time']==[0,2.6,12]
    np.testing.assert_allclose(result['cases']['+Z']['q'],q[:,5])
    np.testing.assert_allclose(result['cases']['+Z']['pre'],tcp[1,5])
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(AssertionError):extract(tmp_path,'policy')


def test_robot_export_retains_origin_axis_scale_and_hides_local_paths(tmp_path):
    path=tmp_path/'model.urdf'
    path.write_text('''<robot><link name="base"><visual><origin xyz="1 2 3" rpy="0 1 0"/>
      <geometry><mesh filename="/private/assets/base.STL" scale=".1 .2 .3"/></geometry></visual></link>
      <joint name="joint1" type="revolute"><parent link="base"/><child link="arm"/>
      <origin xyz="0 0 .1" rpy="1 2 3"/><axis xyz="0 1 0"/></joint></robot>''')
    result=robot_description(path)
    assert result['visuals'][0]['mesh']=='base.STL'
    assert result['visuals'][0]['scale']==[.1,.2,.3]
    assert result['joints'][0]['rpy']==[1,2,3]
    assert result['joints'][0]['axis']==[0,1,0]
    assert '/private' not in json.dumps(result)
