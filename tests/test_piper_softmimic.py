"""Official-source alignment and embodiment boundaries, without Isaac Sim startup."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from sim.piper_softmimic.config import AGENT, UPSTREAM, UPSTREAM_COMMIT, REPO
from sim.piper_softmimic.upstream import assert_version, force_methods
from sim.piper_softmimic.asset import prepare_urdf


def test_benchmark_comparison_checks_optimizer_state_too(tmp_path):
    torch=pytest.importorskip('torch')
    from sim.piper_softmimic.compare_benchmarks import compare
    baseline=tmp_path/'baseline';candidate=tmp_path/'candidate'
    state={'model_state_dict':{'weight':torch.tensor([1.])},'optimizer_state_dict':{'moment':torch.tensor([2.])}}
    for path,seconds in ((baseline,1.),(candidate,.5)):
        path.mkdir();(path/'report.json').write_text(json.dumps({'status':'complete','iterations':3,'mean_collection_s':seconds,'mean_learning_s':.5}))
        np.savez(path/'rollout.npz',q=np.array([1.,2.]))
        torch.save(state,path/'model_3.pt')
    report=compare(baseline,candidate)
    assert report['bitwise_equal'] and report['throughput_speedup']==1.5
    state['optimizer_state_dict']['moment']+=1;torch.save(state,candidate/'model_3.pt')
    report=compare(baseline,candidate)
    assert not report['bitwise_equal'] and report['rollout']['q']['equal']
    assert 'checkpoint.optimizer_state_dict.moment' in report['checkpoint_differences']


def sampling_adapter():
    # Exercise the actual hot-path methods without launching Isaac Sim.
    torch=pytest.importorskip('torch')
    tree=ast.parse((REPO/'sim/piper_softmimic/env.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='PiperEnv')
    cls.bases=[];cls.decorator_list=[]
    cls.body=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ('sample','future_q','_clear_samples')]
    helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='history_indices')
    code=ast.Module(body=[helper,cls],type_ignores=[]);ast.fix_missing_locations(code)
    namespace={'torch':torch};exec(compile(code,'env.py','exec'),namespace)
    return namespace


def test_sampling_cache_preserves_boundaries_and_invalidates_on_clock_change():
    torch=pytest.importorskip('torch');adapter=sampling_adapter()['PiperEnv']()
    adapter.dataset={'lengths':torch.tensor([3,4]),'q_aug':torch.tensor([[[0.],[10.],[20.],[99.]],[[30.],[40.],[50.],[60.]]])}
    adapter.clip=torch.tensor([0,0,1]);adapter.phase=torch.tensor([-.5,.5,5.]);adapter.motion_time=torch.zeros(3);adapter.data_dt=1.
    adapter.num_envs=3
    adapter._samples={};adapter._indices={}
    adapter.command=SimpleNamespace(invalidate=lambda:None)
    result=adapter.sample('q_aug');torch.testing.assert_close(result,torch.tensor([[0.],[5.],[60.]]),rtol=0,atol=0)
    assert adapter.sample('q_aug') is result
    adapter.motion_time+=1;adapter._clear_samples()
    torch.testing.assert_close(adapter.sample('q_aug'),torch.tensor([[5.],[15.],[60.]]),rtol=0,atol=0)
    adapter.future_offsets=torch.tensor([0.,.5,1.,100.])
    expected=torch.stack([adapter.sample('q_aug',float(offset)) for offset in adapter.future_offsets],1)
    torch.testing.assert_close(adapter.future_q(),expected,rtol=0,atol=0)


def test_history_gather_preserves_official_term_major_order():
    torch=pytest.importorskip('torch');indices=sampling_adapter()['history_indices']
    widths=[3,3,6,6,6,6,2,1,1,3,1,1]
    for parts in (widths,widths+[3,3,1,1,3,3,6,2,1,1,3,3,3,15,3,1]):
        history=torch.arange(2*3*sum(parts)).reshape(2,3,sum(parts))
        expected=torch.cat([part.reshape(2,-1) for part in torch.split(history,parts,dim=-1)],dim=-1)
        actual=history.reshape(2,-1)[:,indices(parts)]
        assert torch.equal(actual,expected)




def test_ppo_and_network_values_match_the_pinned_official_runner():
    if not UPSTREAM.exists():pytest.skip('Pinned vendor checkout not present')
    assert_version()
    path=UPSTREAM/'softmimic_gym/softmimic_gym/tasks/locomotion/tracking/config/g1/agents/rsl_rl_ppo_cfg.py'
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='G1StuntRunnerCfg')
    for statement in cls.body:
        if not isinstance(statement,ast.Assign):continue
        key=statement.targets[0].id
        if key in ('policy','algorithm'):
            source={item.arg:ast.literal_eval(item.value) for item in statement.value.keywords}
            assert {name:value for name,value in AGENT[key].items() if name!='class_name'}==source
        elif key in ('num_steps_per_env','max_iterations','save_interval','empirical_normalization'):
            assert AGENT[key]==ast.literal_eval(statement.value)


def test_piper_adapter_preserves_arm_joints_and_inertias():
    original=ET.parse(REPO/'src/piper_ros/src/piper_description/urdf/piper_description.urdf').getroot()
    adapted=ET.parse(prepare_urdf()).getroot()
    for index in range(1,7):
        a=original.find(f"joint[@name='joint{index}']");b=adapted.find(f"joint[@name='joint{index}']")
        assert ET.tostring(a)==ET.tostring(b)
        a=original.find(f"link[@name='link{index}']/inertial");b=adapted.find(f"link[@name='link{index}']/inertial")
        assert ET.tostring(a)==ET.tostring(b)
    assert all(adapted.find(f"joint[@name='joint{i}']").get('type')=='fixed' for i in (7,8))
    assert adapted.find("joint[@name='tcp_fixed']/origin").get('xyz')=='0 0 0.16'


def test_original_forcefield_is_in_equilibrium_and_reacts_to_displacement():
    torch=pytest.importorskip('torch')
    functions=force_methods({'torch':torch,'math_utils':SimpleNamespace()})
    force=torch.tensor([[1.,2.,3.],[0.,0.,0.]])
    desired=torch.tensor([[.1,.2,.3],[.4,.5,.6]])
    stiffness=torch.tensor([100.,20.])
    command=SimpleNamespace(num_envs=2,device='cpu',body_indices=torch.zeros(2,dtype=torch.long),
        active_force_mask=torch.tensor([True,False]),_stiffness=torch.ones(2,1)*100,
        _ff_stiffness=stiffness[:,None,None],_ff_origin=(desired+force/stiffness[:,None])[:,None],
        _ff_normal=torch.zeros(2,1,3),robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=desired[:,None].clone())))
    calculate=functions['_compute_reactive_forcefield_force_w']
    torch.testing.assert_close(calculate(command),force)
    command.robot.data.body_pos_w[0,0,0]+=.01
    torch.testing.assert_close(calculate(command),force-torch.tensor([[1.,0.,0.],[0.,0.,0.]]))


def test_dense_forcefield_matches_release_for_plane_setpoint_and_inactive_rows():
    torch=pytest.importorskip('torch')
    tree=ast.parse((REPO/'sim/piper_softmimic/env.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='dense_forcefield_force')
    namespace={'torch':torch};code=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(code)
    exec(compile(code,'env.py','exec'),namespace)
    original=force_methods({'torch':torch,'math_utils':SimpleNamespace()})['_compute_reactive_forcefield_force_w']
    positions=torch.tensor([[[1.,2.,3.]],[[1.,2.,3.]],[[-1.,2.,3.]],[[1.,2.,3.]]])
    command=SimpleNamespace(num_envs=4,device='cpu',body_indices=torch.zeros(4,dtype=torch.long),
        active_force_mask=torch.tensor([True,True,True,False]),_stiffness=torch.ones(4,1)*100,
        _ff_stiffness=torch.ones(4,1,1)*100,_ff_origin=torch.zeros(4,1,3),
        _ff_normal=torch.tensor([[[0.,0.,0.]],[[1.,0.,0.]],[[1.,0.,0.]],[[0.,0.,0.]]]),
        robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=positions)))
    calculate=namespace['dense_forcefield_force']
    torch.testing.assert_close(calculate(command),original(command),rtol=0,atol=0)
    torch.testing.assert_close(calculate(command),torch.tensor([[-100.,-200.,-300.],[0.,0.,0.],[100.,0.,0.],[0.,0.,0.]]),rtol=0,atol=0)
    command.active_force_mask[:]=False
    assert torch.equal(calculate(command),original(command))
