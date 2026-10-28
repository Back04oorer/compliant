import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_entrypoints import load
from test_tui_data import prompts


def run_dir(tmp_path):
    root=tmp_path/'outputs/piper_compliance/softmimic_official'
    run=root/'train';run.mkdir(parents=True)
    (run/'progress.json').write_text(json.dumps({'status':'paused','iteration':1088,'total_iterations':100000}))
    (run/'manifest.json').write_text('{}')
    (run/'model_500.pt').write_bytes(b'checkpoint')
    (run/'model_1000.pt').write_bytes(b'checkpoint')
    dataset=root/'dataset';dataset.mkdir()
    (dataset/'manifest.json').write_text('{}');(dataset/'train.npz').write_bytes(b'dataset')
    return root,run


@pytest.mark.parametrize('execute',[False,True])
def test_softmimic_menu_preserves_preview_and_interpreter_symlink(tmp_path,monkeypatch,execute):
    mod=load('teleop_tui');python=tmp_path/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    monkeypatch.setattr(sys,'argv',['tui','--softmimic-python',str(python)]+(['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin,'isatty',lambda:False)
    prompts(monkeypatch,['f','q']);calls=[]
    monkeypatch.setattr(mod,'hardware_status',lambda cfg:[])
    monkeypatch.setattr(mod,'run_softmimic_action',lambda *args:calls.append(args))
    assert mod.main()==0 and calls==[(python,execute)]


def test_softmimic_status_never_starts_a_process(tmp_path,monkeypatch,capsys):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);_,run=run_dir(tmp_path)
    prompts(monkeypatch,['status'])
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**kw:pytest.fail('Status probed runtime'))
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Status started training'))
    mod.run_softmimic_action('/missing/python',True)
    text=capsys.readouterr().out
    assert 'paused' in text and '1,088/100,000' in text and str(run/'model_1000.pt') in text


def test_softmimic_status_discovers_named_run_and_ignores_newer_smoke(tmp_path,monkeypatch,capsys):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,run=run_dir(tmp_path)
    run.rename(root/'FirstPPo')
    smoke=root/'smoke_latest';smoke.mkdir();(smoke/'manifest.json').write_text('{}')
    (smoke/'progress.json').write_text('{"status":"complete","iteration":2,"total_iterations":2}')
    prompts(monkeypatch,['status']);mod.run_softmimic_action('/missing/python',True)
    text=capsys.readouterr().out
    assert 'FirstPPo/model_1000.pt' in text and '1,088/100,000' in text


@pytest.mark.parametrize('execute',[False,True])
def test_softmimic_evaluate_routes_to_comparison_without_training(tmp_path,monkeypatch,capsys,execute):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,run=run_dir(tmp_path)
    run=run.rename(root/'FirstPPo')
    (run/'model_9500.pt').write_bytes(b'checkpoint');(run/'model_12500.pt').write_bytes(b'checkpoint')
    protocol=root/'evaluations/xyz_push_v1/protocol.npz';protocol.parent.mkdir(parents=True);protocol.touch()
    output=root/'evaluations/new_comparison'
    prompts(monkeypatch,['evaluate','',run/'model_500.pt','','',output])
    calls=[]
    def probe(*a,**kw):
        assert execute
    def start(command,**kw):
        assert execute;calls.append(command)
        assert 'sim.piper_softmimic.compare' in command and '--resume' not in command
        assert command[command.index('--checkpoint')+1]==str(run/'model_12500.pt')
        assert command[command.index('--noise')+1]=='both'
        output.mkdir();(output/'report.json').write_text('{"status":"complete"}')
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(mod.subprocess,'run',probe);monkeypatch.setattr(mod.subprocess,'Popen',start)
    mod.run_softmimic_action('/simulation/bin/python',execute)
    text=capsys.readouterr().out
    assert 'sim.piper_softmimic.compare' in text and 'model_12500.pt' in text
    assert ('saves a checkpoint' not in text) and ('--resume' not in text)
    assert len(calls)==int(execute) and output.exists()==execute


@pytest.mark.parametrize('mode',['resume','new'])
def test_softmimic_preview_builds_simulation_command_without_writes(tmp_path,monkeypatch,capsys,mode):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,run=run_dir(tmp_path);output=root/'train_next'
    prompts(monkeypatch,[mode]+([run/'model_1000.pt'] if mode=='resume' else [root/'dataset'])+[output])
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Preview started training'))
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**kw:pytest.fail('Preview ran probe'))
    mod.run_softmimic_action('/simulation/bin/python',False)
    text=capsys.readouterr().out
    assert '-m sim.piper_softmimic.train --accept-eula' in text and 'PREVIEW ONLY' in text
    assert ('--resume' in text)==(mode=='resume') and str(output) in text
    assert not output.exists() and 'roslaunch' not in text


@pytest.mark.parametrize('probe_fails',[False,True])
def test_softmimic_execute_uses_isolated_env_and_waits_for_checkpoint_cleanup(tmp_path,monkeypatch,probe_fails):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,_=run_dir(tmp_path)
    prompts(monkeypatch,['new',root/'dataset',root/'train_next']);monkeypatch.setenv('PYTHONPATH','/ros');monkeypatch.setenv('PYTHONHOME','/ros')
    calls=[]
    def probe(command,**kwargs):
        assert command[0]=='/simulation/bin/python' and 'isaaclab' in command[2]
        assert 'PYTHONPATH' not in kwargs['env'] and 'PYTHONHOME' not in kwargs['env']
        if probe_fails:raise subprocess.CalledProcessError(1,command)
    def start(command,**kwargs):
        assert kwargs['cwd']==tmp_path and kwargs['env']['OMP_NUM_THREADS']=='4'
        calls.append(command);waits=iter([KeyboardInterrupt(),0])
        def wait():
            value=next(waits)
            if isinstance(value,BaseException):raise value
            return value
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(mod.subprocess,'run',probe);monkeypatch.setattr(mod.subprocess,'Popen',start)
    if probe_fails:
        with pytest.raises(ValueError,match='--softmimic-python'):mod.run_softmimic_action('/simulation/bin/python',True)
        assert calls==[]
    else:
        mod.run_softmimic_action('/simulation/bin/python',True)
        assert len(calls)==1 and 'sim.piper_softmimic.train' in calls[0]


@pytest.mark.parametrize('mismatch',[None,'dataset','ppo','observation','environment'])
def test_resume_rejects_incompatible_checkpoint_manifests(tmp_path,monkeypatch,mismatch):
    from sim.piper_softmimic import train
    from sim.piper_softmimic.config import AGENT,UPSTREAM_COMMIT
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'manifest.json').write_text('{"sha256":"same"}')
    monkeypatch.setattr(train,'DATASET',dataset)
    run=tmp_path/'train';run.mkdir();checkpoint=run/'model_1000.pt';checkpoint.write_bytes(b'checkpoint')
    manifest={'upstream_commit':UPSTREAM_COMMIT,'num_envs':4096,'physics_dt':.005,'control_dt':.02,
              'actor_observation_dim':117,'critic_observation_dim':793,'dataset':{'sha256':'same'},'agent':copy.deepcopy(AGENT)}
    if mismatch=='dataset':manifest['dataset']['sha256']='other'
    elif mismatch=='ppo':manifest['agent']['algorithm']['gamma']=.9
    elif mismatch=='observation':manifest['actor_observation_dim']=57
    elif mismatch=='environment':manifest['num_envs']=16
    (run/'manifest.json').write_text(json.dumps(manifest))
    if mismatch:
        with pytest.raises(ValueError,match='match'):train.validate_resume(checkpoint,4096)
    else:assert train.validate_resume(checkpoint,4096)==manifest


def test_softmimic_reports_runtime_failure_even_if_isaac_close_returns_zero(tmp_path,monkeypatch,capsys):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,_=run_dir(tmp_path);output=root/'train_failed'
    prompts(monkeypatch,['new',root/'dataset',output]);monkeypatch.setattr(mod.subprocess,'run',lambda *a,**kw:None)
    def start(*a,**kw):
        output.mkdir();(output/'failure.json').write_text('{"error":"invalid simulation state"}')
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(mod.subprocess,'Popen',start)
    mod.run_softmimic_action('/simulation/bin/python',True)
    assert 'SoftMimic failed: invalid simulation state' in capsys.readouterr().out


def test_new_run_defaults_to_complete_balanced_dataset_without_starting_training(tmp_path,monkeypatch,capsys):
    mod=load('teleop_tui');monkeypatch.setattr(mod,'ROOT',tmp_path);root,_=run_dir(tmp_path)
    balanced=root/'dataset_balanced_v1';balanced.mkdir()
    (balanced/'manifest.json').write_text('{"status":"complete"}');(balanced/'train.npz').touch()
    rejected=root/'dataset_rejected';rejected.mkdir()
    (rejected/'manifest.json').write_text('{"status":"rejected"}');(rejected/'train.npz').touch()
    prompts(monkeypatch,['new','',root/'train_next'])
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Preview started training'))
    mod.run_softmimic_action('/simulation/python',False)
    assert '--dataset '+str(balanced) in capsys.readouterr().out
