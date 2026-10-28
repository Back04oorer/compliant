import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.piper_softmimic import compare
from sim.piper_softmimic.evaluate import case_metrics, envelope


def fake_report(checkpoint, output, noise):
    time=np.arange(601)*.02
    rows=[];positions=[];forces=[]
    for axis in range(3):
        for sign in (-1,1):
            force=np.zeros((601,3));force[:,axis]=sign*2*envelope(time)
            for controller in ('policy','analytic'):
                position=force/100
                if controller=='policy':
                    position=position.copy();position[:,(axis+1)%3]+=.001 if checkpoint.stem=='model_12500' else .01
                row=dict(axis='XYZ'[axis],sign=sign,force_N=2,stiffness_N_per_m=100,
                         variant='nominal',controller=controller,task_failure_over_30cm=False)
                row.update(case_metrics(time,position,np.zeros(3),force,row))
                rows.append(row);positions.append(position);forces.append(force)
    output.mkdir()
    np.savez(output/'traces.npz',time=time,tcp=np.stack(positions,1),force=np.stack(forces,1),reference_pos=np.zeros(3))
    return dict(status='complete',checkpoint_name=checkpoint.name,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                protocol_sha256='protocol',model_sha256='model',evaluation_source_sha256='source',
                actor_observation_noise=noise,plant='same',protocol={'timing':'same'},
                pair_properties_identical=True,cases=rows)


def test_comparison_generates_three_controller_plots_and_keeps_null_settling(tmp_path,monkeypatch):
    latest=tmp_path/'model_12500.pt';latest.write_bytes(b'new')
    old=tmp_path/'model_5500.pt';old.write_bytes(b'old');(tmp_path/'manifest.json').write_text('{}')
    protocol=tmp_path/'protocol.npz';protocol.touch();protocol.with_suffix('.json').write_text('{}')
    args=SimpleNamespace(checkpoint=latest,reference_checkpoint=old,protocol=protocol,output=tmp_path/'comparison',noise='both')
    calls=[]
    def evaluate(checkpoint,protocol,output,noise):
        calls.append((checkpoint,noise));return fake_report(checkpoint,output,noise)
    monkeypatch.setattr(compare,'run_evaluation',evaluate)
    compare.compare(args)
    report=json.loads((args.output/'report.json').read_text())
    assert calls==[(latest,False),(old,False),(latest,True),(old,True)]
    assert report['modes']['clean']['latest']['nominal_push_rmse_mm']==pytest.approx(1)
    assert report['modes']['clean']['reference']['nominal_recovery_2mm_passes']==0
    assert report['modes']['clean']['latest']['nominal_recovery_2mm_passes']==6
    assert report['modes']['clean']['analytic_max_repeat_difference_m']==0
    assert len(report['cases'])==36
    assert any(r['recovery_2mm_s'] is None for r in report['cases'])
    for name in ('xyz_positive.png','xyz_negative.png','summary.csv'):
        assert (args.output/name).stat().st_size>100
    with pytest.raises(FileExistsError):compare.compare(args)


@pytest.mark.parametrize('mismatch',['protocol_sha256','actor_observation_noise','cases','pair_properties_identical'])
def test_comparison_rejects_incompatible_evaluations(mismatch):
    report=dict(status='complete',protocol_sha256='p',model_sha256='m',evaluation_source_sha256='s',
                actor_observation_noise=False,plant='same',protocol={},pair_properties_identical=True,
                cases=[dict(axis='X',sign=1,force_N=2,stiffness_N_per_m=100,variant='nominal',controller='policy')])
    other=copy.deepcopy(report)
    other[mismatch]=[] if mismatch=='cases' else 'different' if mismatch=='protocol_sha256' else not other[mismatch]
    with pytest.raises(ValueError):compare.validate_pair(report,other)


def test_child_failure_is_detected_even_when_isaac_returns_zero(tmp_path,monkeypatch):
    output=tmp_path/'failed'
    def start(*a,**kw):
        output.mkdir();(output/'failure.json').write_text('{"error":"invalid physics"}')
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(compare.subprocess,'Popen',start)
    with pytest.raises(RuntimeError,match='invalid physics'):
        compare.run_evaluation(tmp_path/'model.pt',tmp_path/'protocol.npz',output,False)


def test_cancellation_signals_only_the_evaluation_child_session(tmp_path,monkeypatch):
    waits=iter([KeyboardInterrupt(),0]);signals=[]
    def wait(**kw):
        value=next(waits)
        if isinstance(value,BaseException):raise value
        return value
    monkeypatch.setattr(compare.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=12345,wait=wait))
    monkeypatch.setattr(compare.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        compare.run_evaluation(tmp_path/'model.pt',tmp_path/'protocol.npz',tmp_path/'cancelled',False)
    assert signals==[(12345,compare.signal.SIGINT)]


def test_comparison_rejects_different_noise_seeds():
    with pytest.raises(ValueError, match='evaluation_seed'):
        compare.validate_pair({'evaluation_seed': 68123}, {'evaluation_seed': 68124})


def test_evaluation_forwards_fixed_seed(tmp_path, monkeypatch):
    output = tmp_path / 'evaluation'
    commands = []
    def start(command, **kwargs):
        commands.append(command)
        output.mkdir()
        (output / 'report.json').write_text('{"status":"complete"}')
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(compare.subprocess, 'Popen', start)
    compare.run_evaluation(tmp_path / 'model.pt', tmp_path / 'protocol.npz', output, True, seed=68125)
    assert commands[0][-3:] == ['--seed', '68125', '--observation-noise']
