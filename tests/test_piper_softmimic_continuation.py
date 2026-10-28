import json
import sys
from types import SimpleNamespace

import pytest

from sim.piper_softmimic import continue_balanced as job


@pytest.mark.parametrize('paused', [False, True])
def test_continuation_exact_milestones_and_matched_seeds(tmp_path, monkeypatch, paused):
    run = tmp_path / 'BalancedPush_v1'; run.mkdir()
    work = run / 'continuation_30000'; work.mkdir()
    original = run / 'model_19999.pt'; original.write_bytes(b'original')
    preserved = work / 'preserved_20000'; preserved.mkdir()
    (preserved / original.name).write_bytes(original.read_bytes())
    protocol = tmp_path / 'protocol.npz'; protocol.write_bytes(b'frozen')
    (work / 'plan.json').write_text(json.dumps(dict(source_sha256={}, protocol_sha256=job.sha(protocol), checkpoint_sha256=job.sha(original))))
    for key, value in dict(RUN=run, JOB=work, OUTPUT=tmp_path, PROTOCOL=protocol).items():
        monkeypatch.setattr(job, key, value)
    monkeypatch.setattr(sys, 'argv', ['continue_balanced', '--accept-eula'])
    monkeypatch.setattr(job.signal, 'signal', lambda *args: None)
    trained = []; evaluated = []
    def start(command, **kwargs):
        target = int(command[command.index('--iterations')+1]); trained.append(target)
        assert command[command.index('--resume')+1].endswith(f'model_{19999 if target==25000 else 24999}.pt')
        (run / 'progress.json').write_text(json.dumps(dict(status='paused' if paused else 'complete', iteration=target)))
        (run / f'model_{target-1}.pt').write_bytes(str(target).encode())
        return SimpleNamespace(pid=123, wait=lambda: 0)
    monkeypatch.setattr(job.subprocess, 'Popen', start)
    def evaluate(checkpoint, protocol, directory, noise, seed):
        updates = int(checkpoint.stem.split('_')[1])+1
        evaluated.append((updates, noise, seed))
        rows = [dict(controller='policy', variant='nominal', axis=axis, sign=sign, force_N=force,
                     stiffness_N_per_m=100, steady_target_rmse_m=.002, recovery_rmse_m=.001,
                     recovery_2mm_s=None if noise else .5, recovery_5mm_s=.2, task_failure_over_30cm=False)
                for force in (.5, 1, 2, 3, 5) for axis in 'XYZ' for sign in (-1, 1)]
        report = dict(status='complete', completed_training_updates=updates, checkpoint_sha256=job.sha(checkpoint),
                      evaluation_seed=seed, protocol_sha256='same', model_sha256='same', evaluation_source_sha256='same',
                      actor_observation_noise=noise, plant='same', protocol={}, pair_properties_identical=True, cases=rows)
        directory.mkdir(parents=True); (directory / 'report.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(job, 'run_evaluation', evaluate)
    job.main()
    assert original.read_bytes() == b'original'
    state = json.loads((work / 'status.json').read_text())
    if paused:
        assert trained == [25000] and not evaluated and state['status'] == 'stopped'
    else:
        assert trained == [25000, 30000] and state['status'] == 'complete'
        assert evaluated == [(n, noise, seed) for n in (20000, 25000, 30000)
                             for noise, seed in [(False, 68123), (True, 68123), (True, 68124), (True, 68125)]]
        report = json.loads((work / 'report.json').read_text())
        assert len(report['checkpoints']['30000']['clean_68123']['small_force_Z']) == 4
        assert report['checkpoints']['30000']['noisy_68123']['recovery_2mm_passes'] == 0
