import copy
import json
import sys

import pytest

from sim.piper_softmimic.summarize_continuation import SEEDS, render, summarize


def report_fixture(tmp_path):
    result = dict(status='complete', protocol_sha256='p', checkpoints={})
    for stage in (20000, 25000, 30000):
        result['checkpoints'][str(stage)] = {}
        for noisy, seed in [(False, SEEDS[0])] + [(True, seed) for seed in SEEDS]:
            label = f'{"noisy" if noisy else "clean"}_{seed}'
            rows = []
            for force in (.5, 1, 2, 3, 5):
                for axis in 'XYZ':
                    for sign in (-1, 1):
                        error = .004 if stage == 20000 else .002
                        # Non-Z cases stay unchanged, so overall averages must not hide small-Z gains.
                        if axis != 'Z' or force > 1:
                            error = .004
                        if noisy:
                            error += (seed - SEEDS[0]) * .001
                        rows.append(dict(axis=axis, sign=sign, force_N=force, stiffness_N_per_m=100,
                            controller='policy', variant='nominal', steady_target_rmse_m=error,
                            recovery_rmse_m=.003 if noisy else .001, steady_displacement_m=force / 100 - error,
                            expected_displacement_m=force / 100, recovery_2mm_s=None if noisy else .5,
                            recovery_5mm_s=.2, task_failure_over_30cm=False))
            detail = dict(status='complete', completed_training_updates=stage, checkpoint_sha256=str(stage),
                          evaluation_seed=seed, actor_observation_noise=noisy, protocol_sha256='p', model_sha256='m',
                          evaluation_source_sha256='s', plant='same', protocol={}, pair_properties_identical=True, cases=rows)
            path = tmp_path / f'{stage}_{label}.json'
            path.write_text(json.dumps(detail))
            result['checkpoints'][str(stage)][label] = dict(report=str(path), checkpoint_sha256=str(stage))
    return result


def test_summary_exposes_small_z_improvement_noise_spread_and_sustained_recovery(tmp_path):
    rows = summarize(report_fixture(tmp_path))
    latest = next(r for r in rows if r['updates'] == 30000 and r['noise'] == 'noisy' and r['group'] == 'small_Z')
    assert latest['push_error_mm'] == pytest.approx(3)
    assert latest['push_error_mm_seed_std'] == pytest.approx((2 / 3) ** .5)
    assert latest['push_error_mm_change_vs_20000'] == pytest.approx(-2)
    assert latest['recovery_2mm_pass_rate'] == 0 and latest['recovery_5mm_pass_rate'] == 1
    assert latest['seeds'] == 3 and latest['cases_per_seed'] == 4
    individual = next(r for r in rows if r['updates'] == 30000 and r['noise'] == 'clean' and r['group'] == '0.5N_+Z')
    assert individual['expected_deflection_mm'] == 5 and individual['deflection_mm'] == 3
    assert 'Small-force Z details:' in render(rows)


def test_summary_refuses_missing_noise_seed_and_wrong_checkpoint(tmp_path):
    report = report_fixture(tmp_path)
    invalid = copy.deepcopy(report)
    del invalid['checkpoints']['30000']['noisy_68125']
    with pytest.raises(ValueError, match='three paired'):
        summarize(invalid)
    path = tmp_path / '30000_clean_68123.json'
    detail = json.loads(path.read_text()); detail['checkpoint_sha256'] = 'wrong'
    path.write_text(json.dumps(detail))
    with pytest.raises(ValueError, match='Checkpoint'):
        summarize(report)


def test_summary_rejects_changed_noise_seed_and_protocol(tmp_path):
    report = report_fixture(tmp_path)
    path = tmp_path / '30000_noisy_68124.json'
    detail = json.loads(path.read_text()); detail['evaluation_seed'] = 999
    path.write_text(json.dumps(detail))
    with pytest.raises(ValueError, match='seed mismatch'):
        summarize(report)
    detail['evaluation_seed'] = 68124; detail['protocol_sha256'] = 'changed'
    path.write_text(json.dumps(detail))
    with pytest.raises(ValueError, match='protocol mismatch'):
        summarize(report)


def test_completed_supervisor_produces_comparison_artifacts(tmp_path, monkeypatch):
    from sim.piper_softmimic import summarize_continuation as job
    report = report_fixture(tmp_path)
    (tmp_path / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'status.json').write_text('{"status":"complete"}')
    monkeypatch.setattr(job, 'JOB', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['summarize_continuation'])
    job.main()
    output = json.loads((tmp_path / 'comparison.json').read_text())
    assert output['status'] == 'complete' and len(output['rows']) == 36
    assert 'Small-force Z details:' in (tmp_path / 'comparison.txt').read_text()
    assert 'push_error_mm_change_vs_20000' in (tmp_path / 'comparison.csv').read_text()
    assert not (tmp_path / 'comparison_failure.json').exists()
