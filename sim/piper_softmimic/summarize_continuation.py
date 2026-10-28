"""Summarize the frozen 20k/25k/30k continuation, optionally waiting for completion."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
from statistics import fmean, pstdev
import time

from .compare import validate_pair
from .continue_balanced import JOB, SEEDS

STAGES = ('20000', '25000', '30000')
FIELDS = {'push_error_mm': 'steady_target_rmse_m', 'recovery_error_mm': 'recovery_rmse_m',
          'deflection_mm': 'steady_displacement_m', 'expected_deflection_mm': 'expected_displacement_m'}


def summarize(report):
    if report['status'] != 'complete' or set(report['checkpoints']) != set(STAGES):
        raise ValueError('All three completed checkpoints are required')
    labels = ['clean_' + str(SEEDS[0])] + ['noisy_' + str(seed) for seed in SEEDS]
    groups = [('all', lambda row: True), ('small_Z', lambda row: row['axis'] == 'Z' and row['force_N'] <= 1)]
    for force in (.5, 1.):
        for sign in (-1, 1):
            groups.append((f'{force:g}N_{"+" if sign > 0 else "-"}Z',
                           lambda row, force=force, sign=sign: row['axis'] == 'Z' and row['force_N'] == force and row['sign'] == sign))
    records = {}
    for stage in STAGES:
        if set(report['checkpoints'][stage]) != set(labels):
            raise ValueError('Expected clean and all three paired noise seeds')
        for label in labels:
            record = report['checkpoints'][stage][label]
            detail = json.loads(Path(record['report']).read_text())
            if detail['completed_training_updates'] != int(stage) or detail['checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Checkpoint does not match the continuation report')
            if detail['evaluation_seed'] != int(label.split('_')[1]) or detail['actor_observation_noise'] != label.startswith('noisy_'):
                raise ValueError('Noise mode or paired seed mismatch')
            if detail['protocol_sha256'] != report['protocol_sha256']:
                raise ValueError('Frozen evaluation protocol mismatch')
            if stage != '20000':
                validate_pair(detail, records['20000', label])
            records[stage, label] = detail
    result = []
    for stage in STAGES:
        for mode in ('clean', 'noisy'):
            selected = labels[:1] if mode == 'clean' else labels[1:]
            for group, select in groups:
                per_seed = []
                for label in selected:
                    rows = [row for row in records[stage, label]['cases']
                            if row['controller'] == 'policy' and row['variant'] == 'nominal' and select(row)]
                    expected = 30 if group == 'all' else 4 if group == 'small_Z' else 1
                    if len(rows) != expected:
                        raise ValueError(f'Incomplete nominal cases: {stage}/{label}/{group}')
                    values = {name: fmean(row[key] for row in rows) * 1000 for name, key in FIELDS.items()}
                    values.update(recovery_2mm_pass_rate=fmean(row['recovery_2mm_s'] is not None for row in rows),
                                  recovery_5mm_pass_rate=fmean(row['recovery_5mm_s'] is not None for row in rows),
                                  failure_rate=fmean(row['task_failure_over_30cm'] for row in rows))
                    per_seed.append(values)
                summary = dict(updates=int(stage), noise=mode, group=group, seeds=len(selected), cases_per_seed=expected)
                for key in per_seed[0]:
                    summary[key] = fmean(row[key] for row in per_seed)
                    summary[key + '_seed_std'] = pstdev(row[key] for row in per_seed)
                baseline = summary if stage == '20000' else next(row for row in result if row['updates'] == 20000 and row['noise'] == mode and row['group'] == group)
                for key in ('push_error_mm', 'recovery_error_mm'):
                    summary[key + '_change_vs_20000'] = summary[key] - baseline[key]
                result.append(summary)
    return result


def render(rows):
    lines = ['Piper SoftMimic: 20000 / 25000 / 30000 PPO updates',
             'Paired frozen protocol; K=100 N/m; nominal plant. Noisy results: mean of three paired seeds.',
             'Errors and deflection: mm. Negative error change means improvement; std is across noise seeds.',
             'Recovery pass requires a sustained return, not just low average error. Simulation only.\n']
    for row in rows:
        if row['group'] == 'all' or row['group'] == 'small_Z':
            lines.append(f"{row['updates']:5d} | {row['noise']:5s} | {row['group']:7s} | "
                         f"push {row['push_error_mm']:.3f} (delta {row['push_error_mm_change_vs_20000']:+.3f}, seed std {row['push_error_mm_seed_std']:.3f}) | "
                         f"recovery {row['recovery_error_mm']:.3f} (delta {row['recovery_error_mm_change_vs_20000']:+.3f}) | "
                         f"2mm pass {row['recovery_2mm_pass_rate']:.1%} | 5mm pass {row['recovery_5mm_pass_rate']:.1%} | failures {row['failure_rate']:.1%}")
    lines += ['\nSmall-force Z details:', 'updates | noise | force/direction | deflection / target | push error | recovery error']
    for row in rows:
        if row['group'] not in ('all', 'small_Z'):
            lines.append(f"{row['updates']:5d} | {row['noise']:5s} | {row['group']:8s} | "
                         f"{row['deflection_mm']:.3f} / {row['expected_deflection_mm']:.1f} | "
                         f"{row['push_error_mm']:.3f} | {row['recovery_error_mm']:.3f}")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Wait for the existing supervisor; never start or change training')
    args = parser.parse_args()
    try:
        print('Waiting for the existing 30000-update continuation and paired evaluations.', flush=True)
        while True:
            state = json.loads((JOB / 'status.json').read_text())
            if state['status'] == 'complete':
                break
            if state['status'] in ('failed', 'stopped'):
                raise RuntimeError(f'Continuation did not finish: {state}')
            if not args.wait:
                raise RuntimeError('Continuation is still running; use --wait')
            os.kill(json.loads((JOB / 'launch.json').read_text())['pid'], 0)
            time.sleep(15)
        report = json.loads((JOB / 'report.json').read_text())
        rows = summarize(report)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        for name, body in (('comparison.json', json.dumps(dict(status='complete', protocol_sha256=report['protocol_sha256'], rows=rows), indent=2, allow_nan=False)+'\n'),
                           ('comparison.csv', stream.getvalue()), ('comparison.txt', render(rows))):
            pending = JOB / (name + '.pending')
            pending.write_text(body); pending.replace(JOB / name)
        print(render(rows), flush=True)
    except BaseException as error:
        (JOB / 'comparison_failure.json').write_text(json.dumps({'error': repr(error)})+'\n')
        raise


if __name__ == '__main__':
    main()
