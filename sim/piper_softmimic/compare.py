"""Evaluate two frozen checkpoints against the same privileged analytic baseline."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

from .config import OUTPUT


def case_key(row):
    return tuple(row[k] for k in ('axis', 'sign', 'force_N', 'stiffness_N_per_m', 'variant', 'controller'))


def validate_pair(latest, reference):
    if latest.get('evaluation_seed', 68123) != reference.get('evaluation_seed', 68123):
        raise ValueError('Incompatible evaluations: evaluation_seed')
    for key in ('protocol_sha256', 'model_sha256', 'evaluation_source_sha256',
                'actor_observation_noise', 'plant', 'protocol'):
        if latest[key] != reference[key]:
            raise ValueError(f'Incompatible evaluations: {key}')
    for report in (latest, reference):
        if report['status'] != 'complete' or not report['pair_properties_identical']:
            raise ValueError('Incomplete evaluation or unmatched physical properties')
    keys = [case_key(row) for row in latest['cases']]
    other = [case_key(row) for row in reference['cases']]
    if keys != other or len(set(keys)) != len(keys):
        raise ValueError('Evaluation cases differ or contain duplicates')


def summarize(rows):
    nominal = [r for r in rows if r['variant'] == 'nominal' and r['force_N'] == 2
               and r['stiffness_N_per_m'] == 100]
    return dict(nominal_cases=len(nominal),
                nominal_push_rmse_mm=float(np.mean([r['steady_target_rmse_m'] for r in nominal]) * 1000),
                nominal_recovery_rmse_mm=float(np.mean([r['recovery_rmse_m'] for r in nominal]) * 1000),
                nominal_pre_push_bias_mm=float(np.mean([r['pre_push_bias_m'] for r in nominal]) * 1000),
                nominal_recovery_2mm_passes=sum(r['recovery_2mm_s'] is not None for r in nominal),
                all_cases=len(rows),
                all_recovery_2mm_passes=sum(r['recovery_2mm_s'] is not None for r in rows),
                all_failures_over_30cm=sum(r['task_failure_over_30cm'] for r in rows),
                all_finite=all(r['finite'] for r in rows))


def run_evaluation(checkpoint, protocol, output, noise, seed=68123):
    command = [sys.executable, '-u', '-m', 'sim.piper_softmimic.evaluate', '--accept-eula',
               '--checkpoint', str(checkpoint), '--protocol', str(protocol), '--output', str(output), '--seed', str(seed)]
    if noise:
        command.append('--observation-noise')
    print(f'Evaluating {checkpoint.name}, observation noise={noise}; log: {output.with_suffix(".log")}', flush=True)
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait()
        except KeyboardInterrupt:
            # The child has its own session: interrupt this evaluator, never the live trainer.
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    failure = output / 'failure.json'
    if failure.is_file():
        raise RuntimeError(f'{output.name}: {json.loads(failure.read_text())["error"]}')
    report_path = output / 'report.json'
    if result or not report_path.is_file():
        raise RuntimeError(f'{output.name}: exit {result}, missing/failed report; see {output.with_suffix(".log")}')
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete':
        raise RuntimeError(f'{output.name}: incomplete report')
    return report


def plot_comparison(output, latest_dir, reference_dir, latest, reference, mode):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with np.load(latest_dir / 'traces.npz') as new, np.load(reference_dir / 'traces.npz') as old:
        baseline_ids = [i for i, row in enumerate(latest['cases']) if row['controller'] == 'analytic']
        baseline_difference = float(np.abs(new['tcp'][:, baseline_ids] - old['tcp'][:, baseline_ids]).max())
        if baseline_difference > 1e-4:
            raise ValueError(f'Analytic baseline changed between checkpoints: {baseline_difference} m')
        for sign in (1, -1):
            fig = Figure(figsize=(13, 9), dpi=150); FigureCanvasAgg(fig)
            axes = fig.subplots(3, 3)
            for column, axis in enumerate('XYZ'):
                selection = [i for i, row in enumerate(latest['cases']) if row['axis'] == axis
                             and row['sign'] == sign and row['force_N'] == 2
                             and row['stiffness_N_per_m'] == 100 and row['variant'] == 'nominal']
                for data, report, controller, color, label in (
                    (new, latest, 'policy', '#3366bb', latest['checkpoint_name']),
                    (old, reference, 'policy', '#dc8734', reference['checkpoint_name']),
                    (new, latest, 'analytic', '#408566', 'Analytic (privileged)'),
                ):
                    index = next(i for i in selection if report['cases'][i]['controller'] == controller)
                    t = data['time']; pos = data['tcp'][:, index]; ref = data['reference_pos']
                    pre = pos[(t >= 2.5) & (t < 3)].mean(0)
                    target = ref + data['force'][:, index] / 100
                    axes[0, column].plot(t, (pos[:, column] - pre[column]) * sign * 1000, color=color, label=label)
                    axes[1, column].plot(t, np.linalg.norm(pos - target, axis=-1) * 1000, color=color)
                    axes[2, column].plot(t, np.linalg.norm(pos - ref, axis=-1) * 1000, color=color)
                axes[0, column].plot(t, np.linalg.norm(new['force'][:, index], axis=-1) / 100 * 1000,
                                     'k--', linewidth=1, label='F/K target (20 mm)')
                axes[0, column].set_title(f'{"+" if sign > 0 else "-"}{axis}: 2 N, K=100 N/m')
                axes[2, column].set_xlabel('Time [s]')
                for ax in axes[:, column]:
                    ax.axvspan(3, 6, color='#dc8734', alpha=.07)
                    ax.axvline(6, linestyle=':', color='#687168', linewidth=1)
                    ax.grid(alpha=.2)
            axes[0, 0].set_ylabel('Deflection from pre-push [mm]')
            axes[1, 0].set_ylabel('3D compliant target error [mm]')
            axes[2, 0].set_ylabel('3D reference pose error [mm]')
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f'Independent XYZ comparison: {mode} observations')
            fig.tight_layout()
            fig.savefig(output / f'xyz_{"positive" if sign > 0 else "negative"}.png')
    return baseline_difference


def compare(args):
    for checkpoint in (args.checkpoint, args.reference_checkpoint):
        if not checkpoint.is_file() or not (checkpoint.parent / 'manifest.json').is_file():
            raise ValueError(f'Missing checkpoint or training manifest: {checkpoint}')
    if args.checkpoint.resolve() == args.reference_checkpoint.resolve():
        raise ValueError('Choose two different checkpoints')
    if not args.protocol.is_file() or not args.protocol.with_suffix('.json').is_file():
        raise ValueError('Prepare the fixed XYZ protocol first; see sim/piper_softmimic/README.md')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.checkpoint, args.reference_checkpoint)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'compare_source.py').write_bytes(Path(__file__).read_bytes())
    try:
        modes = ('clean', 'noisy') if args.noise == 'both' else (args.noise,)
        report = dict(status='complete', checkpoint=str(args.checkpoint.absolute()),
                      reference_checkpoint=str(args.reference_checkpoint.absolute()), modes={})
        rows = []
        for mode in modes:
            new_dir = args.output / (mode + '_latest'); old_dir = args.output / (mode + '_reference')
            latest = run_evaluation(args.checkpoint, args.protocol, new_dir, mode == 'noisy')
            reference = run_evaluation(args.reference_checkpoint, args.protocol, old_dir, mode == 'noisy')
            validate_pair(latest, reference)
            for checkpoint, result in ((args.checkpoint, latest), (args.reference_checkpoint, reference)):
                if result['checkpoint_sha256'] != hashes[str(checkpoint)]:
                    raise ValueError(f'Checkpoint changed during evaluation: {checkpoint}')
            controllers = {
                'latest': [r for r in latest['cases'] if r['controller'] == 'policy'],
                'reference': [r for r in reference['cases'] if r['controller'] == 'policy'],
                'analytic': [r for r in latest['cases'] if r['controller'] == 'analytic'],
            }
            summary = {name: summarize(cases) for name, cases in controllers.items()}
            summary['checkpoint_sha256'] = latest['checkpoint_sha256']
            summary['reference_checkpoint_sha256'] = reference['checkpoint_sha256']
            summary['protocol_sha256'] = latest['protocol_sha256']
            # The top-level PNGs show clean observations when both modes are requested.
            if mode == modes[0]:
                summary['analytic_max_repeat_difference_m'] = plot_comparison(
                    args.output, new_dir, old_dir, latest, reference, mode)
            report['modes'][mode] = summary
            rows += [dict(comparison_controller=name, observation_mode=mode, **row)
                     for name, cases in controllers.items() for row in cases]
            print(json.dumps({mode: summary}, indent=2), flush=True)
        report['cases'] = rows
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f'Comparison complete: {args.output}', flush=True)
        print('Open xyz_positive.png / xyz_negative.png; summary.csv contains every case and failure.', flush=True)
        try:
            relative = args.output.absolute().relative_to(OUTPUT / 'evaluations')
        except ValueError:
            pass
        else:
            if len(relative.parts) == 1:
                print(f'Dashboard: /media/evaluations/{relative}/xyz_positive.png', flush=True)
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(error)}) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=OUTPUT / 'evaluations/xyz_push_v1/protocol.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--noise', choices=('clean', 'noisy', 'both'), default='both')
    parser.add_argument('--accept-eula', action='store_true')
    args = parser.parse_args()
    if not args.accept_eula:
        parser.error('--accept-eula is required')
    compare(args)


if __name__ == '__main__':
    main()
