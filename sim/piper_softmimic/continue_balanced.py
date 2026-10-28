"""Bounded 20000 -> 25000 -> 30000 continuation with matched force evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from .compare import run_evaluation, validate_pair
from .config import OUTPUT

RUN = OUTPUT / 'BalancedPush_v1'
JOB = RUN / 'continuation_30000'
PROTOCOL = OUTPUT / 'evaluations/force_examples_v1/protocol.npz'
SEEDS = (68123, 68124, 68125)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def focus(report):
    rows = [r for r in report['cases'] if r['controller'] == 'policy' and r['variant'] == 'nominal']
    if len(rows) != 30:
        raise ValueError('Expected all 30 nominal force/direction cases')
    def average(key):
        return sum(r[key] for r in rows) / len(rows) * 1000
    return dict(push_error_mm=average('steady_target_rmse_m'), recovery_error_mm=average('recovery_rmse_m'),
                recovery_2mm_passes=sum(r['recovery_2mm_s'] is not None for r in rows),
                recovery_5mm_passes=sum(r['recovery_5mm_s'] is not None for r in rows),
                failures=sum(r['task_failure_over_30cm'] for r in rows), cases=rows,
                small_force_Z=[r for r in rows if r['axis'] == 'Z' and r['force_N'] <= 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-eula', action='store_true')
    args = parser.parse_args()
    if not args.accept_eula:
        parser.error('--accept-eula is required')
    plan = json.loads((JOB / 'plan.json').read_text())
    state = dict(status='starting', target_completed_updates=30000, started_at_unix=time.time())
    results = dict(status='running', protocol_sha256=plan['protocol_sha256'], noise_seeds=list(SEEDS), checkpoints={})
    def save(**values):
        state.update(values)
        pending = JOB / 'status.pending'
        pending.write_text(json.dumps(state, indent=2)+'\n'); pending.replace(JOB / 'status.json')
    def guard():
        for filename, expected in plan['source_sha256'].items():
            if sha(Path(__file__).with_name(filename)) != expected:
                raise ValueError(f'Source changed during continuation: {filename}')
        if sha(PROTOCOL) != plan['protocol_sha256']:
            raise ValueError('Frozen evaluation protocol changed')
        if sha(RUN / 'model_19999.pt') != plan['checkpoint_sha256']:
            raise ValueError('Original 20000 checkpoint changed')
    def train(checkpoint, target):
        guard()
        command = [sys.executable, '-u', '-m', 'sim.piper_softmimic.train', '--accept-eula',
                   '--resume', str(checkpoint), '--resume-in-place', '--output', str(RUN),
                   '--iterations', str(target), '--num-envs', '4096']
        with (JOB / f'train_{target}.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save(status='training', target_stage=target, child_pid=process.pid, command=command)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGTERM)
                process.wait()
                raise
        progress = json.loads((RUN / 'progress.json').read_text())
        if progress['status'] == 'paused':
            raise KeyboardInterrupt('Training paused; continuation will not restart it')
        if code or progress['status'] != 'complete' or progress['iteration'] != target or (RUN / 'failure.json').exists():
            raise RuntimeError(f'Training to {target} did not complete: {progress}')
        checkpoint = RUN / f'model_{target-1}.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Missing exact milestone checkpoint: {checkpoint}')
        return checkpoint
    def evaluate(checkpoint, updates):
        guard()
        checkpoint_hash = sha(checkpoint)
        records = {}
        for noise, seed in [(False, SEEDS[0])] + [(True, seed) for seed in SEEDS]:
            label = f'{"noisy" if noise else "clean"}_{seed}'
            directory = OUTPUT / 'evaluations' / f'BalancedPush_v1_continue_{updates}_{label}'
            if directory.exists():
                raise ValueError(f'Refusing to overwrite evaluation: {directory}')
            save(status='evaluating', evaluating_updates=updates, evaluation_label=label, evaluation_output=str(directory))
            report = run_evaluation(checkpoint, PROTOCOL, directory, noise, seed=seed)
            if report['checkpoint_sha256'] != checkpoint_hash or report['completed_training_updates'] != updates:
                raise ValueError('Evaluation used the wrong checkpoint')
            if updates != 20000:
                baseline = json.loads(Path(results['checkpoints']['20000'][label]['report']).read_text())
                validate_pair(report, baseline)
            records[label] = dict(report=str(directory / 'report.json'), seed=seed, checkpoint_sha256=checkpoint_hash, **focus(report))
        results['checkpoints'][str(updates)] = records
        results['status'] = 'complete' if updates == 30000 else 'running'
        pending = JOB / 'report.pending'
        pending.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n'); pending.replace(JOB / 'report.json')
    def stop(*_):
        raise KeyboardInterrupt('Continuation interrupted')
    signal.signal(signal.SIGTERM, stop)
    try:
        guard()
        checkpoint = train(RUN / 'model_19999.pt', 25000)
        evaluate(JOB / 'preserved_20000/model_19999.pt', 20000)
        evaluate(checkpoint, 25000)
        checkpoint = train(checkpoint, 30000)
        evaluate(checkpoint, 30000)
        guard()
        save(status='complete', finished_at_unix=time.time())
    except KeyboardInterrupt as error:
        save(status='stopped', error=str(error), finished_at_unix=time.time())
    except BaseException as error:
        save(status='failed', error=repr(error), finished_at_unix=time.time())
        raise


if __name__ == '__main__':
    main()
