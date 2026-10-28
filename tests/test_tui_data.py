import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_entrypoints import load, ROOT


def prompts(monkeypatch, values):
    answers = iter(str(value) for value in values)
    monkeypatch.setattr('builtins.input', lambda label: next(answers, ':back'))


def session_dir(tmp_path):
    session = tmp_path / 'recorded task'
    episode = session / 'episode_001'
    episode.mkdir(parents=True)
    (episode / 'episode.bag').write_bytes(b'')
    return session


def dataset_dir(tmp_path, dims=(28, 14)):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    info = {'features': {'observation.state': {'shape': [dims[0]]}, 'action': {'shape': [dims[1]]}}}
    (dataset / 'meta/info.json').write_text(json.dumps(info))
    return dataset


def test_rgb_monocular_conversion_contract_and_preview_does_not_write(tmp_path, monkeypatch, capsys):
    mod = load('teleop_tui')
    session = session_dir(tmp_path)
    output = tmp_path / 'new dataset'
    task = 'pick $(do-not-run) and place'
    prompts(monkeypatch, [session, output, task, 'mono', '/top/mono/compressed', '640x480'])
    command = mod.data_command('convert-rgb', tmp_path / 'training-python')
    assert command[1].endswith('convert_rosbag_to_lerobot.py')
    for flag, expected in [('--session-dir', str(session)), ('--output-root', str(tmp_path)),
        ('--dataset-id', 'new dataset'), ('--task', task), ('--state-indices', '1-14,22-35'),
        ('--fps', '30'), ('--action-source', 'mux'), ('--image-width', '640'), ('--image-height', '480')]:
        assert command[command.index(flag) + 1] == expected
    assert '--convert-to-v30' in command
    assert '--actfs' not in command and '--lfv-top-crop' not in command
    monkeypatch.setattr(mod, 'data_command', lambda *args: command)
    monkeypatch.setattr(mod.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Preview ran a process'))
    monkeypatch.setattr(mod.subprocess, 'call', lambda *args, **kwargs: pytest.fail('Preview executed conversion'))
    mod.run_data_action('convert-rgb', tmp_path / 'training-python', False)
    assert 'PREVIEW ONLY' in capsys.readouterr().out
    assert not output.exists()


def test_rgb_side_by_side_conversion_uses_crop_without_depth_model(tmp_path, monkeypatch):
    mod = load('teleop_tui')
    session = session_dir(tmp_path)
    calibration, metadata = tmp_path / 'calibration.json', tmp_path / 'processing.json'
    calibration.write_text('{}')
    metadata.write_text('{}')
    prompts(monkeypatch, [session, tmp_path / 'rgb', 'pick', '', '', '', calibration, metadata])
    command = mod.data_command('convert-rgb', sys.executable)
    assert '--lfv-top-crop' in command
    assert command[command.index('--actfs-calibration-json') + 1] == str(calibration)
    assert command[command.index('--actfs-processing-metadata') + 1] == str(metadata)
    assert '--actfs' not in command and '--actfs-model-dir' not in command
    assert '--dino-model' not in command and '--defm-root' not in command


@pytest.mark.parametrize('condition', ['bad', 'active', 'missing-bag', 'existing-output'])
def test_conversion_rejects_unsafe_inputs(tmp_path, monkeypatch, condition):
    mod = load('teleop_tui')
    session = session_dir(tmp_path)
    output = tmp_path / 'rgb'
    episode = session / 'episode_001'
    if condition == 'bad':
        (episode / 'BAD_ROSBAG.txt').write_text('bad')
    elif condition == 'active':
        (episode / 'episode.bag.active').write_bytes(b'')
    elif condition == 'missing-bag':
        (episode / 'episode.bag').unlink()
    else:
        output.mkdir()
        (output / 'keep.txt').write_text('do not overwrite')
    prompts(monkeypatch, [session, output])
    with pytest.raises(mod.Cancelled):
        mod.data_command('convert-rgb', sys.executable)
    if condition == 'existing-output':
        assert (output / 'keep.txt').read_text() == 'do not overwrite'


@pytest.mark.parametrize('size', ['0x320', '640x-1', 'invalid'])
def test_conversion_rejects_invalid_image_size(tmp_path, monkeypatch, size):
    mod = load('teleop_tui')
    prompts(monkeypatch, [session_dir(tmp_path), tmp_path / 'rgb', 'task', 'mono', '/mono/compressed', size, '800x600'])
    command = mod.data_command('convert-rgb', sys.executable)
    assert command[command.index('--image-width') + 1] == '800'
    assert command[command.index('--image-height') + 1] == '600'


@pytest.mark.parametrize('dims,mode', [((14, 7), 'single'), ((28, 14), 'bimanual')])
@pytest.mark.parametrize('action', ['act-smoke', 'act-train'])
def test_act_reuses_standard_trainer_and_infers_arm_mode(tmp_path, monkeypatch, dims, mode, action):
    mod = load('teleop_tui')
    dataset = dataset_dir(tmp_path, dims)
    output = tmp_path / 'train'
    values = [dataset, output] + (['120', '4'] if action == 'act-train' else [])
    prompts(monkeypatch, values)
    command = mod.data_command(action, sys.executable)
    assert command[1] == str(ROOT / 'scripts/train_act.py')
    assert command[command.index('--arm-mode') + 1] == mode
    assert ('--smoke' in command) == (action == 'act-smoke')
    if action == 'act-train':
        assert command[command.index('--steps') + 1] == '120'
        assert command[command.index('--batch-size') + 1] == '4'
    assert not output.exists()


@pytest.mark.parametrize('condition', ['dimensions', 'existing-output', 'steps', 'batch'])
def test_act_rejects_invalid_or_existing_outputs(tmp_path, monkeypatch, condition):
    mod = load('teleop_tui')
    dataset = dataset_dir(tmp_path, (42, 14) if condition == 'dimensions' else (28, 14))
    output = tmp_path / 'train'
    if condition == 'existing-output':
        output.mkdir()
    prompts(monkeypatch, [dataset, output, '0' if condition == 'steps' else '10', '0'])
    with pytest.raises(mod.Cancelled):
        mod.data_command('act-train', sys.executable)


@pytest.mark.parametrize('fails', [False, True])
def test_execute_checks_environment_and_clears_ros_python_paths(monkeypatch, fails):
    mod = load('teleop_tui')
    command = ['/training/bin/python', str(ROOT / 'scripts/train_act.py'), '--smoke']
    monkeypatch.setattr(mod, 'data_command', lambda *args: command)
    monkeypatch.setenv('PYTHONPATH', '/opt/ros/noetic/lib/python3/dist-packages')
    monkeypatch.setenv('PYTHONHOME', '/ros-python')
    calls = []

    def run(probe, **kwargs):
        assert probe[0] == '/training/bin/python'
        assert probe[1] == '-c' and 'sys.version_info' in probe[2]
        assert 'PYTHONPATH' not in kwargs['env'] and 'PYTHONHOME' not in kwargs['env']
        if fails:
            raise subprocess.CalledProcessError(1, probe)

    monkeypatch.setattr(mod.subprocess, 'run', run)
    monkeypatch.setattr(mod.subprocess, 'call', lambda cmd, **kwargs: calls.append((cmd, kwargs)) or 0)
    if fails:
        with pytest.raises(ValueError, match='--training-python'):
            mod.run_data_action('act-smoke', '/training/bin/python', True)
        assert calls == []
    else:
        mod.run_data_action('act-smoke', '/training/bin/python', True)
        assert len(calls) == 1 and calls[0][0] == command
        assert calls[0][1]['cwd'] == ROOT
        assert 'PYTHONPATH' not in calls[0][1]['env']


@pytest.mark.parametrize('key,action', [('c', 'convert-rgb'), ('s', 'act-smoke'), ('t', 'act-train')])
@pytest.mark.parametrize('execute', [False, True])
def test_data_menu_dispatch_preserves_preview_mode_and_venv_symlink(tmp_path, monkeypatch, key, action, execute):
    mod = load('teleop_tui')
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    choices = iter([key, 'q'])
    calls = []
    monkeypatch.setattr(sys, 'argv', ['teleop_tui', '--training-python', str(python)]
                        + (['--execute'] if execute else []))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr('builtins.input', lambda prompt: next(choices))
    monkeypatch.setattr(mod, 'hardware_status', lambda cfg: [])
    monkeypatch.setattr(mod, 'run_data_action', lambda *args: calls.append(args))
    assert mod.main() == 0
    assert calls == [(action, python, execute)]
    assert all('stereo' not in mod.ACTIONS[key][0].lower() for key in ('c', 's', 't'))
