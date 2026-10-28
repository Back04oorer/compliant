"""Keyboard and form regression checks without curses terminals, ROS or hardware."""
import curses
import json
from pathlib import Path

import pytest

from test_entrypoints import load
from test_tui_data import dataset_dir, session_dir
import tui_inputs as ui


class Screen:
    def __init__(self, keys, size=(18, 80)):
        self.keys = iter(keys)
        self.size = size
        self.rows, self.frames = {}, []
    def keypad(self, value): pass
    def timeout(self, value): self.read_timeout = value
    def clear(self): self.erase()
    def getmaxyx(self): return self.size
    def erase(self): self.rows = {}
    def refresh(self): self.frames.append(dict(self.rows))
    def addnstr(self, row, col, text, count, attr=0):
        self.rows[row] = text[:count]
    def get_wch(self): return next(self.keys)


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(ui, 'interactive', lambda: True)
    def run(keys, size=(18, 80)):
        screen = Screen(keys, size)
        monkeypatch.setattr(ui.curses, 'wrapper', lambda fn, *args: fn(screen, *args))
        return screen
    return run


def test_choice_arrows_and_default(terminal):
    terminal([curses.KEY_DOWN, '\n'])
    assert ui.choice('Mode', ('robot', 'dry-run', 'check'), 'robot') == 'dry-run'
    terminal(['\n'])
    assert ui.choice('Device', ('cuda', 'cpu'), 'cpu') == 'cpu'


def test_long_list_filter_empty_result_recovery_and_scrolling(terminal):
    screen = terminal(['/', 'x', curses.KEY_BACKSPACE, '3', curses.KEY_DOWN, '\n'], (10, 44))
    result = ui.pick(screen, 'Items', [(i, 'episode_%03d' % i) for i in range(100)])
    assert result == 13
    assert any('No matches' in ' '.join(frame.values()) for frame in screen.frames)
    screen = terminal([curses.KEY_NPAGE, curses.KEY_NPAGE, '\n'], (10, 44))
    assert ui.pick(screen, 'Items', [(i, str(i)) for i in range(100)]) == 8


@pytest.mark.parametrize('keys', [['\x1b'], ['/', 'x', '\x1b']])
def test_escape_cancels_list_even_while_filtering(terminal, keys):
    terminal(keys)
    with pytest.raises(ui.Cancelled):
        ui.choice('Mode', ('robot', 'check'), 'robot')


def test_small_terminal_cannot_accidentally_confirm(terminal):
    terminal(['\n', curses.KEY_DOWN, '\n', '\x1b'], (5, 20))
    with pytest.raises(ui.Cancelled):
        ui.choice('Mode', ('robot', 'check'), 'robot')


def test_edit_retries_in_place_preserves_bad_value_and_can_clear(terminal):
    screen = terminal(list('nan') + ['\n', '\x15'] + list('0') + ['\n', '\x15'] + list('12') + ['\n'])
    assert ui.number('Steps', '80000', minimum=1, integer=True) == 12
    assert any(frame.get(3) == 'nan' and frame.get(5) for frame in screen.frames)
    assert any(frame.get(3) == '0' and frame.get(5) for frame in screen.frames)


def test_edit_supports_cursor_delete_unicode_and_default(terminal):
    terminal(list('a猫c') + [curses.KEY_LEFT, curses.KEY_DC, 'b', '\n'])
    assert ui.text('Task', 'default') == 'a猫b'
    terminal(['\n'])
    assert ui.image_size('Size', '2560x720', even_width=True) == (2560, 720)


def test_path_catalog_filters_bags_and_selects_without_side_effects(tmp_path, terminal):
    bags = []
    for name in ('episode_001', 'episode_002'):
        p = tmp_path / 'session with space' / name / 'episode.bag'
        p.parent.mkdir(parents=True)
        p.touch()
        bags.append(p)
    terminal(['/', *'002', '\n'])
    assert ui.path('Bag', tmp_path, candidates=bags, suffix='.bag') == bags[1]


def test_browser_descends_directories_and_filters_file_type(tmp_path, terminal):
    bag = tmp_path / 'session' / 'episode_001' / 'episode.bag'
    bag.parent.mkdir(parents=True)
    bag.touch()
    (bag.parent / 'recording.bag.active').touch()
    terminal([curses.KEY_DOWN, '\n', curses.KEY_DOWN, '\n', curses.KEY_DOWN, '\n'])
    assert ui.path('Bag', tmp_path, suffix='.bag') == bag


def test_directory_browser_selects_current_directory(tmp_path, terminal):
    terminal([curses.KEY_DOWN, '\n'])
    assert ui.path('Dataset', tmp_path, kind='directory') == tmp_path


def test_escape_returns_through_browser_history_then_cancels(tmp_path, terminal):
    (tmp_path / 'child').mkdir()
    screen = terminal([curses.KEY_DOWN, '\n', '\x1b', '\x1b'], (18, 200))
    with pytest.raises(ui.Cancelled):
        ui.path('File', tmp_path)
    assert 'child' in screen.frames[-2][0]
    assert screen.frames[-1][0] == 'File | ' + str(tmp_path)


def test_missing_start_directory_escape_returns_immediately(tmp_path, terminal):
    terminal(['\x1b'])
    with pytest.raises(ui.Cancelled):
        ui.path('Output', tmp_path / 'missing', kind='directory', new=True)


def test_manual_quoted_path_and_relative_path_are_never_shell_evaluated(tmp_path, terminal):
    bag = tmp_path / '$(do-not-run) with spaces.bag'
    bag.touch()
    terminal(['\t', *('"' + bag.name + '"'), '\n'])
    assert ui.path('Bag', tmp_path, suffix='.bag') == bag
    assert list(tmp_path.iterdir()) == [bag]


def test_invalid_path_can_be_corrected_without_losing_the_browser(tmp_path, terminal):
    bag = tmp_path / 'episode.bag'
    bag.touch()
    screen = terminal(['\t', *'missing.bag', '\n', '\x15', *bag.name, '\n'])
    assert ui.path('Bag', tmp_path, suffix='.bag') == bag
    assert any('existing file' in frame.get(5, '') for frame in screen.frames)


def test_new_output_browser_checks_overwrite_without_creating_anything(tmp_path, terminal):
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'keep').write_text('keep')
    screen = terminal(['\t', *str(existing), '\n', '\x15', *str(tmp_path / 'new'), '\n'])
    assert ui.path('Output', tmp_path, kind='directory', new=True) == tmp_path / 'new'
    assert not (tmp_path / 'new').exists() and (existing / 'keep').read_text() == 'keep'
    assert any('already exists' in frame.get(5, '') for frame in screen.frames)


def test_new_directory_name_uses_browsed_parent(tmp_path, terminal):
    terminal([curses.KEY_DOWN, '\n', *'new dataset', '\n'])
    assert ui.path('Output', tmp_path, kind='directory', new=True) == tmp_path / 'new dataset'
    assert not (tmp_path / 'new dataset').exists()


def test_permission_error_keeps_parent_and_manual_path_available(tmp_path, terminal, monkeypatch):
    bag = tmp_path / 'episode.bag'
    bag.touch()
    monkeypatch.setattr(Path, 'iterdir', lambda p: (_ for _ in ()).throw(PermissionError('permission denied')))
    screen = terminal(['\t', *str(bag), '\n'])
    assert ui.path('Bag', tmp_path, suffix='.bag') == bag
    assert 'permission denied' in ' '.join(screen.frames[0].values())


def test_plain_input_retries_only_the_current_field(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, 'interactive', lambda: False)
    mod = load('teleop_tui')
    dataset = dataset_dir(tmp_path)
    output = tmp_path / 'new output'
    answers = iter([str(dataset), str(output), 'wrong', '0', '200', 'bad', '8'])
    prompts = []
    def read(label):
        prompts.append(label)
        return next(answers)
    monkeypatch.setattr('builtins.input', read)
    command = mod.data_command('act-train', '/training/python')
    assert command[command.index('--steps') + 1] == '200'
    assert command[command.index('--batch-size') + 1] == '8'
    assert sum('LeRobot dataset' in p for p in prompts) == 1
    assert sum('New training' in p for p in prompts) == 1
    assert sum('Training steps' in p for p in prompts) == 3
    assert not output.exists()


def test_invalid_session_and_json_can_be_reselected(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'interactive', lambda: False)
    mod = load('teleop_tui')
    bad = session_dir(tmp_path / 'bad')
    (bad / 'episode_001/BAD_ROSBAG.txt').touch()
    good = session_dir(tmp_path / 'good')
    invalid = tmp_path / 'bad.json'
    invalid.write_text('invalid json')
    valid = tmp_path / 'valid.json'
    valid.write_text('{}')
    answers = iter([str(bad), str(good), str(tmp_path / 'rgb'), 'task', 'sbs', '', '',
                    str(invalid), str(valid), str(valid)])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    command = mod.data_command('convert-rgb', '/training/python')
    assert command[command.index('--session-dir') + 1] == str(good)
    assert command[command.index('--actfs-calibration-json') + 1] == str(valid)


@pytest.mark.parametrize('action', ['c', 'm', 'b', 'k', 'r'])
def test_form_cancellation_returns_to_menu_without_launch_or_extra_prompt(monkeypatch, action):
    mod = load('teleop_tui')
    monkeypatch.setattr('sys.argv', ['tui', '--execute'])
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    answers = iter([action, ':back', 'q'] if action == 'r' else [action, 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    monkeypatch.setattr(mod, 'hardware_status', lambda cfg: [])
    def cancel(*args, **kwargs): raise ui.Cancelled()
    for name in ('run_data_action', 'run_calibration_action'):
        monkeypatch.setattr(mod, name, cancel)
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **k: pytest.fail('Cancelled action launched'))
    monkeypatch.setattr(mod.subprocess, 'call', lambda *a, **k: pytest.fail('Cancelled action executed'))
    assert mod.main() == 0


def test_real_terminal_arrows_retry_and_escape(tmp_path):
    import errno
    import fcntl
    import os
    import pty
    import select
    import struct
    import subprocess
    import sys
    import termios
    import time
    script = tmp_path / 'terminal_check.py'
    script.write_text('''import sys
sys.path.insert(0, %r)
import tui_inputs as ui
import curses
curses.wrapper(lambda screen: screen.timeout(10))
print('CHOICE=' + ui.choice('Device check', ('cuda', 'cpu'), 'cuda'), flush=True)
print('NUMBER=' + str(ui.number('Speed check', '1', minimum=0, maximum=1)), flush=True)
try:
    ui.choice('Cancel check', ('robot', 'check'), 'robot')
except ui.Cancelled:
    print('CANCELLED', flush=True)
''' % str(Path(ui.__file__).parent))
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 100, 0, 0))
    env = dict(os.environ, TERM='xterm-256color', ESCDELAY='50')
    process = subprocess.Popen([sys.executable, str(script)], stdin=slave, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    output = b''
    def wait_for(marker):
        nonlocal output
        deadline = time.monotonic() + 5
        while marker not in output and time.monotonic() < deadline:
            if select.select([master], [], [], .05)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno == errno.EIO: break
                    raise
                if not data: break
                output += data
        assert marker in output, output.decode(errors='replace')
    try:
        wait_for(b'Device check')
        time.sleep(.05)  # Forms must reset the live menu's timed input.
        os.write(master, b'\x1bOB\n')  # xterm application-cursor Down, then Enter.
        wait_for(b'CHOICE=cpu')
        wait_for(b'Speed check')
        os.write(master, b'nan\n')
        wait_for(b'finite')
        os.write(master, b'\x15.5\n')
        wait_for(b'NUMBER=0.5')
        wait_for(b'Cancel check')
        os.write(master, b'\x1b')
        wait_for(b'CANCELLED')
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        os.close(master)


@pytest.mark.parametrize('invalid', ['nan', 'inf', '-1', '1.1', '0'])
def test_replay_speed_invalid_values_retry_in_the_same_field(monkeypatch, invalid):
    monkeypatch.setattr(ui, 'interactive', lambda: False)
    values = iter([invalid, '.5'])
    monkeypatch.setattr('builtins.input', lambda _: next(values))
    assert ui.number('Speed', '1', minimum=0, maximum=1, exclusive_minimum=True) == .5


def test_new_output_rejects_file_as_parent_and_allows_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'interactive', lambda: False)
    parent = tmp_path / 'file'
    parent.write_text('preserve')
    values = iter([str(parent / 'child'), str(tmp_path / 'correct')])
    monkeypatch.setattr('builtins.input', lambda _: next(values))
    assert ui.path('Output', tmp_path, kind='directory', new=True) == tmp_path / 'correct'
    assert parent.read_text() == 'preserve' and not (tmp_path / 'correct').exists()
