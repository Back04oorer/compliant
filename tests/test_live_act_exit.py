"""Exercise bridge shutdown with real terminal input and SIGINT, without ROS or CAN."""
import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import time

import pytest

from test_entrypoints import ROOT


@pytest.mark.parametrize('control', ['q', 'sigint', 'abort'])
def test_bridge_process_returns_before_exit_and_second_sigint_can_abort(tmp_path, control):
    config = tmp_path/'config.json'
    config.write_text((ROOT/'config/rgb_act_execution.json').read_text())
    reset = tmp_path/'reset.json'
    reset.write_text(json.dumps({'positions': [0, -.4, .3, 0, 0, 0, .04]*2}))
    session = tmp_path/'bridge'
    # Only feedback is simulated; the real bridge main loop, keyboard, guard and signal handler run.
    source = '''
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path[:0] = [str(Path.cwd()/'scripts'), str(Path.cwd()/'tests')]
from test_entrypoints import load
from test_rgb_act_execution import telemetry
from test_direct_act import mit_pose
mod = load('rgb_act_ros_executor')
original = mod.Trial
class SimulatedFeedback(original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measured = mit_pose()
        self.measured[0] = .15
        self.diagnostics = telemetry()[1]
        self.guard.arm(self.measured, self.diagnostics, mod.time.monotonic())
        self.guard.phase = 'policy'
    def ready(self, now):
        return self.measured[:], mod.time.time()
    def publish(self, command):
        self.measured = list(command)
mod.Trial = SimulatedFeedback
subscriber = lambda *a, **kw: SimpleNamespace(unregister=lambda: None)
sys.modules['rospy'] = SimpleNamespace(init_node=lambda *a, **kw:None,
    get_param=lambda name, default:default, is_shutdown=lambda:False,
    Subscriber=subscriber, signal_shutdown=lambda reason:None,
    Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=mod.time.time)))
sys.modules['sensor_msgs.msg'] = SimpleNamespace(CompressedImage=object, JointState=object)
sys.modules['std_msgs.msg'] = SimpleNamespace(Bool=object, String=object)
sys.exit(mod.main())
'''
    master, slave = pty.openpty()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, '-u', '-c', source, '--config', str(config),
        '--reset-pose', str(reset), '--session-dir', str(session), '--live-act', '--port', str(port)],
        cwd=ROOT, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    output = bytearray()
    def wait_for(text):
        deadline = time.monotonic()+5
        while text not in output.decode(errors='replace'):
            if process.poll() is not None:
                pytest.fail(output.decode(errors='replace'))
            assert time.monotonic() < deadline, output.decode(errors='replace')
            if select.select([master], [], [], .1)[0]:
                output.extend(os.read(master, 65536))
    try:
        wait_for('q/Ctrl-C returns to first frame, then exits.')
        if control == 'q':
            os.write(master, b'q')
        else:
            process.send_signal(signal.SIGINT)
        wait_for('Returning slowly to dataset first frame before exit.')
        assert process.poll() is None
        if control == 'abort':
            process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 0
        events = [json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
        names = [event['event'] for event in events]
        assert 'exit_return_started' in names
        if control == 'abort':
            assert 'exit_return_reached' not in names
            assert any(event.get('reason') == 'Exit return cancelled by Ctrl-C' for event in events)
        else:
            assert 'exit_return_reached' in names
            reached = next(event for event in events if event['event'] == 'exit_return_reached')
            assert abs(reached['measured'][0]) < .01
            commands = [event['positions'][0] for event in events if event['event'] == 'dry_run_command']
            assert all(abs(a-b) <= .2/30+1e-8 for a,b in zip([.15]+commands, commands))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
