import importlib.util
import os
from pathlib import Path
import select
import threading
import tty
from types import SimpleNamespace

import pytest

from test_entrypoints import ROOT, load


@pytest.fixture
def recorder_module(monkeypatch):
    directory=ROOT/'workspaces/scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('test_progress_recorder',directory/'record_act_keyboard.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminal_pair():
    master, slave=os.openpty()
    tty.setcbreak(slave)
    try:
        yield master, slave
    finally:
        os.close(master)
        os.close(slave)


def test_record_cli_requests_foreground_save_and_verbose_progress(recorder_module):
    command=load('piper').build_command('record', {}, 'test')
    args=recorder_module.build_arg_parser().parse_args(command[2:])
    assert args.wait_for_save and args.save_progress_mode == 'verbose'
    assert args.save_workers == 1 and args.record_status_interval_sec == .5


def test_recording_elapsed_time_is_printed_in_the_terminal(recorder_module, monkeypatch, capsys):
    mod=recorder_module
    recorder=mod.Recorder.__new__(mod.Recorder)
    recorder.recording=True
    recorder.current_started_monotonic=100
    recorder.current_episode_name='episode_003'
    recorder.last_record_status_ts=100
    recorder.record_status_interval_sec=.5
    recorder._pending_save_count=lambda: 0
    recorder._print_colored=lambda text, **kw: print(text)
    recorder._emit_save_progress=lambda **kw: None
    monkeypatch.setattr(mod.time,'monotonic',lambda:112.5)
    recorder._emit_runtime_status()
    output=capsys.readouterr().out
    assert 'ACTIVE episode_003' in output and 'elapsed=' in output and '12' in output


@pytest.mark.parametrize('wait_for_save',[False,True])
def test_stop_waits_for_episode_finalization_before_ready(
        recorder_module, wait_for_save, capsys, monkeypatch, terminal_pair):
    mod=recorder_module
    recorder=mod.Recorder.__new__(mod.Recorder)
    for name,value in {
        'recording':True,'shutdown_requested':False,'current_episode_name':'episode_001','current_episode_dir':Path('/tmp/episode_001'),
        'current_started_utc':'2026-10-20T00:00:00Z','current_started_monotonic':mod.time.monotonic()-1,
        'rosbag_proc':None,'hz_monitors':{},'rosbag_log_handle':None,'session_name':'test',
        'camera_transport':'compressed','camera_streams':[],'required_topics':[],'camera_record_topics':[],
        'extra_topics':[],'current_topics':[],'current_optional_requested':[],'current_optional_recorded':[],
        'current_optional_missing':[],'camera_health_topics':[],'current_preflight':{},'debug_log_paths':[],
        'profile':{},'current_warnings':[],
    }.items():
        setattr(recorder,name,value)
    recorder.args=SimpleNamespace(hz_window=200,split=False,keep_debug_logs=False,notes='',wait_for_save=wait_for_save)
    events=[]
    recorder._divider=lambda:None
    recorder._print_colored=lambda text, **kw:print(text)
    recorder._submit_save_job=lambda job:(events.append('submitted') or 1)
    recorder._scan_next_episode_index=lambda:2
    recorder._emit_save_progress=lambda **kw:events.append('progress')
    master,slave=terminal_pair
    monkeypatch.setattr(mod.sys,'stdin',SimpleNamespace(isatty=lambda:True,fileno=lambda:slave))

    def finish(episode):
        assert episode=='episode_001' and not recorder.recording
        assert 'READY' not in capsys.readouterr().out
        events.append('finished')
        # These keypresses occur while saving and must not start another episode.
        os.write(master,b'   ')
        print('[PASS] episode_001')

    recorder._wait_for_episode_save_job=finish
    recorder._stop_episode('user_stop')
    output=capsys.readouterr().out
    if wait_for_save:
        assert events==['submitted','finished']
        assert output.index('[PASS]') < output.index('READY')
        assert 'saving in background' not in output
        assert not select.select([slave],[],[],0)[0]
        os.write(master,b' ')
        assert select.select([slave],[],[],.2)[0]
        assert os.read(slave,1)==b' '
    else:
        assert events==['submitted','progress'] and 'saving in background' in output


@pytest.mark.parametrize('save_state',['queued','running','done'])
def test_episode_start_is_blocked_until_previous_save_finishes(recorder_module, save_state, capsys):
    recorder=recorder_module.Recorder.__new__(recorder_module.Recorder)
    recorder.recording=False
    recorder.save_lock=threading.Lock()
    recorder.save_jobs={1:{'state':save_state}}
    recorder._print_colored=lambda text, **kw:print(text)
    recorder._emit_save_progress=lambda **kw:None
    preflights=[]

    def master_status():
        preflights.append('checked')
        return False, 'test stops before opening a real recorder'

    recorder._ros_master_ok=master_status
    recorder._start_episode()
    if save_state in ('queued','running'):
        assert not preflights
        assert 'SAVING previous episode' in capsys.readouterr().out
    else:
        assert preflights==['checked']


def test_quit_during_save_is_preserved_while_spaces_are_discarded(recorder_module, monkeypatch, terminal_pair):
    recorder=recorder_module.Recorder.__new__(recorder_module.Recorder)
    recorder.shutdown_requested=False
    master,slave=terminal_pair
    monkeypatch.setattr(recorder_module.sys,'stdin',SimpleNamespace(isatty=lambda:True,fileno=lambda:slave))
    os.write(master,b'  Q ')
    assert select.select([slave],[],[],.2)[0]
    recorder._drain_saving_keys()
    assert recorder.shutdown_requested
    assert not select.select([slave],[],[],0)[0]


def test_verbose_saving_prints_stage_bar_and_final_pass(recorder_module, capsys):
    recorder=recorder_module.Recorder.__new__(recorder_module.Recorder)
    recorder.save_progress_mode='verbose'
    recorder.last_save_status_ts=0
    recorder.save_status_interval_sec=.5
    recorder._snapshot_save_jobs=lambda:list({
        1:{'state':'running','episode_name':'episode_001','step':3,'total':10,'stage':'bag info','status':'RUN'},
        2:{'state':'done','episode_name':'episode_002','reported_done':False,
           'result':{'save_status':'OK','duration_sec':10,'metadata_path':'/tmp/metadata.json'}}}.items())
    recorder._print_colored=lambda text, **kw:print(text)
    recorder._divider=lambda:None
    reported=[]
    recorder._mark_save_reported=lambda ids:reported.extend(ids)
    recorder._emit_save_progress(force=True)
    output=capsys.readouterr().out
    assert 'bag info' in output and '3/10' in output.replace(' ','')
    assert '[###' in output and '[PASS] episode_002' in output
    assert reported==[2]


def test_selected_camera_recorder_stays_in_terminal_with_visible_save_progress(
        recorder_module, tmp_path, monkeypatch):
    import json
    import sys

    module=load('record_session')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    base=tmp_path/'workspaces/config/rosbag_profiles/lfv_teleop_top_stereo_raw_profile.yaml'
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps({'required_topics':[]}))
    monkeypatch.setattr(module,'ensure_camera_free',lambda device:None)
    monkeypatch.setitem(sys.modules,'rosgraph',SimpleNamespace(Master=lambda name:
                        SimpleNamespace(getPid=lambda:1,getPublishedTopics=lambda ns:[])))
    monkeypatch.setattr(sys,'argv',['record_session','--session','test','--camera','left=/dev/video0'])
    commands=[]

    def launch(command, **kwargs):
        commands.append((command,kwargs))
        return SimpleNamespace(wait=lambda **kw:0,poll=lambda:0)

    monkeypatch.setattr(module.subprocess,'Popen',launch)
    assert module.main()==0
    command,kwargs=commands[-1]
    assert kwargs=={}  # Recorder inherits this terminal and prints directly.
    args=recorder_module.build_arg_parser().parse_args(command[2:])
    assert args.wait_for_save and args.save_progress_mode=='verbose' and args.save_workers==1
