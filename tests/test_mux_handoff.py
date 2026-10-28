"""Local ARM owns mux handoff; idle/dry-run clients cannot switch robot control."""
import io
from types import SimpleNamespace

import pytest

from test_entrypoints import load
from test_rgb_act_execution import config, pose, telemetry


def setup_trial(monkeypatch, mode='act', execute=True):
    mod = load('rgb_act_ros_executor')
    now = [10.]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: now[0])
    calls, sent = [], []
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.)))
    obs, diagnostics, _, gravity = telemetry()
    observations = SimpleNamespace(snapshot=lambda *a: obs)
    pubs = {side: SimpleNamespace(publish=lambda msg, s=side: sent.append((s, msg.position)))
            for side in mod.STATE_TOPICS}
    kwargs = dict(live_act=True) if mode == 'act' else dict(states_only=True)
    if mode == 'recording':
        from test_recorded_replay import plan
        cfg, recording = plan()
        kwargs.update(reset_positions=recording['actions'][0], recording_plan=recording)
    else:
        cfg = config()
    def select(side, topic):
        calls.append((side, topic))
        return topic
    events = io.StringIO()
    trial = mod.Trial(cfg, observations, ros, execute, pubs, {}, events,
                      lambda: SimpleNamespace(header=SimpleNamespace()), mux_select=select, **kwargs)
    trial.diagnostics, trial.gravity = diagnostics, gravity
    trial.selected = {s: '/teleop/arm_%s/joint_states_single' % s for s in mod.STATE_TOPICS}
    trial.check_graph = lambda now: None  # Exclusive graph/driver checks have separate coverage.
    return mod, trial, now, calls, sent


@pytest.mark.parametrize('mode', ['act', 'recording', 'chunk-replay'])
@pytest.mark.parametrize('execute', [False, True])
def test_local_arm_switches_shared_mux_and_stop_restores_teleop(monkeypatch, mode, execute):
    mod, trial, now, calls, sent = setup_trial(monkeypatch, mode, execute)
    assert trial.status()['ready']  # Default teleop selection does not block ARM.
    trial.tick()
    trial.stop('idle client quit')
    assert not calls and not sent
    trial.arm()
    assert trial.guard.active
    assert calls == (list(mod.COMMAND_TOPICS.items()) if execute else [])
    assert len(sent) == (2 if execute else 0)  # Only measured holds during handoff.
    assert trial.status()['ready']
    if execute:
        assert trial.selected == mod.COMMAND_TOPICS
    trial.stop('operator exit')
    assert not trial.guard.active and not trial.mux_owned
    if execute:
        assert set(calls[2:]) == {(s, '/teleop/arm_%s/joint_states_single' % s) for s in mod.STATE_TOPICS}
        assert len(sent) == 4  # ARM hold and stop hold, one per arm.
    else:
        assert not calls and not sent
    before = list(calls), list(sent)
    trial.stop('bridge cleanup')
    assert before == (calls, sent)


@pytest.mark.parametrize('failed_side', ['left', 'right'])
def test_partial_handoff_failure_disarms_and_rolls_back_attempted_arms(monkeypatch, failed_side):
    mod, trial, _, calls, _ = setup_trial(monkeypatch)
    def select(side, topic):
        calls.append((side, topic))
        if side == failed_side and topic == mod.COMMAND_TOPICS[side]:
            raise ValueError('selection acknowledgement missing')
        return topic
    trial.mux_select = select
    with pytest.raises(ValueError, match='acknowledgement'):
        trial.arm()
    assert not trial.guard.active and not trial.mux_owned
    attempted = {'left'} if failed_side == 'left' else {'left', 'right'}
    for side in attempted:
        assert calls.count((side, '/teleop/arm_%s/joint_states_single' % side)) == 1
    trial.tick()
    assert not trial.guard.active


@pytest.mark.parametrize('fault', ['moving', 'disabled', 'gravity', 'stale', 'old-driver'])
def test_bad_feedback_cannot_publish_or_switch_mux(monkeypatch, fault):
    _, trial, _, calls, sent = setup_trial(monkeypatch)
    item = trial.diagnostics['left'][0]
    if fault == 'moving':
        item['velocities_rad_s'][0] = .1
    elif fault == 'disabled':
        item['driver_enabled'] = False
    elif fault == 'gravity':
        trial.gravity['left'] = (False, 10.)
    elif fault == 'old-driver':
        item['command_topic'] = '/robot/arm_left/replay_joint_cmd'
    else:
        item['joint_feedback_age_s'][0] = 1.
    with pytest.raises(ValueError):
        trial.arm()
    assert not calls and not sent and not trial.guard.active


@pytest.mark.parametrize('mode', ['act', 'recording'])
def test_pause_keeps_policy_mux_and_manual_takeover_stops_without_writing(monkeypatch, mode):
    mod, trial, now, calls, sent = setup_trial(monkeypatch, mode)
    trial.arm()
    trial.replay_key('pause')
    assert trial.guard.phase == 'paused' and len(calls) == 2
    before = len(sent)
    trial.selected['left'] = '/intervention/teleop_handoff/arm_left/joint_states_single'
    now[0] += .01
    trial.tick()
    assert not trial.guard.active
    assert len(sent) == before  # No hold or model command after takeover.
    assert calls[2:] == [('right', '/teleop/arm_right/joint_states_single')]
    trial.tick()
    assert len(sent) == before


def test_watchdog_stop_restores_teleop_and_requires_new_local_arm(monkeypatch):
    _, trial, now, calls, sent = setup_trial(monkeypatch)
    trial.arm()
    trial.guard.last_client -= 1.
    now[0] += .01
    trial.tick()
    assert not trial.guard.active and not trial.mux_owned and len(calls) == 4
    before = len(sent)
    trial.tick()
    assert len(sent) == before


def test_failed_restore_is_reported_and_retried_on_cleanup(monkeypatch, capsys):
    _, trial, _, calls, _ = setup_trial(monkeypatch)
    trial.arm()
    select = trial.mux_select
    def failed_restore(side, topic):
        if side == 'left' and topic.startswith('/teleop/'):
            raise ValueError('service unavailable')
        return select(side, topic)
    trial.mux_select = failed_restore
    trial.stop('operator exit')
    assert not trial.guard.active and trial.mux_owned == {'left'}
    assert 'mux_restore_failed' in trial.event_log.getvalue()
    assert 'Teleop menu 7' in capsys.readouterr().out
    trial.mux_select = select
    trial.stop('bridge cleanup')
    assert not trial.mux_owned


@pytest.mark.parametrize('case', ['old-then-target', 'no-confirmation', 'service-failure'])
def test_mux_service_waits_for_target_confirmation_and_unregisters(monkeypatch, case):
    import sys
    import threading
    from types import ModuleType
    mod = load('rgb_act_ros_executor')
    service_type = object()
    service_module = ModuleType('topic_tools.srv')
    service_module.MuxSelect = service_type
    monkeypatch.setitem(sys.modules, 'topic_tools', ModuleType('topic_tools'))
    monkeypatch.setitem(sys.modules, 'topic_tools.srv', service_module)
    callbacks, unregistered, workers = [], [], []
    target = mod.COMMAND_TOPICS['left']
    class ServiceError(Exception):
        pass
    def subscribe(topic, msg_type, callback, **kwargs):
        assert topic == mod.MUX_TOPICS['left'] + '/selected'
        callbacks.append(callback)
        callback(SimpleNamespace(data='/teleop/arm_left/joint_states_single'))
        return SimpleNamespace(unregister=lambda: unregistered.append(True))
    def proxy(service, typ):
        assert service == mod.MUX_TOPICS['left'] + '_select' and typ is service_type
        def call(topic):
            assert topic == target
            if case == 'service-failure':
                raise ServiceError('selection unavailable')
            if case == 'old-then-target':
                worker = threading.Timer(.01, lambda: callbacks[0](SimpleNamespace(data=target)))
                workers.append(worker)
                worker.start()
        return call
    ros = SimpleNamespace(Subscriber=subscribe, ServiceProxy=proxy, wait_for_service=lambda *a, **k: None,
                          ROSException=ServiceError, ServiceException=ServiceError)
    if case == 'old-then-target':
        assert mod.select_mux(ros, 'left', target, object) == target
    else:
        with pytest.raises(ValueError, match='not confirmed|switch failed'):
            mod.select_mux(ros, 'left', target, object)
    for worker in workers:
        worker.join()
    assert unregistered == [True]
