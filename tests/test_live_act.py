"""Live ACT re-observes every step and discards predictions across local controls."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_direct_act import mit_pose
from test_entrypoints import load
from test_rgb_act_execution import config, plan, telemetry


def payload(guard, stamp=100., action=None):
    result = plan(guard.sequence + 1, stamp, action or mit_pose())
    result['action'] = result.pop('actions')[0]
    result['policy_epoch'] = guard.policy_epoch
    return result


def armed(reset=False):
    guard = load('live_act').LiveACTGuard(config(), mit_pose() if reset else None)
    guard.arm(mit_pose(), telemetry()[1], 10.)
    return guard


def test_latest_step_is_held_between_inferences_without_future_queue():
    guard = armed()
    target = mit_pose()
    target[0] = .1
    guard.submit(payload(guard, action=target), mit_pose(), 100., 10.)
    measured = mit_pose()
    for n in range(1, 10):
        now = 10 + n/30
        guard.last_client = now
        measured = guard.step(measured, now)
    assert 0 < measured[0] < .02 and not guard.queue
    assert guard.target[0] == pytest.approx(.02)
    new_target = target[:]
    new_target[0] = .08
    guard.submit(payload(guard, 100.3, new_target), measured, 100.3, now)
    assert guard.target[0] == pytest.approx(.04)
    assert guard.sequence == 2 and not guard.queue


def test_live_gripper_roundoff_does_not_stop_or_publish_an_out_of_bounds_target():
    measured = mit_pose()
    measured[6] = measured[13] = .08
    guard = load('live_act').LiveACTGuard(config())
    guard.arm(measured, telemetry()[1], 10.)
    action = measured[:]
    action[6] = action[13] = float(np.nextafter(np.float32(.08), np.float32(1.)))
    guard.submit(payload(guard, action=action), measured, 100., 10.)
    assert guard.sequence == 1 and guard.active
    assert guard.target[6] == guard.target[13] == .08
    assert guard.step(measured, 10.03) == measured
    invalid = measured[:]
    invalid[6] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        guard.submit(payload(guard, 100.03, invalid), measured, 100.03, 10.03)
    assert guard.sequence == 1


@pytest.mark.parametrize('index', [6, 13])
@pytest.mark.parametrize('value,bound', [(.080004952848, .08), (.0810013338923, .08),
                                       (.2, .08), (1e308, .08),
                                       (-.000004952848, 0.), (-.001001, 0.), (-1e308, 0.)])
def test_live_gripper_finite_overshoot_is_clipped_before_smoothing(index, value, bound):
    measured = mit_pose()
    measured[index] = bound
    guard = load('live_act').LiveACTGuard(config())
    guard.arm(measured, telemetry()[1], 10.)
    action = measured[:]
    action[index] = value
    guard.submit(payload(guard, action=action), measured, 100., 10.)
    assert action[index] == value
    assert guard.target == measured
    assert guard.step(measured, 10.03) == measured
    assert guard.active and guard.sequence == 1


@pytest.mark.parametrize('index', [6, 13])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True])
def test_live_gripper_rejects_nonfinite_or_invalid_targets(index, value):
    guard = armed()
    action = mit_pose()
    action[index] = value
    with pytest.raises(ValueError):
        guard.submit(payload(guard, action=action), mit_pose(), 100., 10.)
    assert guard.target is None and guard.sequence == 0


@pytest.mark.parametrize('index,value', [(6, -.000005), (13, .080005)])
def test_live_gripper_clipping_does_not_relax_feedback_or_reset_limits(index, value):
    guard = armed()
    measured = mit_pose()
    measured[index] = value
    with pytest.raises(ValueError, match='limits'):
        guard.submit(payload(guard), measured, 100., 10.)
    with pytest.raises(ValueError, match='limits'):
        load('live_act').LiveACTGuard(config(), measured)
    assert guard.sequence == 0


def test_bridge_logs_raw_gripper_overshoot_and_still_stops_on_joint_limit():
    mod = load('rgb_act_ros_executor')
    events = io.StringIO()
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.)))
    trial = mod.Trial(config(), None, ros, False, {}, {}, events, None, live_act=True)
    measured = mit_pose()
    measured[6] = .08
    trial.ready = lambda now: (measured, 100.)
    trial.diagnostics = telemetry()[1]
    trial.arm()
    action = measured[:]
    action[6] = .0810013338923
    assert trial.submit(payload(trial.guard, action=action)) is True
    step = next(json.loads(line) for line in events.getvalue().splitlines()
                if json.loads(line)['event'] == 'step')
    assert step['raw_action'][6] == action[6]
    assert step['guarded_target'][6] == .08
    assert trial.guard.active
    action[4] = 1.3
    trial.ready = lambda now: (measured, 100.01)
    with pytest.raises(ValueError, match='limits'):
        trial.submit(payload(trial.guard, 100.01, action))
    assert not trial.guard.active and trial.guard.phase == 'stopped'


def test_live_act_continues_past_legacy_five_and_sixty_second_budgets():
    guard = armed()
    for n in range(1, 2101):
        now, stamp = 10 + n/30, 100 + n/30
        guard.last_client = now
        guard.submit(payload(guard, stamp), mit_pose(), stamp, now)
        assert guard.step(mit_pose(), now) == mit_pose()
    assert guard.active and guard.continuous


@pytest.mark.parametrize('fault', ['plan', 'heartbeat', 'tracking', 'deadline', 'observation', 'nan'])
def test_live_act_keeps_watchdogs_and_validation(fault):
    guard = armed()
    guard.submit(payload(guard), mit_pose(), 100., 10.)
    measured, now = mit_pose(), 10.03
    if fault == 'plan':
        guard.last_plan = 9.
    elif fault == 'heartbeat':
        guard.last_client = 9.
    elif fault == 'tracking':
        measured[0] = .3
    elif fault == 'deadline':
        now = 10.2
    elif fault in ('nan', 'observation'):
        item = payload(guard, 100.01)
        if fault == 'nan':
            item['action'][0] = float('nan')
        else:
            item['source_stamps']['observation.image'] = 99.
        with pytest.raises(ValueError):
            guard.submit(item, measured, 100.01, now)
        return
    with pytest.raises(ValueError):
        guard.step(measured, now)


def test_pause_holds_measured_pose_and_clears_pending_target_across_resume():
    guard = armed()
    guard.submit(payload(guard), mit_pose(), 100., 10.)
    measured = mit_pose()
    measured[0] = .01
    old_epoch = guard.policy_epoch
    guard.toggle_pause(measured, 10.01)
    assert guard.phase == 'paused' and guard.target is guard.previous_target is None
    assert guard.policy_epoch > old_epoch
    for n in range(1, 1001):
        now = 10.01+n/30
        guard.last_client = now
        assert guard.step(measured, now) == measured
    guard.toggle_pause(measured, now)
    assert guard.phase == 'policy' and guard.target is None
    guard.policy_after = 101.
    with pytest.raises(ValueError, match='fresh observations'):
        guard.submit(payload(guard), measured, 101., now+.01)
    guard.submit(payload(guard, 101.01), measured, 101.01, now+.01)
    assert guard.sequence == 2


def test_enter_returns_slowly_and_holds_first_frame_until_space():
    guard = armed(reset=True)
    guard.phase = 'policy'
    guard.submit(payload(guard), mit_pose(), 100., 10.)
    measured = mit_pose()
    measured[0] = .1
    guard.rewind(measured, 10.01)
    assert guard.phase == 'reset' and guard.target is None
    for n in range(1, 1801):
        now = 10.01+n/30
        guard.last_client = now
        previous = measured
        measured = guard.step(measured, now)
        assert max(abs(a-b) for a,b in zip(measured, previous)) <= .2/30 + 1e-8
        if guard.phase == 'paused':
            break
    assert guard.phase == 'paused' and abs(measured[0]) < .01
    guard.toggle_pause(measured, now+.01)
    assert guard.phase == 'policy' and guard.target is None


def test_first_frame_timeout_remains_sixty_seconds():
    guard = armed(reset=True)
    guard.last_tick = guard.last_client = 70.
    with pytest.raises(ValueError, match='positioning timed out'):
        guard.step(mit_pose(), 70.01)


def test_inflight_prediction_from_before_pause_is_discarded_without_disarming():
    mod = load('rgb_act_ros_executor')
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda:100.)))
    trial = mod.Trial(config(), None, ros, False, {}, {}, io.StringIO(), None, live_act=True)
    trial.ready = lambda now: (mit_pose(), 100.)
    trial.diagnostics = telemetry()[1]
    trial.arm()
    item = payload(trial.guard)
    trial.replay_key('pause')
    assert trial.submit(item) is False
    assert trial.guard.active and trial.guard.phase == 'paused' and trial.guard.target is None
    trial.replay_key('pause')
    assert trial.submit(item) is False  # Phase is running, but epoch changed twice.
    assert trial.guard.active and trial.guard.target is None


def test_client_uses_new_observations_single_step_endpoint_and_resets_cache(monkeypatch):
    mod = load('execute_rgb_act')
    def status(phase='policy', epoch=1):
        return {'armed': True, 'ready': True, 'phase': phase, 'policy_epoch': epoch,
                'sequence': 0, 'policy_after': 100.}
    states = iter([status('reset'), status('paused', 2), status(epoch=3),
                   status(epoch=3), status(epoch=4), {'local_exit_requested': True}])
    snapshots = iter([99., 100.1, 100.2])
    calls, observations, resets = [], [], []
    def request(session, route, item=None):
        calls.append((route, item))
        if route == '/status':
            return next(states)
        if route == '/step':
            assert 'actions' not in item and len(item['action']) == 14
            return {'accepted': item['policy_epoch'] == 4, 'sequence': 1}
    def observe(*args, **kw):
        stamp = next(snapshots)
        return {'stamp':stamp}, {'source_stamps': {'left': stamp}, 'observation_stamp': stamp}
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod, 'read_ros_observation', observe)
    monkeypatch.setattr(mod.time, 'sleep', lambda _: None)
    predictor = SimpleNamespace(reset=lambda: resets.append(True),
        select_action=lambda obs: observations.append(obs['stamp']) or np.array(mit_pose()))
    assert mod.run_client({'live_act': True}, predictor, 'unused', config()) == 0
    assert observations == [100.1, 100.2]
    assert len(resets) >= 4
    assert [item['policy_epoch'] for route,item in calls if route == '/step'] == [3,4]
    assert all(route != '/chunk' for route,_ in calls)
    assert calls[-1][0] == '/stop'


def test_http_step_accepts_one_target_and_discards_old_epoch_after_pause():
    from test_rgb_act_execution import command_server
    mod = load('rgb_act_ros_executor')
    client = load('execute_rgb_act')
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda:100.)))
    trial = mod.Trial(config(), None, ros, False, {}, {}, io.StringIO(), None, live_act=True)
    trial.ready = lambda now: (mit_pose(), 100.)
    trial.diagnostics = telemetry()[1]
    trial.arm()
    with command_server(trial) as url:
        session = {'url':url, 'token':'test-private-token', 'execute':False}
        result = client.request(session, '/step', payload(trial.guard))
        assert result['accepted'] and result['sequence'] == 1
        old = payload(trial.guard, 100.01)
        trial.replay_key('pause')
        result = client.request(session, '/step', old)
        assert result['accepted'] is False and result['armed'] and result['phase'] == 'paused'
        assert trial.guard.target is None


def test_pause_during_reset_does_not_consume_its_sixty_second_budget():
    guard = armed(reset=True)
    measured = mit_pose()
    guard.toggle_pause(measured, 10.01)
    now = 10.01
    for _ in range(2100):
        now += 1/30
        guard.last_client = now
        guard.step(measured, now)
    guard.toggle_pause(measured, now)
    assert guard.phase == 'reset'
    assert now - guard.started < .02
    guard.last_client = now+1/30
    guard.step(measured, now+1/30)
    assert guard.active


def test_distant_and_changing_model_targets_are_limited_like_tictac_without_disabling_tracking():
    guard = armed()
    measured = mit_pose()
    target = measured[:]
    target[0], target[6], target[9] = .4, .07, -.4
    # All positions are legal, but exceed the old initial-distance and raw-jump thresholds.
    guard.submit(payload(guard, action=target), measured, 100., 10.)
    first = guard.target[:]
    assert first[0] == pytest.approx(.02)
    assert first[6] == pytest.approx(measured[6]+.002)
    assert first[9] == pytest.approx(measured[9]-.02)
    target[0], target[9] = -.4, .4
    guard.submit(payload(guard, 100.01, target), measured, 100.01, 10.01)
    assert max(abs(a-b) for i,(a,b) in enumerate(zip(guard.target,first)) if i%7!=6) <= .02+1e-8
    assert max(abs(a-b) for i,(a,b) in enumerate(zip(guard.target,first)) if i%7==6) <= .002+1e-8
    command = guard.step(measured, 10.03)
    assert max(abs(a-b) for a,b in zip(command,measured)) <= .2/30
    measured[0] = .3
    guard.last_client = 10.06
    with pytest.raises(ValueError,match='tracking'):
        guard.step(measured,10.06)


@pytest.mark.parametrize('index,value',[(0,3.),(1,12.501),(2,-12.501),(4,1.3)])
def test_tictac_target_shaping_cannot_hide_out_of_bounds_predictions(index,value):
    guard = armed()
    target = mit_pose()
    target[index] = value
    with pytest.raises(ValueError,match='limits'):
        guard.submit(payload(guard,action=target),mit_pose(),100.,10.01)


@pytest.mark.parametrize('phase', ['policy', 'paused', 'reset'])
def test_quit_returns_slowly_without_client_then_restores_mux(monkeypatch, phase):
    mod = load('rgb_act_ros_executor')
    clock, measured, commands, restored = [10.], mit_pose(), [], []
    measured[0] = .15
    monkeypatch.setattr(mod.time, 'monotonic', lambda: clock[0])
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.+clock[0])))
    trial = mod.Trial(config(), None, ros, False, {}, {}, io.StringIO(), None,
                      reset_positions=mit_pose(), live_act=True)
    trial.ready = lambda now: (measured[:], 100.+clock[0])
    trial.diagnostics = telemetry()[1]
    trial.publish = lambda command: commands.append(command[:])
    trial.restore_teleop = lambda: restored.append(measured[:])
    trial.arm()
    trial.guard.phase = phase
    old = payload(trial.guard)
    trial.replay_key('quit')
    assert trial.returning_to_start and trial.guard.phase == 'reset'
    assert trial.guard.target is None and not restored and not trial.local_exit_requested
    epoch, started = trial.guard.policy_epoch, trial.guard.started
    trial.replay_key('quit')
    trial.replay_key('pause')
    assert trial.guard.policy_epoch == epoch and trial.guard.started == started
    assert trial.submit(old) is False
    for _ in range(300):
        previous = measured[:]
        clock[0] += 1/30
        trial.tick()
        measured[:] = commands[-1]
        assert max(abs(a-b) for a,b in zip(measured, previous)) <= .2/30+1e-8
        if trial.exit_requested:
            break
    assert clock[0] > 10.5  # Return succeeds after the disconnected client's heartbeat would expire.
    assert trial.exit_requested and trial.local_exit_requested and not trial.guard.active
    assert not trial.returning_to_start and abs(measured[0]) < .01
    assert len(restored) == 1 and abs(restored[0][0]) < .01
    assert 'exit_return_reached' in trial.event_log.getvalue()


@pytest.mark.parametrize('fault', ['before-arm', 'missing-feedback', 'tracking', 'return-timeout', 'stop'])
def test_exit_return_never_bypasses_faults_or_arms_a_disarmed_robot(monkeypatch, fault):
    mod = load('rgb_act_ros_executor')
    clock = [10.]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: clock[0])
    measured, commands = mit_pose(), []
    measured[0] = .1
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.)))
    trial = mod.Trial(config(), None, ros, False, {}, {}, io.StringIO(), None,
                      reset_positions=mit_pose(), live_act=True)
    trial.ready = lambda now: (measured[:], 100.)
    trial.publish = lambda command: commands.append(command[:])
    trial.diagnostics = telemetry()[1]
    if fault != 'before-arm':
        trial.arm()
    if fault == 'missing-feedback':
        def missing(now):
            raise ValueError('Stale CAN feedback')
        trial.ready = missing
    trial.quit()
    if trial.returning_to_start:
        if fault == 'tracking':
            measured[0] += .5
        elif fault == 'return-timeout':
            trial.guard.started = -51.
        elif fault == 'stop':
            trial.stop('Immediate stop')
        clock[0] += 1/30
        trial.tick()
    assert not trial.guard.active and not trial.returning_to_start
    assert trial.exit_requested and trial.local_exit_requested
    assert 'exit_return_reached' not in trial.event_log.getvalue()
    if fault in ('before-arm', 'missing-feedback'):
        assert commands == []


@pytest.mark.parametrize('immediate', [False, True])
def test_authenticated_quit_returns_normally_but_aborts_on_client_failure(immediate):
    from test_rgb_act_execution import command_server
    mod = load('rgb_act_ros_executor')
    client = load('execute_rgb_act')
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.)))
    trial = mod.Trial(config(), None, ros, False, {}, {}, io.StringIO(), None,
                      reset_positions=mit_pose(), live_act=True)
    trial.ready = lambda now: (mit_pose(), 100.)
    trial.diagnostics = telemetry()[1]
    trial.arm()
    with command_server(trial) as url:
        session = {'url':url, 'token':'test-private-token', 'execute':False}
        result = client.request(session, '/quit', {'immediate': True} if immediate else {})
        if immediate:
            assert not result['returning_to_start'] and 'exit_return_started' not in trial.event_log.getvalue()
        else:
            assert result['returning_to_start'] and result['armed'] and not trial.exit_requested
            client.request(session, '/stop', {})
        assert trial.exit_requested and not trial.guard.active


@pytest.mark.parametrize('failure', [KeyboardInterrupt, ValueError])
def test_client_ctrl_c_returns_home_but_faults_request_immediate_stop(monkeypatch, failure):
    mod = load('execute_rgb_act')
    routes = []
    def request(session, route, item=None):
        routes.append(route)
        if route == '/status':
            raise failure('test cancellation')
    monkeypatch.setattr(mod, 'request', request)
    if failure is KeyboardInterrupt:
        assert mod.run_live_client({}, object(), 'unused', config()) == 0
        assert routes == ['/status', '/quit']
    else:
        with pytest.raises(ValueError):
            mod.run_live_client({}, object(), 'unused', config())
        assert routes == ['/status', '/stop']


def test_client_waits_during_exit_return_without_running_model(monkeypatch):
    mod = load('execute_rgb_act')
    statuses = iter([{'armed':True, 'ready':True, 'returning_to_start':True},
                     {'local_exit_requested':True}])
    routes = []
    def request(session, route, item=None):
        routes.append(route)
        if route == '/status':
            return next(statuses)
    monkeypatch.setattr(mod, 'request', request)
    monkeypatch.setattr(mod.time, 'sleep', lambda _: None)
    assert mod.run_live_client({}, object(), 'unused', config()) == 0
    assert routes == ['/status', '/status', '/stop']
