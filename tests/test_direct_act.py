"""Live ACT through mux keeps motion/feedback guards while matching MIT J2/J3 encoding."""
import io
from types import SimpleNamespace

import pytest

from test_entrypoints import load
from test_rgb_act_execution import config, plan, pose, telemetry


def mit_pose():
    return [0., -.03, .03, 0., 0., 0., .04] * 2


def test_mit_j2_j3_preserves_targets_without_changing_other_position_limits():
    mod = load('rgb_act_execution')
    cfg = config()
    guard = mod.ExecutionGuard(cfg, mit_j2_j3=True)
    target = mit_pose()
    assert guard.validate_actions([target[:] for _ in range(50)], target) == [target] * 50
    assert guard.position([0., -12.5, 12.5, 0., 0., 0., .04] * 2)
    assert cfg['arm_position_min'][1] == 0 and cfg['arm_position_max'][2] == 0
    with pytest.raises(ValueError, match='joint2'):
        mod.ExecutionGuard(cfg).position(target)


@pytest.mark.parametrize('index,value', [(1, -12.501), (2, 12.501), (8, float('nan')),
                                        (9, float('inf')), (0, 3.), (3, 2.), (4, 1.3),
                                        (5, 4.), (6, .081), (13, -.001)])
def test_live_act_still_rejects_invalid_positions(index, value):
    target = mit_pose()
    target[index] = value
    with pytest.raises(ValueError):
        load('rgb_act_execution').ExecutionGuard(config(), mit_j2_j3=True).position(target)


def test_live_act_keeps_target_distance_jump_and_camera_freshness_checks():
    guard = load('rgb_act_execution').ExecutionGuard(config(), mit_j2_j3=True)
    measured = mit_pose()
    actions = [measured[:] for _ in range(50)]
    actions[0][1] += .16
    with pytest.raises(ValueError, match='First target'):
        guard.validate_actions(actions, measured)
    actions[0] = measured[:]
    actions[10][2] += .11
    with pytest.raises(ValueError, match='jump'):
        guard.validate_actions(actions, measured)
    _, diagnostics, _, _ = telemetry()
    guard.arm(measured, diagnostics, 10.)
    payload = plan(target=measured)
    payload['source_stamps']['observation.image'] = 99.
    with pytest.raises(ValueError, match='stale'):
        guard.submit(payload, measured, 100., 10.01)


def test_live_act_reset_can_reach_signed_j2_j3_with_mit_following_error():
    mod = load('rgb_act_execution')
    target = mit_pose()
    guard = mod.ExecutionGuard(config(), target, mit_j2_j3=True)
    measured = target[:]
    measured[1], measured[2], measured[8], measured[9] = -.01, .01, -.01, .01
    _, diagnostics, _, _ = telemetry()
    guard.arm(measured, diagnostics, 10.)
    for n in range(1, 1001):
        now = 10. + n / 30
        guard.last_client = now
        previous = guard.last_command[:]
        command = guard.step(measured, now)
        assert max(abs(a-b) for a,b in zip(command, previous)) <= .2 / 30 + 1e-8
        measured = command[:]
        measured[4] += .028  # MIT's static following error also occurs during bag replay.
        if guard.phase == 'policy':
            break
    assert guard.phase == 'policy'
    assert abs(command[1] - target[1]) < .002 and command[1] < 0
    assert abs(command[2] - target[2]) < .002 and command[2] > 0


@pytest.mark.parametrize('fault', [None, 'mux', 'publisher', 'subscriber', 'connection', 'p-mode', 'torque'])
def test_live_act_requires_exclusive_mux_and_position_only_mit_driver(monkeypatch, fault):
    from test_rgb_act_execution import mux_graph
    mod = load('rgb_act_ros_executor')
    values = {'ctrl_mode': 'p' if fault == 'p-mode' else 'mit', 'mit/enable_pos': True,
              'mit/enable_vel': False, 'mit/enable_tor': fault == 'torque',
              'gripper_cmd_limit_enable': True, 'gripper_cmd_min_raw': 0, 'gripper_cmd_max_raw': 80000}
    ros = SimpleNamespace(get_name=lambda: '/act',
                          get_param_cached=lambda k, d: values.get(k.split('_node/', 1)[1], d))
    pubs = {side: SimpleNamespace(get_num_connections=lambda: 0 if fault == 'connection' else 1)
            for side in ('left', 'right')}
    selected = {side: SimpleNamespace(get_num_connections=lambda: 1) for side in pubs}
    trial = mod.Trial(config(), None, ros, True, pubs, selected, io.StringIO(), None, live_act=True)
    published, subscribed = mux_graph('/act')
    if fault == 'publisher':
        published[mod.COMMAND_TOPICS['left']].append('/rogue')
    elif fault == 'subscriber':
        subscribed[mod.COMMAND_TOPICS['left']] = ['/piper_ctrl_left_node']
    elif fault == 'mux':
        published[mod.MUX_TOPICS['left']] = ['/rogue']
    monkeypatch.setattr(mod, 'graph_connections', lambda node: (published, subscribed))
    if fault:
        with pytest.raises(ValueError):
            trial.check_connections(10.)
    else:
        trial.check_connections(10.)
        assert trial.command_topics == mod.COMMAND_TOPICS
        assert trial.guard.driver_topics == mod.MUX_TOPICS
        assert trial.guard.source_topics == {*mod.IMAGE_TOPICS, *mod.STATE_TOPICS}


@pytest.mark.parametrize('execute', [False, True])
def test_live_act_can_consume_model_steps_through_mux_and_never_publishes_before_arm(execute):
    mod = load('rgb_act_ros_executor')
    published = {'left': [], 'right': []}
    pubs = {s: SimpleNamespace(publish=messages.append) for s, messages in published.items()}
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.)))
    message = lambda: SimpleNamespace(header=SimpleNamespace())
    trial = mod.Trial(config(), None, ros, execute, pubs, {}, io.StringIO(), message, live_act=True,
                      mux_select=lambda side, topic: topic)
    measured = mit_pose()
    _, trial.diagnostics, _, trial.gravity = telemetry()
    trial.ready = lambda now: (measured, 100.)
    trial.tick()
    assert published == {'left': [], 'right': []}
    trial.arm()
    payload = plan(target=measured)
    payload.update(action=measured, policy_epoch=trial.guard.policy_epoch)
    del payload['actions']
    assert trial.submit(payload) is True
    trial.tick()
    assert trial.guard.sequence == 1 and not trial.guard.queue
    for side in ('left', 'right'):
        assert len(published[side]) == 2 * int(execute)
        if execute:
            assert published[side][0].position == measured[:7]
    trial.stop('test')
