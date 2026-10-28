from contextlib import contextmanager
import copy
import io
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pytest

from test_entrypoints import ROOT, load


def config():
    return json.loads((ROOT / "config/rgb_act_execution.json").read_text())


def pose():
    return [0, 1, -1, 0, 0, 0, 0.04] * 2


def telemetry(now=10.0, ros_now=100.0):
    mod = load("rgb_act_execution")
    diagnostics = {}
    for side in mod.STATE_TOPICS:
        diagnostics[side] = ({"schema_version": 1, "published_at": ros_now, "driver_enabled": True,
                             "blocked": False, "position_only": True, "motor_enabled": [True] * 6,
                             "gripper_enabled": True, "motor_faults": [0] * 6, "err_code": 0, "arm_status": 0,
                             "teach_status": 0, "ctrl_mode": 1, "expected_mode_feedback": 4, "mode_feedback": 4,
                             "command_topic": mod.MUX_TOPICS[side], "gripper_topic": mod.MUX_TOPICS[side],
                             "joint_feedback_age_s": [0] * 6, "speed_feedback_age_s": [0] * 6,
                             "enable_feedback_age_s": [0] * 6, "gripper_feedback_age_s": 0, "status_age_s": 0,
                             "velocities_rad_s": [0] * 6, "positions_rad": pose()[:6],
                             "gripper_width_m": 0.04, "gravity_required": True}, now)
    state = pose()[:7] + [0] * 7 + pose()[7:] + [0] * 7
    return {"observation.state": state}, diagnostics, dict(mod.COMMAND_TOPICS), {
        side: (True, now) for side in mod.STATE_TOPICS}


def armed():
    guard = load("rgb_act_execution").ExecutionGuard(config())
    _, diagnostics, _, _ = telemetry()
    guard.arm(pose(), diagnostics, 10.0)
    return guard


def plan(sequence=1, stamp=100.0, target=None):
    mod = load("rgb_act_execution")
    return {"sequence": sequence, "rate_hz": 30, "action_names": mod.ACTION_NAMES,
            "source_stamps": {key: stamp for key in (*mod.IMAGE_TOPICS, *mod.STATE_TOPICS)},
            "observation_stamp": stamp, "actions": [list(target or pose()) for _ in range(50)]}


def test_config_limits_match_both_urdfs_and_reject_expansion():
    mod = load("rgb_act_execution")
    for side in ("left", "right"):
        path = ROOT / ("src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/urdf/"
                       "piper_description_%s.urdf" % side)
        joints = {joint.attrib["name"]: joint for joint in ET.parse(path).getroot().findall("joint")}
        for i in range(6):
            limit = joints["joint%d" % (i + 1)].find("limit")
            assert config()["arm_position_min"][i] >= float(limit.attrib["lower"])
            assert config()["arm_position_max"][i] <= float(limit.attrib["upper"])
    for key, value in (("rate_hz", 10), ("trial_seconds", 100), ("feedback_timeout_s", float("nan")),
                       ("chunk_size", 50.0), ("unexpected", 1), ("joint_velocity_rad_s", 0.3)):
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError):
            mod.ExecutionGuard(cfg)
    cfg = config()
    cfg["arm_position_max"][0] = 99
    with pytest.raises(ValueError, match="bounds"):
        mod.ExecutionGuard(cfg)


def test_trial_limits_stay_inside_the_verified_motor_configuration():
    mod = load('rgb_act_execution')
    cfg = config()
    assert cfg['arm_position_min'][0] >= -math.radians(150)
    assert cfg['arm_position_max'][0] <= math.radians(150)
    assert cfg['arm_position_min'][2] >= -math.radians(170)
    assert cfg['arm_position_min'][4] >= -math.radians(70)
    assert cfg['arm_position_max'][4] <= math.radians(70)
    assert cfg['joint_velocity_rad_s'] <= 0.2
    cfg['arm_position_max'][4] = 1.57
    with pytest.raises(ValueError, match='bounds'):
        mod.ExecutionGuard(cfg)


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('index', [6, 13])
@pytest.mark.parametrize('value,expected', [(0.08000000566244125, .08), (-1e-9, 0.)])
def test_gripper_endpoint_roundoff_is_normalized_inside_limits(direct, index, value, expected):
    guard = load('rgb_act_execution').ExecutionGuard(config(), mit_j2_j3=direct)
    target = pose()
    target[index] = value
    assert guard.position(target)[index] == expected
    assert guard.measured_position(target)[index] == expected
    assert target[index] == value


@pytest.mark.parametrize('index,value', [(6, .08000002), (13, -2e-8), (6, .080004)])
def test_gripper_excess_beyond_roundoff_is_rejected_with_useful_precision(index, value):
    target = pose()
    target[index] = value
    with pytest.raises(ValueError, match='excess') as error:
        load('rgb_act_execution').ExecutionGuard(config()).position(target)
    assert '%.12g' % value in str(error.value)


@pytest.mark.parametrize("failure", [None, "mux", "disabled", "motor", "gripper", "fault", "mode", "teach",
                                     "age", "can", "null-can", "gravity", "velocity", "position", "unknown-gravity",
                                     "arm-fault", "unknown-arm-status"])
def test_feedback_requires_real_fresh_enabled_position_only_drivers(failure):
    guard = armed()
    obs, diagnostics, selected, gravity = telemetry()
    item = diagnostics["left"][0]
    if failure == "mux":
        selected["left"] = "/teleop/arm_left/joint_states_single"
    elif failure == "disabled":
        item["driver_enabled"] = False
    elif failure == "motor":
        item["motor_enabled"][3] = False
    elif failure == "gripper":
        item["gripper_enabled"] = False
    elif failure == "fault":
        item["motor_faults"][0] = 2
    elif failure == "mode":
        item["mode_feedback"] = 1
    elif failure == "teach":
        item["teach_status"] = 1
    elif failure == "age":
        item["published_at"] = 99
    elif failure in ("can", "null-can"):
        item["joint_feedback_age_s"][1] = None if failure == "null-can" else 1.0
    elif failure == "gravity":
        gravity["left"] = (False, 10.0)
    elif failure == "velocity":
        item["velocities_rad_s"][0] = float("nan")
    elif failure == "position":
        obs["observation.state"][0] = 9
    elif failure == "unknown-gravity":
        item.pop("gravity_required")
    elif failure == "arm-fault":
        item["arm_status"] = 7
    elif failure == "unknown-arm-status":
        item.pop("arm_status")
    if failure:
        with pytest.raises(ValueError):
            guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0)
    else:
        assert guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0) == pose()


@pytest.mark.parametrize('teach_status', [0, 2])
def test_disabled_or_stopped_sdk_teaching_state_is_accepted(teach_status):
    guard = armed()
    obs, diagnostics, selected, gravity = telemetry()
    for item, _ in diagnostics.values():
        item['teach_status'] = teach_status
    assert guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0) == pose()


def test_stationary_start_and_explicit_arm_required():
    mod = load("rgb_act_execution")
    guard = mod.ExecutionGuard(config())
    assert not guard.active and guard.step(pose(), 10.0) is None
    with pytest.raises(ValueError, match="not armed"):
        guard.submit(plan(), pose(), 100.0, 10.0)
    _, diagnostics, _, _ = telemetry()
    diagnostics["right"][0]["velocities_rad_s"][2] = 0.04
    with pytest.raises(ValueError, match="stationary"):
        guard.arm(pose(), diagnostics, 10.0)


def test_dry_run_can_read_disabled_drivers_without_changing_mux_or_enabling_motors():
    guard = armed()
    obs, diagnostics, selected, gravity = telemetry()
    for side, (item, _) in diagnostics.items():
        item["driver_enabled"] = False
        item["motor_enabled"] = [False] * 6
        item["gripper_enabled"] = False
        selected[side] = "/teleop/arm_%s/joint_states_single" % side
        gravity[side] = (False, 10.0)
    assert guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0, require_control=False) == pose()
    with pytest.raises(ValueError, match="Mux"):
        guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0)


def test_motion_guard_uses_latest_can_position_not_the_older_camera_aligned_pose():
    guard = armed()
    obs, diagnostics, selected, gravity = telemetry()
    diagnostics["right"][0]["positions_rad"][0] = 0.05
    measured = guard.feedback(obs, diagnostics, selected, gravity, 100.0, 10.0)
    assert measured[7] == 0.05 and obs["observation.state"][14] == 0


@pytest.mark.parametrize("failure", ["nan", "shape", "bounds", "first", "jump", "sequence", "names", "rate",
                                     "missing-stamp", "stale", "future", "skew", "reference"])
def test_invalid_chunks_are_rejected_before_any_action_is_queued(failure):
    guard, payload = armed(), plan()
    if failure == "nan":
        payload["actions"][-1][0] = float("nan")
    elif failure == "shape":
        payload["actions"][-1].pop()
    elif failure == "bounds":
        payload["actions"][-1][2] = 0.01
    elif failure == "first":
        for row in payload["actions"]:
            row[0] = 0.16
    elif failure == "jump":
        payload["actions"][-1][0] = 0.11
    elif failure == "sequence":
        payload["sequence"] = 2
    elif failure == "names":
        payload["action_names"] = list(reversed(payload["action_names"]))
    elif failure == "rate":
        payload["rate_hz"] = 50
    elif failure == "missing-stamp":
        payload["source_stamps"].pop("left")
    elif failure == "stale":
        payload = plan(stamp=99.0)
    elif failure == "future":
        payload = plan(stamp=101.0)
    elif failure == "skew":
        payload["source_stamps"]["left"] = 99.9
    elif failure == "reference":
        payload["observation_stamp"] = 99.99
    with pytest.raises(ValueError):
        guard.submit(payload, pose(), 100.0, 10.0)
    assert not guard.queue and guard.sequence == 0


def test_chunks_cannot_replace_queue_or_replay_an_observation():
    guard, payload = armed(), plan()
    payload["source_stamps"]["left"] = 99.99  # reference need not be the earliest nearest-history sample
    guard.submit(payload, pose(), 100.0, 10.0)
    with pytest.raises(ValueError, match="still executing"):
        guard.submit(plan(2, 100.1), pose(), 100.1, 10.1)
    guard.queue.clear()
    with pytest.raises(ValueError, match="new synchronized"):
        guard.submit(plan(2), pose(), 100.1, 10.1)


def test_commands_enforce_velocity_acceleration_and_gripper_units():
    guard = armed()
    target = pose()
    target[0], target[7], target[6], target[13] = 0.12, -0.12, 0.05, 0.03
    guard.submit(plan(target=target), pose(), 100.0, 10.0)
    previous, old_velocity = pose(), [0] * 14
    vmax = guard.limits(config()['joint_velocity_rad_s'], 0.015)
    amax = guard.limits(1.0, 0.05)
    for step in range(1, 51):
        now = 10 + step / 30
        guard.last_client = now
        result = guard.step(previous, now)
        for i in range(14):
            velocity = (result[i] - previous[i]) * 30
            assert abs(velocity) <= vmax[i] + 1e-8
            assert abs(velocity - old_velocity[i]) <= amax[i] / 30 + 1e-8
            old_velocity[i] = velocity
        previous = result
    assert 0 < previous[0] < target[0] and target[7] < previous[7] < 0
    assert 0.04 < previous[6] <= 0.05 and 0.03 <= previous[13] < 0.04
    assert not guard.queue


@pytest.mark.parametrize("failure", ["late", "trial", "tracking", "plan-watchdog", "client-watchdog"])
def test_control_deadlines_tracking_and_time_budget_stop_motion(failure):
    guard = armed()
    now, measured = 10.03, pose()
    if failure == "late":
        now = 10.2
    elif failure == "trial":
        guard.started = 4.0
    elif failure == "tracking":
        measured[0] = 0.3
    elif failure == "plan-watchdog":
        guard.last_plan = 9.0
    elif failure == "client-watchdog":
        guard.last_client = 9.0
    with pytest.raises(ValueError):
        guard.step(measured, now)


def fake_trial(execute=False):
    mod = load("rgb_act_ros_executor")
    published = {"left": [], "right": []}
    ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: 100.0))
    message_type = lambda: SimpleNamespace(header=SimpleNamespace(), velocity=[], effort=[])
    pubs = {side: SimpleNamespace(publish=values.append) for side, values in published.items()}
    trial = mod.Trial(config(), None, ros, execute, pubs, {}, io.StringIO(), message_type,
                      mux_select=lambda side, topic: topic)
    _, trial.diagnostics, trial.selected, trial.gravity = telemetry()
    trial.ready = lambda now: (pose(), 100.0)
    return trial, published


def test_dry_run_cannot_publish_even_when_armed_and_consuming_actions():
    trial, published = fake_trial()
    trial.arm()
    trial.submit(plan())
    time.sleep(0.002)
    trial.tick()
    assert trial.guard.active and len(trial.guard.queue) == 49
    trial.stop("test")
    assert published == {"left": [], "right": []}


def test_bad_chunk_latches_stop_and_sends_only_measured_hold():
    trial, published = fake_trial(True)
    trial.arm()
    bad = plan()
    bad["actions"][-1][0] = float("nan")
    with pytest.raises(ValueError):
        trial.submit(bad)
    assert not trial.guard.active
    assert [msg.position for msg in published["left"]] == [pose()[:7]] * 2
    assert published["left"][0].velocity == published["left"][0].effort == []
    trial.tick()
    assert len(published["left"]) == 2  # ARM primer + stopped hold; no automatic resume


def test_takeover_does_not_publish_hold_into_the_new_control_source():
    trial, published = fake_trial(True)
    trial.arm()
    def lost_route(now):
        raise ValueError("Mux takeover")
    trial.ready = lost_route
    trial.tick()
    assert not trial.guard.active
    assert all(len(messages) == 1 for messages in published.values())  # ARM primer only


def test_client_failure_stops_even_with_a_full_queued_chunk():
    trial, published = fake_trial(True)
    trial.arm()
    trial.submit(plan())
    trial.guard.last_client -= 1.0
    trial.tick()
    assert not trial.guard.active and not trial.guard.queue
    assert "heartbeat expired" in trial.guard.reason
    assert [msg.position for msg in published["left"]] == [pose()[:7]] * 2


@contextmanager
def command_server(trial):
    mod = load("rgb_act_ros_executor")
    server = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.CommandHandler)
    server.trial, server.token = trial, "test-private-token"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def test_http_requires_token_and_has_no_remote_arm_enable_or_mux_endpoint():
    trial, _ = fake_trial()
    with command_server(trial) as url:
        for route, token, expected in (("/status", "", 403), ("/arm", "test-private-token", 404),
                                        ("/enable", "test-private-token", 404), ("/mux", "test-private-token", 404)):
            req = urllib.request.Request(url + route, headers={"Authorization": "Bearer " + token})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == expected
        assert not trial.guard.active


def test_http_malformed_authorized_request_stops_trial():
    trial, _ = fake_trial()
    trial.arm()
    with command_server(trial) as url:
        req = urllib.request.Request(url + "/chunk", data=b"not-json",
                                     headers={"Authorization": "Bearer test-private-token"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 409
    assert not trial.guard.active


def test_client_requires_separate_execution_opt_in_before_model_loading(monkeypatch, tmp_path):
    mod = load("execute_rgb_act")
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"schema_version": 1, "routing": "mux", "execute": True}))
    session.chmod(0o600)
    monkeypatch.setattr(sys, "argv", ["execute", "--checkpoint", str(tmp_path), "--executor-session", str(session)])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("fault", [None, "publisher", "mux-output", "mux-connection", "command-connection", "gripper-scale",
                                 "selected-publisher", "input-subscriber", "driver-subscriber"])
def test_graph_and_driver_parameter_checks_reject_conflicting_control(monkeypatch, fault):
    mod = load("rgb_act_ros_executor")
    trial, _ = fake_trial(True)
    # Use this module's class so monkeypatching its graph helper affects the instance.
    trial.__class__ = mod.Trial
    values = {"gripper_cmd_limit_enable": True, "gripper_cmd_min_raw": 0, "gripper_cmd_max_raw": 80000}
    trial.ros.get_name = lambda: "/test_executor"
    trial.ros.get_param_cached = lambda key, default: (
        2 if fault == "gripper-scale" and key.endswith("gripper_val_mutiple") else values.get(key.rsplit("/", 1)[-1], default))
    trial.selected_subscribers = {side: SimpleNamespace(get_num_connections=lambda: 0 if fault == "mux-connection" else 1)
                                  for side in mod.STATE_TOPICS}
    for pub in trial.publishers.values():
        pub.get_num_connections = lambda: 0 if fault == "command-connection" else 1
    graph = {topic: ["/test_executor", "/other"] if fault == "publisher" else ["/test_executor"]
             for topic in mod.COMMAND_TOPICS.values()}
    graph.update({topic: ["/rogue"] if fault == "mux-output" else ["/mux_arm_%s_joint" % side]
                  for side, topic in mod.MUX_TOPICS.items()})
    graph.update({topic + '/selected': ['/rogue'] if fault == 'selected-publisher' else ['/mux_arm_%s_joint' % side]
                  for side, topic in mod.MUX_TOPICS.items()})
    subscribers = {topic: ['/rogue'] if fault == 'input-subscriber' else ['/mux_arm_%s_joint' % side]
                   for side, topic in mod.COMMAND_TOPICS.items()}
    subscribers.update({topic: [] if fault == 'driver-subscriber' else ['/piper_ctrl_%s_node' % side]
                        for side, topic in mod.MUX_TOPICS.items()})
    monkeypatch.setattr(mod, "graph_connections", lambda node: (graph, subscribers))
    if fault:
        with pytest.raises(ValueError):
            trial.check_graph(10.0)
    else:
        trial.check_graph(10.0)
        assert trial.graph_error is None


def test_cold_mux_only_holds_after_local_arm_and_must_initialize(monkeypatch):
    mod = load('rgb_act_ros_executor')
    trial, sent = fake_trial(True)
    trial.__class__ = mod.Trial
    trial.ros.get_name = lambda: '/test_executor'
    values = {'gripper_cmd_limit_enable': True, 'gripper_cmd_min_raw': 0, 'gripper_cmd_max_raw': 80000}
    trial.ros.get_param_cached = lambda key, default: values.get(key.rsplit('/', 1)[-1], default)
    trial.selected_subscribers = {side: SimpleNamespace(get_num_connections=lambda: 1) for side in mod.STATE_TOPICS}
    for pub in trial.publishers.values():
        pub.get_num_connections = lambda: 1
    publishers = {topic: ['/test_executor'] for topic in mod.COMMAND_TOPICS.values()}
    publishers.update({topic + '/selected': ['/mux_arm_%s_joint' % side] for side, topic in mod.MUX_TOPICS.items()})
    subscribers = {topic: ['/mux_arm_%s_joint' % side] for side, topic in mod.COMMAND_TOPICS.items()}
    subscribers.update({topic: ['/piper_ctrl_%s_node' % side] for side, topic in mod.MUX_TOPICS.items()})
    monkeypatch.setattr(mod, 'graph_connections', lambda node: (publishers, subscribers))
    now = [10.]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: now[0])
    trial.ready = lambda timestamp: (trial.check_graph(timestamp) or pose(), 100.)
    trial.check_graph(10.)
    assert trial.mux_pending == {'left', 'right'} and trial.first_publish_time is None
    trial.tick()
    assert not any(sent.values()) and not trial.guard.active
    _, diagnostics, _, _ = telemetry()
    trial.guard.arm(pose(), diagnostics, 10.)
    trial.diagnostics = diagnostics
    before = trial.guard.last_command[:]
    now[0] = 10.01
    trial.tick()
    assert trial.guard.last_command == before and trial.first_publish_time == 10.01
    assert all(len(messages) == 1 and messages[0].position == pose()[:7] for messages in sent.values())
    # A real mux must advertise after receiving this first stationary target.
    now[0] = 11.02
    with pytest.raises(ValueError, match='did not initialize') as first:
        trial.check_graph(now[0])
    with pytest.raises(ValueError) as cached:
        trial.check_graph(11.03)
    assert str(cached.value) == str(first.value)
    publishers.update({topic: ['/mux_arm_%s_joint' % side] for side, topic in mod.MUX_TOPICS.items()})
    trial.check_graph(11.3)
    assert not trial.mux_pending and trial.graph_error is None
    del publishers[mod.MUX_TOPICS['left']]
    with pytest.raises(ValueError, match='disappeared'):
        trial.check_graph(11.6)


def test_recording_parameter_error_is_cached_without_alternating_generic_errors(monkeypatch):
    from test_recorded_replay import plan as recording_plan
    cfg, p = recording_plan()
    mod = load('rgb_act_ros_executor')
    ros = SimpleNamespace(get_name=lambda: '/private_test', get_param_cached=lambda key, default: 'changed')
    trial = mod.Trial(cfg, None, ros, False, {}, {}, io.StringIO(), None,
                      reset_positions=p['actions'][0], states_only=True, recording_plan=p)
    monkeypatch.setattr(mod, 'graph_connections', lambda name: ({}, {}))
    errors = []
    for now in (10., 10.1, 10.3, 10.4):
        with pytest.raises(ValueError, match='differs from recording') as exc:
            trial.check_graph(now)
        errors.append(str(exc.value))
    assert len(set(errors)) == 1


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize('states_only,recording_mode,live_act',
                         [(False, False, False), (True, False, False),
                          (True, True, False), (False, False, True)])
def test_mocked_startup_never_sends_a_command_and_default_creates_no_actuator_publishers(monkeypatch, tmp_path, execute, states_only, recording_mode, live_act):
    mod = load("rgb_act_ros_executor")
    publications, subscriptions = [], []
    forbidden = lambda *args, **kwargs: pytest.fail("Unexpected robot command/service during startup")
    def publisher(topic, msg_type, **kwargs):
        publications.append((topic, kwargs))
        return SimpleNamespace(publish=forbidden, unregister=lambda: None)
    def subscriber(topic, msg_type, callback, **kwargs):
        subscriptions.append(topic)
        return SimpleNamespace(unregister=lambda: None)
    rospy = SimpleNamespace(init_node=lambda *args, **kwargs: None, get_param=lambda key, default: default,
                            is_shutdown=lambda: False, signal_shutdown=lambda reason: None,
                            Publisher=publisher, Subscriber=subscriber, ServiceProxy=forbidden)
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    for package, names in (("sensor_msgs", ("CompressedImage", "JointState")), ("std_msgs", ("Bool", "String"))):
        module = ModuleType(package + ".msg")
        for name in names:
            setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, package, ModuleType(package))
        monkeypatch.setitem(sys.modules, package + ".msg", module)
    server = SimpleNamespace(serve_forever=lambda **kwargs: None, shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(mod, "ThreadingHTTPServer", lambda *args: server)
    terminal = SimpleNamespace(isatty=lambda: True, readline=lambda: "")
    monkeypatch.setattr(sys, "stdin", terminal)
    monkeypatch.setattr(mod.select, "select", lambda *args: ([terminal], [], []))
    class Keyboard:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            server.trial.exit_requested = True
            return []
    monkeypatch.setattr(mod, 'ReplayKeyboard', Keyboard)
    session_dir = tmp_path / "session"
    argv = ["bridge", "--config", str(ROOT / "config/rgb_act_execution.json"), "--session-dir", str(session_dir)]
    if recording_mode:
        from test_recorded_replay import plan
        cfg, p = plan()
        (tmp_path / 'config.json').write_text(json.dumps(cfg))
        (tmp_path / 'plan.json').write_text(json.dumps(p))
        (tmp_path / 'reset.json').write_text(json.dumps({'positions': p['actions'][0]}))
        argv[argv.index('--config') + 1] = str(tmp_path / 'config.json')
        argv += ['--recording-plan', str(tmp_path / 'plan.json'), '--reset-pose', str(tmp_path / 'reset.json')]
    if live_act:
        argv += ["--live-act"]
    monkeypatch.setattr(sys, "argv", argv + (["--execute"] if execute else []) + (['--states-only'] if states_only else []))
    assert mod.main() == 0
    assert len(subscriptions) == (8 if states_only else 11)
    assert any(topic in subscriptions for topic in mod.IMAGE_TOPICS.values()) is not states_only
    assert server.trial.guard.source_topics == (set(mod.STATE_TOPICS) if states_only else {*mod.IMAGE_TOPICS, *mod.STATE_TOPICS})
    assert len(publications) == (2 if execute else 0)
    assert all(kwargs["latch"] is False for _, kwargs in publications)
    assert {topic for topic, _ in publications} == (set(
        mod.COMMAND_TOPICS.values()) if execute else set())
    assert server.trial.guard.driver_topics == mod.MUX_TOPICS
    assert server.trial.guard.mit_j2_j3 is live_act
    assert json.loads((session_dir / "session.json").read_text())["live_act"] is live_act
    assert json.loads((session_dir / "session.json").read_text())["routing"] == "mux"
    assert (session_dir / "session.json").stat().st_mode & 0o077 == 0
    assert session_dir.stat().st_mode & 0o077 == 0
    assert not server.trial.guard.active


def test_adapter_has_no_hardware_enable_or_process_launcher_calls():
    import ast
    for name in ("rgb_act_execution", "rgb_act_ros_executor", "execute_rgb_act"):
        tree = ast.parse((ROOT / "scripts" / (name + ".py")).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {entry.name.split(".")[0] for entry in node.names} & {"subprocess", "can", "piper_sdk"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"Service", "EnableArm", "DisableArm", "MotionCtrl_2"}
                if node.func.attr == "Publisher":
                    assert "if args.execute:" in (ROOT / "scripts/rgb_act_ros_executor.py").read_text()


def mux_graph(node):
    mod = load('rgb_act_execution')
    published = {topic: [node] for topic in mod.COMMAND_TOPICS.values()}
    subscribed = {}
    for side, output in mod.MUX_TOPICS.items():
        mux = '/mux_arm_%s_joint' % side
        published[output] = published[output + '/selected'] = [mux]
        subscribed[mod.COMMAND_TOPICS[side]] = [mux]
        subscribed[output] = ['/piper_ctrl_%s_node' % side]
    return published, subscribed
