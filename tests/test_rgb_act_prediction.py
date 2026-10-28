import base64
from contextlib import contextmanager
import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import urllib.error
import urllib.request

import pytest

from test_entrypoints import ROOT, load


def message(stamp=100.0, side="left", image=False):
    header = SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp))
    if image:
        return SimpleNamespace(header=header, data=b"\xff\xd8test-jpeg")
    offset = 0 if side == "left" else 20
    return SimpleNamespace(header=header, name=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
                           position=[offset + i for i in range(7)], effort=[offset + 10 + i for i in range(7)])


def populated(mod, stamp=100.0, received=None):
    buffer = mod.ObservationBuffer()
    for key in mod.IMAGE_TOPICS:
        buffer.add(key, message(stamp, image=True), received=received)
    for key in mod.STATE_TOPICS:
        buffer.add(key, message(stamp, side=key), received=received)
    return buffer


def test_snapshot_matches_training_order_and_rejects_missing_streams():
    mod = load("rgb_act_ros_observer")
    with pytest.raises(mod.NotReady, match="Missing observations"):
        mod.ObservationBuffer().snapshot(100.0)
    result = populated(mod, received=10.0).snapshot(100.1, 10.1)
    assert result["prediction_only"] is True
    assert result["observation.state"] == list(range(7)) + list(range(10, 17)) + list(range(20, 27)) + list(range(30, 37))
    assert result["state_names"][:2] == ["left.joint1.pos", "left.joint2.pos"]
    assert result["state_names"][14] == "right.joint1.pos"
    assert result["max_skew_sec"] == 0
    assert base64.b64decode(result["images"]["observation.image"]) == b"\xff\xd8test-jpeg"


def test_sync_uses_history_to_match_slower_camera():
    mod = load("rgb_act_ros_observer")
    buffer = populated(mod, received=10.0)
    for key in (*mod.IMAGE_TOPICS, *mod.STATE_TOPICS):
        if key != "observation.image":
            buffer.add(key, message(100.2, side=key, image=key in mod.IMAGE_TOPICS), received=10.2)
    result = buffer.snapshot(100.3, 10.3)
    assert set(result["source_stamps"].values()) == {100.0}


@pytest.mark.parametrize("ros_now,mono_now,reason", [(100.1, 10.6, "Stale"), (100.6, 10.1, "Stale"),
                                                    (99.0, 10.1, "future"), (100.1, 9.0, "future")])
def test_snapshot_rejects_stale_or_inconsistent_clocks(ros_now, mono_now, reason):
    mod = load("rgb_act_ros_observer")
    with pytest.raises(mod.NotReady, match=reason):
        populated(mod, received=10.0).snapshot(ros_now, mono_now)


def test_snapshot_rejects_unsynchronized_streams():
    mod = load("rgb_act_ros_observer")
    buffer = populated(mod, received=10.0)
    buffer.invalidate("right", "test")
    buffer.add("right", message(100.2, side="right"), received=10.0)
    with pytest.raises(mod.NotReady, match="synchronization"):
        buffer.snapshot(100.3, 10.1)


@pytest.mark.parametrize("condition", ["nan", "inf", "names", "position", "effort", "timestamp", "zero-stamp", "image"])
def test_bad_observations_are_rejected(condition):
    mod = load("rgb_act_ros_observer")
    buffer = mod.ObservationBuffer()
    key = "observation.image" if condition == "image" else "left"
    sample = message(image=condition == "image")
    if condition in ("nan", "inf"):
        sample.position[0] = float(condition)
    elif condition == "names":
        sample.name.reverse()
    elif condition == "position":
        sample.position.append(0)
    elif condition == "effort":
        sample.effort.pop()
    elif condition in ("timestamp", "zero-stamp"):
        sample.header.stamp.to_sec = lambda: float("nan") if condition == "timestamp" else 0
    else:
        sample.data = b"not a jpeg"
    with pytest.raises(ValueError):
        buffer.add(key, sample)


def test_invalid_latest_sample_cannot_fall_back_to_old_data():
    mod = load("rgb_act_ros_observer")
    buffer = populated(mod, received=10.0)
    with pytest.raises(ValueError, match="Non-increasing"):
        buffer.add("left", message(99.0), received=10.1)
    buffer.invalidate("left", "Non-increasing timestamp")
    with pytest.raises(mod.NotReady, match="left"):
        buffer.snapshot(100.1, 10.1)
    buffer.add("left", message(100.01), received=10.1)
    assert buffer.snapshot(100.1, 10.1)["prediction_only"] is True


@contextmanager
def observer_server(mod, buffer):
    server = mod.make_server(buffer, 0, lambda: 100.1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_exposes_observations_and_health_but_no_command_endpoints():
    mod = load("rgb_act_ros_observer")
    with observer_server(mod, populated(mod)) as url:
        for route in ("/health", "/observation"):
            with urllib.request.urlopen(url + route) as response:
                assert json.load(response)["prediction_only"] is True
        for request, status in [(url + "/enable", 404),
                                (urllib.request.Request(url + "/observation", data=b"{}", method="POST"), 404)]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == status


def test_http_reports_missing_streams_without_synthesizing_observations():
    mod = load("rgb_act_ros_observer")
    with observer_server(mod, mod.ObservationBuffer()) as url:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/observation")
        assert error.value.code == 503
        assert "Missing observations" in json.load(error.value)["error"]


@pytest.mark.parametrize('states_only', [False, True])
def test_ros_entrypoint_only_subscribes_and_cleans_up(monkeypatch, states_only):
    mod = load("rgb_act_ros_observer")
    subscriptions, closed = [], []
    def subscribe(topic, msg_type, callback, **kwargs):
        subscriptions.append((topic, kwargs["callback_args"]))
        return SimpleNamespace(unregister=lambda: closed.append(topic))
    forbidden = lambda *a, **kw: pytest.fail("A predictions-only observer attempted a command interface")
    rospy = SimpleNamespace(init_node=lambda *a, **kw: None, Subscriber=subscribe,
                            Publisher=forbidden, ServiceProxy=forbidden,
                            Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.0)),
                            is_shutdown=lambda: True, signal_shutdown=lambda reason: closed.append(reason))
    sensor = ModuleType("sensor_msgs.msg")
    sensor.CompressedImage = type("CompressedImage", (), {})
    sensor.JointState = type("JointState", (), {})
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor)
    monkeypatch.setattr(mod, "make_server", lambda *a: SimpleNamespace(server_port=8766, server_close=lambda: closed.append("server")))
    monkeypatch.setattr(sys, "argv", ["observer"] + (['--states-only'] if states_only else []))
    assert mod.main() == 0
    topics = mod.STATE_TOPICS if states_only else {**mod.IMAGE_TOPICS, **mod.STATE_TOPICS}
    assert subscriptions == [(topic, key) for key, topic in topics.items()]
    assert len(closed) == len(topics) + 2


@pytest.mark.parametrize("url", ["https://127.0.0.1/observation", "http://example.com/observation",
                                 "http://127.0.0.1/enable", "http://127.0.0.1/observation?execute=true"])
def test_predictor_rejects_nonlocal_observation_sources(url):
    with pytest.raises(ValueError, match="loopback"):
        load("predict_rgb_act").read_ros_observation(url)


@pytest.mark.parametrize("condition", ["protocol", "stale", "names", "future"])
def test_predictor_rejects_invalid_transport_before_loading_images(monkeypatch, condition):
    mod = load("predict_rgb_act")
    payload = {"schema_version": 1, "prediction_only": True, "generated_at_unix": time.time(),
               "state_names": mod.STATE_NAMES}
    if condition == "protocol":
        payload["prediction_only"] = False
    elif condition == "names":
        payload["state_names"] = list(reversed(mod.STATE_NAMES))
    else:
        payload["generated_at_unix"] += -10 if condition == "stale" else 10
    response = SimpleNamespace(read=lambda limit: json.dumps(payload).encode())
    @contextmanager
    def open_response(*a, **kw):
        yield response
    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_response))
    with pytest.raises(ValueError):
        mod.read_ros_observation("http://127.0.0.1/observation")


@pytest.mark.parametrize("flag", ["--execute", "--publish", "--enable-robot"])
def test_predictions_cli_has_no_actuation_option(monkeypatch, tmp_path, flag):
    mod = load("predict_rgb_act")
    monkeypatch.setattr(sys, "argv", ["predict", "--checkpoint", str(tmp_path), "--observation-npz",
                                     str(tmp_path / "input.npz"), "--output-dir", str(tmp_path / "out"), flag])
    with pytest.raises(SystemExit) as error:
        mod.main()
    assert error.value.code == 2


def test_new_scripts_do_not_import_hardware_or_process_launchers():
    import ast
    for name in ("rgb_act_ros_observer", "predict_rgb_act"):
        tree = ast.parse((ROOT / "scripts" / (name + ".py")).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & {"subprocess", "can", "piper_sdk"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"Publisher", "ServiceProxy", "Service", "MotorEnable", "MotionCtrl_2"}


@pytest.mark.parametrize("invalid", [None, "state", "image", "action"])
def test_model_inputs_include_batch_dimension_for_custom_camera_keys(invalid):
    torch = pytest.importorskip("torch")
    import numpy as np
    mod = load("predict_rgb_act")
    predictor = mod.RGBActPredictor.__new__(mod.RGBActPredictor)
    predictor.np, predictor.torch, predictor.area_resize = np, torch, None
    features = {"observation.state": SimpleNamespace(shape=(28,))}
    features.update({key: SimpleNamespace(shape=(3, 16, 16)) for key in mod.IMAGE_TOPICS})
    config = SimpleNamespace(chunk_size=50, input_features=features)
    def preprocess(batch):
        for key, feature in features.items():
            assert list(batch[key].shape) == [1, *feature.shape]
        return batch
    def predict(batch):
        return torch.full((1, 50, 14), float("nan") if invalid == "action" else 0.0)
    predictor.preprocessor = preprocess
    predictor.postprocessor = lambda action: action
    predictor.policy = SimpleNamespace(config=config, predict_action_chunk=predict)
    observation = {"observation.state": np.zeros(28, dtype=np.float32)}
    observation.update({key: np.zeros((16, 16, 3), dtype=np.uint8) for key in mod.IMAGE_TOPICS})
    if invalid == "state":
        observation["observation.state"][0] = np.nan
    elif invalid == "image":
        observation["observation.right_wrist_image"] = np.zeros((3, 16, 16), dtype=np.uint8)
    if invalid:
        with pytest.raises(ValueError):
            predictor.predict(observation)
    else:
        assert predictor.predict(observation).shape == (50, 14)


def test_checkpoint_dataset_relocation_preserves_nested_path(monkeypatch, tmp_path):
    mod = load('predict_rgb_act')
    model = tmp_path / 'model'
    model.mkdir()
    saved = {'dataset': {'root': '/old/checkout/data/lerobot/nested/task'},
             'output_dir': '/old/checkout/outputs/run'}
    (model / 'train_config.json').write_text(json.dumps(saved))
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    dataset = tmp_path / 'data/lerobot/nested/task'
    dataset.mkdir(parents=True)
    assert mod.checkpoint_dataset_root(model) == dataset
    dataset.rmdir()
    with pytest.raises(ValueError, match='dataset is unavailable'):
        mod.checkpoint_dataset_root(model)


def test_checkpoint_dataset_does_not_guess_unrelated_paths(monkeypatch, tmp_path):
    mod = load('predict_rgb_act')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path / 'data/lerobot/task').mkdir(parents=True)
    (tmp_path / 'train_config.json').write_text(json.dumps({
        'dataset': {'root': '/unrelated/task'}, 'output_dir': '/old/checkout/outputs/run'}))
    with pytest.raises(ValueError, match='dataset is unavailable'):
        mod.checkpoint_dataset_root(tmp_path)


def test_requested_cuda_cannot_silently_fall_back_to_cpu(monkeypatch, tmp_path):
    mod = load('predict_rgb_act')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(ValueError, match='--gpus all'):
        mod.RGBActPredictor(tmp_path, 'cuda')
