#!/usr/bin/env python3
"""Predict RGB ACT action chunks from read-only ROS observations or an offline NPZ. Never commands a robot."""
import argparse
import base64
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from rgb_act_ros_observer import ACTION_NAMES, IMAGE_TOPICS, MAX_RESPONSE_BYTES, STATE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_dataset_root(checkpoint):
    """Relocate this repository's saved dataset path when the checkout moves into Docker."""
    train_config = checkpoint / "train_config.json"
    if not train_config.is_file():
        return None
    saved = json.loads(train_config.read_text())
    dataset = saved.get("dataset", {}).get("root")
    if not dataset:
        return None
    dataset = Path(dataset)
    if dataset.is_dir():
        return dataset
    # output_dir identifies the original checkout; preserve the full relative dataset path.
    for parent in Path(saved.get("output_dir") or "").parents:
        if parent.name == "outputs":
            try:
                relative = dataset.relative_to(parent.parent)
            except ValueError:
                break
            if relative.parts and relative.parts[0] == "data" and ".." not in relative.parts:
                relocated = ROOT / relative
                if relocated.is_dir():
                    return relocated
            break
    raise ValueError("Checkpoint dataset is unavailable; mount its data directory inside the container: " + str(dataset))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def read_ros_observation(url, timeout=2.0, max_age=0.5, states_only=False):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or \
            parsed.path != "/observation" or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("Observer URL must be loopback HTTP with path /observation")
    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({})).open(url, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Observer response exceeds 16 MiB")
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or payload.get("prediction_only") is not True:
        raise ValueError("Unexpected observation protocol")
    age = time.time() - float(payload["generated_at_unix"])
    if not math.isfinite(age) or not 0 <= age <= max_age:
        raise ValueError("Observer response is stale or its clock is in the future")
    if payload.get("state_names") != STATE_NAMES:
        raise ValueError("State order differs from the training contract")
    import numpy as np

    observation = {"observation.state": np.asarray(payload["observation.state"], dtype=np.float32)}
    if states_only:
        state = observation['observation.state']
        if state.shape != (len(STATE_NAMES),) or not np.isfinite(state).all():
            raise ValueError('Invalid follower state')
    else:
        import cv2
    for key in (() if states_only else IMAGE_TOPICS):
        image = cv2.imdecode(np.frombuffer(base64.b64decode(payload["images"][key], validate=True), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode JPEG for " + key)
        observation[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return observation, {key: payload[key] for key in
                         ("observation_stamp", "source_stamps", "max_skew_sec", "generated_at_unix")}


class RGBActPredictor:
    def __init__(self, checkpoint, device="cuda", stepwise=False):
        import numpy as np
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable: install CUDA-enabled Torch and create the container with --gpus all")
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.np, self.torch = np, torch
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (self.checkpoint / name).is_file():
                raise ValueError("Incomplete local checkpoint; missing " + name)
        raw = json.loads((self.checkpoint / "config.json").read_text())
        if raw.get("type") != "act":
            raise ValueError("This adapter requires standard RGB ACT (type=act)")
        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        if set(config.input_features) != {"observation.state", *IMAGE_TOPICS}:
            raise ValueError("This adapter requires the three-camera bimanual RGB contract")
        if list(config.input_features["observation.state"].shape) != [28] or \
                list(config.output_features["action"].shape) != [14] or config.n_obs_steps != 1:
            raise ValueError("Expected state28/action14 with n_obs_steps=1")
        self.stepwise = stepwise
        if stepwise:
            config.n_action_steps = 1
            config.temporal_ensemble_coeff = .01
        config.device = device
        # A complete checkpoint already contains the backbone; avoid ImageNet downloads during loading.
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(self.checkpoint, config=config, strict=True, local_files_only=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"},
                                     "unnormalizer_processor": {"device": "cpu"}},
        )
        self.policy.eval()
        self.policy.reset()
        self.area_resize = None
        dataset = checkpoint_dataset_root(self.checkpoint)
        if dataset is not None:
            provenance = dataset / "resize_provenance.json"
            if provenance.is_file():
                resize = json.loads(provenance.read_text())
                if resize.get("filter") != "area":
                    raise ValueError("Unsupported dataset materialization resize filter")
                self.area_resize = tuple(resize["resize"])
                if len(self.area_resize) != 2 or any(type(v) is not int or v <= 0 for v in self.area_resize):
                    raise ValueError("Invalid dataset resize provenance")
                if any(list(config.input_features[key].shape)[1:] != list(self.area_resize) for key in IMAGE_TOPICS):
                    raise ValueError("Dataset resize provenance differs from checkpoint image shapes")
        self.contract = {
            "checkpoint": str(self.checkpoint), "prediction_only": True, "device": device,
            "input_shapes": {key: list(value.shape) for key, value in config.input_features.items()},
            "action_shape": [config.chunk_size, 14], "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "dataset_area_resize": self.area_resize, "saved_processors_loaded": True,
            "stepwise": stepwise, "temporal_ensemble_coeff": config.temporal_ensemble_coeff if stepwise else None,
        }

    def reset(self):
        self.policy.reset()

    def predict(self, observation):
        return self._infer(observation, stepwise=False)

    def select_action(self, observation):
        if not self.stepwise:
            raise ValueError('Stepwise ACT must be enabled when loading the model')
        return self._infer(observation, stepwise=True)

    def _infer(self, observation, stepwise):
        np, torch = self.np, self.torch
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("Expected 28 finite, unnormalized state values")
        # LeRobot's automatic batching does not recognize this project's custom wrist-image keys.
        tensors = {"observation.state": torch.from_numpy(state.copy()).unsqueeze(0)}
        for key in IMAGE_TOPICS:
            image = np.asarray(observation[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or \
                    min(image.shape[:2]) <= 0 or max(image.shape[:2]) > 4096:
                raise ValueError("Expected uint8 RGB HWC image (at most 4096px per dimension) for " + key)
            if self.area_resize is not None:
                import cv2
                height, width = self.area_resize
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            tensors[key] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
        with torch.inference_mode():
            batch = self.preprocessor(tensors)
            for key, feature in self.policy.config.input_features.items():
                if list(batch[key].shape) != [1, *feature.shape] or not torch.isfinite(batch[key]).all():
                    raise ValueError("Invalid processed input: " + key)
            actions = self.postprocessor(self.policy.select_action(batch) if stepwise else self.policy.predict_action_chunk(batch))
            actions = actions.detach().float().cpu().numpy()
        expected = (1, 14) if stepwise else (1, self.policy.config.chunk_size, 14)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError("Policy returned an invalid/non-finite action chunk")
        return actions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local pretrained_model directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observer-url", help="e.g. http://127.0.0.1:8766/observation")
    source.add_argument("--observation-npz", type=Path, help="Offline state28 and three uint8 RGB HWC arrays")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for prediction files")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rate", type=float, default=1.0, help="Live predictions per second; not a control rate")
    parser.add_argument("--max-predictions", type=int, default=10, help="Live prediction count; 0 means until Ctrl-C")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0 or args.max_predictions < 0 or args.cpu_threads < 1:
        parser.error("Rate/CPU threads must be positive; prediction count must be nonnegative")
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        parser.error("Prediction output already exists; choose a new directory")
    import numpy as np
    import torch

    torch.set_num_threads(args.cpu_threads)
    predictor = RGBActPredictor(args.checkpoint, args.device)
    output.mkdir(parents=True, exist_ok=False)
    (output / "contract.json").write_text(json.dumps(predictor.contract, indent=2) + "\n")
    print("PREDICTIONS ONLY: no robot commands, publishers, services, or mux switching", flush=True)
    print(json.dumps(predictor.contract, indent=2), flush=True)
    count, previous_stamp = 0, None
    with (output / "predictions.jsonl").open("x") as log:
        while args.max_predictions == 0 or count < args.max_predictions:
            started = time.monotonic()
            if args.observation_npz:
                with np.load(args.observation_npz, allow_pickle=False) as archive:
                    observation = {key: archive[key] for key in ("observation.state", *IMAGE_TOPICS)}
                metadata = {"source": "offline", "observation_npz": str(args.observation_npz.resolve())}
            else:
                observation, metadata = read_ros_observation(args.observer_url)
                stamp = float(metadata["observation_stamp"])
                if not math.isfinite(stamp) or (previous_stamp is not None and stamp <= previous_stamp):
                    raise ValueError("No new synchronized observation; refusing to reuse an old snapshot")
                previous_stamp = stamp
                metadata["source"] = args.observer_url
            actions = predictor.predict(observation)
            name = "prediction_%06d.npy" % count
            np.save(output / name, actions, allow_pickle=False)
            state = np.asarray(observation["observation.state"])
            measured = np.concatenate((state[:7], state[14:21]))
            delta = actions[0] - measured
            record = {**metadata, "prediction_only": True, "file": name, "shape": list(actions.shape),
                      "elapsed_seconds": time.monotonic() - started, "state": state.tolist(),
                      "first_action": actions[0].tolist(),
                      "max_first_joint_delta_rad": float(np.max(np.abs(delta[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]))),
                      "first_gripper_deltas": delta[[6, 13]].tolist()}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print("Saved %s: shape=%s, elapsed=%.3fs, max first joint delta=%.4frad" %
                  (name, actions.shape, record["elapsed_seconds"], record["max_first_joint_delta_rad"]), flush=True)
            count += 1
            if args.observation_npz:
                break
            time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - started)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Prediction monitoring stopped; no robot commands were sent.")
    except urllib.error.HTTPError as error:
        raise SystemExit("Observer unavailable (%s): %s" % (error.code, error.read(4096).decode(errors="replace")))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        raise SystemExit("Prediction stopped: " + str(error))
