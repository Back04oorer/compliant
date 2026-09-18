"""Small, robot-agnostic checkpoint loader for Stereo-Patch policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import factory as policy_factory

# Imports register config subclasses before PreTrainedConfig parses config.json.
import lerobot_policy_stereo_act  # noqa: F401
import lerobot_policy_stereo_diffusion  # noqa: F401
import lerobot_policy_stereo_flow  # noqa: F401
import lerobot_policy_stereo_flow_v2  # noqa: F401

from .processors import make_stereo_pre_post_processors


SUPPORTED_TYPES = {"stereo_act", "stereo_diffusion", "stereo_flow_v2"}


def resolve_checkpoint(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(
        snapshot_download(
            repo_id=str(checkpoint), revision=revision, token=token, cache_dir=cache_dir
        )
    ).resolve()


def checkpoint_contract(config: PreTrainedConfig) -> dict[str, Any]:
    return {
        "policy_type": config.type,
        "input_features": {
            key: list(feature.shape) for key, feature in config.input_features.items()
        },
        "output_features": {
            key: list(feature.shape) for key, feature in config.output_features.items()
        },
        "fastfs": {
            key: getattr(config, key, None)
            for key in (
                "fastfs_valid_iters",
                "fastfs_scale",
                "fastfs_input_width",
                "fastfs_input_height",
                "fastfs_max_disp",
                "fastfs_max_depth_m",
            )
        },
        "n_obs_steps": getattr(config, "n_obs_steps", None),
        "horizon": getattr(config, "horizon", getattr(config, "chunk_size", None)),
        "n_action_steps": getattr(config, "n_action_steps", None),
    }


@dataclass
class LoadedStereoPolicy:
    checkpoint: Path
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any

    @property
    def contract(self) -> dict[str, Any]:
        return checkpoint_contract(self.policy.config)

    def prepare(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        processed = self.preprocessor(dict(observation))
        result: dict[str, Any] = {}
        for key in self.policy.config.input_features:
            if key not in processed:
                raise KeyError(f"Observation is missing required policy key: {key}")
            value = processed[key]
            result[key] = value.to(self.policy.config.device) if torch.is_tensor(value) else value
        return result

    @torch.inference_mode()
    def predict_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        batch = self.prepare(observation)
        chunk = self.policy.predict_action_chunk(batch)
        output = self.postprocessor(chunk)
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected action chunk [H,A], got {array.shape}")
        return array

    def reset(self) -> None:
        self.policy.reset()


def load_stereo_policy(
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    dino_model: str | Path | None = None,
    defm_root: str | Path | None = None,
    num_inference_steps: int | None = None,
    flow_num_steps: int | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> LoadedStereoPolicy:
    local = resolve_checkpoint(
        checkpoint, revision=revision, token=token, cache_dir=cache_dir
    )
    overrides = [f"--device={device}"]
    if dino_model is not None:
        overrides.append(f"--dino_model_id={Path(dino_model).expanduser().resolve()}")
    if defm_root is not None:
        overrides.append(f"--defm_root={Path(defm_root).expanduser().resolve()}")
    if num_inference_steps is not None:
        overrides.append(f"--num_inference_steps={int(num_inference_steps)}")
    if flow_num_steps is not None:
        overrides.append(f"--flow_num_steps={int(flow_num_steps)}")

    config = PreTrainedConfig.from_pretrained(local, cli_overrides=overrides)
    if config.type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported policy type {config.type!r}; expected {sorted(SUPPORTED_TYPES)}")
    policy_cls = policy_factory.get_policy_class(config.type)
    policy = policy_cls.from_pretrained(local, config=config)
    preprocessor, postprocessor = make_stereo_pre_post_processors(
        config,
        default_factory=policy_factory.make_pre_post_processors,
        pretrained_path=str(local),
    )
    return LoadedStereoPolicy(local, policy, preprocessor, postprocessor)
