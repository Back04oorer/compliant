#!/usr/bin/env python3
"""Train entrypoint that registers the local `stereo_act` policy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for local_path in (REPO_ROOT, SCRIPT_DIR):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

import lerobot_policy_stereo_act  # noqa: F401,E402
import lerobot_policy_stereo_diffusion  # noqa: F401,E402
import lerobot_policy_stereo_flow  # noqa: F401,E402
import lerobot_policy_stereo_flow_v2  # noqa: F401,E402

import accelerate.utils.other as accelerate_other  # noqa: E402
from lerobot.policies import factory as policy_factory  # noqa: E402
from lerobot.scripts import lerobot_train as lerobot_train_module  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats  # noqa: E402

from stereo_patch.processors import make_stereo_pre_post_processors  # noqa: E402
from lerobot_policy_stereo_act.configuration_stereo_act import StereoACTConfig  # noqa: E402
from lerobot_policy_stereo_diffusion.configuration_stereo_diffusion import (  # noqa: E402
    StereoDiffusionConfig,
)
from lerobot_policy_stereo_flow.configuration_stereo_flow import StereoFlowConfig  # noqa: E402


accelerate_other.is_deepspeed_available = lambda: False


ALIASES = {
    "--fusion-mode": "--policy.fusion_mode",
    "--top-rgb-key": "--policy.top_rgb_feature_key",
    "--depth-key": "--policy.depth_feature_key",
    "--depth-feature-is-tokens": "--policy.depth_feature_is_tokens",
    "--target-grid-h": "--policy.target_grid_h",
    "--target-grid-w": "--policy.target_grid_w",
    "--wrist-grid-h": "--policy.wrist_grid_h",
    "--wrist-grid-w": "--policy.wrist_grid_w",
    "--dino-model-id": "--policy.dino_model_id",
    "--defm-root": "--policy.defm_root",
    "--defm-model-name": "--policy.defm_model_name",
}


_ORIG_MAKE_PRE_POST_PROCESSORS = policy_factory.make_pre_post_processors
_ORIG_MAKE_DATASET = lerobot_train_module.make_dataset
_ORIG_MAKE_OPTIMIZER_AND_SCHEDULER = lerobot_train_module.make_optimizer_and_scheduler
_ORIG_DATALOADER = torch.utils.data.DataLoader
_ORIG_EPISODE_AWARE_SAMPLER = lerobot_train_module.EpisodeAwareSampler
_ORIG_SAVE_CHECKPOINT = lerobot_train_module.save_checkpoint
_ORIG_UPDATE_LAST_CHECKPOINT = lerobot_train_module.update_last_checkpoint
_SAMPLING_WEIGHTS_PATH: Path | None = None
_SAMPLING_INDICES_PATH: Path | None = None
_SAMPLING_SEED = 1000
_TRAIN_OUTPUT_DIR: Path | None = None
_SAMPLING_WEIGHTS_CACHE: torch.Tensor | None = None
_SAMPLING_INDICES_CACHE: list[int] | None = None
_SELECTIVE_CHECKPOINT_STEPS: set[int] | None = None
_LAST_CHECKPOINT_WAS_SAVED = True


def _selective_save_checkpoint(*args: Any, **kwargs: Any) -> Any:
    global _LAST_CHECKPOINT_WAS_SAVED
    step = int(kwargs.get("step", args[1] if len(args) > 1 else -1))
    if _SELECTIVE_CHECKPOINT_STEPS is not None and step not in _SELECTIVE_CHECKPOINT_STEPS:
        _LAST_CHECKPOINT_WAS_SAVED = False
        print(f"[stereo-act] selective checkpoint: skip step={step}")
        return None
    _LAST_CHECKPOINT_WAS_SAVED = True
    return _ORIG_SAVE_CHECKPOINT(*args, **kwargs)


def _selective_update_last_checkpoint(*args: Any, **kwargs: Any) -> Any:
    if not _LAST_CHECKPOINT_WAS_SAVED:
        return None
    return _ORIG_UPDATE_LAST_CHECKPOINT(*args, **kwargs)


lerobot_train_module.save_checkpoint = _selective_save_checkpoint
lerobot_train_module.update_last_checkpoint = _selective_update_last_checkpoint


def _is_stereo_policy(policy_cfg: Any) -> bool:
    return isinstance(policy_cfg, (StereoACTConfig, StereoDiffusionConfig, StereoFlowConfig)) or getattr(
        policy_cfg, "type", None
    ) in {"stereo_act", "stereo_diffusion", "stereo_flow", "stereo_flow_v2"}


class _SubsetAwareEpisodeSampler:
    """Episode sampler whose indices match LeRobot's selected HF subset.

    LeRobotDataset stores selected episodes contiguously in ``hf_dataset``, but
    the stock EpisodeAwareSampler emits full-dataset absolute frame indices.
    That is only correct when no episode subset is requested.
    """

    def __init__(
        self,
        dataset_from_indices,
        dataset_to_indices,
        episode_indices_to_use=None,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=False,
    ) -> None:
        if episode_indices_to_use is None:
            self._delegate = _ORIG_EPISODE_AWARE_SAMPLER(
                dataset_from_indices,
                dataset_to_indices,
                episode_indices_to_use=None,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                shuffle=shuffle,
            )
            self.indices = None
            self.shuffle = shuffle
            return
        self._delegate = None
        self.indices: list[int] = []
        relative_start = 0
        for episode_index in episode_indices_to_use:
            absolute_start = int(dataset_from_indices[episode_index])
            absolute_end = int(dataset_to_indices[episode_index])
            episode_length = absolute_end - absolute_start
            start = relative_start + int(drop_n_first_frames)
            end = relative_start + episode_length - int(drop_n_last_frames)
            if end > start:
                self.indices.extend(range(start, end))
            relative_start += episode_length
        self.shuffle = bool(shuffle)

    def __iter__(self):
        if self._delegate is not None:
            yield from self._delegate
            return
        if self.shuffle:
            for index in torch.randperm(len(self.indices)):
                yield self.indices[int(index)]
        else:
            yield from self.indices

    def __len__(self) -> int:
        return len(self._delegate) if self._delegate is not None else len(self.indices)


class _FeatureMmapDataset:
    def __init__(
        self,
        dataset: Any,
        *,
        feature_key: str,
        mmap_path: Path,
        shape: tuple[int, ...],
        dtype: str,
        source_feature: dict[str, Any] | None = None,
        observation_delta_indices: tuple[int, ...] = (0,),
    ) -> None:
        self.dataset = dataset
        self.feature_key = feature_key
        self.mmap_path = mmap_path
        self.shape = shape
        self.dtype = dtype
        self.observation_delta_indices = observation_delta_indices
        self._array_cache: np.memmap | None = None
        if source_feature and hasattr(dataset, "meta") and isinstance(getattr(dataset.meta, "features", None), dict):
            dataset.meta.features[feature_key] = source_feature

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_array_cache"] = None
        return state

    def _np_dtype(self):
        if self.dtype == "float16":
            return np.float16
        if self.dtype == "float32":
            return np.float32
        if self.dtype == "uint8":
            return np.uint8
        raise ValueError(f"Unsupported mmap dtype for {self.feature_key}: {self.dtype}")

    def _array(self) -> np.memmap:
        if self._array_cache is None:
            self._array_cache = np.memmap(self.mmap_path, dtype=self._np_dtype(), mode="r", shape=self.shape)
        return self._array_cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        raw_index = item["index"]
        abs_index = int(raw_index.item() if hasattr(raw_index, "item") else raw_index)
        # mmap-only features are intentionally absent from the Hugging Face
        # table, so LeRobot cannot apply policy observation deltas to them.
        # Reproduce LeRobotDataset._get_query_indices here, including endpoint
        # clamping within the current episode.  A single [0] observation keeps
        # the historical [H,W] shape; temporal policies receive [S,H,W].
        if len(self.observation_delta_indices) == 1:
            mmap_value = np.array(self._array()[abs_index], copy=True)
        else:
            raw_episode = item["episode_index"]
            episode_index = int(
                raw_episode.item() if hasattr(raw_episode, "item") else raw_episode
            )
            episode = self.dataset.meta.episodes[episode_index]
            episode_start = int(episode["dataset_from_index"])
            episode_end = int(episode["dataset_to_index"])
            mmap_indices = [
                max(episode_start, min(episode_end - 1, abs_index + delta))
                for delta in self.observation_delta_indices
            ]
            mmap_value = np.array(self._array()[mmap_indices], copy=True)
        item[self.feature_key] = torch.from_numpy(mmap_value)
        return item


def _wrap_feature_mmap_dataset(
    dataset: Any,
    *,
    feature_key: str,
    dataset_root: Path,
    spec: dict[str, Any],
    label: str,
    observation_delta_indices: tuple[int, ...] = (0,),
) -> Any:
    mmap_path = dataset_root / str(spec.get("path", ""))
    shape_raw = spec.get("shape")
    dtype = str(spec.get("dtype", "float32"))
    if not mmap_path.is_file() or not isinstance(shape_raw, list) or len(shape_raw) < 2:
        raise FileNotFoundError(f"Invalid {label} mmap spec for {feature_key}: {spec}")
    shape = tuple(int(x) for x in shape_raw)
    source_feature = spec.get("source_feature") if isinstance(spec.get("source_feature"), dict) else None
    print(f"[stereo-act] using mmap feature: {feature_key} {mmap_path} shape={shape} dtype={dtype}")
    return _FeatureMmapDataset(
        dataset,
        feature_key=feature_key,
        mmap_path=mmap_path,
        shape=shape,
        dtype=dtype,
        source_feature=source_feature,
        observation_delta_indices=observation_delta_indices,
    )


def _maybe_wrap_stereo_mmap_dataset(cfg: Any, dataset: Any) -> Any:
    policy_cfg = getattr(cfg, "policy", None)
    if not _is_stereo_policy(policy_cfg):
        return dataset
    if not bool(getattr(policy_cfg, "use_depth", True)):
        print("[stereo-act] use_depth=false: depth mmap is not loaded")
        return dataset
    depth_key = str(getattr(policy_cfg, "depth_feature_key", "observation.fastfs_depth_image"))
    dataset_root = Path(getattr(getattr(cfg, "dataset", None), "root", "") or "")
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return dataset
    import json

    info = json.loads(info_path.read_text(encoding="utf-8"))
    observation_delta_indices = tuple(
        int(index) for index in (getattr(policy_cfg, "observation_delta_indices", None) or [0])
    )

    feature_mmaps = info.get("feature_mmaps")
    if isinstance(feature_mmaps, dict) and isinstance(feature_mmaps.get(depth_key), dict):
        return _wrap_feature_mmap_dataset(
            dataset,
            feature_key=depth_key,
            dataset_root=dataset_root,
            spec=feature_mmaps[depth_key],
            label="feature_mmaps",
            observation_delta_indices=observation_delta_indices,
        )

    if bool(getattr(policy_cfg, "depth_feature_is_tokens", False)):
        spec = info.get("defm_tokens_mmap")
        if isinstance(spec, dict) and str(spec.get("token_key", "")) == depth_key:
            return _wrap_feature_mmap_dataset(
                dataset,
                feature_key=depth_key,
                dataset_root=dataset_root,
                spec=spec,
                label="defm_tokens_mmap",
                observation_delta_indices=observation_delta_indices,
            )
    return dataset


def _extract_episode_stats(row: dict[str, Any], feature_dtypes: dict[str, str]) -> dict[str, dict[str, np.ndarray]]:
    episode_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not str(key).startswith("stats/"):
            continue
        _, feature_name, stat_name = str(key).split("/", 2)
        # Custom mmap materialization removes the original depth/video feature
        # from meta.features while retaining its historical per-episode stats.
        # It is loaded in metric units by _FeatureMmapDataset and deliberately
        # excluded from policy normalization, so do not feed those stale stats
        # into LeRobot's subset aggregator.
        if feature_name not in feature_dtypes:
            continue
        array = np.asarray(value)
        feature_dtype = str(feature_dtypes.get(feature_name, "")).strip().lower()
        if feature_dtype in {"image", "video"} and stat_name != "count":
            if array.dtype == object:
                flat_values: list[float] = []
                for item in array:
                    while isinstance(item, np.ndarray):
                        item = item.flatten()[0]
                    flat_values.append(float(item))
                array = np.asarray(flat_values, dtype=np.float64).reshape(3, 1, 1)
            elif array.shape == (3,):
                array = array.astype(np.float64).reshape(3, 1, 1)
        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)
        episode_stats.setdefault(feature_name, {})[stat_name] = array
    return episode_stats


def _apply_episode_subset_stats(cfg: Any, dataset: Any) -> Any:
    selected = getattr(getattr(cfg, "dataset", None), "episodes", None)
    if selected is None:
        return dataset
    selected = sorted(set(int(index) for index in selected))
    if not selected:
        raise ValueError("dataset.episodes cannot be empty")
    episodes_meta = getattr(dataset.meta, "episodes", None)
    if episodes_meta is None:
        raise ValueError("Dataset metadata does not contain per-episode rows")

    feature_dtypes = {
        key: str(spec.get("dtype", "")).strip().lower()
        for key, spec in (getattr(dataset.meta, "features", {}) or {}).items()
        if isinstance(spec, dict)
    }
    groups: dict[tuple[int, int], list[int]] = {}
    for episode_index in selected:
        episode_meta = episodes_meta[episode_index]
        chunk_index = int(episode_meta["meta/episodes/chunk_index"])
        file_index = int(episode_meta["meta/episodes/file_index"])
        groups.setdefault((chunk_index, file_index), []).append(episode_index)

    per_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for (chunk_index, file_index), episode_indices in sorted(groups.items()):
        path = (
            Path(dataset.root)
            / "meta/episodes"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.parquet"
        )
        table = pd.read_parquet(path).set_index("episode_index", drop=False)
        for episode_index in episode_indices:
            row = table.loc[episode_index]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            per_episode_stats.append(_extract_episode_stats(row.to_dict(), feature_dtypes))

    dataset.meta.stats = {
        key: value
        for key, value in aggregate_stats(per_episode_stats).items()
        if key in (getattr(dataset.meta, "features", {}) or {})
    }
    print(
        f"[stereo-act] recomputed normalization stats for episode subset: "
        f"episodes={len(selected)} indices={selected}"
    )
    return dataset


def _make_dataset(cfg: Any) -> Any:
    policy_cfg = getattr(cfg, "policy", None)
    if _is_stereo_policy(policy_cfg):
        # RGB is normalized per encoder inside Stereo-ACT; metric depth must
        # remain in meters instead of receiving global visual normalization.
        cfg.dataset.use_imagenet_stats = False
    dataset = _ORIG_MAKE_DATASET(cfg)
    dataset = _apply_episode_subset_stats(cfg, dataset)
    return _maybe_wrap_stereo_mmap_dataset(cfg, dataset)


def _make_optimizer_and_scheduler(cfg: Any, policy: Any):
    optimizer, scheduler = _ORIG_MAKE_OPTIMIZER_AND_SCHEDULER(cfg, policy)
    scheduler_name = "None" if scheduler is None else type(scheduler).__name__
    behavior = "constant_lr_no_warmup_no_decay" if scheduler is None else "step_after_every_optimizer_update"
    print(
        f"[stereo-act] scheduler={scheduler_name} behavior={behavior} "
        f"resume={bool(getattr(cfg, 'resume', False))}"
    )
    for index, group in enumerate(optimizer.param_groups):
        print(
            f"[stereo-act] optimizer_runtime_group={group.get('name', index)} "
            f"initial_lr={float(group['lr']):.2e}"
        )
    return optimizer, scheduler

def _make_pre_post_processors(policy_cfg, pretrained_path: str | None = None, **kwargs: Any):
    return make_stereo_pre_post_processors(
        policy_cfg,
        default_factory=_ORIG_MAKE_PRE_POST_PROCESSORS,
        pretrained_path=pretrained_path,
        **kwargs,
    )


policy_factory.make_pre_post_processors = _make_pre_post_processors
lerobot_train_module.make_pre_post_processors = _make_pre_post_processors
lerobot_train_module.make_dataset = _make_dataset
lerobot_train_module.make_optimizer_and_scheduler = _make_optimizer_and_scheduler
lerobot_train_module.EpisodeAwareSampler = _SubsetAwareEpisodeSampler
lerobot_train_main = lerobot_train_module.main


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _WeightedDataLoader(_ORIG_DATALOADER):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _SAMPLING_WEIGHTS_CACHE, _SAMPLING_INDICES_CACHE
        if _SAMPLING_WEIGHTS_PATH is None and _SAMPLING_INDICES_PATH is None:
            super().__init__(*args, **kwargs)
            return
        if _SAMPLING_WEIGHTS_PATH is not None and _SAMPLING_INDICES_PATH is not None:
            raise RuntimeError("sampling weights and deterministic sampling indices are mutually exclusive")
        dataset = args[0] if args else kwargs.get("dataset")
        if dataset is None:
            raise RuntimeError("Weighted DataLoader could not resolve dataset")
        existing_sampler = kwargs.get("sampler")
        if existing_sampler is not None and not isinstance(existing_sampler, _SubsetAwareEpisodeSampler):
            raise RuntimeError(
                "sampling weights cannot replace an unknown custom sampler: "
                f"{type(existing_sampler).__name__}"
            )
        if _SAMPLING_INDICES_PATH is not None and _SAMPLING_INDICES_CACHE is None:
            raw_indices = np.load(_SAMPLING_INDICES_PATH, allow_pickle=False)
            if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
                raise RuntimeError(
                    f"Sampling indices must be a 1-D integer array: {raw_indices.shape} {raw_indices.dtype}"
                )
            raw_indices = raw_indices.astype(np.int64, copy=False)
            if len(raw_indices) == 0:
                raise RuntimeError("Sampling indices cannot be empty")
            if np.any(raw_indices < 0) or np.any(raw_indices >= len(dataset)):
                raise RuntimeError("Sampling indices contain out-of-range dataset rows")
            if len(np.unique(raw_indices)) != len(raw_indices):
                raise RuntimeError("Sampling indices must be unique")
            if np.any(np.diff(raw_indices) <= 0):
                raise RuntimeError("Sampling indices must be strictly increasing")
            _SAMPLING_INDICES_CACHE = raw_indices.tolist()
            print(
                "[stereo-act] deterministic anchor sampler: "
                f"path={_SAMPLING_INDICES_PATH} kept={len(raw_indices)} "
                f"dropped={len(dataset) - len(raw_indices)} seed={_SAMPLING_SEED}"
            )
            if _TRAIN_OUTPUT_DIR is not None:
                _TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                summary_path = _SAMPLING_INDICES_PATH.with_name("weights_summary.json")
                summary = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.is_file()
                    else None
                )
                if summary is not None and int(summary.get("total_frames", -1)) != len(dataset):
                    raise RuntimeError("Sampling summary frame count does not match the training dataset")
                if summary is not None:
                    for source in summary.get("source_parquet", []):
                        source_path = Path(source["path"])
                        expected_sha256 = str(source["sha256"])
                        if not source_path.is_file() or _file_sha256(source_path) != expected_sha256:
                            raise RuntimeError(
                                f"Sampling indices are stale for source parquet: {source_path}"
                            )
                manifest = {
                    "sampler": "torch.utils.data.SubsetRandomSampler",
                    "replacement": False,
                    "seed": int(_SAMPLING_SEED),
                    "dataset_frames": len(dataset),
                    "kept_frames": len(raw_indices),
                    "dropped_frames": len(dataset) - len(raw_indices),
                    "indices_path": str(_SAMPLING_INDICES_PATH),
                    "indices_sha256": _file_sha256(_SAMPLING_INDICES_PATH),
                    "summary": summary,
                }
                (_TRAIN_OUTPUT_DIR / "sampling_sampler_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
        if _SAMPLING_WEIGHTS_PATH is not None and _SAMPLING_WEIGHTS_CACHE is None:
            raw = np.load(_SAMPLING_WEIGHTS_PATH, allow_pickle=False)
            if raw.ndim != 1 or len(raw) != len(dataset):
                raise RuntimeError(
                    f"Sampling weights shape mismatch: weights={raw.shape} dataset_len={len(dataset)}"
                )
            if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
                raise RuntimeError("Sampling weights must be finite and strictly positive")
            summary_candidates = [
                _SAMPLING_WEIGHTS_PATH.with_name(f"{_SAMPLING_WEIGHTS_PATH.stem}_summary.json"),
                _SAMPLING_WEIGHTS_PATH.with_suffix(".summary.json"),
                _SAMPLING_WEIGHTS_PATH.with_name("grasp_preclose_100f_summary.json"),
            ]
            summary_path = next((path for path in summary_candidates if path.is_file()), None)
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path is not None
                else None
            )
            if summary is not None:
                if int(summary.get("total_frames", -1)) != len(dataset):
                    raise RuntimeError("Sampling summary frame count does not match the training dataset")
                for source in summary.get("source_parquet", []):
                    source_path = Path(source["path"])
                    expected_sha256 = str(source["sha256"])
                    if not source_path.is_file() or _file_sha256(source_path) != expected_sha256:
                        raise RuntimeError(f"Sampling weights are stale for source parquet: {source_path}")
            _SAMPLING_WEIGHTS_CACHE = torch.from_numpy(raw.astype(np.float64, copy=False))
            print(
                "[stereo-act] weighted training sampler: "
                f"path={_SAMPLING_WEIGHTS_PATH} frames={len(raw)} "
                f"min={float(raw.min()):.4f} max={float(raw.max()):.4f} "
                f"mean={float(raw.mean()):.4f} seed={_SAMPLING_SEED}"
            )
            if _TRAIN_OUTPUT_DIR is not None:
                _TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "sampler": "torch.utils.data.WeightedRandomSampler",
                    "replacement": True,
                    "num_samples": len(raw),
                    "seed": int(_SAMPLING_SEED),
                    "weights_path": str(_SAMPLING_WEIGHTS_PATH),
                    "weights_sha256": _file_sha256(_SAMPLING_WEIGHTS_PATH),
                    "weights_shape": list(raw.shape),
                    "weights_min": float(raw.min()),
                    "weights_max": float(raw.max()),
                    "weights_mean": float(raw.mean()),
                    "summary": summary,
                }
                (_TRAIN_OUTPUT_DIR / "sampling_sampler_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
        kwargs["shuffle"] = False
        if existing_sampler is not None:
            print(
                "[stereo-act] replacing EpisodeAwareSampler with audited frame weights; "
                "the weight vector already spans the full selected dataset"
            )
        if _SAMPLING_INDICES_PATH is not None:
            kwargs["sampler"] = torch.utils.data.SubsetRandomSampler(
                _SAMPLING_INDICES_CACHE,
                generator=torch.Generator().manual_seed(int(_SAMPLING_SEED)),
            )
        else:
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                _SAMPLING_WEIGHTS_CACHE,
                num_samples=len(_SAMPLING_WEIGHTS_CACHE),
                replacement=True,
                generator=torch.Generator().manual_seed(int(_SAMPLING_SEED)),
            )
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = _WeightedDataLoader


def _rewrite_alias_args(argv: list[str]) -> list[str]:
    global _SAMPLING_WEIGHTS_PATH, _SAMPLING_INDICES_PATH, _SAMPLING_SEED
    global _TRAIN_OUTPUT_DIR, _SELECTIVE_CHECKPOINT_STEPS
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {
            "--sampling-weights-npy",
            "--sampling-indices-npy",
            "--sampling-seed",
            "--checkpoint-steps",
        }:
            if i + 1 >= len(argv):
                raise SystemExit(f"Missing value for {token}")
            value = argv[i + 1]
            if token == "--sampling-weights-npy":
                _SAMPLING_WEIGHTS_PATH = Path(value).expanduser().resolve()
                if not _SAMPLING_WEIGHTS_PATH.is_file():
                    raise SystemExit(f"Sampling weights do not exist: {_SAMPLING_WEIGHTS_PATH}")
            elif token == "--sampling-indices-npy":
                _SAMPLING_INDICES_PATH = Path(value).expanduser().resolve()
                if not _SAMPLING_INDICES_PATH.is_file():
                    raise SystemExit(f"Sampling indices do not exist: {_SAMPLING_INDICES_PATH}")
            elif token == "--checkpoint-steps":
                try:
                    parsed = {int(item.strip()) for item in value.split(",") if item.strip()}
                except ValueError as error:
                    raise SystemExit(f"Invalid --checkpoint-steps={value!r}") from error
                if not parsed or any(step <= 0 for step in parsed):
                    raise SystemExit(f"Invalid --checkpoint-steps={value!r}")
                _SELECTIVE_CHECKPOINT_STEPS = parsed
            else:
                _SAMPLING_SEED = int(value)
            i += 2
            continue
        if token.startswith("--sampling-weights-npy="):
            _SAMPLING_WEIGHTS_PATH = Path(token.split("=", 1)[1]).expanduser().resolve()
            if not _SAMPLING_WEIGHTS_PATH.is_file():
                raise SystemExit(f"Sampling weights do not exist: {_SAMPLING_WEIGHTS_PATH}")
            i += 1
            continue
        if token.startswith("--sampling-indices-npy="):
            _SAMPLING_INDICES_PATH = Path(token.split("=", 1)[1]).expanduser().resolve()
            if not _SAMPLING_INDICES_PATH.is_file():
                raise SystemExit(f"Sampling indices do not exist: {_SAMPLING_INDICES_PATH}")
            i += 1
            continue
        if token.startswith("--sampling-seed="):
            _SAMPLING_SEED = int(token.split("=", 1)[1])
            i += 1
            continue
        if token.startswith("--checkpoint-steps="):
            value = token.split("=", 1)[1]
            try:
                parsed = {int(item.strip()) for item in value.split(",") if item.strip()}
            except ValueError as error:
                raise SystemExit(f"Invalid --checkpoint-steps={value!r}") from error
            if not parsed or any(step <= 0 for step in parsed):
                raise SystemExit(f"Invalid --checkpoint-steps={value!r}")
            _SELECTIVE_CHECKPOINT_STEPS = parsed
            i += 1
            continue
        if token in {"--output_dir", "--output-dir"} and i + 1 < len(argv):
            _TRAIN_OUTPUT_DIR = Path(argv[i + 1]).expanduser().resolve()
        elif token.startswith("--output_dir=") or token.startswith("--output-dir="):
            _TRAIN_OUTPUT_DIR = Path(token.split("=", 1)[1]).expanduser().resolve()
        rewritten = False
        for alias, target in ALIASES.items():
            if token == alias:
                if i + 1 >= len(argv):
                    raise SystemExit(f"Missing value for {alias}")
                out.extend([target, argv[i + 1]])
                i += 2
                rewritten = True
                break
            if token.startswith(alias + "="):
                out.append(target + "=" + token.split("=", 1)[1])
                i += 1
                rewritten = True
                break
        if not rewritten:
            out.append(token)
            i += 1
    if _SAMPLING_WEIGHTS_PATH is not None and _SAMPLING_INDICES_PATH is not None:
        raise SystemExit("sampling weights and deterministic sampling indices are mutually exclusive")
    return out


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_rewrite_alias_args(sys.argv[1:])]
    lerobot_train_main()
