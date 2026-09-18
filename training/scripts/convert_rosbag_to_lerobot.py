#!/usr/bin/env python3
# ┌──────────────────────────────────────────────────────────┐
# │          convert_rosbag_to_lerobot.py  数据流            │
# └──────────────────────────────────────────────────────────┘
#
#   session_dir/episode_*/episode.bag
#          │
#          ▼
#   ┌──────────────────────────────────────────────────────┐
#   │  ROS Topics (4关节流 + 3相机流)                      │
#   │  ├─ /robot/arm_left    JointState  pos(7)+effort(7) │
#   │  ├─ /robot/arm_right   JointState  pos(7)+effort(7) │
#   │  ├─ /teleop/arm_left 或 /robot/arm_left/joint_cmd_mux   JointState  pos(7) │
#   │  ├─ /teleop/arm_right 或 /robot/arm_right/joint_cmd_mux JointState  pos(7) │
#   │  ├─ camera_top         CompressedImage               │
#   │  ├─ camera_left_wrist  CompressedImage               │
#   │  └─ camera_right_wrist CompressedImage               │
#   └──────────────────────┬───────────────────────────────┘
#                          │  build_joint_and_image_streams()
#                          ▼
#         joint_ts/pos/effort/vel  +  image_ts/frames(BGR, resized)
#                          │
#          计算公共时间窗口 (全部 7 流参与门控)
#          t_start = max(各流首帧时间戳)
#          t_end   = min(各流末帧时间戳)
#                          │
#                          ▼
#          按 FPS 均匀采样  →  sample_ts[N_raw]
#                          │
#          nearest_indices()  对全部 7 流最近邻匹配
#                          │
#                          ▼
#          同步延迟过滤: lag > max_sync_lag_sec → 丢弃
#          kept_count < min_frames → 跳过此 episode
#                          │
#                          ▼
#   observation.state = [left_pos(7) | left_effort(7) | left_vel(7)
#                        right_pos(7)| right_effort(7)| right_vel(7)]  = 42维
#   action            = [left_action(7) | right_action(7)] = 14维
#                          │
#                          ▼
#   ┌──────────────────────────────────────────────────────┐
#   │  写出 episode 文件                                   │
#   │  data/.../episode_XXXXXX.parquet                     │
#   │  videos/.../observation.image/...mp4                 │
#   │  videos/.../observation.left_wrist_image/...mp4      │
#   │  videos/.../observation.right_wrist_image/...mp4     │
#   └──────────────────────┬───────────────────────────────┘
#                          │  所有 episode 完成
#                          ▼
#          rewrite_global_indices()  修正全局帧索引
#                          │
#                          ▼
#   meta/info.json  tasks.jsonl  episodes.jsonl
#   episodes_stats.jsonl  README.md
#                          │  可选 --convert-to-v30
#                          ▼
#   <output-root>/<dataset-id>/  LeRobot v2.1 或 v3.0

"""Convert ROS1 dual-arm episode bags into a LeRobot v2.1 style local dataset.

This converter exports one bimanual episode per input bag episode:
  - observation.state = [L_pos, L_effort_raw, L_vel, R_pos, R_effort_raw, R_vel]
  - action            = [L_action_pos, R_action_pos]
  - observation.image (top camera)
  - observation.left_wrist_image
  - observation.right_wrist_image
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from conversion_feature_selection import (
    ACTFS_DEPTH_OBS_KEY,
    ACTFS_OBS_KEY_ORDER,
    OBS_KEY_ORDER,
    OBS_KEY_ALIASES,
    build_feature_catalog,
    resolve_feature_selection,
)


@dataclass(frozen=True)
class JointTopicSpec:
    pos_dim: int
    effort_dim: int  # 0 for topics where effort is not required.
    role: str  # "robot" | "action"
    vel_dim: int = 0  # 0 means velocity is not required for this topic.


@dataclass
class ConvertedEpisode:
    episode_index: int
    length: int
    parquet_path: Path
    video_paths: Dict[str, Path]
    stats: Dict[str, Dict]


class EpisodeSkipError(RuntimeError):
    """Expected per-episode skip condition."""


def release_episode_memory() -> None:
    """Release large decoded-frame arrays before the next rosbag episode."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    default_output_root = Path(os.environ.get("LEROBOT_ROOT", "./lerobot_data"))
    parser = argparse.ArgumentParser(
        description="Convert recorded ROS bags to local LeRobot v2.1 dataset files."
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Path to a rosbag session directory (contains episode_*/episode.bag).",
    )
    parser.add_argument(
        "--output-root",
        default=str(default_output_root),
        help="Root directory where dataset_id folder is created.",
    )
    parser.add_argument(
        "--dataset-id",
        default="act_dataset",
        help="Dataset folder name under output-root.",
    )
    parser.add_argument(
        "--task",
        default="teleop bimanual pick and place",
        help="Task text written to meta/tasks.jsonl.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target dataset FPS after temporal resampling.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=32,
        help="Skip converted episodes shorter than this many valid synchronized frames.",
    )
    parser.add_argument(
        "--max-sync-lag-sec",
        type=float,
        default=0.08,
        help="Drop sampled frames if any stream nearest timestamp exceeds this lag.",
    )
    parser.add_argument(
        "--pos-dim-per-arm",
        type=int,
        default=7,
        help="Robot position dims per arm to keep in observation.state.",
    )
    parser.add_argument(
        "--effort-dim-per-arm",
        type=int,
        default=7,
        help="Robot raw effort dims per arm to keep in observation.state.",
    )
    parser.add_argument(
        "--action-dim-per-arm",
        type=int,
        default=7,
        help="Action position dims per arm to keep in action.",
    )
    parser.add_argument(
        "--action-source",
        choices=["teleop", "mux"],
        default="mux",
        help="Select action supervision topics from teleop streams or robot joint_cmd_mux streams.",
    )
    parser.add_argument(
        "--state-history-size",
        type=int,
        default=1,
        help=(
            "How many state frames to concatenate in observation.state "
            "(1=current only, 2=previous+current)."
        ),
    )
    parser.add_argument(
        "--obs-indices",
        default="all",
        help=(
            "Observation image indices to keep (1=top, 2=left wrist, 3=right wrist). "
            "Accepts comma-separated 1-based indices/ranges, `all`, or `none`."
        ),
    )
    parser.add_argument(
        "--state-indices",
        default="all",
        help=(
            "Base observation.state indices to keep before history concat. "
            "Accepts comma-separated 1-based indices/ranges or `all`."
        ),
    )
    parser.add_argument(
        "--action-indices",
        default="all",
        help=(
            "Action indices to keep. Accepts comma-separated 1-based indices/ranges or `all`."
        ),
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=256,
        help="Resize output videos to this width.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=256,
        help="Resize output videos to this height.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="OpenCV fourcc codec used for output mp4 files (default: mp4v).",
    )
    parser.add_argument(
        "--image-use-bag-time",
        action="store_true",
        help=(
            "Use bag receive time (instead of message header stamp) for image topics. "
            "Useful when camera header timestamps are invalid or not synchronized."
        ),
    )
    parser.add_argument(
        "--top-camera-topic",
        default="/top_stereo/color/image_raw/compressed",
        help="Overhead camera topic (CompressedImage).",
    )
    parser.add_argument(
        "--left-wrist-topic",
        default="/fisheye_left/usb_cam_node/image_raw/compressed",
        help="Left wrist camera topic (CompressedImage).",
    )
    parser.add_argument(
        "--right-wrist-topic",
        default="/fisheye_right/usb_cam_node/image_raw/compressed",
        help="Right wrist camera topic (CompressedImage).",
    )
    parser.add_argument(
        "--actfs",
        action="store_true",
        help=(
            "Enable ACT-FS conversion: observation.image becomes LFV rectified/cropped "
            "top-stereo left view, and observation.fastfs_depth_image is generated by "
            "Fast-FoundationStereo with the same rectified/crop contract."
        ),
    )
    parser.add_argument(
        "--lfv-top-crop",
        action="store_true",
        help="Store the LFV rectified/cropped top left view without computing or storing depth.",
    )
    parser.add_argument(
        "--actfs-calibration-json",
        default="",
        help="Stereo calibration JSON used for ACT-FS LFV rectification.",
    )
    parser.add_argument(
        "--actfs-processing-metadata",
        default="",
        help="Processing metadata defining rectified output size, P1/P2, crop and rotation.",
    )
    parser.add_argument(
        "--actfs-model-dir",
        default="",
        help="Fast-FoundationStereo checkpoint path for ACT-FS depth.",
    )
    parser.add_argument(
        "--actfs-fast-foundationstereo-root",
        default="",
        help="Fast-FoundationStereo repository root.",
    )
    parser.add_argument(
        "--actfs-valid-iters",
        type=int,
        default=12,
        help="FastFS valid_iters used for ACT-FS depth conversion.",
    )
    parser.add_argument(
        "--actfs-scale",
        type=float,
        default=1.0,
        help="FastFS input scale used for ACT-FS depth conversion.",
    )
    parser.add_argument(
        "--actfs-max-disp",
        type=int,
        default=192,
        help="FastFS max_disp used for ACT-FS depth conversion.",
    )
    parser.add_argument(
        "--actfs-max-depth-m",
        type=float,
        default=2.5,
        help="Max depth in meters used for ACT-FS colorized depth image.",
    )
    parser.add_argument(
        "--actfs-depth-output-format",
        choices=("preview", "defm_metric"),
        default="preview",
        help=(
            "How to store observation.fastfs_depth_image. "
            "'preview' keeps the legacy 3-channel colorized mp4. "
            "'defm_metric' stores raw FastFS metric depth as float32 [H,W] in the parquet, "
            "which is the format expected before DeFM preprocessing."
        ),
    )
    parser.add_argument(
        "--actfs-warmup-frames",
        type=int,
        default=10,
        help="CUDA warmup frames before ACT-FS depth conversion stats/output.",
    )
    parser.add_argument(
        "--robot-left-topic",
        default="/robot/arm_left/joint_states_single",
        help="Left robot state topic (JointState).",
    )
    parser.add_argument(
        "--robot-right-topic",
        default="/robot/arm_right/joint_states_single",
        help="Right robot state topic (JointState).",
    )
    parser.add_argument(
        "--teleop-left-topic",
        default="/teleop/arm_left/joint_states_single",
        help="Left teleop command topic (JointState).",
    )
    parser.add_argument(
        "--teleop-right-topic",
        default="/teleop/arm_right/joint_states_single",
        help="Right teleop command topic (JointState).",
    )
    parser.add_argument(
        "--mux-left-topic",
        default="/robot/arm_left/joint_cmd_mux",
        help="Left mux action topic (JointState).",
    )
    parser.add_argument(
        "--mux-right-topic",
        default="/robot/arm_right/joint_cmd_mux",
        help="Right mux action topic (JointState).",
    )
    parser.add_argument(
        "--convert-to-v30",
        action="store_true",
        help=(
            "After writing v2.1 files, run LeRobot's in-place conversion to v3.0 "
            "(requires lerobot installed in this Python environment)."
        ),
    )
    parser.add_argument(
        "--v30-push-to-hub",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="When --convert-to-v30 is set, push converted dataset to Hub (default: false).",
    )
    parser.add_argument(
        "--v30-force-conversion",
        action="store_true",
        help="Forward --force-conversion to LeRobot converter (mainly useful for hub-based conversion).",
    )
    return parser.parse_args()


def _normalize_action_source(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "mux", "joint_cmd_mux"}:
        return "mux"
    if raw in {"teleop", "master"}:
        return "teleop"
    raise RuntimeError(f"--action-source must be teleop or mux, got {value}")


def _resolve_action_topics(args: argparse.Namespace) -> tuple[str, str, str]:
    action_source = _normalize_action_source(getattr(args, "action_source", "mux"))
    if action_source == "mux":
        left_topic = str(args.mux_left_topic)
        right_topic = str(args.mux_right_topic)
    else:
        left_topic = str(args.teleop_left_topic)
        right_topic = str(args.teleop_right_topic)
    return action_source, left_topic, right_topic


class _CompatBagTime:
    def __init__(self, sec: float) -> None:
        self._sec = float(sec)

    def to_sec(self) -> float:
        return self._sec


class _RosbagsCompatBag:
    """Small rosbag.Bag-compatible adapter backed by the pure-Python rosbags package."""

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        if mode != "r":
            raise RuntimeError("rosbags compatibility adapter only supports read mode")
        from rosbags.highlevel import AnyReader  # type: ignore

        self._path = Path(path).expanduser().resolve()
        self._reader = AnyReader([self._path])
        self._reader.open()

    def read_messages(self, topics: Sequence[str] | None = None):
        topic_set = set(topics or [])
        if topic_set:
            connections = [conn for conn in self._reader.connections if conn.topic in topic_set]
        else:
            connections = list(self._reader.connections)
        for conn, timestamp_ns, rawdata in self._reader.messages(connections=connections):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            yield conn.topic, msg, _CompatBagTime(float(timestamp_ns) * 1e-9)

    def close(self) -> None:
        self._reader.close()


class _RosbagsCompatModule:
    Bag = _RosbagsCompatBag


def import_runtime_deps():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing runtime deps. Install cv2/OpenCV in the conversion Python environment."
        ) from exc
    try:
        import rosbag  # type: ignore
    except Exception:
        try:
            import rosbags.highlevel  # noqa: F401
            rosbag = _RosbagsCompatModule()
            print("[convert] info using rosbags fallback reader for ROS1 bag files")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing runtime deps. Install either ROS1 python rosbag or pure-Python rosbags."
            ) from exc
    # pandas parquet writer needs one engine: pyarrow or fastparquet.
    try:
        import pyarrow  # noqa: F401
    except Exception:
        try:
            import fastparquet  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing parquet engine. Install 'pyarrow' (recommended) or 'fastparquet'."
            ) from exc
    return cv2, rosbag


def sanitize_video_key(key: str) -> str:
    # Keep dots for LeRobot-style keys, only replace path separators and spaces.
    return key.replace("/", "_").replace(" ", "_")


def msg_time_sec(msg, bag_time) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        if hasattr(stamp, "to_sec"):
            secs = float(stamp.to_sec())
        else:
            sec = float(getattr(stamp, "sec", getattr(stamp, "secs", 0.0)) or 0.0)
            nsec = float(
                getattr(stamp, "nanosec", getattr(stamp, "nsec", getattr(stamp, "nsecs", 0.0))) or 0.0
            )
            secs = sec + nsec * 1e-9
        if secs > 0.0:
            return secs
    return float(bag_time.to_sec())


def decode_compressed_image(msg, cv2, width: int, height: int):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if width > 0 and height > 0:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


class ActFSProcessor:
    """Generate ACT-FS top RGB and aligned FastFS depth images from raw side-by-side topcam frames."""

    def __init__(
        self,
        *,
        cv2_module: Any,
        calibration_json: Path,
        processing_metadata: Path,
        model_dir: Path,
        fast_foundationstereo_root: Path,
        valid_iters: int,
        scale: float,
        max_disp: int,
        max_depth_m: float,
        depth_output_format: str,
        warmup_frames: int,
        enable_depth: bool,
    ) -> None:
        self.cv2 = cv2_module
        self.calibration_json = calibration_json.expanduser().resolve()
        self.processing_metadata = processing_metadata.expanduser().resolve()
        self.model_dir = model_dir.expanduser().resolve()
        self.fast_foundationstereo_root = fast_foundationstereo_root.expanduser().resolve()
        self.valid_iters = int(valid_iters)
        self.scale = float(scale)
        self.max_disp = int(max_disp)
        self.max_depth_m = float(max_depth_m)
        self.depth_output_format = str(depth_output_format)
        self.warmup_frames = max(0, int(warmup_frames))
        self.enable_depth = bool(enable_depth)
        if self.depth_output_format not in {"preview", "defm_metric"}:
            raise RuntimeError(f"Unsupported ACT-FS depth output format: {self.depth_output_format}")
        for label, path in [
            ("ACT-FS calibration json", self.calibration_json),
            ("ACT-FS processing metadata", self.processing_metadata),
        ]:
            if not path.exists():
                raise RuntimeError(f"{label} not found: {path}")
        if self.enable_depth:
            for label, path in [
                ("ACT-FS model", self.model_dir),
                ("Fast-FoundationStereo root", self.fast_foundationstereo_root),
            ]:
                if not path.exists():
                    raise RuntimeError(f"{label} not found: {path}")

        try:
            from fastfs_utils import (
                build_image_undistort_like_maps,
                colorize_depth,
                import_ffs,
                infer_depth,
                rectify_crop,
                set_default_normalize,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import the bundled FastFS integration helpers."
            ) from exc
        self.rectify_crop = rectify_crop

        calib = json.loads(self.calibration_json.read_text(encoding="utf-8"))
        processing = json.loads(self.processing_metadata.read_text(encoding="utf-8"))
        self.camera, self.left_maps, self.right_maps = build_image_undistort_like_maps(calib, processing)
        self.colorize_depth = None
        self.infer_depth = None
        self.InputPadder = None
        self.AMP_DTYPE = None
        self.model = None
        if self.enable_depth:
            try:
                import torch  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "ACT-FS depth conversion requires torch and the external Fast-FoundationStereo repository."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError("ACT-FS depth conversion requires CUDA for Fast-FoundationStereo.")
            self.colorize_depth = colorize_depth
            self.infer_depth = infer_depth
            self.InputPadder, self.AMP_DTYPE, _vis_disparity = import_ffs(self.fast_foundationstereo_root)
            self.model = torch.load(str(self.model_dir), map_location="cpu", weights_only=False)
            set_default_normalize(self.model)
            self.model.args.valid_iters = int(self.valid_iters)
            self.model.args.max_disp = int(self.max_disp)
            self.model.cuda().eval()
        self._warmed = False
        crop = self.camera["crop"]
        if self.enable_depth:
            print(
                "[convert][actfs] enabled: raw top -> LFV rectified/crop %dx%d + FastFS depth "
                "valid_iters=%d scale=%.3f max_disp=%d max_depth_m=%.2f"
                % (
                    int(crop["width"]),
                    int(crop["height"]),
                    self.valid_iters,
                    self.scale,
                    self.max_disp,
                    self.max_depth_m,
                )
            )
        else:
            print(
                "[convert][actfs] enabled: raw top -> LFV rectified/crop %dx%d only; "
                "FastFS depth is not computed because depth obs is not selected."
                % (int(crop["width"]), int(crop["height"]))
            )

    def process_top(self, raw_top_bgr: np.ndarray) -> np.ndarray:
        left, _right = self.rectify_crop(raw_top_bgr, self.left_maps, self.right_maps, self.camera)
        return left

    def process(self, raw_top_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_depth or self.model is None or self.infer_depth is None:
            raise RuntimeError("ACT-FS depth processing was not enabled for this conversion.")
        left, right = self.rectify_crop(raw_top_bgr, self.left_maps, self.right_maps, self.camera)
        if not self._warmed:
            print(f"[convert][actfs] warming FastFS for {self.warmup_frames} frame(s)...")
            for _ in range(self.warmup_frames):
                self.infer_depth(
                    self.model,
                    self.InputPadder,
                    self.AMP_DTYPE,
                    left,
                    right,
                    self.camera,
                    self.scale,
                    self.valid_iters,
                    self.max_depth_m,
                )
            self._warmed = True
            print("[convert][actfs] warmup done.")
        depth, left_out, _right_out, _timing = self.infer_depth(
            self.model,
            self.InputPadder,
            self.AMP_DTYPE,
            left,
            right,
            self.camera,
            self.scale,
            self.valid_iters,
            self.max_depth_m,
        )
        if self.depth_output_format == "defm_metric":
            return left_out, depth.astype(np.float32, copy=False)
        depth_bgr = self.colorize_depth(depth, self.max_depth_m)  # type: ignore[misc]
        return left_out, depth_bgr


def nearest_indices(source_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    if source_ts.size == 0:
        raise ValueError("nearest_indices: source_ts is empty")
    if source_ts.size == 1:
        return np.zeros(query_ts.shape[0], dtype=np.int64)
    idx = np.searchsorted(source_ts, query_ts, side="left")
    idx = np.clip(idx, 1, source_ts.size - 1)
    prev_idx = idx - 1
    prev_dist = np.abs(query_ts - source_ts[prev_idx])
    next_dist = np.abs(source_ts[idx] - query_ts)
    choose_prev = prev_dist <= next_dist
    idx[choose_prev] = prev_idx[choose_prev]
    return idx.astype(np.int64)


def vector_stats(x: np.ndarray) -> Dict[str, List[float]]:
    # x shape: [N, D]
    return {
        "min": x.min(axis=0).astype(np.float64).tolist(),
        "max": x.max(axis=0).astype(np.float64).tolist(),
        "mean": x.mean(axis=0).astype(np.float64).tolist(),
        "std": x.std(axis=0).astype(np.float64).tolist(),
        "count": [int(x.shape[0])],
    }


def scalar_stats(x: np.ndarray) -> Dict[str, List[float]]:
    # x shape: [N]
    return {
        "min": [float(np.min(x))],
        "max": [float(np.max(x))],
        "mean": [float(np.mean(x))],
        "std": [float(np.std(x))],
        "count": [int(x.shape[0])],
    }


def image_channel_stats(frames: Sequence[np.ndarray]) -> Dict[str, List]:
    if not frames:
        raise ValueError("image_channel_stats: empty frames")
    mins = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    maxs = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    for frame in frames:
        # BGR uint8 -> RGB float [0, 1]
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        flat = rgb.reshape(-1, 3)
        mins = np.minimum(mins, flat.min(axis=0))
        maxs = np.maximum(maxs, flat.max(axis=0))
        sums += flat.sum(axis=0)
        sums2 += (flat * flat).sum(axis=0)
    n_pixels = float(frames[0].shape[0] * frames[0].shape[1] * len(frames))
    means = sums / n_pixels
    var = np.maximum(sums2 / n_pixels - means * means, 0.0)
    stds = np.sqrt(var)
    # LeRobot v2.1 commonly stores image stats as [C,1,1]
    return {
        "min": [[[float(mins[0])]], [[float(mins[1])]], [[float(mins[2])]]],
        "max": [[[float(maxs[0])]], [[float(maxs[1])]], [[float(maxs[2])]]],
        "mean": [[[float(means[0])]], [[float(means[1])]], [[float(means[2])]]],
        "std": [[[float(stds[0])]], [[float(stds[1])]], [[float(stds[2])]]],
        "count": [int(len(frames))],
    }


def depth_map_scalar_stats(frames: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    """Global scalar stats for raw metric depth maps.

    The values are scalar over all valid depth pixels, but min/max/mean/std are
    returned as three identical [C,1,1] entries because LeRobot's v2.1->v3.0
    converter validates any feature key containing "image" as image-like stats.
    The raw depth data itself remains float32 [H,W] in parquet.
    """
    if not frames:
        raise ValueError("depth_map_scalar_stats: empty frames")
    count = 0
    min_v = float("inf")
    max_v = float("-inf")
    sum_v = 0.0
    sum2_v = 0.0
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        count += int(arr.size)
        min_v = min(min_v, float(arr.min()))
        max_v = max(max_v, float(arr.max()))
        sum_v += float(arr.sum(dtype=np.float64))
        sum2_v += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))
    if count == 0:
        min_v = max_v = mean_v = std_v = 0.0
    else:
        mean_v = sum_v / float(count)
        var_v = max(sum2_v / float(count) - mean_v * mean_v, 0.0)
        std_v = math.sqrt(var_v)
    def as_image_stat(value: float) -> List[List[List[float]]]:
        return [[[float(value)]], [[float(value)]], [[float(value)]]]

    return {
        "min": as_image_stat(min_v),
        "max": as_image_stat(max_v),
        "mean": as_image_stat(mean_v),
        "std": as_image_stat(std_v),
        "count": [int(len(frames))],
    }


def write_mp4(video_path: Path, frames: Sequence[np.ndarray], fps: float, codec: str, cv2) -> None:
    if not frames:
        raise ValueError(f"No frames for video {video_path}")
    h, w = frames[0].shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path} with codec={codec}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def parse_episode_num(name: str) -> int:
    m = re.match(r"episode_(\d+)$", name)
    if not m:
        return 10**9
    return int(m.group(1))


def collect_bag_episode_dirs(session_dir: Path) -> List[Path]:
    eps = [p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    eps.sort(key=lambda p: parse_episode_num(p.name))
    out = []
    for ep in eps:
        bag = ep / "episode.bag"
        if bag.is_file():
            out.append(ep)
    return out


def build_joint_and_image_streams(
    *,
    bag_path: Path,
    joint_specs: Mapping[str, JointTopicSpec],
    image_topics: Sequence[str],
    image_width: int,
    image_height: int,
    image_use_bag_time: bool,
    cv2,
    rosbag,
    raw_image_topics: Optional[set[str]] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[np.ndarray]],
    Dict[str, int],
]:
    joint_ts: Dict[str, List[float]] = {t: [] for t in joint_specs}
    joint_pos: Dict[str, List[np.ndarray]] = {t: [] for t in joint_specs}
    joint_effort: Dict[str, List[np.ndarray]] = {
        t: [] for t, spec in joint_specs.items() if spec.effort_dim > 0
    }
    joint_vel: Dict[str, List[np.ndarray]] = {}
    for topic, spec in joint_specs.items():
        required_vel_dim = int(spec.vel_dim) if int(spec.vel_dim) > 0 else (spec.pos_dim if spec.role == "robot" else 0)
        if required_vel_dim > 0:
            joint_vel[topic] = []
    image_ts: Dict[str, List[float]] = {t: [] for t in image_topics}
    image_frames: Dict[str, List[np.ndarray]] = {t: [] for t in image_topics}
    dropped_short: Dict[str, int] = {t: 0 for t in joint_specs}
    raw_image_topics = set(raw_image_topics or set())

    topics = list(joint_specs.keys()) + list(image_topics)
    bag = rosbag.Bag(str(bag_path), "r")
    try:
        for topic, msg, t in bag.read_messages(topics=topics):
            if topic in joint_specs:
                ts = msg_time_sec(msg, t)
                spec = joint_specs[topic]
                pos = list(getattr(msg, "position", []))
                if len(pos) < spec.pos_dim:
                    dropped_short[topic] += 1
                    continue
                joint_ts[topic].append(ts)
                joint_pos[topic].append(np.asarray(pos[: spec.pos_dim], dtype=np.float32))
                if spec.effort_dim > 0:
                    effort = list(getattr(msg, "effort", []))
                    if len(effort) < spec.effort_dim:
                        joint_ts[topic].pop()
                        joint_pos[topic].pop()
                        dropped_short[topic] += 1
                        continue
                    joint_effort[topic].append(np.asarray(effort[: spec.effort_dim], dtype=np.float32))
                required_vel_dim = int(spec.vel_dim) if int(spec.vel_dim) > 0 else (spec.pos_dim if spec.role == "robot" else 0)
                if required_vel_dim > 0:
                    velocity = list(getattr(msg, "velocity", []))
                    if len(velocity) < required_vel_dim:
                        joint_ts[topic].pop()
                        joint_pos[topic].pop()
                        if spec.effort_dim > 0:
                            joint_effort[topic].pop()
                        dropped_short[topic] += 1
                        continue
                    joint_vel[topic].append(np.asarray(velocity[: required_vel_dim], dtype=np.float32))
            elif topic in image_ts:
                ts = float(t.to_sec()) if image_use_bag_time else msg_time_sec(msg, t)
                if topic in raw_image_topics:
                    frame = decode_compressed_image(msg, cv2, 0, 0)
                else:
                    frame = decode_compressed_image(msg, cv2, image_width, image_height)
                if frame is None:
                    continue
                image_ts[topic].append(ts)
                image_frames[topic].append(frame)
    finally:
        bag.close()

    jt = {k: np.asarray(v, dtype=np.float64) for k, v in joint_ts.items()}
    jp = {k: np.asarray(v, dtype=np.float32) for k, v in joint_pos.items()}
    je = {k: np.asarray(v, dtype=np.float32) for k, v in joint_effort.items()}
    jv = {k: np.asarray(v, dtype=np.float32) for k, v in joint_vel.items()}
    it = {k: np.asarray(v, dtype=np.float64) for k, v in image_ts.items()}
    return jt, jp, je, jv, it, image_frames, dropped_short


def ensure_nonempty_streams(
    *,
    joint_ts: Mapping[str, np.ndarray],
    joint_pos: Mapping[str, np.ndarray],
    joint_effort: Mapping[str, np.ndarray],
    joint_vel: Mapping[str, np.ndarray],
    image_ts: Mapping[str, np.ndarray],
    image_frames: Mapping[str, List[np.ndarray]],
    joint_specs: Mapping[str, JointTopicSpec],
    image_topics: Sequence[str],
) -> None:
    for topic, spec in joint_specs.items():
        if topic not in joint_ts or joint_ts[topic].size == 0:
            raise EpisodeSkipError(f"Missing joint stream: {topic}")
        if topic not in joint_pos or joint_pos[topic].shape[0] == 0:
            raise EpisodeSkipError(f"Missing joint positions: {topic}")
        if spec.effort_dim > 0:
            if topic not in joint_effort or joint_effort[topic].shape[0] == 0:
                raise EpisodeSkipError(f"Missing joint effort stream: {topic}")
        required_vel_dim = int(spec.vel_dim) if int(spec.vel_dim) > 0 else (spec.pos_dim if spec.role == "robot" else 0)
        if required_vel_dim > 0:
            if topic not in joint_vel or joint_vel[topic].shape[0] == 0:
                raise EpisodeSkipError(f"Missing joint velocity stream: {topic}")
    for topic in image_topics:
        if topic not in image_ts or image_ts[topic].size == 0:
            raise EpisodeSkipError(f"Missing image timestamps: {topic}")
        if topic not in image_frames or not image_frames[topic]:
            raise EpisodeSkipError(f"Missing image frames: {topic}")


def _joint_label(i: int) -> str:
    if i < 6:
        return f"joint{i + 1}"
    if i == 6:
        return "gripper"
    return f"joint{i + 1}"


def _make_joint_names(prefix: str, dim: int, suffix: str) -> List[str]:
    return [f"{prefix}.{_joint_label(i)}.{suffix}" for i in range(dim)]


def _with_state_history(obs_state: np.ndarray, history_size: int) -> np.ndarray:
    """Concatenate previous states along feature dim, clamping at episode start."""
    if history_size <= 1:
        return obs_state.astype(np.float32, copy=False)
    if obs_state.ndim != 2:
        raise RuntimeError(f"obs_state must be 2D, got shape={obs_state.shape}")
    frame_count, state_dim = int(obs_state.shape[0]), int(obs_state.shape[1])
    if frame_count <= 0:
        return obs_state.astype(np.float32, copy=False)
    offsets = np.arange(-(history_size - 1), 1, dtype=np.int64)  # oldest ... current
    base_idx = np.arange(frame_count, dtype=np.int64)[:, None]
    gather_idx = np.clip(base_idx + offsets[None, :], 0, frame_count - 1)
    stacked = obs_state[gather_idx]  # [T, H, D]
    return stacked.reshape(frame_count, history_size * state_dim).astype(np.float32, copy=False)


def _expand_state_names_with_history(base_names: Sequence[str], history_size: int) -> List[str]:
    if history_size <= 1:
        return list(base_names)
    names: List[str] = []
    for offset in range(-(history_size - 1), 1):
        slot = "t0" if offset == 0 else f"t{offset}"
        for base in base_names:
            names.append(f"{slot}.{base}")
    return names


def convert_episode(
    *,
    bag_episode_dir: Path,
    episode_index: int,
    dataset_dir: Path,
    fps: float,
    min_frames: int,
    max_sync_lag_sec: float,
    pos_dim_per_arm: int,
    effort_dim_per_arm: int,
    action_dim_per_arm: int,
    state_history_size: int,
    selected_obs_keys: Sequence[str],
    selected_state_indices: Sequence[int],
    selected_action_indices: Sequence[int],
    topic_robot_left: str,
    topic_robot_right: str,
    topic_action_left: str,
    topic_action_right: str,
    topic_top: str,
    topic_left_wrist: str,
    topic_right_wrist: str,
    image_width: int,
    image_height: int,
    image_use_bag_time: bool,
    video_codec: str,
    actfs_processor: Optional[ActFSProcessor],
    cv2,
    rosbag,
) -> ConvertedEpisode:
    bag_path = bag_episode_dir / "episode.bag"
    obs_key_to_topic = {
        "observation.image": topic_top,
        "observation.left_wrist_image": topic_left_wrist,
        "observation.right_wrist_image": topic_right_wrist,
        ACTFS_DEPTH_OBS_KEY: topic_top,
    }
    selected_obs_keys = [str(key) for key in selected_obs_keys]
    image_topics = [obs_key_to_topic[key] for key in selected_obs_keys]
    image_topics = list(dict.fromkeys(image_topics))
    actfs_enabled_for_episode = actfs_processor is not None
    raw_image_topics = {topic_top} if actfs_enabled_for_episode and any(
        key in {"observation.image", ACTFS_DEPTH_OBS_KEY} for key in selected_obs_keys
    ) else set()
    selected_state_indices = [int(index) for index in selected_state_indices]
    selected_action_indices = [int(index) for index in selected_action_indices]
    joint_specs = {
        topic_robot_left: JointTopicSpec(
            pos_dim=pos_dim_per_arm,
            effort_dim=effort_dim_per_arm,
            role="robot",
            vel_dim=pos_dim_per_arm,
        ),
        topic_robot_right: JointTopicSpec(
            pos_dim=pos_dim_per_arm,
            effort_dim=effort_dim_per_arm,
            role="robot",
            vel_dim=pos_dim_per_arm,
        ),
        topic_action_left: JointTopicSpec(
            pos_dim=action_dim_per_arm,
            effort_dim=0,
            role="action",
        ),
        topic_action_right: JointTopicSpec(
            pos_dim=action_dim_per_arm,
            effort_dim=0,
            role="action",
        ),
    }
    joint_ts, joint_pos, joint_effort, joint_vel, image_ts, image_frames, dropped_short = build_joint_and_image_streams(
        bag_path=bag_path,
        joint_specs=joint_specs,
        image_topics=image_topics,
        image_width=image_width,
        image_height=image_height,
        image_use_bag_time=image_use_bag_time,
        cv2=cv2,
        rosbag=rosbag,
        raw_image_topics=raw_image_topics,
    )

    for topic, n in dropped_short.items():
        if n > 0:
            print(f"[convert] info {bag_episode_dir.name}: dropped_short_dims topic={topic} count={n}")

    ensure_nonempty_streams(
        joint_ts=joint_ts,
        joint_pos=joint_pos,
        joint_effort=joint_effort,
        joint_vel=joint_vel,
        image_ts=image_ts,
        image_frames=image_frames,
        joint_specs=joint_specs,
        image_topics=image_topics,
    )

    required_ts = [
        joint_ts[topic_robot_left],
        joint_ts[topic_robot_right],
        joint_ts[topic_action_left],
        joint_ts[topic_action_right],
        *[image_ts[topic] for topic in image_topics],
    ]
    t_start = max(float(ts[0]) for ts in required_ts)
    t_end = min(float(ts[-1]) for ts in required_ts)
    if t_end <= t_start:
        raise EpisodeSkipError("No overlapping time window across required streams.")

    raw_frame_count = int(math.floor((t_end - t_start) * fps)) + 1
    if raw_frame_count <= 0:
        raise EpisodeSkipError("No frames after overlap/fps computation.")
    sample_ts = t_start + (np.arange(raw_frame_count, dtype=np.float64) / fps)

    robot_left_idx = nearest_indices(joint_ts[topic_robot_left], sample_ts)
    robot_right_idx = nearest_indices(joint_ts[topic_robot_right], sample_ts)
    action_left_idx = nearest_indices(joint_ts[topic_action_left], sample_ts)
    action_right_idx = nearest_indices(joint_ts[topic_action_right], sample_ts)
    image_indices = {
        key: nearest_indices(image_ts[obs_key_to_topic[key]], sample_ts) for key in selected_obs_keys
    }

    lag_arrays = [
        np.abs(joint_ts[topic_robot_left][robot_left_idx] - sample_ts),
        np.abs(joint_ts[topic_robot_right][robot_right_idx] - sample_ts),
        np.abs(joint_ts[topic_action_left][action_left_idx] - sample_ts),
        np.abs(joint_ts[topic_action_right][action_right_idx] - sample_ts),
        *[
            np.abs(image_ts[obs_key_to_topic[key]][image_indices[key]] - sample_ts)
            for key in selected_obs_keys
        ],
    ]
    keep = np.ones(raw_frame_count, dtype=bool)
    for lag in lag_arrays:
        keep &= lag <= max_sync_lag_sec

    kept_count = int(np.sum(keep))
    dropped_lag = int(raw_frame_count - kept_count)
    if dropped_lag > 0:
        print(
            f"[convert] info {bag_episode_dir.name}: dropped_lag_frames={dropped_lag} "
            f"max_sync_lag_sec={max_sync_lag_sec}"
        )

    if kept_count < min_frames:
        raise EpisodeSkipError(
            f"Too few valid synchronized frames: kept={kept_count} min_frames={min_frames}"
        )

    sample_ts = sample_ts[keep]
    robot_left_idx = robot_left_idx[keep]
    robot_right_idx = robot_right_idx[keep]
    action_left_idx = action_left_idx[keep]
    action_right_idx = action_right_idx[keep]
    image_indices = {key: index_array[keep] for key, index_array in image_indices.items()}

    left_pos = joint_pos[topic_robot_left][robot_left_idx]
    left_effort = joint_effort[topic_robot_left][robot_left_idx]
    left_vel = joint_vel[topic_robot_left][robot_left_idx]
    right_pos = joint_pos[topic_robot_right][robot_right_idx]
    right_effort = joint_effort[topic_robot_right][robot_right_idx]
    right_vel = joint_vel[topic_robot_right][robot_right_idx]
    left_action_pos = joint_pos[topic_action_left][action_left_idx]
    right_action_pos = joint_pos[topic_action_right][action_right_idx]

    left_state = np.concatenate([left_pos, left_effort, left_vel], axis=1)
    right_state = np.concatenate([right_pos, right_effort, right_vel], axis=1)
    base_obs_state = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
    base_obs_state = base_obs_state[:, selected_state_indices]
    obs_state = _with_state_history(base_obs_state, history_size=state_history_size)
    action_target = np.concatenate([left_action_pos, right_action_pos], axis=1).astype(np.float32)
    action_target = action_target[:, selected_action_indices]

    image_frame_map: Dict[str, List[np.ndarray]] = {}
    actfs_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    actfs_top_cache: Dict[int, np.ndarray] = {}
    actfs_metric_depth_enabled = (
        actfs_enabled_for_episode
        and actfs_processor is not None
        and getattr(actfs_processor, "depth_output_format", "preview") == "defm_metric"
    )
    actfs_depth_selected = ACTFS_DEPTH_OBS_KEY in selected_obs_keys
    for key in selected_obs_keys:
        topic = obs_key_to_topic[key]
        if actfs_enabled_for_episode and key in {"observation.image", ACTFS_DEPTH_OBS_KEY}:
            frames: List[np.ndarray] = []
            for raw_index in image_indices[key]:
                raw_idx = int(raw_index)
                raw_frame = image_frames[topic][raw_idx]
                if actfs_depth_selected:
                    if raw_idx not in actfs_cache:
                        actfs_cache[raw_idx] = actfs_processor.process(raw_frame)  # type: ignore[union-attr]
                    top_rgb_bgr, depth_out = actfs_cache[raw_idx]
                    frame = top_rgb_bgr if key == "observation.image" else depth_out
                else:
                    if raw_idx not in actfs_top_cache:
                        actfs_top_cache[raw_idx] = actfs_processor.process_top(raw_frame)  # type: ignore[union-attr]
                    frame = actfs_top_cache[raw_idx]
                if image_width > 0 and image_height > 0:
                    # Metric depth must follow the same resize contract as live
                    # deployment.  Nearest-neighbor preserves measured depth
                    # discontinuities instead of averaging foreground and
                    # background into synthetic edge depths.  RGB continues to
                    # use area interpolation for downsampling.
                    resize_interpolation = (
                        cv2.INTER_NEAREST
                        if key == ACTFS_DEPTH_OBS_KEY and actfs_metric_depth_enabled
                        else cv2.INTER_AREA
                    )
                    frame = cv2.resize(frame, (image_width, image_height), interpolation=resize_interpolation)
                if key == ACTFS_DEPTH_OBS_KEY and actfs_metric_depth_enabled:
                    frame = np.nan_to_num(frame.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                frames.append(frame)
            image_frame_map[key] = frames
        else:
            image_frame_map[key] = [
                image_frames[topic][int(i)] for i in image_indices[key]
            ]

    frame_count = obs_state.shape[0]
    ep_chunk = episode_index // 1000
    parquet_path = (
        dataset_dir
        / "data"
        / f"chunk-{ep_chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    video_obs_keys = [
        key
        for key in selected_obs_keys
        if not (key == ACTFS_DEPTH_OBS_KEY and actfs_metric_depth_enabled)
    ]
    video_paths = {
        key: (
            dataset_dir
            / "videos"
            / f"chunk-{ep_chunk:03d}"
            / sanitize_video_key(key)
            / f"episode_{episode_index:06d}.mp4"
        )
        for key in video_obs_keys
    }
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = np.arange(frame_count, dtype=np.int64)
    # After we drop unsynchronized samples, the kept frames are re-encoded into a dense mp4
    # stream at exactly `fps`. The LeRobot video loader expects timestamps to match that dense
    # stream; preserving the sparse original sample times can push lookups past the end of a
    # concatenated video shard near episode boundaries.
    rel_ts = (frame_index.astype(np.float64) / float(fps)).astype(np.float32)
    frame_columns: Dict[str, Any] = {
        "observation.state": [row.tolist() for row in obs_state],
        "action": [row.tolist() for row in action_target],
        "timestamp": rel_ts.astype(np.float32),
        "frame_index": frame_index,
        "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
        "index": np.zeros(frame_count, dtype=np.int64),  # global index filled later
        "task_index": np.zeros(frame_count, dtype=np.int64),
    }
    if actfs_metric_depth_enabled and ACTFS_DEPTH_OBS_KEY in selected_obs_keys:
        frame_columns[ACTFS_DEPTH_OBS_KEY] = [
            np.asarray(frame, dtype=np.float32).tolist() for frame in image_frame_map[ACTFS_DEPTH_OBS_KEY]
        ]
    df = pd.DataFrame(frame_columns)
    df.to_parquet(parquet_path, index=False)

    for key in video_obs_keys:
        write_mp4(video_paths[key], image_frame_map[key], fps=fps, codec=video_codec, cv2=cv2)

    stats = {
        "action": vector_stats(action_target),
        "observation.state": vector_stats(obs_state),
        "timestamp": scalar_stats(rel_ts.astype(np.float64)),
        "frame_index": scalar_stats(frame_index.astype(np.float64)),
        "episode_index": scalar_stats(np.full(frame_count, episode_index, dtype=np.float64)),
        "index": scalar_stats(frame_index.astype(np.float64)),  # placeholder; corrected later
        "task_index": scalar_stats(np.zeros(frame_count, dtype=np.float64)),
    }
    for key in selected_obs_keys:
        if key == ACTFS_DEPTH_OBS_KEY and actfs_metric_depth_enabled:
            stats[key] = depth_map_scalar_stats(image_frame_map[key])
        else:
            stats[key] = image_channel_stats(image_frame_map[key])

    return ConvertedEpisode(
        episode_index=episode_index,
        length=frame_count,
        parquet_path=parquet_path,
        video_paths=video_paths,
        stats=stats,
    )


def rewrite_global_indices(converted: Sequence[ConvertedEpisode]) -> None:
    global_index = 0
    for ep in converted:
        df = pd.read_parquet(ep.parquet_path)
        n = len(df)
        idx = np.arange(global_index, global_index + n, dtype=np.int64)
        df["index"] = idx
        df.to_parquet(ep.parquet_path, index=False)
        ep.stats["index"] = scalar_stats(idx.astype(np.float64))
        global_index += n


def make_feature_desc(dtype: str, shape: List[int], names):
    return {"dtype": dtype, "shape": shape, "names": names}


def write_metadata_files(
    *,
    dataset_dir: Path,
    dataset_id: str,
    fps: float,
    action_source: str,
    state_history_size: int,
    selected_obs_keys: Sequence[str],
    selected_state_indices: Sequence[int],
    selected_state_names: Sequence[str],
    selected_action_indices: Sequence[int],
    selected_action_names: Sequence[str],
    obs_selection_spec: str,
    state_selection_spec: str,
    action_selection_spec: str,
    task_text: str,
    converted: Sequence[ConvertedEpisode],
    image_height: int,
    image_width: int,
    video_codec: str,
    actfs_metadata: Optional[dict[str, Any]] = None,
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = len(converted)
    total_frames = int(sum(ep.length for ep in converted))
    total_chunks = max(1, math.ceil(total_episodes / 1000))
    actfs_depth_output_format = str((actfs_metadata or {}).get("depth_output_format", "preview"))
    raw_metric_depth_enabled = bool(actfs_metadata) and actfs_depth_output_format == "defm_metric"
    video_obs_keys = [
        key
        for key in selected_obs_keys
        if not (key == ACTFS_DEPTH_OBS_KEY and raw_metric_depth_enabled)
    ]
    total_videos = total_episodes * len(video_obs_keys)
    base_state_names = list(selected_state_names)
    state_names = _expand_state_names_with_history(base_state_names, history_size=state_history_size)
    action_names = list(selected_action_names)
    state_total_dim = len(state_names)
    base_state_total_dim = len(base_state_names)
    action_total_dim = len(action_names)

    video_info = {
        "video.fps": float(fps),
        "video.height": int(image_height),
        "video.width": int(image_width),
        "video.channels": 3,
        "video.codec": str(video_codec),
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    video_feature = {
        "dtype": "video",
        "shape": [int(image_height), int(image_width), 3],
        "names": ["height", "width", "channels"],
        # Both keys appear in v2.1 datasets; keeping both improves compatibility.
        "video_info": video_info,
        "info": video_info,
    }
    metric_depth_feature = {
        "dtype": "float32",
        "shape": [int(image_height), int(image_width)],
        "names": ["height", "width"],
        "info": {
            "is_metric_depth": True,
            "unit": "meter",
            "producer": "Fast-FoundationStereo",
            "intended_consumer": "DeFM preprocess_depth_image",
            "note": "Raw metric depth before colorize_depth; clipped by actfs.max_depth_m.",
        },
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "omy",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        # Keep legacy aliases because LeRobot's upstream v2.1->v3.0 converter
        # still reads them directly during local upgrade.
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "storage_total_chunks": total_chunks,
        "storage_chunk_size_episodes": 1000,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "action_source": str(action_source),
        "state_history_kind": "dense_state_concat",
        "state_history_size": int(state_history_size),
        "state_base_dim": int(base_state_total_dim),
        "feature_selection": {
            "layout": "2arm4cam_actfs" if actfs_metadata else "2arm3cam",
            "obs_selection": str(obs_selection_spec),
            "obs_keys": list(selected_obs_keys),
            "obs_indices": [
                (ACTFS_OBS_KEY_ORDER if actfs_metadata else OBS_KEY_ORDER).index(key) + 1
                for key in selected_obs_keys
                if key in (ACTFS_OBS_KEY_ORDER if actfs_metadata else OBS_KEY_ORDER)
            ],
            "state_selection": str(state_selection_spec),
            "state_indices": [int(index) + 1 for index in selected_state_indices],
            "state_base_names": list(selected_state_names),
            "action_selection": str(action_selection_spec),
            "action_indices": [int(index) + 1 for index in selected_action_indices],
            "action_names": list(selected_action_names),
        },
        "features": {
            "action": make_feature_desc("float32", [action_total_dim], action_names),
            "observation.state": make_feature_desc("float32", [state_total_dim], state_names),
            "timestamp": make_feature_desc("float32", [1], None),
            "frame_index": make_feature_desc("int64", [1], None),
            "episode_index": make_feature_desc("int64", [1], None),
            "index": make_feature_desc("int64", [1], None),
            "task_index": make_feature_desc("int64", [1], None),
        },
    }
    if actfs_metadata:
        info["actfs"] = dict(actfs_metadata)
    for key in selected_obs_keys:
        if key == ACTFS_DEPTH_OBS_KEY and raw_metric_depth_enabled:
            info["features"][key] = metric_depth_feature
        else:
            info["features"][key] = video_feature
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in converted:
            row = {"episode_index": ep.episode_index, "tasks": [task_text], "length": ep.length}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for ep in converted:
            row = {"episode_index": ep.episode_index, "stats": ep.stats}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = (
        "# Local LeRobot v2.1 bimanual dataset\n\n"
        f"- dataset_id: `{dataset_id}`\n"
        f"- total_episodes: `{total_episodes}`\n"
        f"- total_frames: `{total_frames}`\n"
        f"- fps: `{fps}`\n"
        f"- obs keys: `{', '.join(selected_obs_keys) if selected_obs_keys else 'none'}`\n"
        f"- obs selection: `{obs_selection_spec}`\n"
        f"- ACT-FS depth output format: `{actfs_depth_output_format}`\n"
        f"- observation.state dim: `{state_total_dim}` "
        f"(base selection `{state_selection_spec}`, history_size=`{state_history_size}`)\n"
        f"- action dim: `{action_total_dim}` (selection `{action_selection_spec}`, source `{action_source}`)\n"
        "- state history boundary handling: clamp to earliest frame in the episode\n"
    )
    (dataset_dir / "README.md").write_text(readme, encoding="utf-8")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")


def _ensure_legacy_v21_info_aliases(dataset_dir: Path) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Dataset info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    updated = False

    total_chunks = info.get("total_chunks")
    if total_chunks is None and info.get("storage_total_chunks") is not None:
        info["total_chunks"] = int(info["storage_total_chunks"])
        updated = True

    chunks_size = info.get("chunks_size")
    if chunks_size is None and info.get("storage_chunk_size_episodes") is not None:
        info["chunks_size"] = int(info["storage_chunk_size_episodes"])
        updated = True

    if updated:
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[convert] patched legacy v2.1 info aliases: {info_path}")


def convert_local_dataset_v21_to_v30(
    *,
    dataset_id: str,
    output_root: Path,
    push_to_hub: bool,
    force_conversion: bool,
) -> None:
    try:
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    except Exception as exc:
        raise RuntimeError(
            "Failed to import LeRobot v2.1->v3.0 converter. Install lerobot in this environment."
        ) from exc

    print(f"[convert] upgrading dataset to v3.0: {dataset_id}")
    _ensure_legacy_v21_info_aliases(output_root / dataset_id)
    convert_dataset(
        repo_id=dataset_id,
        root=output_root,
        push_to_hub=push_to_hub,
        force_conversion=force_conversion,
    )


def main() -> int:
    args = parse_args()
    cv2, rosbag = import_runtime_deps()

    pos_dim_per_arm = int(args.pos_dim_per_arm)
    effort_dim_per_arm = int(args.effort_dim_per_arm)
    action_dim_per_arm = int(args.action_dim_per_arm)
    state_history_size = int(args.state_history_size)

    _validate_positive("pos-dim-per-arm", pos_dim_per_arm)
    _validate_positive("effort-dim-per-arm", effort_dim_per_arm)
    _validate_positive("action-dim-per-arm", action_dim_per_arm)
    _validate_positive("state-history-size", state_history_size)
    _validate_positive("min-frames", int(args.min_frames))
    if float(args.fps) <= 0:
        raise RuntimeError(f"--fps must be > 0, got {args.fps}")
    if float(args.max_sync_lag_sec) < 0:
        raise RuntimeError(f"--max-sync-lag-sec must be >= 0, got {args.max_sync_lag_sec}")

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        raise RuntimeError(f"session-dir not found: {session_dir}")
    action_source, action_left_topic, action_right_topic = _resolve_action_topics(args)
    actfs_enabled = bool(getattr(args, "actfs", False))
    lfv_top_crop_enabled = actfs_enabled or bool(getattr(args, "lfv_top_crop", False))
    feature_catalog = build_feature_catalog(
        "2arm3cam",
        pos_dim_per_arm=pos_dim_per_arm,
        effort_dim_per_arm=effort_dim_per_arm,
        action_dim_per_arm=action_dim_per_arm,
    )
    if actfs_enabled:
        feature_catalog["obs_keys"] = list(ACTFS_OBS_KEY_ORDER)
    try:
        _selected_obs_indices, selected_obs_keys, obs_selection_spec = resolve_feature_selection(
            args.obs_indices,
            feature_catalog["obs_keys"],
            allow_empty=True,
            alias_map=OBS_KEY_ALIASES,
        )
        selected_state_indices, selected_state_names, state_selection_spec = resolve_feature_selection(
            args.state_indices,
            feature_catalog["state_names"],
            allow_empty=False,
        )
        selected_action_indices, selected_action_names, action_selection_spec = resolve_feature_selection(
            args.action_indices,
            feature_catalog["action_names"],
            allow_empty=False,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if (not actfs_enabled) and ACTFS_DEPTH_OBS_KEY in selected_obs_keys:
        raise RuntimeError(f"{ACTFS_DEPTH_OBS_KEY} requires --actfs")
    if lfv_top_crop_enabled and "observation.image" in selected_obs_keys and ACTFS_DEPTH_OBS_KEY not in selected_obs_keys:
        print("[convert] LFV top crop enabled; FastFS depth is not selected.")

    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / args.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = collect_bag_episode_dirs(session_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episode_*/episode.bag found under {session_dir}")

    converted: List[ConvertedEpisode] = []
    skip_reasons: Dict[str, int] = {}
    next_ep_idx = 0
    if state_history_size > 1:
        print(
            f"[convert] info state history enabled: size={state_history_size} "
            "(concatenate previous states, clamped at episode start)"
        )
    actfs_processor: Optional[ActFSProcessor] = None
    actfs_metadata: Optional[dict[str, Any]] = None
    if lfv_top_crop_enabled and any(key in {"observation.image", ACTFS_DEPTH_OBS_KEY} for key in selected_obs_keys):
        actfs_processor = ActFSProcessor(
            cv2_module=cv2,
            calibration_json=Path(str(args.actfs_calibration_json)),
            processing_metadata=Path(str(args.actfs_processing_metadata)),
            model_dir=Path(str(args.actfs_model_dir)),
            fast_foundationstereo_root=Path(str(args.actfs_fast_foundationstereo_root)),
            valid_iters=int(args.actfs_valid_iters),
            scale=float(args.actfs_scale),
            max_disp=int(args.actfs_max_disp),
            max_depth_m=float(args.actfs_max_depth_m),
            depth_output_format=str(args.actfs_depth_output_format),
            warmup_frames=int(args.actfs_warmup_frames),
            enable_depth=actfs_enabled and ACTFS_DEPTH_OBS_KEY in selected_obs_keys,
        )
        crop = actfs_processor.camera["crop"]
        actfs_metadata = {
            "schema_version": 1,
            "top_source_topic": str(args.top_camera_topic),
            "top_output_key": "observation.image",
            "depth_output_key": ACTFS_DEPTH_OBS_KEY,
            "top_processing": "raw side-by-side decode -> LFV rectified/crop left view",
            "depth_processing": (
                "raw side-by-side decode -> LFV rectified/crop -> Fast-FoundationStereo -> raw metric depth"
                if str(args.actfs_depth_output_format) == "defm_metric"
                else "raw side-by-side decode -> LFV rectified/crop -> Fast-FoundationStereo -> fixed colorized RGB depth"
            ),
            "depth_output_format": str(args.actfs_depth_output_format),
            "rectified_crop_width": int(crop["width"]),
            "rectified_crop_height": int(crop["height"]),
            "calibration_json": str(Path(str(args.actfs_calibration_json)).expanduser().resolve()),
            "processing_metadata": str(Path(str(args.actfs_processing_metadata)).expanduser().resolve()),
            "model_dir": str(Path(str(args.actfs_model_dir)).expanduser().resolve()),
            "fast_foundationstereo_root": str(Path(str(args.actfs_fast_foundationstereo_root)).expanduser().resolve()),
            "valid_iters": int(args.actfs_valid_iters),
            "scale": float(args.actfs_scale),
            "max_disp": int(args.actfs_max_disp),
            "max_depth_m": float(args.actfs_max_depth_m),
            "stored_video_width": int(args.image_width),
            "stored_video_height": int(args.image_height),
            "stored_depth_width": int(args.image_width),
            "stored_depth_height": int(args.image_height),
            "depth_image_encoding": (
                "float32 metric depth in meters, shape [H,W], clipped by max_depth_m"
                if str(args.actfs_depth_output_format) == "defm_metric"
                else "BGR uint8 TURBO colormap, near=bright, clipped by max_depth_m"
            ),
        } if actfs_enabled else None
    print(
        f"[convert] info action source: {action_source} "
        f"(left={action_left_topic}, right={action_right_topic})"
    )
    print(
        f"[convert] info feature retention: obs={obs_selection_spec} "
        f"state={state_selection_spec} action={action_selection_spec}"
    )
    if not selected_obs_keys:
        print("[convert] info observation image selection is empty: state/action only dataset")
    else:
        obs_topic_map = {
            "observation.image": (
                str(args.top_camera_topic) + " -> LFV rectified/crop left"
                if lfv_top_crop_enabled
                else str(args.top_camera_topic)
            ),
            "observation.left_wrist_image": str(args.left_wrist_topic),
            "observation.right_wrist_image": str(args.right_wrist_topic),
            ACTFS_DEPTH_OBS_KEY: str(args.top_camera_topic) + " -> FastFS depth",
        }
        print("[convert] info observation topics:")
        for key in selected_obs_keys:
            print(f"[convert]   {key} <- {obs_topic_map.get(key, '')}")

    for bag_episode_dir in episode_dirs:
        print(f"[convert] {bag_episode_dir.name} -> episode_{next_ep_idx:06d}")
        try:
            out = convert_episode(
                bag_episode_dir=bag_episode_dir,
                episode_index=next_ep_idx,
                dataset_dir=dataset_dir,
                fps=float(args.fps),
                min_frames=int(args.min_frames),
                max_sync_lag_sec=float(args.max_sync_lag_sec),
                pos_dim_per_arm=pos_dim_per_arm,
                effort_dim_per_arm=effort_dim_per_arm,
                action_dim_per_arm=action_dim_per_arm,
                state_history_size=state_history_size,
                selected_obs_keys=selected_obs_keys,
                selected_state_indices=selected_state_indices,
                selected_action_indices=selected_action_indices,
                topic_robot_left=str(args.robot_left_topic),
                topic_robot_right=str(args.robot_right_topic),
                topic_action_left=action_left_topic,
                topic_action_right=action_right_topic,
                topic_top=str(args.top_camera_topic),
                topic_left_wrist=str(args.left_wrist_topic),
                topic_right_wrist=str(args.right_wrist_topic),
                image_width=int(args.image_width),
                image_height=int(args.image_height),
                image_use_bag_time=bool(args.image_use_bag_time),
                video_codec=str(args.video_codec),
                actfs_processor=actfs_processor,
                cv2=cv2,
                rosbag=rosbag,
            )
            converted.append(out)
            next_ep_idx += 1
        except EpisodeSkipError as exc:
            reason = str(exc)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(f"[convert] WARN skip {bag_episode_dir.name}: {reason}")
        except Exception as exc:
            reason = f"Unexpected error: {exc}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(f"[convert] WARN skip {bag_episode_dir.name}: {reason}")
        finally:
            release_episode_memory()

    if not converted:
        details = ", ".join([f"{k} x{v}" for k, v in skip_reasons.items()]) or "no details"
        raise RuntimeError(f"No converted episodes were produced. skip_reasons: {details}")

    rewrite_global_indices(converted)
    write_metadata_files(
        dataset_dir=dataset_dir,
        dataset_id=str(args.dataset_id),
        fps=float(args.fps),
        action_source=action_source,
        state_history_size=state_history_size,
        selected_obs_keys=selected_obs_keys,
        selected_state_indices=selected_state_indices,
        selected_state_names=selected_state_names,
        selected_action_indices=selected_action_indices,
        selected_action_names=selected_action_names,
        obs_selection_spec=obs_selection_spec,
        state_selection_spec=state_selection_spec,
        action_selection_spec=action_selection_spec,
        task_text=str(args.task),
        converted=converted,
        image_height=int(args.image_height),
        image_width=int(args.image_width),
        video_codec=str(args.video_codec),
        actfs_metadata=actfs_metadata,
    )

    if bool(args.convert_to_v30):
        convert_local_dataset_v21_to_v30(
            dataset_id=str(args.dataset_id),
            output_root=output_root,
            push_to_hub=bool(args.v30_push_to_hub),
            force_conversion=bool(args.v30_force_conversion),
        )

    output_version = "v3.0" if bool(args.convert_to_v30) else "v2.1"
    print(f"[convert] done: {len(converted)} episodes -> {dataset_dir} ({output_version})")
    if skip_reasons:
        print("[convert] skip summary:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"[convert]   {count} x {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
