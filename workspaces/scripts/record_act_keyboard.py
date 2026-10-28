#!/usr/bin/env python3
"""Keyboard-driven ACT rosbag recorder.

Controls:
  SPACE: start/stop current episode recording
  K:     delete the latest recorded episode in the current session
  Q:     stop recording if needed and quit
"""

# ============================================================
# 【初学者速读】record_act_keyboard.py 是什么？
# ============================================================
#
# 这是一个"按键控制的机器人数据录制程序"。
# 目的：用键盘控制 rosbag 录制，把机器人演示的过程保存成数据集，
#       供后续训练 AI 模型（ACT Policy）使用。
#
# 一句话：人工遥操作机器人时，按空格开始录，再按空格停止，
#         反复操作攒够 N 条 episode 数据，拿去训练 AI。
#
# ============================================================
# 数据流（最重要）
# ============================================================
#
#  ┌──────────────────────────────────────────────────────────┐
#  │                    整体数据流                             │
#  │                                                          │
#  │  ROS 话题（实时流）                                       │
#  │  ├─ /robot/arm_left/joint_states_single   ← 机器人左臂   │
#  │  ├─ /robot/arm_right/joint_states_single  ← 机器人右臂   │
#  │  ├─ /teleop/arm_left/joint_states_single  ← 示教臂左     │
#  │  ├─ /teleop/arm_right/joint_states_single ← 示教臂右     │
#  │  ├─ /realsense_*/color/image_raw/compressed ← 摄像头图像 │
#  │  └─ ... (由 YAML profile 文件定义)                       │
#  │        │                                                  │
#  │        ▼  按空格触发                                      │
#  │  rosbag record --lz4 -O episode/episode.bag ...          │
#  │  （把所有话题数据压缩写入 .bag 文件）                     │
#  │        │                                                  │
#  │        ▼  按空格停止后，后台异步处理                      │
#  │  _finalize_episode_job()                                  │
#  │  ├─ 等待 .bag 文件写完（.bag.active → .bag）              │
#  │  ├─ 统计每个话题的消息数、帧率                            │
#  │  ├─ 检测是否有摄像头掉帧（bag_hz < 阈值）                 │
#  │  └─ 写出 metadata.json（记录本次录制的完整信息）          │
#  │                                                          │
#  │  磁盘输出结构：                                           │
#  │  session_root/                                           │
#  │  └─ act_20260307_120000/          ← 本次 session         │
#  │     ├─ episode_001/               ← 第1条演示            │
#  │     │   ├─ episode.bag            ← 原始 ROS 录制包      │
#  │     │   └─ metadata.json          ← 录制元信息           │
#  │     ├─ episode_002/               ← 第2条演示            │
#  │     └─ ...                                               │
#  └──────────────────────────────────────────────────────────┘
#
# ============================================================
# 主要模块说明
# ============================================================
#
#  Recorder（主类）
#  ├─ __init__()         加载配置、初始化话题列表、启动看门狗线程
#  ├─ run()              主循环：监听键盘，派发 start/stop
#  │
#  ├─ 录制控制
#  │   ├─ _start_episode()     预检 → 启动 rosbag 进程 → 开始录制
#  │   └─ _stop_episode()      停止 rosbag → 提交异步保存任务
#  │
#  ├─ 预检（preflight）
#  │   └─ _wait_preflight()    开始录制前确认：
#  │                            ROS master 在线？
#  │                            必要话题存在且有发布者？
#  │                            磁盘剩余空间够？
#  │
#  ├─ 异步保存（episode 结束后后台执行）
#  │   ├─ _submit_save_job()        提交保存任务到线程池
#  │   └─ _finalize_episode_job()   实际执行：
#  │                                  等待 bag 写完
#  │                                  统计 bag 信息（话题/帧率/大小）
#  │                                  解析 hz monitor 日志
#  │                                  检测掉帧
#  │                                  写 metadata.json
#  │
#  ├─ 看门狗（后台线程，每秒检查）
#  │   └─ _watchdog_loop()     录制期间监控：
#  │                             ROS master 断连 → 自动停止
#  │                             磁盘满     → 自动停止
#  │                             rosbag 崩溃 → 自动停止
#  │
#  └─ Profile（YAML 配置文件）
#      定义"要录哪些话题"，由 --profile 参数指定
#      camera_streams: 相机列表（image_base + camera_info）
#      required_topics: 必须存在的话题（缺失则 preflight 失败）
#      optional_topics: 可选话题（存在就录，没有也没关系）
#      camera_transport: compressed（压缩图）或 raw（原始图）
#
# ============================================================
# 关键参数速查
# ============================================================
#
#  --profile          YAML 文件，定义录哪些话题（最重要）
#  --session-root     录制数据保存的根目录
#  --session-name     本次 session 名（默认自动生成时间戳）
#  --split            是否按大小拆分 bag 文件（大 session 时开）
#  --min-free-gb      磁盘剩余低于此值则拒绝开始/自动停止
#  --expected-camera-fps  预期相机帧率（用于掉帧检测）
#  --dropframe-warn-ratio 实际帧率低于 expected × ratio 时报警
#
# ============================================================
# 一次典型录制流程
# ============================================================
#
#  1. 启动脚本，等待 SPACE
#  2. 按 SPACE → preflight 检查 → rosbag record 启动 → 录制中
#  3. 人工遥操作机器人完成一次演示
#  4. 按 SPACE → rosbag 停止 → 后台异步保存 episode_001/
#  5. 重复 2-4，攒够所有 episode
#  6. 按 Q → 等待后台保存完成 → 退出
#
# ============================================================

from __future__ import annotations

import argparse
import atexit
import copy
import datetime as dt
import json
import math
import os
import pwd
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
import xmlrpc.client
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required (python3-yaml).") from exc

from ros_runtime_snapshot import (
    DEFAULT_PARAM_NAMESPACES as SNAPSHOT_DEFAULT_PARAM_NAMESPACES,
    DEFAULT_KEY_ROSPARAMS as SNAPSHOT_DEFAULT_KEY_ROSPARAMS,
    DEFAULT_KEY_SERVICES as SNAPSHOT_DEFAULT_KEY_SERVICES,
    DEFAULT_KEY_TOPICS as SNAPSHOT_DEFAULT_KEY_TOPICS,
    DEFAULT_SELECTED_TOPICS as SNAPSHOT_DEFAULT_SELECTED_TOPICS,
    capture_runtime_snapshot,
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def local_now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_topic(topic: str) -> str:
    return topic.strip("/").replace("/", "_").replace(":", "_") or "root"


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_ros_ns(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("Camera namespace must not be empty.")
    if not text.startswith("/"):
        text = "/" + text
    return text.rstrip("/")


def camera_stream_from_ns(camera_ns: str) -> Dict[str, str]:
    ns = normalize_ros_ns(camera_ns)
    return {
        "image_base": f"{ns}/color/image_raw",
        "camera_info": f"{ns}/color/camera_info",
    }


def dedup_camera_streams(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for entry in items:
        image_base = str(entry.get("image_base", "")).strip()
        camera_info = str(entry.get("camera_info", "")).strip()
        if not image_base or not camera_info:
            continue
        key = (image_base, camera_info)
        if key in seen:
            continue
        seen.add(key)
        out.append({"image_base": image_base, "camera_info": camera_info})
    return out


PER_ARM_DIM = 7
ARM_ORDER = ("left", "right", "opp")
ARM_TOPIC_SPEC = {
    "left": {
        "teleop": "/teleop/arm_left/joint_states_single",
        "cmd": "/robot/arm_left/joint_cmd_mux",
        "state": "/robot/arm_left/joint_states_single",
        "selected": "/robot/arm_left/joint_cmd_mux/selected",
    },
    "right": {
        "teleop": "/teleop/arm_right/joint_states_single",
        "cmd": "/robot/arm_right/joint_cmd_mux",
        "state": "/robot/arm_right/joint_states_single",
        "selected": "/robot/arm_right/joint_cmd_mux/selected",
    },
    "opp": {
        "teleop": "/teleop/arm_opp/joint_states_single",
        "cmd": "/robot/arm_opp/joint_cmd_mux",
        "state": "/robot/arm_opp/joint_states_single",
        "selected": "/robot/arm_opp/joint_cmd_mux/selected",
    },
}


def msg_time_sec(msg: object, bag_time: object) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        sec = float(stamp.to_sec())
        if sec > 0.0:
            return sec
    if hasattr(bag_time, "to_sec"):
        return float(bag_time.to_sec())
    return 0.0


def hz_estimate(ts: np.ndarray) -> float:
    if ts.size < 2:
        return 0.0
    duration = float(ts[-1] - ts[0])
    if duration <= 0.0:
        return 0.0
    return float((ts.size - 1) / duration)


def max_period_ms(ts: np.ndarray) -> float:
    if ts.size < 2:
        return 0.0
    return float(np.max(np.diff(ts)) * 1000.0)


def big_gap_metrics(ts: np.ndarray, *, min_gap_sec: float = 0.03, median_mult: float = 3.0) -> Tuple[int, float, float, float]:
    min_gap = max(0.0, float(min_gap_sec))
    mult = max(0.0, float(median_mult))
    if ts.size < 2:
        return 0, min_gap, 0.0, 0.0
    diffs = np.diff(ts)
    median = float(np.median(diffs))
    threshold = max(min_gap, mult * median)
    max_gap = float(np.max(diffs)) if diffs.size > 0 else 0.0
    count = int(np.sum(diffs > threshold))
    return count, threshold, max_gap, median


def nearest_indices(source_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    if source_ts.size == 0:
        return np.zeros(query_ts.size, dtype=np.int64)
    if source_ts.size == 1:
        return np.zeros(query_ts.size, dtype=np.int64)
    idx = np.clip(np.searchsorted(source_ts, query_ts), 1, source_ts.size - 1)
    prev = idx - 1
    choose_prev = np.abs(query_ts - source_ts[prev]) <= np.abs(source_ts[idx] - query_ts)
    idx[choose_prev] = prev[choose_prev]
    return idx.astype(np.int64)


def longest_repeat_ratio(values: np.ndarray, tol: float) -> float:
    if values.shape[0] <= 1:
        return 0.0
    longest = 1
    current = 1
    for idx in range(1, values.shape[0]):
        if float(np.max(np.abs(values[idx] - values[idx - 1]))) <= tol:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return float(longest) / float(values.shape[0])


def joint_motion_span(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.mean(values[:, :6].max(axis=0) - values[:, :6].min(axis=0)))


def lag_grid_seconds(search_ms: float, step_ms: float) -> np.ndarray:
    bound = max(0.0, float(search_ms)) / 1000.0
    step = max(1e-6, float(step_ms)) / 1000.0
    if bound <= 0.0:
        return np.asarray([0.0], dtype=np.float64)
    count = int(math.floor(bound / step)) + 1
    return np.linspace(0.0, bound, count, dtype=np.float64)


def estimate_follow_metrics(
    src_ts: np.ndarray,
    src_pos: np.ndarray,
    dst_ts: np.ndarray,
    dst_pos: np.ndarray,
    *,
    lag_values: np.ndarray,
    motion_span_warn: float,
) -> Dict[str, object]:
    if src_ts.size < 8 or dst_ts.size < 8:
        return {"status": "insufficient"}

    src_span = joint_motion_span(src_pos)
    if src_span < motion_span_warn:
        return {"status": "idle", "motion_span": src_span}

    best_score = None
    best_lag = 0.0
    best_abs = None
    best_idx = None
    for lag in lag_values:
        idx = nearest_indices(dst_ts, src_ts + lag)
        abs_err = np.abs(src_pos[:, :6] - dst_pos[idx, :6])
        score = float(np.mean(abs_err))
        if best_score is None or score < best_score:
            best_score = score
            best_lag = float(lag)
            best_abs = abs_err
            best_idx = idx

    assert best_abs is not None
    assert best_idx is not None
    matched_dt = dst_ts[best_idx] - src_ts
    return {
        "status": "ok",
        "motion_span": src_span,
        "lag_ms": best_lag * 1000.0,
        "mae": float(np.mean(best_abs)),
        "p95": float(np.percentile(best_abs, 95.0)),
        "saturated": bool(np.isclose(best_lag, lag_values[-1])),
        "matched_dt_std_ms": float(np.std(matched_dt) * 1000.0),
    }


class RawTerminal:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self) -> "RawTerminal":
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive TTY is required for keyboard control.")
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


class Recorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.profile = self._load_profile(args.profile)
        if args.camera_transport != "profile":
            self.profile["camera_transport"] = args.camera_transport
        self._apply_extra_camera_args(args)
        self._validate_profile(self.profile)

        self.camera_transport = self.profile["camera_transport"]
        self.required_topics = list(self.profile.get("required_topics", []))
        self.camera_streams = list(self.profile.get("camera_streams", []))
        self.optional_topics = list(self.profile.get("optional_topics", []))
        self.hz_monitor_topics = list(self.profile.get("hz_monitor_topics", []))
        self.extra_topics = dedup_keep_order([str(x).strip() for x in getattr(args, "extra_topic", []) if str(x).strip()])
        self.record_all_optional_topics = bool(getattr(args, "record_all_optional_topics", False))

        self.camera_record_topics = self._build_camera_record_topics()
        self.camera_health_topics = [self._camera_image_topic(entry["image_base"]) for entry in self.camera_streams]
        self.camera_info_topics = [
            entry["camera_info"]
            for entry in self.camera_streams
            if bool(entry.get("camera_info_required", True))
        ]
        self.hz_monitor_topics = self._normalize_hz_monitor_topics(self.hz_monitor_topics)
        self.required_presence_topics = self._build_required_presence_topics()
        # Only enforce joint publishers explicitly required by the selected
        # profile.  Teleoperation recordings require teleop topics, while an
        # open-loop demonstration replay intentionally stops the teleop nodes
        # and uses robot joint_cmd_mux as its action source.
        known_joint_topics = (
            "/robot/arm_left/joint_states_single",
            "/robot/arm_right/joint_states_single",
            "/teleop/arm_left/joint_states_single",
            "/teleop/arm_right/joint_states_single",
            "/robot/arm_opp/joint_states_single",
            "/robot/arm_opp/joint_states_opp",
            "/teleop/arm_opp/joint_states_single",
            "/teleop/arm_opp/joint_states_opp",
            "/robot/arm_left/joint_cmd_mux",
            "/robot/arm_right/joint_cmd_mux",
            "/robot/arm_opp/joint_cmd_mux",
        )
        self.required_joint_topics = [
            topic for topic in known_joint_topics if topic in self.required_topics
        ]
        self.profile_record_topics = dedup_keep_order(self.required_topics + self.camera_record_topics + self.extra_topics)

        self.session_root = Path(args.session_root).expanduser().resolve()
        self.output_owner = self._resolve_output_owner()
        self.output_owner_user = self._resolve_output_owner_user()
        self.output_dir_mode, self.output_file_mode = self._resolve_output_modes()
        self.output_runtime_home = self.session_root
        self.output_ros_home = self.session_root / ".ros"
        self.session_name = args.session_name or f"{args.prefix}_{local_now_tag()}"
        self.session_dir = self.session_root / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._apply_output_policy(self.session_dir)
        self.output_ros_home.mkdir(parents=True, exist_ok=True)
        self._apply_output_policy(self.output_ros_home)
        self.next_episode_index = self._scan_next_episode_index()

        self.recording = False
        self.shutdown_requested = False
        self.signal_requested = False
        self.pending_stop_reason: Optional[str] = None
        self.pending_lock = threading.Lock()

        self.rosbag_proc: Optional[subprocess.Popen] = None
        self.rosbag_log_handle = None
        self.hz_monitors: Dict[str, Tuple[subprocess.Popen, Path, object]] = {}
        self.debug_log_paths: List[Path] = []

        self.current_episode_name: Optional[str] = None
        self.current_episode_dir: Optional[Path] = None
        self.current_started_utc: Optional[str] = None
        self.current_started_monotonic: Optional[float] = None
        self.current_topics: List[str] = []
        self.current_optional_requested: List[str] = []
        self.current_optional_recorded: List[str] = []
        self.current_optional_missing: List[str] = []
        self.current_warnings: List[str] = []
        self.current_preflight: Dict[str, object] = {}
        self.last_stop_monotonic: Optional[float] = None
        self.last_stop_reason: Optional[str] = None

        self.debounce_sec = max(0.0, args.debounce_ms / 1000.0)
        self.last_space_ts = 0.0
        self.record_status_interval_sec = max(0.1, args.record_status_interval_sec)
        self.save_status_interval_sec = max(0.1, args.save_status_interval_sec)
        self.last_record_status_ts = 0.0
        self.last_save_status_ts = 0.0
        self.use_color = self._supports_color()
        self.save_progress_mode = str(getattr(args, "save_progress_mode", "compact")).strip().lower()
        self.expected_camera_fps = float(args.expected_camera_fps)
        if self.expected_camera_fps <= 0.0:
            raise RuntimeError("--expected-camera-fps must be > 0")
        self.dropframe_warn_ratio = float(args.dropframe_warn_ratio)
        if self.dropframe_warn_ratio <= 0.0:
            raise RuntimeError("--dropframe-warn-ratio must be > 0")
        self.warn_preview_lines = max(1, int(args.warn_preview_lines))
        self.health_audit_mode = str(args.health_audit_mode).strip().lower()
        self.health_action_source = str(args.health_action_source).strip().lower()
        self.health_expected_mux_source = str(args.health_expected_mux_source).strip().lower()
        self.health_require_teleop_state_sync = bool(args.health_require_teleop_state_sync)
        self.health_min_joint_hz = float(args.health_min_joint_hz)
        self.health_min_action_hz = float(args.health_min_action_hz)
        if self.health_min_action_hz <= 0.0:
            self.health_min_action_hz = self.health_min_joint_hz
        self.health_min_state_hz = float(args.health_min_state_hz)
        if self.health_min_state_hz <= 0.0:
            self.health_min_state_hz = self.health_min_joint_hz
        self.health_motion_span_warn = float(args.health_motion_span_warn)
        self.health_repeat_warn_ratio = float(args.health_repeat_warn_ratio)
        self.health_repeat_tol = float(args.health_repeat_tol)
        self.health_big_gap_min_sec = float(args.health_big_gap_min_sec)
        self.health_big_gap_median_mult = float(args.health_big_gap_median_mult)
        self.health_big_gap_max_count = max(0, int(args.health_big_gap_max_count))
        if self.health_big_gap_min_sec < 0.0:
            raise RuntimeError("--health-big-gap-min-sec must be >= 0")
        if self.health_big_gap_median_mult < 0.0:
            raise RuntimeError("--health-big-gap-median-mult must be >= 0")
        self.health_teleop_cmd_lag_warn_ms = float(args.health_teleop_cmd_lag_warn_ms)
        self.health_cmd_state_lag_warn_ms = float(args.health_cmd_state_lag_warn_ms)
        self.health_teleop_state_lag_warn_ms = float(args.health_teleop_state_lag_warn_ms)
        self.health_teleop_cmd_mae_warn = float(args.health_teleop_cmd_mae_warn)
        self.health_cmd_state_mae_warn = float(args.health_cmd_state_mae_warn)
        self.health_teleop_state_mae_warn = float(args.health_teleop_state_mae_warn)
        self.health_lag_values = lag_grid_seconds(args.health_lag_search_ms, args.health_lag_step_ms)

        self.key_rosparams = [
            "/piper_ctrl_left_node/can_port",
            "/piper_ctrl_right_node/can_port",
            "/piper_teleop_left_node/can_port",
            "/piper_teleop_right_node/can_port",
            "/piper_teleop_left_node/gripper_reverse",
            "/piper_teleop_right_node/gripper_reverse",
            "/piper_teleop_left_node/mit/enable_tor",
            "/piper_teleop_right_node/mit/enable_tor",
            "/piper_teleop_left_node/mit/gravity_mix_mode",
            "/piper_teleop_right_node/mit/gravity_mix_mode",
            "/piper_teleop_left_node/mit/torque_scale",
            "/piper_teleop_right_node/mit/torque_scale",
            "/piper_teleop_left_node/mit/torque_feedback_sign",
            "/piper_teleop_right_node/mit/torque_feedback_sign",
            "/piper_teleop_left_node/master_slave/enable",
            "/piper_teleop_right_node/master_slave/enable",
        ]

        self.save_workers = max(1, min(int(args.save_workers), 8))
        self.save_executor = ThreadPoolExecutor(
            max_workers=self.save_workers,
            thread_name_prefix="act_save",
        )
        self.save_executor_shutdown = False
        self.save_lock = threading.Lock()
        self.next_save_job_id = 1
        self.save_jobs: Dict[int, Dict[str, object]] = {}
        self.save_futures: Dict[int, Future] = {}

        self._ensure_tools()
        atexit.register(self._atexit_cleanup)

        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def _apply_extra_camera_args(self, args: argparse.Namespace) -> None:
        extra_camera_ns = [normalize_ros_ns(x) for x in getattr(args, "extra_camera_ns", [])]
        if not extra_camera_ns:
            return

        extra_streams = [camera_stream_from_ns(ns) for ns in extra_camera_ns]
        merged_streams = list(self.profile.get("camera_streams", []))
        merged_streams.extend(extra_streams)
        self.profile["camera_streams"] = dedup_camera_streams(merged_streams)

        hz_topics = list(self.profile.get("hz_monitor_topics", []))
        hz_topics.extend(entry["image_base"] for entry in extra_streams)
        self.profile["hz_monitor_topics"] = dedup_keep_order(hz_topics)

    def _load_profile(self, path: str) -> dict:
        profile_path = Path(path).expanduser().resolve()
        with profile_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["_profile_path"] = str(profile_path)
        return data

    def _validate_profile(self, profile: dict) -> None:
        required_keys = ["profile_name", "camera_transport", "required_topics", "camera_streams"]
        for key in required_keys:
            if key not in profile:
                raise RuntimeError(f"Profile missing key: {key}")
        if profile["camera_transport"] not in ("compressed", "raw"):
            raise RuntimeError("camera_transport must be 'compressed' or 'raw'")
        for entry in profile["camera_streams"]:
            if "image_base" not in entry or "camera_info" not in entry:
                raise RuntimeError("camera_streams entries must include image_base and camera_info")

    def _build_camera_record_topics(self) -> List[str]:
        topics: List[str] = []
        for entry in self.camera_streams:
            image_base = entry["image_base"]
            camera_info = entry["camera_info"]
            topics.append(self._camera_image_topic(image_base))
            topics.append(camera_info)
        return dedup_keep_order(topics)

    def _camera_image_topic(self, image_base: str) -> str:
        if self.camera_transport == "compressed":
            return f"{image_base}/compressed"
        return image_base

    def _normalize_hz_monitor_topics(self, topics: List[str]) -> List[str]:
        camera_bases = {str(entry["image_base"]) for entry in self.camera_streams}
        normalized: List[str] = []
        for topic in topics:
            topic_s = str(topic)
            if topic_s in camera_bases:
                normalized.append(self._camera_image_topic(topic_s))
            else:
                normalized.append(topic_s)
        return dedup_keep_order(normalized)

    def _build_required_presence_topics(self) -> List[str]:
        topics = list(self.required_topics) + list(self.camera_health_topics) + list(self.camera_info_topics)
        return dedup_keep_order(topics)

    def _resolve_output_owner(self) -> Optional[Tuple[int, int]]:
        if os.geteuid() != 0:
            return None

        sudo_uid = os.environ.get("SUDO_UID")
        sudo_gid = os.environ.get("SUDO_GID")
        if sudo_uid and sudo_gid:
            try:
                uid = int(sudo_uid)
                gid = int(sudo_gid)
                if uid >= 0 and gid >= 0:
                    return uid, gid
            except ValueError:
                pass

        probe = self.session_root
        while True:
            if probe.exists():
                st = probe.stat()
                if st.st_uid != 0 or st.st_gid != 0:
                    return st.st_uid, st.st_gid
                return None
            if probe.parent == probe:
                return None
            probe = probe.parent

    def _resolve_output_owner_user(self) -> Optional[str]:
        if self.output_owner is None:
            return None
        uid, _gid = self.output_owner
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return None

    def _resolve_output_modes(self) -> Tuple[int, int]:
        try:
            dir_mode = self.session_root.stat().st_mode & 0o777
        except FileNotFoundError:
            dir_mode = 0o775

        # Regular files should stay non-executable while remaining host-readable/writable.
        file_mode = dir_mode & 0o666
        if file_mode == 0:
            file_mode = 0o664
        if dir_mode == 0:
            dir_mode = 0o775
        return dir_mode, file_mode

    def _apply_output_policy(
        self,
        path: Path,
        *,
        recursive: bool = False,
        warnings: Optional[List[str]] = None,
    ) -> None:
        if not path.exists():
            return

        targets: List[Path] = [path]
        if recursive and path.is_dir():
            for root, dirnames, filenames in os.walk(path):
                root_path = Path(root)
                for dirname in dirnames:
                    targets.append(root_path / dirname)
                for filename in filenames:
                    targets.append(root_path / filename)

        for target in targets:
            try:
                if self.output_owner is not None and os.geteuid() == 0:
                    uid, gid = self.output_owner
                    os.chown(target, uid, gid)
                mode = self.output_dir_mode if target.is_dir() else self.output_file_mode
                os.chmod(target, mode)
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"Failed to normalize output access for {target}: {exc}")
                else:
                    self.current_warnings.append(f"Failed to normalize output access for {target}: {exc}")

    def _subprocess_preexec(self) -> None:
        os.setsid()
        os.umask(0o002)
        if self.output_owner is None or os.geteuid() != 0:
            return

        uid, gid = self.output_owner
        if self.output_owner_user:
            os.initgroups(self.output_owner_user, gid)
        else:
            os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    def _build_subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["HOME"] = str(self.output_runtime_home)
        env["ROS_HOME"] = str(self.output_ros_home)
        if self.output_owner is not None:
            uid, _gid = self.output_owner
            env["USER"] = self.output_owner_user or str(uid)
            env["LOGNAME"] = self.output_owner_user or str(uid)
        return env

    def _ensure_tools(self) -> None:
        missing = []
        for cmd in ("rosbag", "rostopic", "rosparam"):
            if shutil.which(cmd) is None:
                missing.append(cmd)
        if missing:
            raise RuntimeError(f"Missing required command(s) in PATH: {', '.join(missing)}")

    def _scan_next_episode_index(self) -> int:
        used_indices = set()
        for entry in self.session_dir.iterdir():
            if not entry.is_dir():
                continue
            match = re.match(r"^episode_(\d+)$", entry.name)
            if match:
                used_indices.add(int(match.group(1)))

        idx = max(1, int(getattr(self.args, "min_episode_index", 1)))
        while idx in used_indices:
            idx += 1
        return idx

    def _run_cmd(
        self,
        cmd: List[str],
        timeout: Optional[float] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            joined = " ".join(cmd)
            raise RuntimeError(f"Command failed ({joined}): {result.stderr.strip()}")
        return result

    def _ros_master_ok(self) -> Tuple[bool, str]:
        uri = os.environ.get("ROS_MASTER_URI", "")
        if not uri:
            return False, "ROS_MASTER_URI is not set."
        try:
            proxy = xmlrpc.client.ServerProxy(uri)
            code, msg, _val = proxy.getUri("/record_act_keyboard")
            if code == 1:
                return True, msg
            return False, msg
        except Exception as exc:
            return False, str(exc)

    def _topic_list(self) -> List[str]:
        res = self._run_cmd(["rostopic", "list"], timeout=8.0)
        if res.returncode != 0:
            return []
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    def _topic_has_publishers(self, topic: str) -> bool:
        res = self._run_cmd(["rostopic", "info", topic], timeout=8.0)
        if res.returncode != 0:
            return False
        output = res.stdout
        if re.search(r"Publishers:\s*None", output):
            return False

        in_publishers = False
        pub_count = 0
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Publishers:"):
                in_publishers = True
                continue
            if stripped.startswith("Subscribers:"):
                break
            if in_publishers and stripped.startswith("*"):
                pub_count += 1
        return pub_count > 0

    def _check_disk_ok(self) -> Tuple[bool, float]:
        usage = shutil.disk_usage(self.session_root)
        free_gb = usage.free / float(1024 ** 3)
        return free_gb >= self.args.min_free_gb, free_gb

    def _fetch_rosparams(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key in self.key_rosparams:
            res = self._run_cmd(["rosparam", "get", key], timeout=2.0)
            if res.returncode == 0:
                value = res.stdout.strip()
                try:
                    parsed = yaml.safe_load(value)
                    out[key] = parsed
                except Exception:
                    out[key] = value
            else:
                out[key] = None
        return out

    def _build_repro_snapshot_targets(self, job_data: Dict[str, object]) -> Dict[str, List[str]]:
        recorded_topics = [str(x).strip() for x in list(job_data.get("recorded_topics", [])) if str(x).strip()]
        camera_param_namespaces: List[str] = []
        for entry in list(job_data.get("camera_streams", [])):
            if not isinstance(entry, dict):
                continue
            image_base = str(entry.get("image_base", "")).strip()
            if not image_base:
                continue
            # Expected camera stream shape: /<camera_ns>/color/image_raw
            parts = image_base.strip("/").split("/")
            if len(parts) >= 3 and parts[-1] == "image_raw" and parts[-2] == "color":
                camera_ns = "/" + "/".join(parts[:-2])
                camera_param_namespaces.append(f"{camera_ns}/realsense2_camera")

        key_topics = dedup_keep_order(
            list(SNAPSHOT_DEFAULT_KEY_TOPICS)
            + list(self.required_joint_topics)
            + recorded_topics
        )
        selected_topics = dedup_keep_order(
            list(SNAPSHOT_DEFAULT_SELECTED_TOPICS)
            + [topic for topic in key_topics if topic.endswith("/joint_cmd_mux/selected")]
        )
        key_services = dedup_keep_order(list(SNAPSHOT_DEFAULT_KEY_SERVICES))
        key_rosparams = dedup_keep_order(list(SNAPSHOT_DEFAULT_KEY_ROSPARAMS) + list(self.key_rosparams))
        key_param_namespaces = dedup_keep_order(
            list(SNAPSHOT_DEFAULT_PARAM_NAMESPACES)
            + camera_param_namespaces
        )
        return {
            "topics": key_topics,
            "selected_topics": selected_topics,
            "services": key_services,
            "rosparams": key_rosparams,
            "param_namespaces": key_param_namespaces,
        }

    def _capture_repro_snapshot(self, job_data: Dict[str, object]) -> Dict[str, object]:
        targets = self._build_repro_snapshot_targets(job_data)
        try:
            snapshot = capture_runtime_snapshot(
                key_topics=targets["topics"],
                selected_topics=targets["selected_topics"],
                key_services=targets["services"],
                key_rosparams=targets["rosparams"],
                key_param_namespaces=targets["param_namespaces"],
                command_timeout_sec=1.2,
            )
            snapshot["capture_status"] = "ok"
            return snapshot
        except Exception as exc:
            return {
                "snapshot_schema_version": 1,
                "captured_at_utc": utc_now_iso(),
                "capture_status": "error",
                "capture_error": str(exc),
                "ros": {
                    "targets": targets,
                },
            }

    @staticmethod
    def _local_now_str() -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_duration(duration_sec: Optional[float]) -> str:
        if duration_sec is None:
            return "unknown"
        return f"{duration_sec:.2f}s"

    @staticmethod
    def _supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        term = os.environ.get("TERM", "")
        return bool(sys.stdout.isatty() and term and term.lower() != "dumb")

    def _colorize(self, text: str, color: Optional[str]) -> str:
        if not self.use_color or not color:
            return text
        palette = {
            "bold": "\033[1m",
            "dim": "\033[2m",
            "green": "\033[32m",
            "red": "\033[31m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "reset": "\033[0m",
        }
        codes = [palette.get(part) for part in color.split("+")]
        if any(code is None for code in codes):
            return text
        return f"{''.join(codes)}{text}{palette['reset']}"

    def _divider(self) -> None:
        self._print_colored("-" * 72, color="dim")

    def _status_color(self, status: str) -> Optional[str]:
        s = status.upper()
        if s in ("OK", "DONE", "PASS", "KEPT"):
            return "green"
        if s in ("RUN",):
            return "cyan"
        if s in ("WAIT",):
            return "dim"
        if s in ("WARN", "BAD", "FAIL", "FAILED", "ERROR"):
            return "red"
        return None

    def _print_colored(self, text: str, color: Optional[str] = None) -> None:
        print(self._colorize(text, color))

    @staticmethod
    def _progress_bar(done: int, total: int, width: int = 24) -> str:
        if total <= 0:
            total = 1
        done = max(0, min(done, total))
        filled = int(round((done / float(total)) * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _print_save_progress(self, done: int, total: int, label: str, status: str) -> None:
        bar = self._progress_bar(done, total)
        color = self._status_color(status)
        self._print_colored(f"[save] {bar} {done:>2}/{total:<2} {label} [{status}]", color=color)
        sys.stdout.flush()

    def _emit_save_completion(self, episode_name: str, result: Dict[str, object]) -> None:
        status_text = str(result.get("save_status", "OK")).upper()
        duration_sec = result.get("duration_sec")
        duration_text = self._format_duration(float(duration_sec)) if duration_sec is not None else "--"
        warning_count = int(result.get("warning_count", 0))
        fatal_count = int(result.get("fatal_reason_count", 0))
        metadata_path = str(result.get("metadata_path", ""))

        self._divider()
        if status_text == "OK":
            self._print_colored(
                f"[PASS] {episode_name} | duration={duration_text} | metadata={metadata_path}",
                color="green+bold",
            )
            return

        self._print_colored(
            f"[ALERT] {episode_name} | status={status_text} | duration={duration_text} "
            f"| warnings={warning_count} | fatal={fatal_count}",
            color="red+bold",
        )

        preview_lines: List[str] = []
        fatal_reasons = result.get("fatal_reasons", [])
        warnings = result.get("warnings", [])
        if isinstance(fatal_reasons, list):
            preview_lines.extend([str(item) for item in fatal_reasons[:2] if str(item).strip()])
        if isinstance(warnings, list) and len(preview_lines) < 2:
            for item in warnings:
                line = str(item).strip()
                if line and line not in preview_lines:
                    preview_lines.append(line)
                if len(preview_lines) >= 2:
                    break

        for line in preview_lines:
            print(f"  - {line}")
        if metadata_path:
            self._print_colored(f"[DETAIL] {metadata_path}", color="red")
        bad_marker_path = str(result.get("bad_marker_path", ""))
        if bad_marker_path:
            self._print_colored(f"[BAD_ROSBAG] {bad_marker_path}", color="red")
        sys.stdout.flush()

    def _update_save_job(
        self,
        job_id: int,
        *,
        state: Optional[str] = None,
        step: Optional[int] = None,
        total: Optional[int] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, object]] = None,
    ) -> None:
        with self.save_lock:
            job = self.save_jobs.get(job_id)
            if job is None:
                return
            if state is not None:
                job["state"] = state
            if step is not None:
                job["step"] = step
            if total is not None:
                job["total"] = total
            if stage is not None:
                job["stage"] = stage
            if status is not None:
                job["status"] = status
            if error is not None:
                job["error"] = error
            if result is not None:
                job["result"] = result
            job["updated_monotonic"] = time.monotonic()

    def _snapshot_save_jobs(self) -> List[Tuple[int, Dict[str, object]]]:
        with self.save_lock:
            return sorted(
                [(job_id, dict(info)) for job_id, info in self.save_jobs.items()],
                key=lambda x: x[0],
            )

    def _pending_save_count(self) -> int:
        with self.save_lock:
            return sum(1 for info in self.save_jobs.values() if info.get("state") in ("queued", "running"))

    def _mark_save_reported(self, job_ids: List[int]) -> None:
        if not job_ids:
            return
        with self.save_lock:
            for job_id in job_ids:
                info = self.save_jobs.get(job_id)
                if info is not None:
                    info["reported_done"] = True

    def _emit_chinese_save_warning(self, job_id: int, episode_name: str, result: Dict[str, object]) -> None:
        if self.save_progress_mode == "compact":
            self._emit_save_completion(episode_name, result)
            return
        status_text = str(result.get("save_status", "WARN")).upper()
        warning_count = int(result.get("warning_count", 0))
        metadata_path = str(result.get("metadata_path", ""))
        bad_marker_path = str(result.get("bad_marker_path", ""))
        drop_topics_raw = result.get("drop_frame_topics", [])
        warnings_raw = result.get("warnings", [])
        fatal_raw = result.get("fatal_reasons", [])
        drop_topics = list(drop_topics_raw) if isinstance(drop_topics_raw, list) else []
        warnings = list(warnings_raw) if isinstance(warnings_raw, list) else []
        fatal_reasons = list(fatal_raw) if isinstance(fatal_raw, list) else []

        self._print_colored(
            f"[警告][save][job={job_id}][{episode_name}] 检测到 rosbag 保存异常"
            f"（status={status_text}，warning_count={warning_count}）。",
            color="red",
        )
        if fatal_reasons:
            self._print_colored("[警告] 以下问题已判定该 episode 不适合训练使用:", color="red")
            for line in fatal_reasons[: self.warn_preview_lines]:
                print(f"  - {line}")
            if len(fatal_reasons) > self.warn_preview_lines:
                print(f"  - ...(还有 {len(fatal_reasons) - self.warn_preview_lines} 条)")
        if drop_topics:
            self._print_colored(f"[警告] 检测到可能掉帧话题（{len(drop_topics)}）:", color="red")
            for line in drop_topics[: self.warn_preview_lines]:
                print(f"  - {line}")
            if len(drop_topics) > self.warn_preview_lines:
                print(f"  - ...(还有 {len(drop_topics) - self.warn_preview_lines} 条)")

        if warnings:
            self._print_colored("[警告] 异常详情（节选）:", color="red")
            for line in warnings[: self.warn_preview_lines]:
                print(f"  - {line}")
            if len(warnings) > self.warn_preview_lines:
                print(f"  - ...(还有 {len(warnings) - self.warn_preview_lines} 条)")

        if metadata_path:
            self._print_colored(f"[警告] 详细信息见: {metadata_path}", color="red")
        if bad_marker_path:
            self._print_colored(f"[警告] 已写入坏包标记: {bad_marker_path}", color="red")
        sys.stdout.flush()

    def _on_save_future_done(self, job_id: int, fut: Future) -> None:
        try:
            result = fut.result()
            with self.save_lock:
                total = int(self.save_jobs.get(job_id, {}).get("total", 1))
            self._update_save_job(
                job_id,
                state="done",
                step=total,
                stage="complete",
                status=str(result.get("save_status", "OK")),
                result=result,
            )
        except Exception as exc:
            self._update_save_job(
                job_id,
                state="failed",
                stage="error",
                status="FAIL",
                error=str(exc),
            )

    def _submit_save_job(self, job_data: Dict[str, object]) -> int:
        with self.save_lock:
            job_id = self.next_save_job_id
            self.next_save_job_id += 1
            self.save_jobs[job_id] = {
                "episode_name": job_data.get("episode_name"),
                "state": "queued",
                "step": 0,
                "total": 6,
                "stage": "queued",
                "status": "WAIT",
                "reported_done": False,
                "error": "",
                "result": None,
                "updated_monotonic": time.monotonic(),
            }
        fut = self.save_executor.submit(self._finalize_episode_job, job_id, job_data)
        fut.add_done_callback(lambda done_fut, jid=job_id: self._on_save_future_done(jid, done_fut))
        with self.save_lock:
            self.save_futures[job_id] = fut
        return job_id

    def _emit_save_progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_save_status_ts) < self.save_status_interval_sec:
            return

        snapshots = self._snapshot_save_jobs()
        if not snapshots:
            return

        active = []
        finished = []
        for job_id, info in snapshots:
            state = str(info.get("state", ""))
            if state in ("queued", "running"):
                active.append((job_id, info))
            elif state in ("done", "failed") and not bool(info.get("reported_done")):
                finished.append((job_id, info))

        if not active and not finished and not force:
            return

        if self.save_progress_mode != "compact":
            for job_id, info in active:
                bar = self._progress_bar(int(info.get("step", 0)), int(info.get("total", 1)))
                line = (
                    f"[save][job={job_id}][{info.get('episode_name')}] "
                    f"{bar} {int(info.get('step', 0)):>2}/{int(info.get('total', 1)):<2} "
                    f"{info.get('stage', '')} [{info.get('status', 'RUN')}]"
                )
                self._print_colored(line, color=self._status_color(str(info.get("status", "RUN"))))

        reported_ids: List[int] = []
        for job_id, info in finished:
            reported_ids.append(job_id)
            if info.get("state") == "done":
                result = info.get("result") or {}
                status_text = str(result.get("save_status", "OK"))
                self._emit_save_completion(str(info.get("episode_name")), result)
                if self.save_progress_mode != "compact" and status_text.upper() in ("WARN", "BAD"):
                    self._emit_chinese_save_warning(job_id, str(info.get("episode_name")), result)
            else:
                self._divider()
                line = f"[ALERT] {info.get('episode_name')} | FAILED | error={info.get('error', 'unknown')}"
                self._print_colored(line, color="red+bold")

        self._mark_save_reported(reported_ids)
        self.last_save_status_ts = now
        sys.stdout.flush()

    def _emit_runtime_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if self.recording and self.current_started_monotonic is not None:
            if force or (now - self.last_record_status_ts) >= self.record_status_interval_sec:
                elapsed = now - self.current_started_monotonic
                self._print_colored(
                    f"[record] ACTIVE {self.current_episode_name} "
                    f"| elapsed={self._format_duration(elapsed)} "
                    f"| pending_saves={self._pending_save_count()}",
                    color="cyan",
                )
                self.last_record_status_ts = now
                sys.stdout.flush()
        self._emit_save_progress(force=force)

    def _print_idle_help(self) -> None:
        print("")
        self._divider()
        self._print_colored("ACT Keyboard Recorder", color="magenta+bold")
        print(f"Session      : {self.session_name}")
        print(f"Session dir  : {self.session_dir}")
        if self.output_owner is not None:
            uid, gid = self.output_owner
            owner_label = self.output_owner_user or str(uid)
            print(f"Output owner : {owner_label} (uid={uid} gid={gid})")
        print(f"Output access: dir={self.output_dir_mode:o} file={self.output_file_mode:o}")
        print(f"Cameras      : {len(self.camera_streams)} ({self.camera_transport})")
        if self.args.extra_camera_ns:
            print(f"Extra camera namespaces: {', '.join(normalize_ros_ns(x) for x in self.args.extra_camera_ns)}")
        if self.extra_topics:
            print(f"Extra record topics: {', '.join(self.extra_topics)}")
        print(f"Optional     : {'all' if self.record_all_optional_topics else 'present-only'}")
        print(f"Next episode : episode_{self.next_episode_index:03d}")
        print(f"Save workers : {self.save_workers} ({self.save_progress_mode} ui)")
        if self.args.wait_for_save:
            print("Save mode    : foreground; wait for progress and PASS/ALERT before the next episode")
        self._print_colored("Keys         : SPACE start/stop | K delete last | Q quit", color="dim")
        self._divider()
        print("")
        sys.stdout.flush()

    def _latest_episode_dir(self) -> Optional[Tuple[int, str, Path]]:
        latest: Optional[Tuple[int, str, Path]] = None
        for entry in self.session_dir.iterdir():
            if not entry.is_dir():
                continue
            match = re.match(r"^episode_(\d+)$", entry.name)
            if not match:
                continue
            idx = int(match.group(1))
            item = (idx, entry.name, entry)
            if latest is None or idx > latest[0]:
                latest = item
        return latest

    def _find_save_job_for_episode(self, episode_name: str) -> Optional[int]:
        with self.save_lock:
            candidates = [
                job_id
                for job_id, info in self.save_jobs.items()
                if str(info.get("episode_name", "")) == episode_name
            ]
        if not candidates:
            return None
        return max(candidates)

    def _drain_saving_keys(self) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        while select.select([fd], [], [], 0)[0]:
            key = os.read(fd, 1)
            if not key:
                break
            if key in (b"q", b"Q"):
                self.shutdown_requested = True

    def _wait_for_episode_save_job(self, episode_name: str) -> None:
        job_id = self._find_save_job_for_episode(episode_name)
        if job_id is None:
            return

        while True:
            with self.save_lock:
                info = dict(self.save_jobs.get(job_id, {}))
            state = str(info.get("state", ""))
            if state not in ("queued", "running"):
                break
            if self.args.wait_for_save:
                self._drain_saving_keys()
            self._emit_save_progress(force=False)
            time.sleep(0.2)
        self._emit_save_progress(force=True)

    def _delete_last_episode(self) -> None:
        if self.recording:
            print("[record] delete-last is unavailable while recording; stop the current episode first.")
            sys.stdout.flush()
            return

        latest = self._latest_episode_dir()
        if latest is None:
            print("[record] no episode exists in the current session; nothing to delete.")
            sys.stdout.flush()
            return

        idx, episode_name, episode_dir = latest
        self._wait_for_episode_save_job(episode_name)

        try:
            shutil.rmtree(episode_dir)
        except Exception as exc:
            print(f"[record] failed to delete {episode_name}: {exc}")
            sys.stdout.flush()
            return

        self.next_episode_index = self._scan_next_episode_index()
        self._print_colored(f"[record] DELETE {episode_name}", color="green")
        print(f"[record] deleted_path={episode_dir}")
        print(f"[record] next=episode_{self.next_episode_index:03d}")
        sys.stdout.flush()

    def _wait_preflight(
        self,
        timeout_sec: float,
        skip_camera_publishers: bool = False,
    ) -> Tuple[bool, Dict[str, object]]:
        started = time.time()
        deadline = started + timeout_sec
        poll_sec = 0.2 if timeout_sec <= 5.0 else 1.0
        last_wait_report = 0.0
        last_master_msg = ""
        last_missing_presence: List[str] = []
        last_joint_no_publisher: List[str] = []
        last_camera_unhealthy: List[str] = []

        def report_wait_status(reason: str, details: str) -> None:
            nonlocal last_wait_report
            now = time.monotonic()
            if (now - last_wait_report) < 1.0:
                return
            print(f"[record] preflight waiting: {reason}: {details}")
            sys.stdout.flush()
            last_wait_report = now

        while time.time() < deadline:
            ok_master, master_msg = self._ros_master_ok()
            if not ok_master:
                last_master_msg = master_msg
                report_wait_status("ros master", master_msg)
                time.sleep(poll_sec)
                continue

            topics = set(self._topic_list())
            missing_presence = sorted([t for t in self.required_presence_topics if t not in topics])
            if missing_presence:
                last_missing_presence = missing_presence
                report_wait_status("missing topics", str(missing_presence[:5]))
                time.sleep(poll_sec)
                continue
            last_missing_presence = []

            joint_no_publisher: List[str] = []
            for topic in self.required_joint_topics:
                if topic in topics and not self._topic_has_publishers(topic):
                    joint_no_publisher.append(topic)
            if joint_no_publisher:
                last_joint_no_publisher = joint_no_publisher
                report_wait_status("joint topics without publishers", str(joint_no_publisher[:5]))
                time.sleep(poll_sec)
                continue
            last_joint_no_publisher = []

            camera_unhealthy: List[str] = []
            if not skip_camera_publishers:
                for cam_topic in self.camera_health_topics:
                    if not self._topic_has_publishers(cam_topic):
                        camera_unhealthy.append(cam_topic)
                if camera_unhealthy:
                    last_camera_unhealthy = camera_unhealthy
                    report_wait_status("camera unhealthy", str(camera_unhealthy[:5]))
                    time.sleep(poll_sec)
                    continue
            last_camera_unhealthy = []

            disk_ok, free_gb = self._check_disk_ok()
            if not disk_ok:
                return False, {
                    "reason": "disk_low",
                    "free_gb": free_gb,
                    "min_free_gb": self.args.min_free_gb,
                }

            optional_present = sorted([t for t in self.optional_topics if t in topics])
            optional_missing = sorted([t for t in self.optional_topics if t not in topics])
            return True, {
                "reason": "ok",
                "wait_sec": round(time.time() - started, 3),
                "ros_master_msg": master_msg,
                "required_presence_topics": list(self.required_presence_topics),
                "required_joint_topics": list(self.required_joint_topics),
                "camera_health_topics": list(self.camera_health_topics),
                "skip_camera_publishers": bool(skip_camera_publishers),
                "optional_present": optional_present,
                "optional_missing": optional_missing,
            }

        return False, {
            "reason": "preflight_timeout",
            "wait_timeout_sec": timeout_sec,
            "required_presence_topics": list(self.required_presence_topics),
            "required_joint_topics": list(self.required_joint_topics),
            "camera_health_topics": list(self.camera_health_topics),
            "skip_camera_publishers": bool(skip_camera_publishers),
            "last_master_msg": last_master_msg,
            "last_missing_presence_topics": last_missing_presence,
            "last_joint_no_publisher_topics": last_joint_no_publisher,
            "last_camera_unhealthy_topics": last_camera_unhealthy,
        }

    def _get_next_episode_slot(self) -> Tuple[int, str, Path]:
        idx = self._scan_next_episode_index()
        while True:
            name = f"episode_{idx:03d}"
            ep_dir = self.session_dir / name
            if not ep_dir.exists():
                return idx, name, ep_dir
            idx += 1

    def _start_hz_monitors(self, episode_dir: Path) -> None:
        self.hz_monitors = {}
        for topic in self.hz_monitor_topics:
            log_path = episode_dir / f"hz_{sanitize_topic(topic)}.log"
            fh = log_path.open("w", encoding="utf-8")
            self._apply_output_policy(log_path)
            self.debug_log_paths.append(log_path)
            cmd = ["rostopic", "hz", "-w", str(self.args.hz_window), topic]
            proc = subprocess.Popen(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=self._build_subprocess_env(),
                preexec_fn=self._subprocess_preexec,
                text=True,
            )
            self.hz_monitors[topic] = (proc, log_path, fh)

    def _stop_hz_monitors(self) -> Dict[str, Path]:
        log_paths: Dict[str, Path] = {}
        for topic, (proc, log_path, fh) in self.hz_monitors.items():
            self._stop_process_group(proc, name=f"hz:{topic}")
            fh.flush()
            fh.close()
            log_paths[topic] = log_path
        self.hz_monitors = {}
        return log_paths

    def _request_process_group_stop(self, proc: subprocess.Popen, name: str) -> bool:
        if proc.poll() is not None:
            return True
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            return proc.poll() is not None

        try:
            os.killpg(pgid, signal.SIGINT)
            return True
        except ProcessLookupError:
            return True
        except Exception as exc:
            self.current_warnings.append(f"Failed to send SIGINT to {name}: {exc}")
            return False

    def _cleanup_debug_logs(self, paths: List[Path], warnings: Optional[List[str]] = None) -> List[str]:
        removed: List[str] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"Failed to remove debug log {path.name}: {exc}")
                else:
                    self.current_warnings.append(f"Failed to remove debug log {path.name}: {exc}")
        return removed

    def _parse_hz_log(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {"status": "no_log"}
        content = path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(
            r"average rate:\s*([0-9.]+).*?"
            r"min:\s*([0-9.]+)s\s+max:\s*([0-9.]+)s\s+std dev:\s*([0-9.]+)s\s+window:\s*([0-9]+)",
            re.DOTALL,
        )
        matches = list(pattern.finditer(content))
        if not matches:
            if "no new messages" in content.lower():
                return {"status": "no_messages"}
            return {"status": "unavailable"}
        match = matches[-1]
        return {
            "status": "ok",
            "avg_hz": float(match.group(1)),
            "min_period_s": float(match.group(2)),
            "max_period_s": float(match.group(3)),
            "std_dev_s": float(match.group(4)),
            "window": int(match.group(5)),
        }

    def _stop_process_group(self, proc: subprocess.Popen, name: str) -> bool:
        if proc.poll() is not None:
            return True
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            return proc.poll() is not None

        sequence = [
            (signal.SIGINT, 5.0),
            (signal.SIGTERM, 3.0),
            (signal.SIGKILL, 1.0),
        ]
        for sig, wait_sec in sequence:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return True
            except Exception:
                pass

            deadline = time.time() + wait_sec
            while time.time() < deadline:
                if proc.poll() is not None:
                    return True
                time.sleep(0.1)

        return proc.poll() is not None

    def _wait_episode_processes(
        self,
        job_data: Dict[str, object],
        warnings: List[str],
    ) -> bool:
        rosbag_ok = True
        rosbag_proc = job_data.get("rosbag_proc")
        if isinstance(rosbag_proc, subprocess.Popen):
            rosbag_ok = self._stop_process_group(rosbag_proc, name="rosbag")
            if not rosbag_ok:
                warnings.append("Failed to terminate rosbag process cleanly.")

        hz_monitor_procs = job_data.get("hz_monitor_procs")
        if isinstance(hz_monitor_procs, dict):
            for topic, proc in hz_monitor_procs.items():
                if not isinstance(proc, subprocess.Popen):
                    continue
                ok = self._stop_process_group(proc, name=f"hz:{topic}")
                if not ok:
                    warnings.append(f"Failed to terminate hz monitor cleanly: {topic}")
        return rosbag_ok

    def _collect_bag_files(self, episode_dir: Path, include_active: bool = False) -> List[Path]:
        if include_active:
            files = sorted(episode_dir.glob("episode*.bag*"))
            return [f for f in files if f.is_file()]
        files = sorted(episode_dir.glob("episode*.bag"))
        return [f for f in files if f.is_file()]

    def _wait_for_bag_finalize(
        self,
        episode_dir: Path,
        timeout_sec: float = 8.0,
        poll_sec: float = 0.2,
    ) -> Tuple[List[Path], List[Path], bool]:
        timeout_sec = max(0.0, float(timeout_sec))
        poll_sec = max(0.05, float(poll_sec))
        deadline = time.monotonic() + timeout_sec

        while True:
            ready_bags = self._collect_bag_files(episode_dir, include_active=False)
            active_bags = sorted(
                [f for f in episode_dir.glob("episode*.bag.active") if f.is_file()]
            )
            if ready_bags and not active_bags:
                return ready_bags, active_bags, False
            if time.monotonic() >= deadline:
                return ready_bags, active_bags, True
            time.sleep(poll_sec)

    def _aggregate_bag_stats(
        self,
        bag_files: List[Path],
        on_bag_done: Optional[Callable[[int, int, Path], None]] = None,
    ) -> Dict[str, object]:
        total_size = 0
        total_messages = 0
        min_start = None
        max_end = None
        per_topic: Dict[str, Dict[str, object]] = {}
        parse_warnings: List[str] = []

        total_bags = len(bag_files)
        for idx, bag_file in enumerate(bag_files, start=1):
            try:
                try:
                    total_size += bag_file.stat().st_size
                except Exception:
                    pass

                res = self._run_cmd(["rosbag", "info", "--yaml", str(bag_file)], timeout=20.0)
                if res.returncode != 0:
                    parse_warnings.append(f"rosbag info failed: {bag_file.name}")
                    continue
                try:
                    info = yaml.safe_load(res.stdout) or {}
                except Exception:
                    parse_warnings.append(f"yaml parse failed: {bag_file.name}")
                    continue

                start = info.get("start")
                end = info.get("end")
                messages = info.get("messages", 0)
                if isinstance(start, (int, float)):
                    min_start = start if min_start is None else min(min_start, start)
                if isinstance(end, (int, float)):
                    max_end = end if max_end is None else max(max_end, end)
                if isinstance(messages, int):
                    total_messages += messages

                for t in info.get("topics", []) or []:
                    topic_name = t.get("topic")
                    if not topic_name:
                        continue
                    entry = per_topic.setdefault(
                        topic_name,
                        {
                            "messages": 0,
                            "type": t.get("type"),
                            "bags": 0,
                        },
                    )
                    entry["messages"] += int(t.get("messages", 0))
                    entry["bags"] += 1
            finally:
                if on_bag_done is not None:
                    on_bag_done(idx, total_bags, bag_file)

        duration = None
        if min_start is not None and max_end is not None:
            duration = float(max_end) - float(min_start)

        return {
            "bag_files": [str(f) for f in bag_files],
            "bag_count": len(bag_files),
            "total_size_bytes": total_size,
            "total_messages": total_messages,
            "start_time": min_start,
            "end_time": max_end,
            "duration_sec": duration,
            "per_topic": per_topic,
            "parse_warnings": parse_warnings,
        }

    def _load_bag_health_series(
        self,
        bag_files: List[Path],
        topics: List[str],
    ) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
        try:
            import rosbag  # type: ignore
        except Exception as exc:
            return {}, [f"rosbag python module unavailable for bag health audit: {exc}"]

        wanted_topics = sorted(dedup_keep_order(topics))
        joint_series: Dict[str, Dict[str, object]] = {
            topic: {"ts": [], "pos": [], "message_count": 0, "dropped_short": 0}
            for topic in wanted_topics
            if not topic.endswith("/selected")
        }
        selected_series: Dict[str, Dict[str, object]] = {
            topic: {"ts": [], "values": [], "message_count": 0}
            for topic in wanted_topics
            if topic.endswith("/selected")
        }
        warnings: List[str] = []

        for bag_file in bag_files:
            try:
                with rosbag.Bag(str(bag_file), "r") as bag:
                    for topic, msg, bag_time in bag.read_messages(topics=wanted_topics):
                        if topic in joint_series:
                            pos = list(getattr(msg, "position", []))
                            if len(pos) < PER_ARM_DIM:
                                joint_series[topic]["dropped_short"] = int(joint_series[topic]["dropped_short"]) + 1
                                continue
                            joint_series[topic]["ts"].append(msg_time_sec(msg, bag_time))
                            joint_series[topic]["pos"].append(np.asarray(pos[:PER_ARM_DIM], dtype=np.float32))
                            joint_series[topic]["message_count"] = int(joint_series[topic]["message_count"]) + 1
                        elif topic in selected_series:
                            selected_series[topic]["ts"].append(msg_time_sec(msg, bag_time))
                            selected_series[topic]["values"].append(str(getattr(msg, "data", "")).strip())
                            selected_series[topic]["message_count"] = int(selected_series[topic]["message_count"]) + 1
            except Exception as exc:
                warnings.append(f"Bag health audit failed to read {bag_file.name}: {exc}")

        result: Dict[str, Dict[str, object]] = {}
        for topic, raw in joint_series.items():
            ts = np.asarray(raw["ts"], dtype=np.float64)
            if raw["pos"]:
                pos = np.asarray(raw["pos"], dtype=np.float32).reshape(-1, PER_ARM_DIM)
            else:
                pos = np.zeros((0, PER_ARM_DIM), dtype=np.float32)
            result[topic] = {
                "kind": "joint",
                "timestamps": ts,
                "positions": pos,
                "message_count": int(raw["message_count"]),
                "dropped_short": int(raw["dropped_short"]),
            }
        for topic, raw in selected_series.items():
            result[topic] = {
                "kind": "selected",
                "timestamps": np.asarray(raw["ts"], dtype=np.float64),
                "values": [str(x) for x in raw["values"]],
                "message_count": int(raw["message_count"]),
            }
        return result, warnings

    def _joint_series_summary(self, topic: str, series: Dict[str, object]) -> Dict[str, object]:
        timestamps = np.asarray(series.get("timestamps", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        positions = np.asarray(
            series.get("positions", np.zeros((0, PER_ARM_DIM), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, PER_ARM_DIM)
        count = int(series.get("message_count", positions.shape[0]))
        gap_count, gap_threshold_sec, max_gap_sec, median_period_sec = big_gap_metrics(
            timestamps,
            min_gap_sec=self.health_big_gap_min_sec,
            median_mult=self.health_big_gap_median_mult,
        )
        return {
            "topic": topic,
            "message_count": count,
            "dropped_short": int(series.get("dropped_short", 0)),
            "hz": hz_estimate(timestamps),
            "max_period_ms": max_period_ms(timestamps),
            "median_period_ms": float(median_period_sec * 1000.0),
            "max_gap_ms": float(max_gap_sec * 1000.0),
            "big_gap_threshold_ms": float(gap_threshold_sec * 1000.0),
            "big_gap_count": int(gap_count),
            "repeat_ratio": longest_repeat_ratio(positions, self.health_repeat_tol),
            "motion_span": joint_motion_span(positions),
        }

    def _analyze_saved_bag_health(
        self,
        *,
        bag_files: List[Path],
        recorded_topics: List[str],
        bag_stats: Dict[str, object],
        hz_stats: Dict[str, object],
    ) -> Dict[str, object]:
        recorded_topic_set = set(recorded_topics)
        active_arms = [
            arm
            for arm in ARM_ORDER
            if ARM_TOPIC_SPEC[arm]["teleop"] in recorded_topic_set
            or ARM_TOPIC_SPEC[arm]["state"] in recorded_topic_set
            or ARM_TOPIC_SPEC[arm]["cmd"] in recorded_topic_set
        ]
        if not active_arms:
            return {
                "available": False,
                "status": "BAD",
                "usable_for_training": False,
                "fatal_reasons": ["No arm topics found for bag health audit."],
                "warnings": [],
                "arms": {},
                "cameras": {},
            }

        topics_to_load: List[str] = []
        for arm in active_arms:
            spec = ARM_TOPIC_SPEC[arm]
            topics_to_load.extend([spec["teleop"], spec["state"]])
            if spec["cmd"] in recorded_topic_set:
                topics_to_load.append(spec["cmd"])
            if spec["selected"] in recorded_topic_set:
                topics_to_load.append(spec["selected"])

        series_by_topic, read_warnings = self._load_bag_health_series(bag_files, topics_to_load)
        fatal_reasons: List[str] = []
        warnings: List[str] = list(read_warnings)
        arms_summary: Dict[str, Dict[str, object]] = {}
        strict_health_checks = self.health_audit_mode == "strict"

        for arm in active_arms:
            spec = ARM_TOPIC_SPEC[arm]
            teleop_topic = spec["teleop"]
            state_topic = spec["state"]
            cmd_topic = spec["cmd"]
            selected_topic = spec["selected"]

            teleop_series = series_by_topic.get(teleop_topic, {"timestamps": np.zeros((0,)), "positions": np.zeros((0, PER_ARM_DIM)), "message_count": 0})
            state_series = series_by_topic.get(state_topic, {"timestamps": np.zeros((0,)), "positions": np.zeros((0, PER_ARM_DIM)), "message_count": 0})
            cmd_series = series_by_topic.get(cmd_topic, {"timestamps": np.zeros((0,)), "positions": np.zeros((0, PER_ARM_DIM)), "message_count": 0})
            selected_series = series_by_topic.get(selected_topic, {"timestamps": np.zeros((0,)), "values": [], "message_count": 0})

            teleop_summary = self._joint_series_summary(teleop_topic, teleop_series)
            state_summary = self._joint_series_summary(state_topic, state_series)
            cmd_summary = self._joint_series_summary(cmd_topic, cmd_series)
            if self.health_action_source == "cmd":
                action_topic = cmd_topic
                action_summary = cmd_summary
                action_label = "joint_cmd_mux action"
            else:
                action_topic = teleop_topic
                action_summary = teleop_summary
                action_label = "teleop action"

            teleop_ts = np.asarray(teleop_series.get("timestamps", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            teleop_pos = np.asarray(teleop_series.get("positions", np.zeros((0, PER_ARM_DIM), dtype=np.float32)), dtype=np.float32).reshape(-1, PER_ARM_DIM)
            state_ts = np.asarray(state_series.get("timestamps", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            state_pos = np.asarray(state_series.get("positions", np.zeros((0, PER_ARM_DIM), dtype=np.float32)), dtype=np.float32).reshape(-1, PER_ARM_DIM)
            cmd_ts = np.asarray(cmd_series.get("timestamps", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            cmd_pos = np.asarray(cmd_series.get("positions", np.zeros((0, PER_ARM_DIM), dtype=np.float32)), dtype=np.float32).reshape(-1, PER_ARM_DIM)

            teleop_cmd = estimate_follow_metrics(
                teleop_ts,
                teleop_pos,
                cmd_ts,
                cmd_pos,
                lag_values=self.health_lag_values,
                motion_span_warn=self.health_motion_span_warn,
            )
            cmd_state = estimate_follow_metrics(
                cmd_ts,
                cmd_pos,
                state_ts,
                state_pos,
                lag_values=self.health_lag_values,
                motion_span_warn=self.health_motion_span_warn,
            )
            teleop_state = estimate_follow_metrics(
                teleop_ts,
                teleop_pos,
                state_ts,
                state_pos,
                lag_values=self.health_lag_values,
                motion_span_warn=self.health_motion_span_warn,
            )

            selected_values = [str(x).strip() for x in list(selected_series.get("values", [])) if str(x).strip()]
            if self.health_expected_mux_source == "left_robot_cmd_right_teleop":
                expected_selected = (
                    "/robot/arm_left/robot_cmd"
                    if arm == "left"
                    else "/teleop/arm_right/joint_states_single"
                )
            elif self.health_expected_mux_source == "robot_cmd":
                expected_selected = f"/robot/arm_{arm}/robot_cmd"
            elif self.health_expected_mux_source == "vla":
                expected_selected = f"/robot/arm_{arm}/vla_joint_cmd"
            elif self.health_expected_mux_source == "teleop":
                expected_selected = teleop_topic
            else:
                expected_selected = (
                    f"/robot/arm_{arm}/vla_joint_cmd"
                    if self.health_action_source == "cmd"
                    else teleop_topic
                )
            mismatch_values = sorted({value for value in selected_values if value != expected_selected})

            if int(action_summary["message_count"]) <= 0:
                fatal_reasons.append(f"{arm}: {action_label} topic has zero messages ({action_topic})")
            if int(state_summary["message_count"]) <= 0:
                fatal_reasons.append(f"{arm}: robot state topic has zero messages ({state_topic})")
            if int(action_summary["dropped_short"]) > 0:
                fatal_reasons.append(
                    f"{arm}: {action_label} topic has {action_summary['dropped_short']} short JointState message(s)"
                )
            if int(state_summary["dropped_short"]) > 0:
                fatal_reasons.append(
                    f"{arm}: robot state topic has {state_summary['dropped_short']} short JointState message(s)"
                )
            if float(action_summary["hz"]) < self.health_min_action_hz:
                fatal_reasons.append(
                    f"{arm}: {action_label} hz too low ({action_summary['hz']:.1f} < {self.health_min_action_hz:.1f})"
                )
            if float(state_summary["hz"]) < self.health_min_state_hz:
                fatal_reasons.append(
                    f"{arm}: robot state hz too low ({state_summary['hz']:.1f} < {self.health_min_state_hz:.1f})"
                )
            if int(action_summary["big_gap_count"]) > self.health_big_gap_max_count:
                fatal_reasons.append(
                    f"{arm}: {action_label} stream has {action_summary['big_gap_count']} large timestamp gap(s) "
                    f"(allowed <= {self.health_big_gap_max_count})"
                )
            if int(state_summary["big_gap_count"]) > self.health_big_gap_max_count:
                fatal_reasons.append(
                    f"{arm}: robot state stream has {state_summary['big_gap_count']} large timestamp gap(s) "
                    f"(allowed <= {self.health_big_gap_max_count})"
                )
            if 0 < int(action_summary["big_gap_count"]) <= self.health_big_gap_max_count:
                warnings.append(
                    f"{arm}: {action_label} stream has {action_summary['big_gap_count']} tolerated gap(s) "
                    f"(<= {self.health_big_gap_max_count}, max_gap_ms={float(action_summary['max_gap_ms']):.2f}, "
                    f"threshold_ms={float(action_summary['big_gap_threshold_ms']):.2f})"
                )
            if 0 < int(state_summary["big_gap_count"]) <= self.health_big_gap_max_count:
                warnings.append(
                    f"{arm}: robot state stream has {state_summary['big_gap_count']} tolerated gap(s) "
                    f"(<= {self.health_big_gap_max_count}, max_gap_ms={float(state_summary['max_gap_ms']):.2f}, "
                    f"threshold_ms={float(state_summary['big_gap_threshold_ms']):.2f})"
                )
            if strict_health_checks:
                if (
                    float(action_summary["motion_span"]) >= self.health_motion_span_warn
                    and float(action_summary["repeat_ratio"]) >= self.health_repeat_warn_ratio
                ):
                    warnings.append(
                        f"{arm}: {action_label} stream shows long repeated runs during motion "
                        f"(repeat={action_summary['repeat_ratio']:.2f}, span={action_summary['motion_span']:.3f})"
                    )
                if int(cmd_summary["message_count"]) <= 0 and cmd_topic in recorded_topic_set:
                    warnings.append(f"{arm}: joint_cmd_mux topic has zero messages ({cmd_topic})")
                if int(cmd_summary["dropped_short"]) > 0:
                    warnings.append(
                        f"{arm}: joint_cmd_mux topic has {cmd_summary['dropped_short']} short JointState message(s)"
                    )
                if float(cmd_summary["hz"]) > 0.0 and float(cmd_summary["hz"]) < self.health_min_joint_hz:
                    warnings.append(
                        f"{arm}: joint_cmd_mux hz is low ({cmd_summary['hz']:.1f} < {self.health_min_joint_hz:.1f})"
                    )
                if int(cmd_summary["big_gap_count"]) > 0:
                    warnings.append(
                        f"{arm}: joint_cmd_mux stream has {cmd_summary['big_gap_count']} large timestamp gap(s)"
                    )
                if mismatch_values:
                    fatal_reasons.append(
                        f"{arm}: mux selected unexpected source(s): {mismatch_values} (expected {expected_selected})"
                    )
                elif selected_topic in recorded_topic_set and not selected_values:
                    warnings.append(f"{arm}: mux selected topic recorded but contained no messages ({selected_topic})")

                enforce_teleop_state_sync = (
                    self.health_action_source == "teleop"
                    or self.health_require_teleop_state_sync
                )
                if enforce_teleop_state_sync and teleop_state.get("status") == "ok":
                    if float(teleop_state["lag_ms"]) > self.health_teleop_state_lag_warn_ms:
                        fatal_reasons.append(
                            f"{arm}: teleop->state lag too high ({teleop_state['lag_ms']:.1f} ms)"
                        )
                    if float(teleop_state["mae"]) > self.health_teleop_state_mae_warn:
                        fatal_reasons.append(
                            f"{arm}: teleop->state mae too high ({teleop_state['mae']:.4f})"
                        )
                    if bool(teleop_state.get("saturated", False)):
                        warnings.append(
                            f"{arm}: teleop->state lag search hit upper bound ({self.args.health_lag_search_ms:.0f} ms)"
                        )
                elif enforce_teleop_state_sync and teleop_state.get("status") == "insufficient":
                    warnings.append(f"{arm}: teleop->state audit status={teleop_state.get('status')}")

                if self.health_action_source == "teleop" and teleop_cmd.get("status") == "ok":
                    if float(teleop_cmd["lag_ms"]) > self.health_teleop_cmd_lag_warn_ms:
                        warnings.append(f"{arm}: teleop->cmd lag too high ({teleop_cmd['lag_ms']:.1f} ms)")
                    if float(teleop_cmd["mae"]) > self.health_teleop_cmd_mae_warn:
                        warnings.append(f"{arm}: teleop->cmd mae too high ({teleop_cmd['mae']:.4f})")
                    if bool(teleop_cmd.get("saturated", False)):
                        warnings.append(
                            f"{arm}: teleop->cmd lag search hit upper bound ({self.args.health_lag_search_ms:.0f} ms)"
                        )
                elif (
                    self.health_action_source == "teleop"
                    and cmd_topic in recorded_topic_set
                    and teleop_cmd.get("status") == "insufficient"
                ):
                    warnings.append(f"{arm}: teleop->cmd audit status={teleop_cmd.get('status')}")

                if cmd_state.get("status") == "ok":
                    if float(cmd_state["lag_ms"]) > self.health_cmd_state_lag_warn_ms:
                        warnings.append(f"{arm}: cmd->state lag too high ({cmd_state['lag_ms']:.1f} ms)")
                    if float(cmd_state["mae"]) > self.health_cmd_state_mae_warn:
                        warnings.append(f"{arm}: cmd->state mae too high ({cmd_state['mae']:.4f})")
                    if bool(cmd_state.get("saturated", False)):
                        warnings.append(
                            f"{arm}: cmd->state lag search hit upper bound ({self.args.health_lag_search_ms:.0f} ms)"
                        )
                elif (
                    cmd_topic in recorded_topic_set
                    and int(cmd_summary["message_count"]) > 0
                    and cmd_state.get("status") == "insufficient"
                ):
                    warnings.append(f"{arm}: cmd->state audit status={cmd_state.get('status')}")

            arms_summary[arm] = {
                "action_source": self.health_action_source,
                "require_teleop_state_sync": self.health_require_teleop_state_sync,
                "action_topic": action_topic,
                "teleop": teleop_summary,
                "cmd": cmd_summary,
                "state": state_summary,
                "selected": {
                    "topic": selected_topic,
                    "message_count": int(selected_series.get("message_count", 0)),
                    "expected": expected_selected,
                    "observed_unique": sorted(set(selected_values)),
                    "mismatch_values": mismatch_values,
                },
                "teleop_to_cmd": teleop_cmd,
                "cmd_to_state": cmd_state,
                "teleop_to_state": teleop_state,
            }

        per_topic = bag_stats.get("per_topic", {})
        bag_duration_raw = bag_stats.get("duration_sec")
        try:
            bag_duration = float(bag_duration_raw) if bag_duration_raw is not None else None
        except Exception:
            bag_duration = None
        camera_threshold_hz = self.expected_camera_fps * self.dropframe_warn_ratio
        cameras_summary: Dict[str, Dict[str, object]] = {}
        for entry in self.camera_streams:
            image_base = str(entry.get("image_base", "")).strip()
            if not image_base:
                continue
            camera_topic = f"{image_base}/compressed" if self.camera_transport == "compressed" else image_base
            info = per_topic.get(camera_topic)
            msg_count = int(info.get("messages", 0)) if isinstance(info, dict) else 0
            bag_hz = (float(msg_count) / bag_duration) if (bag_duration is not None and bag_duration > 0.0) else None

            hz_entry = hz_stats.get(camera_topic) or hz_stats.get(image_base) or {}
            hz_monitor = None
            hz_status = ""
            if isinstance(hz_entry, dict):
                hz_status = str(hz_entry.get("status", ""))
                avg_hz = hz_entry.get("avg_hz")
                if avg_hz is not None:
                    try:
                        hz_monitor = float(avg_hz)
                    except Exception:
                        hz_monitor = None

            status = "OK"
            if msg_count <= 0:
                status = "BAD"
                fatal_reasons.append(f"camera topic has zero messages: {camera_topic}")
            elif bag_hz is not None and bag_hz < camera_threshold_hz:
                status = "BAD"
                fatal_reasons.append(
                    f"camera fps too low: {camera_topic} ({bag_hz:.2f} < {camera_threshold_hz:.2f} Hz)"
                )
            elif hz_monitor is not None and hz_monitor < camera_threshold_hz:
                status = "BAD"
                fatal_reasons.append(
                    f"camera hz monitor too low: {camera_topic} ({hz_monitor:.2f} < {camera_threshold_hz:.2f} Hz)"
                )
            elif hz_status in ("no_messages", "no_log", "unavailable"):
                status = "WARN"
                warnings.append(f"camera hz monitor abnormal: {camera_topic} status={hz_status}")

            cameras_summary[camera_topic] = {
                "message_count": msg_count,
                "bag_hz": bag_hz,
                "hz_monitor": hz_monitor,
                "hz_monitor_topic": camera_topic if camera_topic in hz_stats else image_base,
                "hz_monitor_status": hz_status,
                "threshold_hz": camera_threshold_hz,
                "status": status,
            }

        fatal_reasons = dedup_keep_order([str(item) for item in fatal_reasons])
        warnings = dedup_keep_order([str(item) for item in warnings if str(item).strip()])
        status = "BAD" if fatal_reasons else ("WARN" if warnings else "OK")
        return {
            "available": True,
            "status": status,
            "usable_for_training": len(fatal_reasons) == 0,
            "fatal_reasons": fatal_reasons,
            "warnings": warnings,
            "arms": arms_summary,
            "cameras": cameras_summary,
            "thresholds": {
                "audit_mode": self.health_audit_mode,
                "action_source": self.health_action_source,
                "min_joint_hz": self.health_min_joint_hz,
                "min_action_hz": self.health_min_action_hz,
                "min_state_hz": self.health_min_state_hz,
                "motion_span_warn": self.health_motion_span_warn,
                "repeat_warn_ratio": self.health_repeat_warn_ratio,
                "repeat_tol": self.health_repeat_tol,
                "big_gap_min_sec": self.health_big_gap_min_sec,
                "big_gap_median_mult": self.health_big_gap_median_mult,
                "big_gap_max_count": self.health_big_gap_max_count,
                "teleop_cmd_lag_warn_ms": self.health_teleop_cmd_lag_warn_ms,
                "cmd_state_lag_warn_ms": self.health_cmd_state_lag_warn_ms,
                "teleop_state_lag_warn_ms": self.health_teleop_state_lag_warn_ms,
                "teleop_cmd_mae_warn": self.health_teleop_cmd_mae_warn,
                "cmd_state_mae_warn": self.health_cmd_state_mae_warn,
                "teleop_state_mae_warn": self.health_teleop_state_mae_warn,
                "camera_threshold_hz": camera_threshold_hz,
                "lag_search_ms": float(self.args.health_lag_search_ms),
                "lag_step_ms": float(self.args.health_lag_step_ms),
            },
        }

    def _finalize_episode_job(self, job_id: int, job_data: Dict[str, object]) -> Dict[str, object]:
        self._update_save_job(job_id, state="running", total=1, step=0, stage="wait recorder shutdown", status="RUN")

        episode_name = str(job_data["episode_name"])
        episode_dir = Path(str(job_data["episode_dir"]))
        warnings = list(job_data.get("warnings", []))
        job_data["rosbag_ok"] = self._wait_episode_processes(job_data, warnings)

        bag_files, active_bags, bag_finalize_timeout = self._wait_for_bag_finalize(episode_dir)
        hz_stats: Dict[str, object] = {}
        hz_log_paths = dict(job_data.get("hz_log_paths", {}))
        if bag_finalize_timeout and active_bags:
            active_names = ", ".join([p.name for p in active_bags])
            warnings.append(
                "Timed out waiting for rosbag finalization; active file(s): "
                f"{active_names}"
            )
            if not bag_files:
                bag_files = list(active_bags)
        bag_steps = max(1, len(bag_files))
        hz_steps = max(1, len(hz_log_paths))
        total_steps = 1 + bag_steps + hz_steps + 6
        step = 1
        self._update_save_job(
            job_id,
            total=total_steps,
            step=step,
            stage=f"collect bag files ({len(bag_files)})",
            status="OK" if bag_files else "WARN",
        )

        if bag_files:
            def _on_bag_done(idx: int, total: int, bag_file: Path) -> None:
                nonlocal step
                step += 1
                self._update_save_job(
                    job_id,
                    step=step,
                    stage=f"bag info {idx}/{total}: {bag_file.name}",
                    status="RUN",
                )

            bag_stats = self._aggregate_bag_stats(bag_files, on_bag_done=_on_bag_done)
        else:
            bag_stats = self._aggregate_bag_stats([])
            step += 1
            self._update_save_job(
                job_id,
                step=step,
                stage="bag info skipped (no bag files)",
                status="WARN",
            )

        for warning in bag_stats.get("parse_warnings", []):
            warnings.append(str(warning))
        if not bag_files:
            warnings.append("No bag files found in episode directory.")

        if hz_log_paths:
            total_hz = len(hz_log_paths)
            for idx, (topic, log_path) in enumerate(hz_log_paths.items(), start=1):
                hz_result = self._parse_hz_log(Path(str(log_path)))
                hz_stats[str(topic)] = hz_result
                step += 1
                hz_status = "OK" if hz_result.get("status") == "ok" else "WARN"
                self._update_save_job(
                    job_id,
                    step=step,
                    stage=f"hz {idx}/{total_hz}: {topic}",
                    status=hz_status,
                )
        else:
            step += 1
            self._update_save_job(
                job_id,
                step=step,
                stage="hz logs skipped (none)",
                status="WARN",
            )

        debug_paths = [Path(str(p)) for p in list(job_data.get("debug_log_paths", []))]
        existing_debug_logs = [p for p in debug_paths if p.exists()]
        removed_debug_logs: List[str] = []
        if bool(job_data.get("keep_debug_logs", False)):
            kept_debug_logs = [str(p) for p in existing_debug_logs]
            debug_log_status = "KEPT"
        else:
            removed_debug_logs = self._cleanup_debug_logs(existing_debug_logs, warnings=warnings)
            kept_debug_logs = [str(p) for p in existing_debug_logs if p.exists()]
            debug_log_status = "OK"
        step += 1
        self._update_save_job(job_id, step=step, stage="debug logs", status=debug_log_status)

        if str(job_data.get("camera_transport")) == "compressed":
            per_topic = bag_stats.get("per_topic", {})
            for entry in list(job_data.get("camera_streams", [])):
                if not isinstance(entry, dict):
                    continue
                image_base = entry.get("image_base")
                if not image_base:
                    continue
                compressed_topic = f"{image_base}/compressed"
                info = per_topic.get(compressed_topic)
                if not info or int(info.get("messages", 0)) == 0:
                    warnings.append(f"Compressed stream has zero messages: {compressed_topic}")

        per_topic = bag_stats.get("per_topic", {})
        for topic in self.required_joint_topics:
            info = per_topic.get(topic)
            msg_count = int(info.get("messages", 0)) if isinstance(info, dict) else 0
            if msg_count <= 0:
                warnings.append(f"Required joint topic has zero messages: {topic}")

        drop_frame_topics: List[str] = []
        bag_duration_raw = bag_stats.get("duration_sec")
        try:
            bag_duration = float(bag_duration_raw) if bag_duration_raw is not None else None
        except Exception:
            bag_duration = None
        drop_threshold_hz = self.expected_camera_fps * self.dropframe_warn_ratio
        camera_transport = str(job_data.get("camera_transport"))
        for entry in list(job_data.get("camera_streams", [])):
            if not isinstance(entry, dict):
                continue
            image_base = str(entry.get("image_base", "")).strip()
            if not image_base:
                continue
            camera_topic = f"{image_base}/compressed" if camera_transport == "compressed" else image_base

            info = per_topic.get(camera_topic)
            msg_count = int(info.get("messages", 0)) if isinstance(info, dict) else 0
            bag_hz = (float(msg_count) / bag_duration) if (bag_duration is not None and bag_duration > 0.0) else None

            hz_entry = hz_stats.get(camera_topic) or hz_stats.get(image_base)
            hz_avg = None
            hz_status = ""
            if isinstance(hz_entry, dict):
                hz_status = str(hz_entry.get("status", ""))
                avg_val = hz_entry.get("avg_hz")
                if avg_val is not None:
                    try:
                        hz_avg = float(avg_val)
                    except Exception:
                        hz_avg = None

            low_from_bag = bag_hz is not None and bag_hz < drop_threshold_hz
            low_from_hz = hz_avg is not None and hz_avg < drop_threshold_hz
            if low_from_bag or low_from_hz:
                detail_parts: List[str] = []
                if bag_hz is not None:
                    detail_parts.append(f"bag_hz={bag_hz:.2f}")
                if hz_avg is not None:
                    detail_parts.append(f"hz_monitor={hz_avg:.2f}")
                detail = (
                    f"{camera_topic} ({', '.join(detail_parts)} < 阈值{drop_threshold_hz:.2f}Hz)"
                    if detail_parts
                    else f"{camera_topic} (< 阈值{drop_threshold_hz:.2f}Hz)"
                )
                drop_frame_topics.append(detail)
                warnings.append(f"Possible frame drop detected: {detail}")
            elif hz_status in ("no_messages", "no_log", "unavailable"):
                warnings.append(f"Camera hz monitor abnormal: {camera_topic} status={hz_status}")

        health_check = self._analyze_saved_bag_health(
            bag_files=bag_files,
            recorded_topics=list(job_data.get("recorded_topics", [])),
            bag_stats=bag_stats,
            hz_stats=hz_stats,
        )
        for item in list(health_check.get("warnings", [])):
            warnings.append(str(item))
        warnings = dedup_keep_order([str(item) for item in warnings if str(item).strip()])
        fatal_reasons = dedup_keep_order([str(item) for item in list(health_check.get("fatal_reasons", [])) if str(item).strip()])
        step += 1
        self._update_save_job(
            job_id,
            step=step,
            stage="bag health audit",
            status=str(health_check.get("status", "OK")),
        )

        key_rosparams = self._fetch_rosparams()
        step += 1
        repro_snapshot = self._capture_repro_snapshot(job_data)
        snapshot_status = "OK" if str(repro_snapshot.get("capture_status", "")) == "ok" else "WARN"
        if snapshot_status != "OK":
            warnings.append(
                "Failed to capture full runtime reproducibility snapshot: "
                f"{repro_snapshot.get('capture_error', 'unknown error')}"
            )
        self._update_save_job(job_id, step=step, stage="capture repro snapshot", status=snapshot_status)

        step += 1
        self._update_save_job(job_id, step=step, stage="build metadata", status="OK")

        metadata_path = episode_dir / "metadata.json"
        repro_snapshot_path = episode_dir / "repro_snapshot.json"
        bad_marker_path = episode_dir / "BAD_ROSBAG.txt" if fatal_reasons else None
        if bad_marker_path is not None:
            health_check["bad_marker_path"] = str(bad_marker_path)

        metadata = {
            "schema_version": 2,
            "session_name": job_data.get("session_name"),
            "episode_name": episode_name,
            "episode_index": int(episode_name.split("_")[-1]),
            "episode_dir": str(episode_dir),
            "started_at_utc": job_data.get("started_at_utc"),
            "ended_at_utc": job_data.get("ended_at_utc"),
            "duration_sec": job_data.get("duration_sec"),
            "stop_reason": job_data.get("stop_reason"),
            "recording": {
                "compression": "lz4",
                "split_enabled": bool(job_data.get("split_enabled", False)),
                "split_size_mb": job_data.get("split_size_mb"),
                "camera_transport": job_data.get("camera_transport"),
                "keep_debug_logs": bool(job_data.get("keep_debug_logs", False)),
            },
            "topics": {
                "profile_required": list(job_data.get("required_topics", [])),
                "required_joint_topics": list(self.required_joint_topics),
                "profile_camera_record_topics": list(job_data.get("camera_record_topics", [])),
                "extra_topics": list(job_data.get("extra_topics", [])),
                "recorded_topics": list(job_data.get("recorded_topics", [])),
                "optional_requested": list(job_data.get("optional_requested", [])),
                "optional_recorded": list(job_data.get("optional_recorded", [])),
                "optional_missing": list(job_data.get("optional_missing", [])),
                "camera_health_topics": list(job_data.get("camera_health_topics", [])),
            },
            "preflight": dict(job_data.get("preflight", {})),
            "hz_monitor": {
                "window": job_data.get("hz_window"),
                "topics": hz_stats,
            },
            "bag_stats": bag_stats,
            "environment": {
                "host": socket.gethostname(),
                "user": os.environ.get("USER"),
                "ros_master_uri": os.environ.get("ROS_MASTER_URI"),
                "profile_path": job_data.get("profile_path"),
                "script_args": dict(job_data.get("script_args", {})),
                "key_rosparams": key_rosparams,
            },
            "repro_snapshot_file": str(repro_snapshot_path),
            "repro_snapshot": repro_snapshot,
            "notes": job_data.get("notes", ""),
            "debug_logs": {
                "kept": kept_debug_logs,
                "removed": removed_debug_logs,
            },
            "warnings": warnings,
            "drop_frame_topics": drop_frame_topics,
            "health_check": health_check,
            "rosbag_process_terminated_cleanly": bool(job_data.get("rosbag_ok", False)),
            "save_job_id": job_id,
        }

        repro_snapshot_path.write_text(json.dumps(repro_snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
        self._apply_output_policy(repro_snapshot_path)
        step += 1
        self._update_save_job(job_id, step=step, stage="write repro snapshot", status="OK")

        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        self._apply_output_policy(metadata_path)
        if fatal_reasons:
            bad_marker_lines = [
                f"episode={episode_name}",
                f"metadata={metadata_path}",
                "This rosbag failed integrated action/state/obs health audit.",
                "",
                "Fatal reasons:",
            ]
            bad_marker_lines.extend([f"- {reason}" for reason in fatal_reasons])
            bad_marker_path.write_text("\n".join(bad_marker_lines) + "\n", encoding="utf-8")
            self._apply_output_policy(bad_marker_path)
        self._apply_output_policy(episode_dir, recursive=True, warnings=warnings)
        step += 1
        self._update_save_job(job_id, step=step, stage="write metadata", status="OK")

        warning_count = len(warnings)
        bag_count = int(bag_stats.get("bag_count", 0))
        total_size_mb = float(bag_stats.get("total_size_bytes", 0)) / float(1024 ** 2)
        save_status = "OK"
        if fatal_reasons:
            save_status = "BAD"
        elif warning_count > 0 or bag_count == 0 or not bool(job_data.get("rosbag_ok", False)):
            save_status = "WARN"

        return {
            "episode_name": episode_name,
            "episode_dir": str(episode_dir),
            "metadata_path": str(metadata_path),
            "repro_snapshot_path": str(repro_snapshot_path),
            "bad_marker_path": str(bad_marker_path) if bad_marker_path is not None else "",
            "bag_count": bag_count,
            "total_size_mb": round(total_size_mb, 2),
            "warning_count": warning_count,
            "warnings": warnings,
            "drop_frame_topics": drop_frame_topics,
            "fatal_reason_count": len(fatal_reasons),
            "fatal_reasons": fatal_reasons,
            "health_check_status": str(health_check.get("status", "OK")),
            "usable_for_training": bool(health_check.get("usable_for_training", False)),
            "save_status": save_status,
            "started_at_utc": job_data.get("started_at_utc"),
            "ended_at_utc": job_data.get("ended_at_utc"),
            "duration_sec": job_data.get("duration_sec"),
            "stop_reason": job_data.get("stop_reason"),
        }

    def _reset_current_episode_state(self) -> None:
        self.recording = False
        self.rosbag_proc = None
        self.current_episode_name = None
        self.current_episode_dir = None
        self.current_started_utc = None
        self.current_started_monotonic = None
        self.current_topics = []
        self.current_optional_requested = []
        self.current_optional_recorded = []
        self.current_optional_missing = []
        self.current_preflight = {}
        self.current_warnings = []
        self.debug_log_paths = []

    def _wait_for_save_jobs(self) -> None:
        pending = self._pending_save_count()
        if pending <= 0:
            self._emit_save_progress(force=True)
            return
        self._divider()
        self._print_colored(f"[save] waiting for {pending} pending save job(s)...", color="dim")
        sys.stdout.flush()
        while True:
            self._emit_save_progress(force=False)
            pending = self._pending_save_count()
            if pending <= 0:
                break
            time.sleep(0.2)
        self._emit_save_progress(force=True)
        self._print_colored("[save] all save jobs completed.", color="green")
        sys.stdout.flush()

    def _shutdown_save_executor(self) -> None:
        if self.save_executor_shutdown:
            return
        self.save_executor.shutdown(wait=True)
        self.save_executor_shutdown = True

    def _set_pending_stop(self, reason: str) -> None:
        with self.pending_lock:
            if self.pending_stop_reason is None:
                self.pending_stop_reason = reason

    def _pop_pending_stop(self) -> Optional[str]:
        with self.pending_lock:
            reason = self.pending_stop_reason
            self.pending_stop_reason = None
            return reason

    def _watchdog_loop(self) -> None:
        while not self.shutdown_requested:
            time.sleep(1.0)
            if not self.recording:
                continue

            ok_master, _msg = self._ros_master_ok()
            if not ok_master:
                self._set_pending_stop("ros_master_lost")
                continue

            disk_ok, _free = self._check_disk_ok()
            if not disk_ok:
                self._set_pending_stop("disk_low")
                continue

            if self.rosbag_proc is not None and self.rosbag_proc.poll() is not None:
                self._set_pending_stop("process_error")

    def _start_episode(self) -> None:
        if self.recording:
            return
        if self._pending_save_count():
            self._print_colored("[record] SAVING previous episode; wait for completion, then press SPACE again.", color="yellow")
            self._emit_save_progress(force=True)
            return

        ok_master, master_msg = self._ros_master_ok()
        if not ok_master:
            print(f"[record] ROS master unavailable: {master_msg}")
            return

        disk_ok, free_gb = self._check_disk_ok()
        if not disk_ok:
            print(
                f"[record] Disk free too low: {free_gb:.2f} GB < {self.args.min_free_gb:.2f} GB"
            )
            return

        now = time.monotonic()
        quick_restart = (
            self.last_stop_reason == "user_stop"
            and self.last_stop_monotonic is not None
            and (now - self.last_stop_monotonic) <= self.args.restart_preflight_grace_sec
        )
        preflight_timeout = self.args.restart_preflight_timeout_sec if quick_restart else self.args.wait_timeout_sec
        skip_camera_publishers = bool(quick_restart)
        if quick_restart:
            print(
                f"[record] quick restart preflight enabled "
                f"(timeout={preflight_timeout:.1f}s skip_camera_publishers=true)"
            )
        else:
            print(f"[record] SPACE received, running preflight (timeout={preflight_timeout:.1f}s)")
            sys.stdout.flush()
        ok, preflight = self._wait_preflight(
            timeout_sec=preflight_timeout,
            skip_camera_publishers=skip_camera_publishers,
        )
        if not ok:
            reason = preflight.get("reason", "preflight_failed")
            print(f"[record] Preflight failed: {reason}")
            missing = list(preflight.get("last_missing_presence_topics", []))
            no_publisher = list(preflight.get("last_joint_no_publisher_topics", []))
            unhealthy = list(preflight.get("last_camera_unhealthy_topics", []))
            if missing:
                print(f"[record] preflight missing topics: {missing[:5]}")
            if no_publisher:
                print(f"[record] preflight joint topics without publishers: {no_publisher[:5]}")
            if unhealthy:
                print(f"[record] preflight camera unhealthy: {unhealthy[:5]}")
            return

        idx, ep_name, ep_dir = self._get_next_episode_slot()
        ep_dir.mkdir(parents=True, exist_ok=False)
        self._apply_output_policy(ep_dir)

        self.current_episode_name = ep_name
        self.current_episode_dir = ep_dir
        self.current_started_utc = utc_now_iso()
        self.current_started_monotonic = time.monotonic()
        self.current_preflight = preflight
        self.current_warnings = []
        self.current_optional_requested = (
            list(self.optional_topics)
            if self.record_all_optional_topics
            else list(preflight.get("optional_present", []))
        )
        self.current_optional_recorded = list(preflight.get("optional_present", []))
        self.current_optional_missing = list(preflight.get("optional_missing", []))
        self.current_topics = dedup_keep_order(self.profile_record_topics + self.current_optional_requested)
        self.debug_log_paths = []

        bag_prefix = ep_dir / "episode"
        rosbag_log_path = ep_dir / "rosbag_record.log"
        self.rosbag_log_handle = rosbag_log_path.open("w", encoding="utf-8")
        self._apply_output_policy(rosbag_log_path)
        self.debug_log_paths.append(rosbag_log_path)

        cmd = ["rosbag", "record", "--lz4", "-O", str(bag_prefix)]
        if self.args.split:
            cmd.extend(["--split", "--size", str(self.args.split_size_mb)])
        cmd.extend(self.current_topics)

        self.rosbag_proc = subprocess.Popen(
            cmd,
            stdout=self.rosbag_log_handle,
            stderr=subprocess.STDOUT,
            env=self._build_subprocess_env(),
            preexec_fn=self._subprocess_preexec,
            text=True,
        )

        self._start_hz_monitors(ep_dir)
        self.recording = True
        self.next_episode_index = idx

        self._divider()
        self._print_colored(f"[record] START {ep_name}", color="green+bold")
        print(f"[record] start_utc={self.current_started_utc}")
        print(f"[record] start_local={self._local_now_str()}")
        print(f"[record] episode_path={ep_dir}")
        print(f"[record] bag_prefix={bag_prefix}")
        print(
            f"[record] topics={len(self.current_topics)} "
            f"optional_requested={len(self.current_optional_requested)} "
            f"optional_present={len(self.current_optional_recorded)}"
        )

    def _stop_episode(self, stop_reason: str) -> None:
        if not self.recording:
            return

        episode_name = self.current_episode_name or f"episode_{self.next_episode_index:03d}"
        episode_dir = self.current_episode_dir
        if episode_dir is None:
            return

        start_utc = self.current_started_utc
        stop_local = self._local_now_str()
        ended_utc = utc_now_iso()
        stop_monotonic = time.monotonic()
        stop_color = "green+bold" if stop_reason == "user_stop" else "red+bold"
        self._divider()
        self._print_colored(f"[record] STOP {episode_name} reason={stop_reason}", color=stop_color)
        print(f"[record] episode_path={episode_dir}")

        rosbag_proc = self.rosbag_proc
        rosbag_ok = True
        if rosbag_proc is not None:
            # Stop signal is sent immediately; detailed wait/metadata finalization runs in save worker.
            rosbag_ok = self._request_process_group_stop(rosbag_proc, name="rosbag")

        hz_log_paths: Dict[str, Path] = {}
        hz_monitor_procs: Dict[str, subprocess.Popen] = {}
        for topic, (proc, log_path, fh) in self.hz_monitors.items():
            hz_monitor_procs[str(topic)] = proc
            self._request_process_group_stop(proc, name=f"hz:{topic}")
            try:
                fh.flush()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
            hz_log_paths[str(topic)] = log_path
        self.hz_monitors = {}

        if self.rosbag_log_handle is not None:
            try:
                self.rosbag_log_handle.flush()
            except Exception:
                pass
            try:
                self.rosbag_log_handle.close()
            except Exception:
                pass
            self.rosbag_log_handle = None

        duration_sec = None
        if self.current_started_monotonic is not None:
            duration_sec = stop_monotonic - self.current_started_monotonic

        job_data: Dict[str, object] = {
            "session_name": self.session_name,
            "episode_name": episode_name,
            "episode_dir": str(episode_dir),
            "started_at_utc": start_utc,
            "ended_at_utc": ended_utc,
            "duration_sec": duration_sec,
            "stop_reason": stop_reason,
            "camera_transport": self.camera_transport,
            "camera_streams": copy.deepcopy(self.camera_streams),
            "required_topics": list(self.required_topics),
            "camera_record_topics": list(self.camera_record_topics),
            "extra_topics": list(self.extra_topics),
            "recorded_topics": list(self.current_topics),
            "optional_requested": list(self.current_optional_requested),
            "optional_recorded": list(self.current_optional_recorded),
            "optional_missing": list(self.current_optional_missing),
            "camera_health_topics": list(self.camera_health_topics),
            "preflight": copy.deepcopy(self.current_preflight),
            "hz_window": self.args.hz_window,
            "hz_log_paths": {topic: str(path) for topic, path in hz_log_paths.items()},
            "debug_log_paths": [str(p) for p in self.debug_log_paths],
            "split_enabled": bool(self.args.split),
            "split_size_mb": self.args.split_size_mb if self.args.split else None,
            "keep_debug_logs": bool(self.args.keep_debug_logs),
            "notes": self.args.notes,
            "profile_path": self.profile.get("_profile_path"),
            "script_args": dict(vars(self.args)),
            "warnings": list(self.current_warnings),
            "rosbag_ok": rosbag_ok,
            "rosbag_proc": rosbag_proc,
            "hz_monitor_procs": hz_monitor_procs,
        }

        save_job_id = self._submit_save_job(job_data)

        self.last_stop_monotonic = time.monotonic()
        self.last_stop_reason = stop_reason
        self._reset_current_episode_state()
        self.next_episode_index = self._scan_next_episode_index()

        print(f"[record] start_utc={start_utc}")
        print(f"[record] stop_utc={ended_utc}")
        print(f"[record] stop_local={stop_local}")
        print(f"[record] duration={self._format_duration(duration_sec)}")
        if self.args.wait_for_save:
            self._print_colored(f"[save] SAVING {episode_name}; waiting for file finalization and checks...", color="cyan")
            sys.stdout.flush()
            self._wait_for_episode_save_job(episode_name)
            self._drain_saving_keys()
            if not self.shutdown_requested:
                print(f"[record] READY next=episode_{self.next_episode_index:03d} | SPACE start | Q quit", flush=True)
        else:
            self._print_colored(
                f"[save] queued job={save_job_id} episode={episode_name} (saving in background)",
                color="dim",
            )
            print(f"[record] next=episode_{self.next_episode_index:03d}")
            self._emit_save_progress(force=True)

    def _atexit_cleanup(self) -> None:
        if self.recording:
            self._stop_episode("signal_interrupt")
        self._wait_for_save_jobs()
        self._shutdown_save_executor()

    def _signal_handler(self, _sig, _frame) -> None:
        self.signal_requested = True
        if self.recording:
            self._set_pending_stop("signal_interrupt")
        self.shutdown_requested = True

    def start_external_episode(self) -> Path:
        """Start one episode for a programmatic replay/record orchestrator."""
        self._start_episode()
        if not self.recording or self.current_episode_dir is None:
            raise RuntimeError("Recorder did not enter ACTIVE state; inspect preflight output above.")
        return self.current_episode_dir

    def stop_external_episode(self, stop_reason: str = "user_stop") -> None:
        """Stop the active external episode and queue its normal health audit."""
        if self.recording:
            self._stop_episode(stop_reason)

    def close_external(self, stop_reason: str = "signal_interrupt") -> None:
        """Finalize all external recording work and stop recorder workers."""
        self.shutdown_requested = True
        if self.recording:
            self._stop_episode(stop_reason)
        self._wait_for_save_jobs()
        self._shutdown_save_executor()

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._print_idle_help()

        try:
            with RawTerminal():
                while not self.shutdown_requested:
                    pending = self._pop_pending_stop()
                    if pending and self.recording:
                        self._stop_episode(pending)

                    readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if not readable:
                        self._emit_runtime_status(force=False)
                        continue

                    # Read one byte so Python cannot buffer SPACE through a save.
                    ch = os.read(sys.stdin.fileno(), 1).decode("ascii", errors="ignore")
                    now = time.monotonic()

                    if ch == " ":
                        if now - self.last_space_ts < self.debounce_sec:
                            continue
                        self.last_space_ts = now
                        if self.recording:
                            self._stop_episode("user_stop")
                        else:
                            self._start_episode()
                    elif ch in ("k", "K"):
                        self._delete_last_episode()
                    elif ch in ("q", "Q"):
                        if self.recording:
                            self._stop_episode("user_stop")
                        self.shutdown_requested = True
                    self._emit_runtime_status(force=False)
        finally:
            if self.recording:
                reason = "signal_interrupt" if self.signal_requested else "user_stop"
                self._stop_episode(reason)
            self._wait_for_save_jobs()
            self._shutdown_save_executor()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard-driven ACT rosbag recorder")
    default_profile = str((Path(__file__).resolve().parents[1] / "config/rosbag_profiles/act_rgb_profile.yaml").resolve())
    default_session_root = str((Path(__file__).resolve().parents[2] / "rosbag_data").resolve())
    parser.add_argument(
        "--profile",
        default=default_profile,
        help="Path to YAML topic profile.",
    )
    parser.add_argument(
        "--camera-transport",
        choices=("profile", "compressed", "raw"),
        default="profile",
        help="Override camera transport from profile. Use 'compressed' or 'raw' (default: profile).",
    )
    parser.add_argument(
        "--extra-camera-ns",
        action="append",
        default=[],
        help=(
            "Append an extra camera namespace that publishes <ns>/color/image_raw "
            "and <ns>/color/camera_info. Repeat for multiple extra cameras."
        ),
    )
    parser.add_argument(
        "--extra-topic",
        action="append",
        default=[],
        help="Append an extra ROS topic to record verbatim. Repeat for multiple topics.",
    )
    parser.add_argument(
        "--record-all-optional-topics",
        action="store_true",
        help="Record every optional profile topic even if it is missing during preflight.",
    )
    parser.add_argument(
        "--session-root",
        default=default_session_root,
        help="Root directory for rosbag sessions.",
    )
    parser.add_argument("--prefix", default="act", help="Session name prefix when --session-name is not provided.")
    parser.add_argument("--session-name", default="", help="Optional explicit session directory name.")
    parser.add_argument(
        "--min-episode-index",
        type=int,
        default=1,
        help="Lowest episode number used for new recordings when appending through a staging directory.",
    )
    parser.add_argument("--notes", default="", help="Optional notes written to metadata.")
    parser.add_argument("--wait-timeout-sec", type=float, default=30.0, help="Preflight topic wait timeout.")
    parser.add_argument(
        "--restart-preflight-grace-sec",
        type=float,
        default=8.0,
        help="If a new start happens within this many seconds after user stop, use fast preflight.",
    )
    parser.add_argument(
        "--restart-preflight-timeout-sec",
        type=float,
        default=3.0,
        help="Fast preflight timeout used during quick restart.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Enable rosbag file splitting (default: disabled).",
    )
    parser.add_argument("--split-size-mb", type=int, default=512, help="rosbag split size in MB when --split is used.")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free disk threshold before/while recording.")
    parser.add_argument("--debounce-ms", type=int, default=250, help="SPACE key debounce in milliseconds.")
    parser.add_argument("--hz-window", type=int, default=200, help="Window size used by rostopic hz monitors.")
    parser.add_argument(
        "--record-status-interval-sec",
        type=float,
        default=0.5,
        help="Interval for recording status printing while active.",
    )
    parser.add_argument(
        "--wait-for-save",
        action="store_true",
        help="Wait in the foreground for each episode's save progress and health result before accepting the next episode.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=2,
        help="Number of asynchronous save workers for episode finalization.",
    )
    parser.add_argument(
        "--save-status-interval-sec",
        type=float,
        default=0.5,
        help="Interval for save progress printing while recording.",
    )
    parser.add_argument(
        "--save-progress-mode",
        choices=("compact", "verbose"),
        default="compact",
        help="compact only prints final PASS/ALERT result cards; verbose prints save-stage progress lines.",
    )
    parser.add_argument(
        "--keep-debug-logs",
        action="store_true",
        help="Keep rosbag_record.log and hz_*.log files in episode directory (default: auto-clean).",
    )
    parser.add_argument(
        "--expected-camera-fps",
        type=float,
        default=30.0,
        help="Expected camera fps for drop-frame warning detection.",
    )
    parser.add_argument(
        "--dropframe-warn-ratio",
        type=float,
        default=0.85,
        help="Warn if measured camera fps is lower than expected_fps * ratio.",
    )
    parser.add_argument(
        "--warn-preview-lines",
        type=int,
        default=6,
        help="Number of warning lines previewed in terminal when save status is WARN or BAD.",
    )
    parser.add_argument(
        "--health-audit-mode",
        choices=("recording", "strict"),
        default="recording",
        help=(
            "Health audit policy. 'recording' only checks arm-topic presence/JointState hz/gaps and camera fps; "
            "'strict' also enforces repeat, mux, and lag/mae follow checks."
        ),
    )
    parser.add_argument(
        "--health-action-source",
        choices=("teleop", "cmd"),
        default="teleop",
        help=(
            "Joint stream treated as the action label by the saved-bag health audit. "
            "Use 'cmd' for direct demonstration replay recordings where teleop nodes are stopped."
        ),
    )
    parser.add_argument(
        "--health-expected-mux-source",
        choices=("auto", "teleop", "vla", "robot_cmd", "left_robot_cmd_right_teleop"),
        default="auto",
        help=(
            "Expected value of /robot/arm_*/joint_cmd_mux/selected in strict audit. "
            "Use 'left_robot_cmd_right_teleop' for left-arm intervention; 'robot_cmd' "
            "means both arms use the arbiter; 'auto' preserves the legacy "
            "teleop/cmd-derived expectation."
        ),
    )
    parser.add_argument(
        "--health-require-teleop-state-sync",
        action="store_true",
        help=(
            "In strict audit, enforce leader(teleop)->follower(robot state) lag/MAE limits "
            "even when --health-action-source=cmd. Intended for intervention rollouts."
        ),
    )
    parser.add_argument(
        "--health-min-joint-hz",
        type=float,
        default=80.0,
        help="Bag health audit threshold for teleop/state JointState frequency.",
    )
    parser.add_argument(
        "--health-min-action-hz",
        type=float,
        default=0.0,
        help="Action-stream frequency threshold; <=0 inherits --health-min-joint-hz.",
    )
    parser.add_argument(
        "--health-min-state-hz",
        type=float,
        default=0.0,
        help="Robot-state frequency threshold; <=0 inherits --health-min-joint-hz.",
    )
    parser.add_argument(
        "--health-motion-span-warn",
        type=float,
        default=0.08,
        help="Minimum mean 6-joint motion span before lag/hold checks become active.",
    )
    parser.add_argument(
        "--health-repeat-warn-ratio",
        type=float,
        default=0.65,
        help="Warn when the longest repeated teleop segment exceeds this fraction of the episode.",
    )
    parser.add_argument(
        "--health-repeat-tol",
        type=float,
        default=1e-6,
        help="Consecutive JointState samples within this tolerance are treated as repeats.",
    )
    parser.add_argument(
        "--health-big-gap-min-sec",
        type=float,
        default=0.03,
        help="Absolute lower bound (seconds) for counting a JointState period as a large timestamp gap.",
    )
    parser.add_argument(
        "--health-big-gap-median-mult",
        type=float,
        default=3.0,
        help="Relative large-gap threshold multiplier: threshold=max(min_sec, median_period*mult).",
    )
    parser.add_argument(
        "--health-big-gap-max-count",
        type=int,
        default=0,
        help="Reject the episode only when large-gap count exceeds this value.",
    )
    parser.add_argument(
        "--health-lag-search-ms",
        type=float,
        default=300.0,
        help="Upper bound for action/state lag search in the integrated bag health audit.",
    )
    parser.add_argument(
        "--health-lag-step-ms",
        type=float,
        default=5.0,
        help="Lag-search step size used by the integrated bag health audit.",
    )
    parser.add_argument(
        "--health-teleop-cmd-lag-warn-ms",
        type=float,
        default=80.0,
        help="Warn when teleop->joint_cmd_mux lag exceeds this threshold.",
    )
    parser.add_argument(
        "--health-cmd-state-lag-warn-ms",
        type=float,
        default=170.0,
        help="Warn when joint_cmd_mux->robot_state lag exceeds this threshold.",
    )
    parser.add_argument(
        "--health-teleop-state-lag-warn-ms",
        type=float,
        default=220.0,
        help="Reject the episode when teleop->robot_state lag exceeds this threshold.",
    )
    parser.add_argument(
        "--health-teleop-cmd-mae-warn",
        type=float,
        default=0.08,
        help="Warn when teleop->joint_cmd_mux mean absolute joint error exceeds this value.",
    )
    parser.add_argument(
        "--health-cmd-state-mae-warn",
        type=float,
        default=0.10,
        help="Warn when joint_cmd_mux->robot_state mean absolute joint error exceeds this value.",
    )
    parser.add_argument(
        "--health-teleop-state-mae-warn",
        type=float,
        default=0.12,
        help="Reject the episode when teleop->robot_state mean absolute joint error exceeds this value.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.min_episode_index < 1:
        print("[record] ERROR: --min-episode-index must be >= 1", file=sys.stderr)
        return 2
    try:
        recorder = Recorder(args)
        recorder.run()
        return 0
    except Exception as exc:
        print(f"[record] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
