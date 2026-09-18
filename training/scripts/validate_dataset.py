#!/usr/bin/env python3
"""Validate the Stereo-ACT LeRobot v3 data contract before GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEPTH_KEY = "observation.fastfs_depth_image"
TOP_KEY = "observation.image"
LEFT_WRIST_KEY = "observation.left_wrist_image"
RIGHT_WRIST_KEY = "observation.right_wrist_image"


def load_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    dataset_root: Path, arm_mode: str, variant: str, active_arm: str
) -> dict[str, Any]:
    info = load_info(dataset_root)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("Stereo-ACT expects a LeRobot v3 dataset")
    features = info.get("features") or {}
    state_dim, action_dim = (14, 7) if arm_mode == "single" else (28, 14)
    if list(features.get("observation.state", {}).get("shape", [])) != [state_dim]:
        raise ValueError(f"observation.state must be [{state_dim}] for arm_mode={arm_mode}")
    if list(features.get("action", {}).get("shape", [])) != [action_dim]:
        raise ValueError(f"action must be [{action_dim}] for arm_mode={arm_mode}")
    if variant in {"fused", "dino-only"} and TOP_KEY not in features:
        raise ValueError(f"Missing top RGB feature: {TOP_KEY}")

    expected_wrists = (
        [RIGHT_WRIST_KEY]
        if arm_mode == "single"
        else [LEFT_WRIST_KEY, RIGHT_WRIST_KEY]
    )
    missing_wrists = [key for key in expected_wrists if key not in features]
    if missing_wrists:
        suffix = (
            " Single-arm datasets use the canonical right-wrist slot even "
            f"when active_arm={active_arm}."
            if arm_mode == "single"
            else ""
        )
        raise ValueError(f"Missing wrist feature(s): {missing_wrists}.{suffix}")

    if variant in {"fused", "defm-only"}:
        spec = (info.get("feature_mmaps") or {}).get(DEPTH_KEY)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing mmap feature: {DEPTH_KEY}")
        shape = [int(value) for value in spec.get("shape", [])]
        total_frames = int(info.get("total_frames", -1))
        if len(shape) != 3 or shape[0] != total_frames:
            raise ValueError(f"Invalid depth mmap shape: {shape}")
        if str(spec.get("dtype")) not in {"float16", "float32"}:
            raise ValueError(f"Depth mmap must be float16/float32, got {spec.get('dtype')}")
        mmap_path = dataset_root / str(spec.get("path", ""))
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize
        if not mmap_path.is_file() or mmap_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Depth mmap size mismatch: {mmap_path} expected_bytes={expected_bytes}"
            )
    return {
        "episodes": int(info.get("total_episodes", -1)),
        "frames": int(info.get("total_frames", -1)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "wrists": expected_wrists,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--arm-mode", choices=["single", "bimanual"], required=True)
    parser.add_argument("--active-arm", choices=["left", "right"], default="right")
    parser.add_argument(
        "--variant", choices=["fused", "dino-only", "defm-only"], required=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_contract(
        args.dataset_root.expanduser().resolve(), args.arm_mode, args.variant, args.active_arm
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
