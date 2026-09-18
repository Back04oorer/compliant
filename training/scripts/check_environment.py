#!/usr/bin/env python3
"""Check local training/conversion dependencies before a long Stereo-ACT run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"[OK] {label}: {path} sha256={sha256(path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--defm-root", type=Path, required=True)
    parser.add_argument("--defm-checkpoint", type=Path, default=None)
    parser.add_argument("--fastfs-root", type=Path, default=None)
    parser.add_argument("--fastfs-checkpoint", type=Path, default=None)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--processing-metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        raise RuntimeError(f"Expected Python 3.10-3.12, got {sys.version}")

    for module_name in (
        "torch",
        "torchvision",
        "lerobot",
        "transformers",
        "einops",
        "numpy",
        "pandas",
        "pyarrow",
        "cv2",
        "rosbags",
        "omegaconf",
    ):
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] import {module_name}: {version}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Stereo-ACT/FastFS training requires a CUDA GPU")
    print(f"[OK] CUDA: torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    dino = args.dino_model.expanduser().resolve()
    require_file(dino / "config.json", "DINO config")
    require_file(dino / "model.safetensors", "DINO weights")

    defm = args.defm_root.expanduser().resolve()
    require_file(defm / "defm" / "model_factory.py", "DeFM source")
    if args.defm_checkpoint is not None:
        require_file(args.defm_checkpoint.expanduser().resolve(), "DeFM weights")

    if (args.fastfs_root is None) != (args.fastfs_checkpoint is None):
        raise ValueError("Provide both --fastfs-root and --fastfs-checkpoint, or neither")
    if args.fastfs_root is not None:
        fastfs = args.fastfs_root.expanduser().resolve()
        require_file(fastfs / "core" / "utils" / "utils.py", "FastFS source")
        require_file(args.fastfs_checkpoint.expanduser().resolve(), "FastFS weights")

    if (args.calibration_json is None) != (args.processing_metadata is None):
        raise ValueError("Provide calibration and processing metadata together")
    if args.calibration_json is not None:
        calibration = args.calibration_json.expanduser().resolve()
        processing = args.processing_metadata.expanduser().resolve()
        require_file(calibration, "stereo calibration")
        require_file(processing, "rectification/crop metadata")
        calibration_data = json.loads(calibration.read_text(encoding="utf-8"))
        processing_data = json.loads(processing.read_text(encoding="utf-8"))
        for key in ("image_width", "image_height", "K_left", "D_left", "K_right", "D_right", "R1", "R2"):
            if key not in calibration_data:
                raise ValueError(f"Calibration is missing {key}")
        for key in ("rectified_image_size_per_eye", "crop", "rectification"):
            if key not in processing_data:
                raise ValueError(f"Processing metadata is missing {key}")
        for key in ("P1", "P2"):
            if key not in processing_data["rectification"]:
                raise ValueError(f"Processing metadata rectification is missing {key}")
        print("[OK] calibration/rectification schema")

    print("[PASS] Stereo-ACT environment preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
