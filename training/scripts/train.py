#!/usr/bin/env python3
"""Stable Stereo-ACT launcher for single-arm and bimanual datasets."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from validate_dataset import validate_contract


ROOT = Path(__file__).resolve().parents[1]


def parse_steps(raw: str, total_steps: int) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[-1] != total_steps or any(value <= 0 for value in values):
        raise ValueError("--checkpoint-steps must be positive and include --steps")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="", help="Defaults to dataset-root directory name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-mode", choices=["single", "bimanual"], required=True)
    parser.add_argument(
        "--active-arm",
        choices=["left", "right"],
        default="right",
        help="Provenance only. A prepared single-arm dataset is already canonicalized.",
    )
    parser.add_argument(
        "--variant", choices=["fused", "dino-only", "defm-only"], required=True
    )
    parser.add_argument("--dino-model", type=Path, default=None)
    parser.add_argument("--defm-root", type=Path, default=None)
    parser.add_argument("--defm-checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--checkpoint-steps", default="40000,60000")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fastfs-valid-iters", type=int, default=8)
    parser.add_argument("--fastfs-scale", type=float, default=0.5161290322580645)
    parser.add_argument("--fastfs-input-width", type=int, default=640)
    parser.add_argument("--fastfs-input-height", type=int, default=320)
    parser.add_argument("--fastfs-max-disp", type=int, default=64)
    parser.add_argument("--fastfs-max-depth-m", type=float, default=2.5)
    parser.add_argument("--target-grid-height", type=int, default=20)
    parser.add_argument("--target-grid-width", type=int, default=40)
    parser.add_argument("--wrist-grid-height", type=int, default=8)
    parser.add_argument("--wrist-grid-width", type=int, default=8)
    parser.add_argument("--resume-config", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one real dataloader + forward + backward + optimizer step without checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print command only")
    return parser.parse_args()


def require_path(path: Path | None, label: str, directory: bool = True) -> Path:
    if path is None:
        raise ValueError(f"{label} is required for this variant")
    path = path.expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_command(args: argparse.Namespace) -> list[str]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_contract(dataset_root, args.arm_mode, args.variant, args.active_arm)

    use_top_rgb = args.variant in {"fused", "dino-only"}
    use_depth = args.variant in {"fused", "defm-only"}
    fusion_mode = "cross_relbias" if args.variant == "fused" else "direct"
    dino_model = require_path(args.dino_model, "--dino-model") if use_top_rgb else None
    defm_root = require_path(args.defm_root, "--defm-root") if use_depth else None
    checkpoint_steps = parse_steps(args.checkpoint_steps, args.steps)
    save_frequency = math.gcd(*checkpoint_steps)
    dataset_id = args.dataset_id or dataset_root.name
    steps = 1 if args.smoke else args.steps
    batch_size = 1 if args.smoke else args.batch_size
    workers = 0 if args.smoke else args.num_workers

    command = [
        sys.executable,
        str(ROOT / "scripts" / "lerobot_train_stereo_act.py"),
        "--dataset.repo_id",
        dataset_id,
        "--dataset.root",
        str(dataset_root),
        "--dataset.video_backend",
        "torchcodec",
        "--dataset.use_imagenet_stats=false",
        "--output_dir",
        str(output_dir),
        "--save_checkpoint",
        str(not args.smoke).lower(),
        "--save_freq",
        str(save_frequency),
        "--checkpoint-steps",
        ",".join(map(str, checkpoint_steps)),
        "--policy.type",
        "stereo_act",
        "--policy.push_to_hub=false",
        "--policy.device",
        "cuda",
        "--policy.repo_id",
        f"local/{output_dir.name}",
        "--policy.use_amp=true",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size",
        str(args.chunk_size),
        "--policy.n_action_steps",
        str(args.chunk_size),
        "--policy.vision_backbone=resnet18",
        "--policy.wrist_encoder=resnet18",
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
        "--policy.optimizer_lr=1e-5",
        "--policy.optimizer_lr_backbone=1e-5",
        "--policy.n_encoder_layers=4",
        "--policy.n_decoder_layers=7",
        "--policy.fusion_mode",
        fusion_mode,
        "--policy.use_top_rgb",
        str(use_top_rgb).lower(),
        "--policy.use_depth",
        str(use_depth).lower(),
        "--policy.fastfs_valid_iters",
        str(args.fastfs_valid_iters),
        "--policy.fastfs_scale",
        str(args.fastfs_scale),
        "--policy.fastfs_input_width",
        str(args.fastfs_input_width),
        "--policy.fastfs_input_height",
        str(args.fastfs_input_height),
        "--policy.fastfs_max_disp",
        str(args.fastfs_max_disp),
        "--policy.fastfs_max_depth_m",
        str(args.fastfs_max_depth_m),
        "--policy.top_rgb_feature_key=observation.image",
        "--policy.depth_feature_key=observation.fastfs_depth_image",
        "--policy.depth_feature_is_tokens=false",
        "--policy.dino_freeze=true",
        "--policy.defm_model_name=defm_vit_s14",
        "--policy.defm_freeze=true",
        "--policy.target_grid_h",
        str(args.target_grid_height),
        "--policy.target_grid_w",
        str(args.target_grid_width),
        "--policy.wrist_grid_h",
        str(args.wrist_grid_height),
        "--policy.wrist_grid_w",
        str(args.wrist_grid_width),
        "--policy.fusion_layers=2",
        "--policy.fusion_heads=8",
        "--policy.fusion_ffn_dim=2048",
        "--steps",
        str(steps),
        "--log_freq",
        "1" if args.smoke else "100",
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(workers),
        "--wandb.mode=disabled",
        "--seed",
        str(args.seed),
    ]
    if dino_model is not None:
        command.extend(["--policy.dino_model_id", str(dino_model)])
    if defm_root is not None:
        command.extend(["--policy.defm_root", str(defm_root)])
    if args.defm_checkpoint is not None:
        checkpoint = require_path(args.defm_checkpoint, "--defm-checkpoint", directory=False)
        command.extend(["--policy.defm_pretrained_path", str(checkpoint)])
    if args.resume_config is not None:
        resume = require_path(args.resume_config, "--resume-config", directory=False)
        command.extend(["--resume=true", f"--config_path={resume}"])
    return command


def main() -> int:
    args = parse_args()
    positive_values = {
        "steps": args.steps,
        "batch-size": args.batch_size,
        "chunk-size": args.chunk_size,
        "fastfs-valid-iters": args.fastfs_valid_iters,
        "fastfs-scale": args.fastfs_scale,
        "fastfs-input-width": args.fastfs_input_width,
        "fastfs-input-height": args.fastfs_input_height,
        "fastfs-max-disp": args.fastfs_max_disp,
        "fastfs-max-depth-m": args.fastfs_max_depth_m,
        "target-grid-height": args.target_grid_height,
        "target-grid-width": args.target_grid_width,
        "wrist-grid-height": args.wrist_grid_height,
        "wrist-grid-width": args.wrist_grid_width,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    command = build_command(args)
    print("[stereo-act] " + " ".join(command), flush=True)
    if args.dry_run:
        return 0
    # LeRobot owns output directory creation and refuses to overwrite an
    # existing directory. Only create its parent here.
    args.output_dir.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
