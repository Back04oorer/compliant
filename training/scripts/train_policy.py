#!/usr/bin/env python3
"""Unified Stereo-Patch launcher for ACT, Diffusion, and Flow Matching V2."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

try:  # Installed console entry point.
    from .validate_dataset import validate_contract
except ImportError:  # Direct `python scripts/train_policy.py` execution.
    from validate_dataset import validate_contract


ROOT = Path(__file__).resolve().parents[1]


def _steps(raw: str, total: int) -> list[int]:
    values = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not values or values[-1] != total or any(value <= 0 for value in values):
        raise ValueError("--checkpoint-steps must be positive and include --steps")
    return values


def _required_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.expanduser().is_dir():
        raise FileNotFoundError(f"{label} directory is required: {path}")
    return path.expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-head", choices=["act", "dp", "flow-v2"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-mode", choices=["single", "bimanual"], required=True)
    parser.add_argument("--active-arm", choices=["left", "right"], default="right")
    parser.add_argument("--variant", choices=["fused", "dino-only", "defm-only"], default="fused")
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--defm-root", type=Path)
    parser.add_argument("--defm-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--checkpoint-steps", default="60000,80000")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--target-grid", default="20x40", help="HxW, for example 20x40")
    parser.add_argument("--wrist-grid", default="8x8")
    parser.add_argument("--horizon", type=int, default=32, help="DP/Flow action horizon")
    parser.add_argument("--execute", type=int, default=None, help="Actions queued per replan")
    parser.add_argument("--chunk-size", type=int, default=50, help="ACT action horizon")
    parser.add_argument("--ddim-steps", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--fastfs-valid-iters", type=int, default=8)
    parser.add_argument("--fastfs-scale", type=float, default=0.5161290322580645)
    parser.add_argument("--fastfs-input-width", type=int, default=640)
    parser.add_argument("--fastfs-input-height", type=int, default=320)
    parser.add_argument("--fastfs-max-disp", type=int, default=64)
    parser.add_argument("--fastfs-max-depth-m", type=float, default=2.5)
    parser.add_argument("--resume-config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _grid(raw: str, label: str) -> tuple[int, int]:
    try:
        height, width = (int(value) for value in raw.lower().split("x", 1))
    except Exception as exc:
        raise ValueError(f"{label} must use HxW syntax") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} dimensions must be positive")
    return height, width


def build_command(args: argparse.Namespace) -> list[str]:
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    validate_contract(dataset, args.arm_mode, args.variant, args.active_arm)
    use_rgb = args.variant in {"fused", "dino-only"}
    use_depth = args.variant in {"fused", "defm-only"}
    dino = _required_dir(args.dino_model, "--dino-model") if use_rgb else None
    defm = _required_dir(args.defm_root, "--defm-root") if use_depth else None
    checkpoints = _steps(args.checkpoint_steps, args.steps)
    top_h, top_w = _grid(args.target_grid, "--target-grid")
    wrist_h, wrist_w = _grid(args.wrist_grid, "--wrist-grid")
    policy_type = {"act": "stereo_act", "dp": "stereo_diffusion", "flow-v2": "stereo_flow_v2"}[args.action_head]
    entry = "lerobot_train_stereo_flow_v2.py" if args.action_head == "flow-v2" else "lerobot_train_stereo_act.py"
    execute = args.execute if args.execute is not None else {"act": args.chunk_size, "dp": 8, "flow-v2": 20}[args.action_head]
    total_steps = 1 if args.smoke else args.steps

    command = [
        sys.executable, str(ROOT / "scripts" / entry),
        "--dataset.repo_id", args.dataset_id or dataset.name,
        "--dataset.root", str(dataset),
        "--dataset.video_backend", "torchcodec",
        "--dataset.use_imagenet_stats=false",
        "--output_dir", str(output),
        "--save_checkpoint", str(not args.smoke).lower(),
        "--save_freq", str(math.gcd(*checkpoints)),
        "--checkpoint-steps", ",".join(map(str, checkpoints)),
        "--policy.type", policy_type,
        "--policy.push_to_hub=false", "--policy.device=cuda", "--policy.use_amp=true",
        "--policy.repo_id", f"local/{output.name}",
        "--policy.n_obs_steps=1",
        "--policy.n_action_steps", str(execute),
        "--policy.fusion_mode", "cross_relbias" if args.variant == "fused" else "direct",
        "--policy.use_top_rgb", str(use_rgb).lower(),
        "--policy.use_depth", str(use_depth).lower(),
        "--policy.top_rgb_feature_key=observation.image",
        "--policy.depth_feature_key=observation.fastfs_depth_image",
        "--policy.depth_feature_is_tokens=false",
        "--policy.dino_freeze=true", "--policy.defm_freeze=true",
        "--policy.defm_model_name=defm_vit_s14",
        "--policy.fastfs_valid_iters", str(args.fastfs_valid_iters),
        "--policy.fastfs_scale", str(args.fastfs_scale),
        "--policy.fastfs_input_width", str(args.fastfs_input_width),
        "--policy.fastfs_input_height", str(args.fastfs_input_height),
        "--policy.fastfs_max_disp", str(args.fastfs_max_disp),
        "--policy.fastfs_max_depth_m", str(args.fastfs_max_depth_m),
        "--policy.target_grid_h", str(top_h), "--policy.target_grid_w", str(top_w),
        "--policy.wrist_grid_h", str(wrist_h), "--policy.wrist_grid_w", str(wrist_w),
        "--policy.wrist_encoder=resnet18",
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
        "--policy.optimizer_lr_backbone=1e-5",
        "--policy.fusion_layers=2", "--policy.fusion_heads=8", "--policy.fusion_ffn_dim=2048",
        "--steps", str(total_steps), "--log_freq", "1" if args.smoke else "100",
        "--batch_size", "1" if args.smoke else str(args.batch_size),
        "--num_workers", "0" if args.smoke else str(args.num_workers),
        "--wandb.mode=disabled", "--seed", str(args.seed),
    ]
    if args.action_head == "act":
        command += [
            "--policy.chunk_size", str(args.chunk_size),
            "--policy.n_encoder_layers=4", "--policy.n_decoder_layers=7",
            "--policy.optimizer_lr=1e-5",
        ]
    elif args.action_head == "dp":
        command += [
            "--policy.horizon", str(args.horizon),
            "--policy.num_train_timesteps=100",
            "--policy.num_inference_steps", str(args.ddim_steps),
            "--policy.noise_scheduler_type=DDIM",
            "--policy.down_dims=[128,256,512]",
            "--policy.optimizer_lr=1e-4",
        ]
    else:
        command += [
            "--policy.horizon", str(args.horizon),
            "--policy.flow_num_steps", str(args.flow_steps),
            "--policy.n_encoder_layers=4", "--policy.flow_decoder_layers=4",
            "--policy.optimizer_lr=1e-4",
            "--policy.scheduler_decay_steps", str(args.steps),
        ]
    if dino:
        command += ["--policy.dino_model_id", str(dino)]
    if defm:
        command += ["--policy.defm_root", str(defm)]
    if args.defm_checkpoint:
        command += ["--policy.defm_pretrained_path", str(args.defm_checkpoint.expanduser().resolve())]
    if args.resume_config:
        command += ["--resume=true", f"--config_path={args.resume_config.expanduser().resolve()}"]
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print("[stereo-patch] " + " ".join(command), flush=True)
    if args.dry_run:
        return 0
    args.output_dir.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
