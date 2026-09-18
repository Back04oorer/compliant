#!/usr/bin/env python3
"""Portable LeRobot RGB ACT entry. Stereo-ACT uses training/scripts/train_policy.py."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(args):
    info = json.loads((args.dataset / "meta/info.json").read_text())
    f = info["features"]
    dims = (14, 7) if args.arm_mode == "single" else (28, 14)
    if f["observation.state"]["shape"] != [dims[0]] or f["action"]["shape"] != [dims[1]]:
        raise ValueError("Unexpected state/action dimensions; convert/subset first")
    keys = ["observation.image", "observation.right_wrist_image"]
    if args.arm_mode == "bimanual":
        keys.append("observation.left_wrist_image")
    inputs = {"observation.state": {"type": "STATE", "shape": [dims[0]]}}
    for key in keys:
        feature = f[key]
        shape = list(feature["shape"])
        names = feature.get("names")
        if names == ["height", "width", "channels"] or (len(shape) == 3 and shape[-1] == 3):
            shape = [shape[-1], shape[0], shape[1]]
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Unexpected RGB shape for {key}: {shape}")
        inputs[key] = {"type": "VISUAL", "shape": shape}
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("Steps and batch size must be positive")
    return [sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false",
        "--dataset.repo_id=" + args.dataset.name, "--dataset.root=" + str(args.dataset.resolve()),
        "--dataset.video_backend=torchcodec", "--output_dir=" + str(args.output.resolve()),
        "--policy.input_features=" + json.dumps(inputs),
        "--policy.output_features=" + json.dumps({"action": {"type": "ACTION", "shape": [dims[1]]}}),
        "--policy.chunk_size=50", "--policy.n_action_steps=50",
        "--policy.optimizer_lr=1e-5", "--policy.optimizer_lr_backbone=1e-5",
        "--steps=" + str(1 if args.smoke else args.steps),
        "--batch_size=" + str(1 if args.smoke else args.batch_size),
        "--num_workers=" + str(0 if args.smoke else 8), "--save_freq=20000",
        "--save_checkpoint=" + str(not args.smoke).lower(), "--wandb.mode=disabled", "--seed=1000"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm-mode", choices=["single", "bimanual"], required=True)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    return 0 if args.dry_run else subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
