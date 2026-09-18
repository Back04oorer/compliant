#!/usr/bin/env python3
"""Inspect a checkpoint or run one action-chunk prediction from an NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stereo_patch.runtime import load_stereo_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local directory or Hugging Face repo ID")
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--defm-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--flow-num-steps", type=int)
    parser.add_argument("--observation-npz", type=Path)
    parser.add_argument("--output-npy", type=Path)
    args = parser.parse_args()
    loaded = load_stereo_policy(
        args.checkpoint,
        device=args.device,
        dino_model=args.dino_model,
        defm_root=args.defm_root,
        num_inference_steps=args.num_inference_steps,
        flow_num_steps=args.flow_num_steps,
    )
    print(json.dumps(loaded.contract, indent=2))
    if args.observation_npz is None:
        return 0
    with np.load(args.observation_npz) as archive:
        observation = {key: archive[key] for key in archive.files}
    chunk = loaded.predict_chunk(observation)
    print(f"action_chunk shape={chunk.shape} min={chunk.min():.6g} max={chunk.max():.6g}")
    if args.output_npy:
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_npy, chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
