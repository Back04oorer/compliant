from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from conversion_feature_selection import build_feature_catalog, resolve_feature_selection
from validate_dataset import validate_contract
from train import build_command
from train_policy import build_command as build_unified_command


def make_dataset(root: Path, *, arm_mode: str, depth: bool = True, top: bool = True) -> Path:
    state_dim, action_dim = (14, 7) if arm_mode == "single" else (28, 14)
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
        "observation.right_wrist_image": {
            "dtype": "video",
            "shape": [3, 320, 640],
        },
    }
    if top:
        features["observation.image"] = {"dtype": "video", "shape": [3, 320, 640]}
    if arm_mode == "bimanual":
        features["observation.left_wrist_image"] = {
            "dtype": "video",
            "shape": [3, 320, 640],
        }
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 3,
        "features": features,
    }
    if depth:
        mmap_path = root / "depth_mmaps" / "depth.float16.mmap"
        mmap_path.parent.mkdir(parents=True)
        np.memmap(mmap_path, mode="w+", dtype=np.float16, shape=(3, 2, 4)).flush()
        info["feature_mmaps"] = {
            "observation.fastfs_depth_image": {
                "path": "depth_mmaps/depth.float16.mmap",
                "shape": [3, 2, 4],
                "dtype": "float16",
            }
        }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("arm_mode", "state_dim", "action_dim"),
    [("single", 14, 7), ("bimanual", 28, 14)],
)
def test_valid_contract(tmp_path: Path, arm_mode: str, state_dim: int, action_dim: int):
    summary = validate_contract(make_dataset(tmp_path, arm_mode=arm_mode), arm_mode, "fused", "left")
    assert summary["state_dim"] == state_dim
    assert summary["action_dim"] == action_dim


def test_dino_only_does_not_require_depth(tmp_path: Path):
    validate_contract(make_dataset(tmp_path, arm_mode="single", depth=False), "single", "dino-only", "right")


def test_fused_rejects_missing_depth(tmp_path: Path):
    with pytest.raises(ValueError, match="Missing mmap"):
        validate_contract(make_dataset(tmp_path, arm_mode="single", depth=False), "single", "fused", "right")


def test_defm_only_does_not_require_top_rgb(tmp_path: Path):
    validate_contract(
        make_dataset(tmp_path, arm_mode="single", top=False),
        "single",
        "defm-only",
        "right",
    )


def test_bimanual_feature_selection_matches_28d_contract():
    catalog = build_feature_catalog("2arm3cam")
    indices, names, normalized = resolve_feature_selection(
        "1-14,22-35", catalog["state_names"], allow_empty=False
    )
    assert len(indices) == 28
    assert len(names) == 28
    assert normalized == ",".join(map(str, [*range(1, 15), *range(22, 36)]))


def test_hardware_fastfs_contract_is_forwarded_to_checkpoint_config(tmp_path: Path):
    dataset = make_dataset(tmp_path / "dataset", arm_mode="single")
    dino = tmp_path / "dino"
    defm = tmp_path / "defm"
    dino.mkdir()
    defm.mkdir()
    args = Namespace(
        dataset_root=dataset,
        dataset_id="",
        output_dir=tmp_path / "output",
        arm_mode="single",
        active_arm="right",
        variant="fused",
        dino_model=dino,
        defm_root=defm,
        defm_checkpoint=None,
        steps=60_000,
        checkpoint_steps="40000,60000",
        batch_size=16,
        num_workers=8,
        chunk_size=50,
        seed=1000,
        fastfs_valid_iters=6,
        fastfs_scale=0.75,
        fastfs_input_width=960,
        fastfs_input_height=480,
        fastfs_max_disp=128,
        fastfs_max_depth_m=3.5,
        target_grid_height=20,
        target_grid_width=40,
        wrist_grid_height=8,
        wrist_grid_width=8,
        resume_config=None,
        smoke=False,
        dry_run=True,
    )
    command = build_command(args)
    joined = " ".join(command)
    assert "--policy.fastfs_valid_iters 6" in joined
    assert "--policy.fastfs_scale 0.75" in joined
    assert "--policy.fastfs_input_width 960" in joined
    assert "--policy.fastfs_input_height 480" in joined
    assert "--policy.fastfs_max_disp 128" in joined
    assert "--policy.fastfs_max_depth_m 3.5" in joined


@pytest.mark.parametrize(
    ("head", "policy_type"),
    [("act", "stereo_act"), ("dp", "stereo_diffusion"), ("flow-v2", "stereo_flow_v2")],
)
def test_unified_launcher_selects_action_head(tmp_path: Path, head: str, policy_type: str):
    dataset = make_dataset(tmp_path / "dataset", arm_mode="bimanual")
    dino, defm = tmp_path / "dino", tmp_path / "defm"
    dino.mkdir()
    defm.mkdir()
    args = Namespace(
        action_head=head,
        dataset_root=dataset,
        dataset_id="",
        output_dir=tmp_path / "output",
        arm_mode="bimanual",
        active_arm="right",
        variant="fused",
        dino_model=dino,
        defm_root=defm,
        defm_checkpoint=None,
        steps=80_000,
        checkpoint_steps="60000,80000",
        batch_size=16,
        num_workers=8,
        seed=1000,
        target_grid="20x40",
        wrist_grid="8x8",
        horizon=32,
        execute=None,
        chunk_size=50,
        ddim_steps=16,
        flow_steps=10,
        fastfs_valid_iters=8,
        fastfs_scale=0.5161290322580645,
        fastfs_input_width=640,
        fastfs_input_height=320,
        fastfs_max_disp=64,
        fastfs_max_depth_m=2.5,
        resume_config=None,
        smoke=False,
        dry_run=True,
    )
    command = build_unified_command(args)
    assert command[command.index("--policy.type") + 1] == policy_type
    assert command[command.index("--policy.fastfs_input_width") + 1] == "640"
