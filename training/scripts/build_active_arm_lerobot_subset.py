#!/usr/bin/env python3
"""Build an episode subset with only the moving arm in a canonical 1-arm slot.

The source is a bimanual LeRobot v3 dataset with a frame-indexed FastFS mmap.
Selected left-arm episodes use left state/action/wrist data; selected right-arm
episodes use the corresponding right data. Both are exposed with the same
14-D state, 7-D action and observation.right_wrist_image schema so the outputs
can be merged into one single-arm Stereo-ACT dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEPTH_KEY = "observation.fastfs_depth_image"
TOP_KEY = "observation.image"
LEFT_WRIST_KEY = "observation.left_wrist_image"
RIGHT_WRIST_KEY = "observation.right_wrist_image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset LeRobot v3 episodes and canonicalize their active arm."
    )
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--episode-indices",
        required=True,
        help="JSON list of zero-based source episode indices, for example '[0,1,10]'.",
    )
    parser.add_argument("--active-arm", required=True, choices=["left", "right"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the mapping without creating an output dataset.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_episode_indices(raw: str, total_episodes: int) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("--episode-indices must be a non-empty JSON list")
    if any(type(item) is not int for item in value):
        raise ValueError("--episode-indices must contain integers only")
    indices = sorted(value)
    if len(indices) != len(set(indices)):
        raise ValueError("--episode-indices contains duplicates")
    if indices[0] < 0 or indices[-1] >= total_episodes:
        raise ValueError(f"Episode indices out of range 0..{total_episodes - 1}: {indices}")
    return indices


def episode_rows(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(root / "meta" / "episodes")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return frame.sort_values("episode_index").reset_index(drop=True)


def source_contract(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = load_json(info_path)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"Expected LeRobot v3 dataset: {root}")
    if list(info["features"]["observation.state"]["shape"]) != [28]:
        raise ValueError("Source observation.state must be 28-D")
    if list(info["features"]["action"]["shape"]) != [14]:
        raise ValueError("Source action must be 14-D")
    for key in (TOP_KEY, LEFT_WRIST_KEY, RIGHT_WRIST_KEY):
        if key not in info["features"]:
            raise ValueError(f"Missing source video feature: {key}")
    spec = info.get("feature_mmaps", {}).get(DEPTH_KEY)
    if not isinstance(spec, dict):
        raise ValueError(f"Missing source mmap feature: {DEPTH_KEY}")
    shape = [int(item) for item in spec["shape"]]
    if shape != [int(info["total_frames"]), 320, 640] or spec.get("dtype") != "float16":
        raise ValueError(f"Unexpected depth mmap contract: {spec}")
    mmap_path = root / str(spec["path"])
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype("float16").itemsize
    if not mmap_path.is_file() or mmap_path.stat().st_size != expected_bytes:
        raise ValueError(f"Invalid depth mmap: {mmap_path}")
    rows = episode_rows(root)
    if len(rows) != int(info["total_episodes"]):
        raise ValueError("Episode metadata count does not match info.json")
    return info, rows


def make_source_metadata_view(source_root: Path, parent: Path) -> Path:
    """Create a read-only source view with data file indices derived from parquet.

    Some older converted datasets use file-001..file-N on disk while their
    episode metadata says file-000..file-(N-1). LeRobot's delete_episodes
    trusts the metadata. Repair only a temporary metadata copy; never mutate
    the reusable source dataset.
    """
    view_root = Path(tempfile.mkdtemp(prefix=f".{source_root.name}.metadata-view-", dir=parent))
    shutil.copytree(source_root / "meta", view_root / "meta", dirs_exist_ok=True)
    os.symlink(source_root / "data", view_root / "data", target_is_directory=True)
    os.symlink(source_root / "videos", view_root / "videos", target_is_directory=True)

    data_locations: dict[int, tuple[int, int]] = {}
    for path in sorted((source_root / "data").glob("chunk-*/*.parquet")):
        chunk_index = int(path.parent.name.split("-", 1)[1])
        file_index = int(path.stem.split("-", 1)[1])
        values = pd.read_parquet(path, columns=["episode_index"])["episode_index"].unique()
        for value in values:
            episode_index = int(value)
            if episode_index in data_locations:
                raise ValueError(f"Episode spans multiple data files: {episode_index}")
            data_locations[episode_index] = (chunk_index, file_index)

    for path in sorted((view_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(path)
        # Depth lives in a custom mmap and is intentionally absent from
        # LeRobot's ordinary feature table. Stale per-episode depth stats from
        # the pre-mmap dataset confuse the official subset aggregator.
        stale_depth_stats = [
            column for column in frame.columns if str(column).startswith(f"stats/{DEPTH_KEY}/")
        ]
        if stale_depth_stats:
            frame = frame.drop(columns=stale_depth_stats)
        for row_index, episode_index_raw in frame["episode_index"].items():
            episode_index = int(episode_index_raw)
            if episode_index not in data_locations:
                raise ValueError(f"No data parquet found for episode {episode_index}")
            chunk_index, file_index = data_locations[episode_index]
            frame.at[row_index, "data/chunk_index"] = chunk_index
            frame.at[row_index, "data/file_index"] = file_index
        # Older conversion runs stored per-video RGB stats as flat (3,)
        # vectors. Current LeRobot aggregation requires channel-first
        # (3,1,1), matching global image statistics.
        for column in frame.columns:
            name = str(column)
            if not name.startswith("stats/observation.") or name.endswith("/count"):
                continue
            if not any(key in name for key in (TOP_KEY, LEFT_WRIST_KEY, RIGHT_WRIST_KEY)):
                continue

            def normalize_image_stat(value: Any) -> Any:
                array = np.asarray(value)
                if array.shape == (3,):
                    return array.reshape(3, 1, 1).tolist()
                return value

            frame[column] = frame[column].map(normalize_image_stat)
        frame.to_parquet(path, index=False)
    return view_root


def select_vector_cell(value: Any, indices: list[int]) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    selected = array[indices]
    return selected.tolist()


def rewrite_data(root: Path, state_indices: list[int], action_indices: list[int]) -> None:
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(path)
        states = np.asarray(frame["observation.state"].tolist(), dtype=np.float32)
        actions = np.asarray(frame["action"].tolist(), dtype=np.float32)
        frame["observation.state"] = [row.tolist() for row in states[:, state_indices]]
        frame["action"] = [row.tolist() for row in actions[:, action_indices]]
        frame.to_parquet(path, index=False)


def rewrite_episode_metadata(
    root: Path,
    *,
    active_wrist_key: str,
    inactive_wrist_key: str,
    state_indices: list[int],
    action_indices: list[int],
) -> None:
    active_video_prefix = f"videos/{active_wrist_key}/"
    inactive_video_prefix = f"videos/{inactive_wrist_key}/"
    active_stats_prefix = f"stats/{active_wrist_key}/"
    inactive_stats_prefix = f"stats/{inactive_wrist_key}/"
    canonical_video_prefix = f"videos/{RIGHT_WRIST_KEY}/"
    canonical_stats_prefix = f"stats/{RIGHT_WRIST_KEY}/"

    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(path)
        drop = [
            column
            for column in frame.columns
            if str(column).startswith(inactive_video_prefix)
            or str(column).startswith(inactive_stats_prefix)
        ]
        if drop:
            frame = frame.drop(columns=drop)
        for column in list(frame.columns):
            name = str(column)
            if name.startswith("stats/observation.state/") and not name.endswith("/count"):
                frame[column] = frame[column].map(lambda value: select_vector_cell(value, state_indices))
            elif name.startswith("stats/action/") and not name.endswith("/count"):
                frame[column] = frame[column].map(lambda value: select_vector_cell(value, action_indices))
        rename: dict[str, str] = {}
        if active_wrist_key != RIGHT_WRIST_KEY:
            for column in frame.columns:
                name = str(column)
                if name.startswith(active_video_prefix):
                    rename[name] = canonical_video_prefix + name[len(active_video_prefix) :]
                elif name.startswith(active_stats_prefix):
                    rename[name] = canonical_stats_prefix + name[len(active_stats_prefix) :]
        if rename:
            frame = frame.rename(columns=rename)
        frame.to_parquet(path, index=False)


def rewrite_stats(
    root: Path,
    *,
    active_wrist_key: str,
    state_indices: list[int],
    action_indices: list[int],
) -> None:
    path = root / "meta" / "stats.json"
    stats = load_json(path)
    rewritten: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "observation.state":
            rewritten[key] = {
                stat: (select_vector_cell(item, state_indices) if stat != "count" else item)
                for stat, item in value.items()
            }
        elif key == "action":
            rewritten[key] = {
                stat: (select_vector_cell(item, action_indices) if stat != "count" else item)
                for stat, item in value.items()
            }
        elif key == active_wrist_key:
            rewritten[RIGHT_WRIST_KEY] = value
        elif key in (LEFT_WRIST_KEY, RIGHT_WRIST_KEY):
            continue
        else:
            rewritten[key] = value
    write_json(path, rewritten)


def write_subset_mmap(
    source_root: Path,
    output_root: Path,
    source_info: dict[str, Any],
    source_episode_rows: pd.DataFrame,
    keep_indices: list[int],
) -> dict[str, Any]:
    source_spec = source_info["feature_mmaps"][DEPTH_KEY]
    dtype = np.dtype(str(source_spec["dtype"]))
    source_shape = tuple(int(item) for item in source_spec["shape"])
    source = np.memmap(source_root / source_spec["path"], dtype=dtype, mode="r", shape=source_shape)
    by_episode = source_episode_rows.set_index("episode_index", drop=False)
    lengths = [int(by_episode.loc[index]["length"]) for index in keep_indices]
    output_shape = (sum(lengths), *source_shape[1:])
    rel_path = Path("depth_mmaps") / f"{DEPTH_KEY.replace('.', '_')}.{dtype.name}.mmap"
    path = output_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(path, dtype=dtype, mode="w+", shape=output_shape)
    offset = 0
    for episode_index, length in zip(keep_indices, lengths, strict=True):
        row = by_episode.loc[episode_index]
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        if stop - start != length:
            raise ValueError(f"Episode bounds mismatch: episode={episode_index}")
        output[offset : offset + length] = source[start:stop]
        offset += length
    output.flush()
    del output
    return {
        "path": str(rel_path),
        "shape": list(output_shape),
        "dtype": dtype.name,
        "source_feature": source_spec.get("source_feature"),
    }


def remove_inactive_and_canonicalize_wrist(
    root: Path,
    active_wrist_key: str,
    inactive_wrist_key: str,
) -> None:
    videos = root / "videos"
    inactive = videos / inactive_wrist_key
    if inactive.exists():
        shutil.rmtree(inactive)
    active = videos / active_wrist_key
    canonical = videos / RIGHT_WRIST_KEY
    if active_wrist_key != RIGHT_WRIST_KEY:
        if canonical.exists():
            shutil.rmtree(canonical)
        active.rename(canonical)


def rewrite_info(
    root: Path,
    *,
    source_info: dict[str, Any],
    active_wrist_key: str,
    state_indices: list[int],
    action_indices: list[int],
    mmap_spec: dict[str, Any],
) -> None:
    path = root / "meta" / "info.json"
    info = load_json(path)
    source_features = source_info["features"]
    features: dict[str, Any] = {}
    for key, value in info["features"].items():
        if key in ("observation.state", "action", TOP_KEY, LEFT_WRIST_KEY, RIGHT_WRIST_KEY):
            continue
        features[key] = value
    state = dict(source_features["observation.state"])
    action = dict(source_features["action"])
    state["shape"] = [14]
    action["shape"] = [7]
    # Use the right-arm labels as the canonical public slot for both sides.
    state["names"] = [source_features["observation.state"]["names"][index] for index in range(14, 28)]
    action["names"] = [source_features["action"]["names"][index] for index in range(7, 14)]
    features["action"] = action
    features["observation.state"] = state
    features[TOP_KEY] = source_features[TOP_KEY]
    features[RIGHT_WRIST_KEY] = source_features[RIGHT_WRIST_KEY]
    info["features"] = features
    info["feature_mmaps"] = {DEPTH_KEY: mmap_spec}
    info["state_base_dim"] = 14
    info["feature_selection"] = {
        "layout": "single_active_arm_stereo_act",
        "obs_selection": "top,active_wrist,fastfs_depth",
        "obs_keys": [TOP_KEY, RIGHT_WRIST_KEY, DEPTH_KEY],
        "obs_indices": [1, 2, 3],
        "state_selection": "canonical_active_arm_14d",
        "state_indices": list(range(1, 15)),
        "state_base_names": state["names"],
        "action_selection": "canonical_active_arm_7d",
        "action_indices": list(range(1, 8)),
        "action_names": action["names"],
    }
    for key in ("action_source", "state_history_kind", "state_history_size", "actfs"):
        if key in source_info:
            info[key] = source_info[key]
    info["total_videos"] = sum(1 for item in (root / "videos").rglob("*.mp4"))
    info["video_files_size_in_mb"] = sum(
        item.stat().st_size for item in (root / "videos").rglob("*.mp4")
    ) / (1024.0 * 1024.0)
    write_json(path, info)


def validate_output(root: Path, expected_episodes: int) -> None:
    info = load_json(root / "meta" / "info.json")
    if int(info["total_episodes"]) != expected_episodes:
        raise ValueError("Output episode count mismatch")
    if list(info["features"]["observation.state"]["shape"]) != [14]:
        raise ValueError("Output state is not 14-D")
    if list(info["features"]["action"]["shape"]) != [7]:
        raise ValueError("Output action is not 7-D")
    if LEFT_WRIST_KEY in info["features"] or (root / "videos" / LEFT_WRIST_KEY).exists():
        raise ValueError("Inactive/left wrist feature remains in canonical output")
    spec = info["feature_mmaps"][DEPTH_KEY]
    shape = [int(item) for item in spec["shape"]]
    if shape != [int(info["total_frames"]), 320, 640]:
        raise ValueError(f"Output mmap shape mismatch: {shape}")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize
    if (root / spec["path"]).stat().st_size != expected_bytes:
        raise ValueError("Output mmap byte size mismatch")
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(path, columns=["observation.state", "action"])
        if frame.empty:
            continue
        if len(frame.iloc[0]["observation.state"]) != 14 or len(frame.iloc[0]["action"]) != 7:
            raise ValueError(f"Output parquet vector shape mismatch: {path}")
    print(
        f"[validate] canonical active-arm dataset ok: episodes={expected_episodes} "
        f"frames={info['total_frames']} state=14 action=7 depth={shape}"
    )


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_dataset_dir).expanduser().resolve()
    output_parent = Path(args.output_root).expanduser().resolve()
    dataset_id = str(args.dataset_id).strip()
    output_root = output_parent / dataset_id
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not dataset_id:
        raise ValueError("--dataset-id cannot be empty")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    source_info, rows = source_contract(source_root)
    keep_indices = parse_episode_indices(args.episode_indices, int(source_info["total_episodes"]))
    by_episode = rows.set_index("episode_index")
    frame_count = sum(int(by_episode.loc[index]["length"]) for index in keep_indices)
    print(
        f"[active-arm] source={source_root.name} arm={args.active_arm} "
        f"episodes={keep_indices} frames={frame_count}"
    )
    if args.dry_run:
        print("[active-arm] dry-run validation passed")
        return 0

    try:
        from lerobot.datasets.dataset_tools import delete_episodes
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError("The active Python environment must provide lerobot") from exc

    output_parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.building-", dir=output_parent))
    # delete_episodes requires a non-existing destination.
    build_root.rmdir()
    source_view: Path | None = None
    try:
        source_view = make_source_metadata_view(source_root, output_parent)
        dataset = LeRobotDataset(repo_id=source_root.name, root=source_view, download_videos=False)
        keep_set = set(keep_indices)
        drop = [index for index in range(int(source_info["total_episodes"])) if index not in keep_set]
        delete_episodes(dataset, episode_indices=drop, output_dir=build_root, repo_id=dataset_id)

        mmap_spec = write_subset_mmap(source_root, build_root, source_info, rows, keep_indices)
        if args.active_arm == "left":
            state_indices = list(range(0, 14))
            action_indices = list(range(0, 7))
            active_wrist_key, inactive_wrist_key = LEFT_WRIST_KEY, RIGHT_WRIST_KEY
        else:
            state_indices = list(range(14, 28))
            action_indices = list(range(7, 14))
            active_wrist_key, inactive_wrist_key = RIGHT_WRIST_KEY, LEFT_WRIST_KEY

        rewrite_data(build_root, state_indices, action_indices)
        rewrite_episode_metadata(
            build_root,
            active_wrist_key=active_wrist_key,
            inactive_wrist_key=inactive_wrist_key,
            state_indices=state_indices,
            action_indices=action_indices,
        )
        rewrite_stats(
            build_root,
            active_wrist_key=active_wrist_key,
            state_indices=state_indices,
            action_indices=action_indices,
        )
        remove_inactive_and_canonicalize_wrist(build_root, active_wrist_key, inactive_wrist_key)
        rewrite_info(
            build_root,
            source_info=source_info,
            active_wrist_key=active_wrist_key,
            state_indices=state_indices,
            action_indices=action_indices,
            mmap_spec=mmap_spec,
        )
        provenance = {
            "source_dataset_dir": str(source_root),
            "source_dataset_id": source_root.name,
            "source_episode_indices_zero_based": keep_indices,
            "active_arm": args.active_arm,
            "canonical_arm_slot": "right",
            "canonical_wrist_key": RIGHT_WRIST_KEY,
        }
        write_json(build_root / "meta" / "active_arm_subset.json", provenance)
        validate_output(build_root, len(keep_indices))
        build_root.rename(output_root)
    except Exception:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise
    finally:
        if source_view is not None and source_view.exists():
            shutil.rmtree(source_view)
    print(f"[active-arm] output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
