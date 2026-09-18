#!/usr/bin/env python3
"""Shared helpers for conversion-time obs/state/action feature selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OBS_KEY_ORDER = [
    "observation.image",
    "observation.left_wrist_image",
    "observation.right_wrist_image",
]

ACTFS_DEPTH_OBS_KEY = "observation.fastfs_depth_image"
ACTFS_OBS_KEY_ORDER = [
    "observation.image",
    "observation.left_wrist_image",
    "observation.right_wrist_image",
    ACTFS_DEPTH_OBS_KEY,
]

OBS_KEY_LABELS = {
    "observation.image": "top camera",
    "observation.left_wrist_image": "left wrist camera",
    "observation.right_wrist_image": "right wrist camera",
    ACTFS_DEPTH_OBS_KEY: "Fast-FoundationStereo depth image",
}

OBS_KEY_ALIASES = {
    "observation.image": "observation.image",
    "image": "observation.image",
    "top": "observation.image",
    "top_image": "observation.image",
    "front": "observation.image",
    "front_image": "observation.image",
    "observation.left_wrist_image": "observation.left_wrist_image",
    "left_wrist": "observation.left_wrist_image",
    "left": "observation.left_wrist_image",
    "left_image": "observation.left_wrist_image",
    "observation.right_wrist_image": "observation.right_wrist_image",
    "right_wrist": "observation.right_wrist_image",
    "right": "observation.right_wrist_image",
    "right_image": "observation.right_wrist_image",
    ACTFS_DEPTH_OBS_KEY: ACTFS_DEPTH_OBS_KEY,
    "fastfs_depth": ACTFS_DEPTH_OBS_KEY,
    "depth": ACTFS_DEPTH_OBS_KEY,
    "depth_image": ACTFS_DEPTH_OBS_KEY,
    "top_depth": ACTFS_DEPTH_OBS_KEY,
    "fast_foundationstereo_depth": ACTFS_DEPTH_OBS_KEY,
}


def _normalize_layout_name(layout: str) -> str:
    raw = str(layout or "").strip().lower()
    if raw in {"", "auto", "2arm3cam", "2arm", "bimanual", "dualarm", "dual-arm"}:
        return "2arm3cam" if raw != "auto" else "auto"
    if raw in {"3arm3cam", "3arm", "triarm", "tri-arm"}:
        return "3arm3cam"
    raise ValueError(f"Unsupported layout: {layout}")


def joint_label(index: int) -> str:
    if index < 6:
        return f"joint{index + 1}"
    if index == 6:
        return "gripper"
    return f"joint{index + 1}"


def make_joint_names(prefix: str, dim: int, suffix: str) -> list[str]:
    return [f"{prefix}.{joint_label(i)}.{suffix}" for i in range(int(dim))]


def obs_key_display_name(key: str) -> str:
    label = OBS_KEY_LABELS.get(key, key)
    return f"{label} ({key})"


def build_feature_catalog(
    layout: str,
    *,
    pos_dim_per_arm: int = 7,
    effort_dim_per_arm: int = 7,
    opp_effort_dim_per_arm: int | None = None,
    action_dim_per_arm: int = 7,
) -> dict[str, list[str]]:
    resolved_layout = _normalize_layout_name(layout)
    if resolved_layout == "auto":
        raise ValueError("build_feature_catalog requires a concrete layout")

    opp_effort_dim = effort_dim_per_arm if opp_effort_dim_per_arm is None else int(opp_effort_dim_per_arm)
    state_names = (
        make_joint_names("left", pos_dim_per_arm, "pos")
        + make_joint_names("left", effort_dim_per_arm, "effort_raw")
        + make_joint_names("left", pos_dim_per_arm, "vel")
        + make_joint_names("right", pos_dim_per_arm, "pos")
        + make_joint_names("right", effort_dim_per_arm, "effort_raw")
        + make_joint_names("right", pos_dim_per_arm, "vel")
    )
    action_names = (
        make_joint_names("left", action_dim_per_arm, "pos")
        + make_joint_names("right", action_dim_per_arm, "pos")
    )
    if resolved_layout == "3arm3cam":
        state_names.extend(make_joint_names("opp", pos_dim_per_arm, "pos"))
        state_names.extend(make_joint_names("opp", opp_effort_dim, "effort_raw"))
        state_names.extend(make_joint_names("opp", pos_dim_per_arm, "vel"))
        action_names.extend(make_joint_names("opp", action_dim_per_arm, "pos"))

    return {
        "obs_keys": list(OBS_KEY_ORDER),
        "state_names": state_names,
        "action_names": action_names,
    }


def format_selection_spec_from_indices(indices: Sequence[int], total_count: int) -> str:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index in seen:
            continue
        normalized.append(index)
        seen.add(index)
    if normalized == list(range(int(total_count))):
        return "all"
    return ",".join(str(index + 1) for index in normalized)


def _alias_index_map(items: Sequence[str], alias_map: Mapping[str, str] | None) -> dict[str, int]:
    index_by_key = {str(item): idx for idx, item in enumerate(items)}
    out: dict[str, int] = {}
    for key, idx in index_by_key.items():
        out[key.lower()] = idx
    if alias_map:
        for raw_alias, canonical in alias_map.items():
            canonical_idx = index_by_key.get(str(canonical))
            if canonical_idx is None:
                continue
            out[str(raw_alias).strip().lower()] = canonical_idx
    return out


def resolve_feature_selection(
    spec: Any,
    items: Sequence[str],
    *,
    allow_empty: bool,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[list[int], list[str], str]:
    values = list(items)
    if not values:
        return [], [], "none"

    raw_text = str(spec or "").strip()
    if not raw_text or raw_text.lower() in {"all", "*"}:
        indices = list(range(len(values)))
        return indices, list(values), "all"
    if raw_text.lower() in {"none", "0"}:
        if not allow_empty:
            raise ValueError("Selection cannot be empty")
        return [], [], "none"

    alias_index = _alias_index_map(values, alias_map)
    chosen: list[int] = []
    seen: set[int] = set()
    tokens = [token.strip() for token in raw_text.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Selection is empty")

    for token in tokens:
        if "-" in token:
            left, right = token.split("-", 1)
            if left.strip().isdigit() and right.strip().isdigit():
                start = int(left)
                end = int(right)
                if start > end:
                    start, end = end, start
                for one_based in range(start, end + 1):
                    if one_based < 1 or one_based > len(values):
                        raise ValueError(f"Index out of range: {one_based} (1..{len(values)})")
                    zero_based = one_based - 1
                    if zero_based not in seen:
                        chosen.append(zero_based)
                        seen.add(zero_based)
                continue
        if token.isdigit():
            one_based = int(token)
            if one_based < 1 or one_based > len(values):
                raise ValueError(f"Index out of range: {one_based} (1..{len(values)})")
            zero_based = one_based - 1
            if zero_based not in seen:
                chosen.append(zero_based)
                seen.add(zero_based)
            continue

        matched_index = alias_index.get(token.lower())
        if matched_index is None:
            raise ValueError(f"Unknown selection token: {token}")
        if matched_index not in seen:
            chosen.append(matched_index)
            seen.add(matched_index)

    if not chosen and not allow_empty:
        raise ValueError("Selection cannot be empty")

    normalized_spec = format_selection_spec_from_indices(chosen, len(values)) if chosen else "none"
    return chosen, [values[index] for index in chosen], normalized_spec
