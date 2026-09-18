"""Shared pre/post-processor selection used by training and deployment."""

from __future__ import annotations

from typing import Any, Callable

from lerobot.processor.converters import transition_to_batch

from lerobot_policy_act_hf.processor_act_hf import (
    _batch_to_transition_with_meta,
    make_act_hf_pre_post_processors,
)


def without_metric_depth_stats(policy_cfg: Any, dataset_stats: Any) -> Any:
    """Keep metric depth in metres instead of applying image statistics."""
    if not isinstance(dataset_stats, dict):
        return dataset_stats
    depth_key = getattr(policy_cfg, "depth_feature_key", "observation.fastfs_depth_image")
    if depth_key not in dataset_stats:
        return dataset_stats
    filtered = dict(dataset_stats)
    filtered.pop(depth_key, None)
    return filtered


def make_stereo_pre_post_processors(
    policy_cfg: Any,
    *,
    default_factory: Callable[..., Any],
    pretrained_path: str | None = None,
    **kwargs: Any,
):
    """Select the checkpoint-compatible processor pair for every action head."""
    policy_type = getattr(policy_cfg, "type", None)
    if policy_type not in {"stereo_act", "stereo_diffusion", "stereo_flow", "stereo_flow_v2"}:
        return default_factory(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    stats = without_metric_depth_stats(policy_cfg, kwargs.get("dataset_stats"))
    if policy_type != "stereo_act":
        return default_factory(
            policy_cfg,
            pretrained_path=pretrained_path,
            **{**kwargs, "dataset_stats": stats},
        )

    if pretrained_path:
        preprocessor, postprocessor = default_factory(
            policy_cfg,
            pretrained_path=pretrained_path,
            **kwargs,
        )
        # ACT-HF checkpoints store the correct pipeline steps, but the runtime
        # conversion functions are Python callables and are not serialized.
        preprocessor.to_transition = _batch_to_transition_with_meta
        preprocessor.to_output = transition_to_batch
        return preprocessor, postprocessor

    return make_act_hf_pre_post_processors(config=policy_cfg, dataset_stats=stats)
