#!/usr/bin/env python3
# ┌──────────────────────────────────────────────────────────┐
# │   lerobot_policy_act_hf/processor_act_hf.py  数据流      │
# └──────────────────────────────────────────────────────────┘
#
#   make_act_hf_pre_post_processors(config, dataset_stats)
#          │
#          ▼
#   make_act_pre_post_processors()  标准 ACT 处理器
#   → preprocessor, postprocessor
#          │
#          ▼  替换 preprocessor 转换函数:
#   preprocessor.to_transition = _batch_to_transition_with_meta
#   preprocessor.to_output     = transition_to_batch
#          │
#          ▼
#   (preprocessor, postprocessor)
#
#   ─────────────────────────────────────────────────────
#   preprocessor 执行流:
#
#   batch dict
#     {obs.*, action, timestamp, frame_index,
#      phase_label, episode_index, ...}
#          │
#          ▼  _batch_to_transition_with_meta()
#   batch_to_transition()  打包 obs/action
#          │
#          ▼  遍历 batch 剩余字段
#   非 obs.* / 非核心字段 → COMPLEMENTARY_DATA
#   (timestamp / frame_index / phase_label 等元数据保留)
#          │
#          ▼  归一化 + 设备对齐
#          │  transition_to_batch()
#          ▼
#   processed batch dict  (含元数据, 供 ACTHFPolicy.forward 使用)
#
#   ─────────────────────────────────────────────────────
#   postprocessor 执行流:
#
#   policy 输出 action tensor
#     → 反归一化 → numpy/tensor

from __future__ import annotations

from typing import Any

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import ACTION, DONE, INFO, OBS_PREFIX, REWARD, TRUNCATED


_CORE_BATCH_KEYS = {ACTION, REWARD, DONE, TRUNCATED, INFO}


def _batch_to_transition_with_meta(batch: dict[str, Any]):
    """Keep non-observation metadata fields (e.g. timestamp/frame_index/phase labels)."""
    transition = batch_to_transition(batch)
    comp_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
    for key, value in batch.items():
        if key in _CORE_BATCH_KEYS:
            continue
        if key.startswith(OBS_PREFIX):
            continue
        if key in comp_data:
            continue
        comp_data[key] = value
    transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
    return transition


def make_act_hf_pre_post_processors(config, dataset_stats=None):
    """Reuse ACT processors for the `act_hf` policy."""
    preprocessor, postprocessor = make_act_pre_post_processors(
        config=config,
        dataset_stats=dataset_stats,
    )
    preprocessor.to_transition = _batch_to_transition_with_meta
    preprocessor.to_output = transition_to_batch
    return preprocessor, postprocessor
