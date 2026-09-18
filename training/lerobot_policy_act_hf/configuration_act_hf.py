#!/usr/bin/env python3
# ┌──────────────────────────────────────────────────────────────┐
# │   lerobot_policy_act_hf/configuration_act_hf.py  数据流      │
# └──────────────────────────────────────────────────────────────┘
#
#   ACTHFConfig(**kwargs)
#          │
#          ▼  dataclass 字段赋值
#   ┌──────────────────────────────────────────────────────────┐
#   │  损失权重:  alpha_fast / lambda_vel / lambda_acc         │
#   │             lambda_dct / dct_hi_start / dct_p            │
#   │  基础损失:  base_loss  ("mse" | "l1")                    │
#   │  夹爪索引:  gripper_indices  "6,13"                      │
#   │  相位标签:  phase_label_keys (逗号分隔 batch key 名)     │
#   │  CSV标签:   burst_csv_dir / glob / regex / phase_hz      │
#   │  遗留别名:  hf_vel_lambda / hf_acc_lambda / ...          │
#   └──────────────────────────┬───────────────────────────────┘
#                              │
#                              ▼  __post_init__()
#   super().__post_init__()   (ACTConfig 基类校验)
#                              │
#                              ▼
#   _apply_legacy_aliases()
#   hf_vel_lambda  → lambda_vel   (仅当规范字段为默认值时)
#   hf_burst_alpha → alpha_fast   等
#                              │
#                              ▼
#   范围校验: alpha_fast≥0 / lambda_*≥0 / eps>0
#   枚举校验: base_loss ∈ {"mse","l1"}
#                              │
#                              ▼
#   合法的 ACTHFConfig 实例
#   → 传入 ACTHFPolicy / 序列化为 config.json

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


_DEFAULT_ALPHA_FAST = 2.0
_DEFAULT_LAMBDA_VEL = 0.02
_DEFAULT_LAMBDA_ACC = 0.01
_DEFAULT_LAMBDA_DCT = 0.0
_DEFAULT_DCT_HI_START = 0
_DEFAULT_DCT_P = 2.0
_DEFAULT_BASE_LOSS = "mse"
_DEFAULT_GRIPPER_INDICES = "6,13"
_DEFAULT_EPS = 1e-6


@PreTrainedConfig.register_subclass("act_hf")
@dataclass
class ACTHFConfig(ACTConfig):
    """ACT config with phase-aware burst losses for continuous actions."""

    # Canonical knobs (preferred).
    alpha_fast: float = _DEFAULT_ALPHA_FAST
    lambda_vel: float = _DEFAULT_LAMBDA_VEL
    lambda_acc: float = _DEFAULT_LAMBDA_ACC
    lambda_dct: float = _DEFAULT_LAMBDA_DCT
    dct_hi_start: int = _DEFAULT_DCT_HI_START
    dct_p: float = _DEFAULT_DCT_P

    base_loss: str = _DEFAULT_BASE_LOSS  # choices: "mse" | "l1"
    gripper_indices: str = _DEFAULT_GRIPPER_INDICES
    eps: float = _DEFAULT_EPS

    # Burst label integration.
    # If labels already exist in batch, the first matching key is used.
    phase_label_keys: str = "b_chunk,phase_label,burst_label,fast_label"
    # Optional: load per-episode burst intervals from CSV and convert to b_t online.
    burst_csv_dir: str = ""
    burst_csv_glob: str = "*.csv"
    burst_episode_regex: str = r"episode[_-]?(\d+)"
    phase_hz: float = 30.0

    # Legacy knobs (for backward compatibility with existing configs/checkpoints).
    hf_dct_lambda: float | None = None
    hf_vel_lambda: float | None = None
    hf_acc_lambda: float | None = None
    hf_dct_p: float | None = None
    hf_dct_hi_start: int | None = None
    hf_use_burst_weight: bool | None = None
    hf_burst_alpha: float | None = None
    hf_gripper_indices: str | None = None
    hf_base_loss: str | None = None
    hf_eps: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_legacy_aliases()

        if self.alpha_fast < 0:
            raise ValueError("alpha_fast must be >= 0")
        if self.lambda_vel < 0:
            raise ValueError("lambda_vel must be >= 0")
        if self.lambda_acc < 0:
            raise ValueError("lambda_acc must be >= 0")
        if self.lambda_dct < 0:
            raise ValueError("lambda_dct must be >= 0")
        if self.dct_p < 0:
            raise ValueError("dct_p must be >= 0")
        if self.dct_hi_start < 0:
            raise ValueError("dct_hi_start must be >= 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.base_loss not in {"mse", "l1"}:
            raise ValueError("base_loss must be one of: mse, l1")
        if self.phase_hz <= 0:
            raise ValueError("phase_hz must be > 0")

    def _apply_legacy_aliases(self) -> None:
        # Old lambda names.
        if self.hf_vel_lambda is not None and self.lambda_vel == _DEFAULT_LAMBDA_VEL:
            self.lambda_vel = self.hf_vel_lambda
        if self.hf_acc_lambda is not None and self.lambda_acc == _DEFAULT_LAMBDA_ACC:
            self.lambda_acc = self.hf_acc_lambda
        if self.hf_dct_lambda is not None and self.lambda_dct == _DEFAULT_LAMBDA_DCT:
            self.lambda_dct = self.hf_dct_lambda
        if self.hf_dct_p is not None and self.dct_p == _DEFAULT_DCT_P:
            self.dct_p = self.hf_dct_p
        if self.hf_dct_hi_start is not None and self.dct_hi_start == _DEFAULT_DCT_HI_START:
            self.dct_hi_start = self.hf_dct_hi_start

        # Old burst weighting knobs.
        if self.hf_burst_alpha is not None and self.alpha_fast == _DEFAULT_ALPHA_FAST:
            self.alpha_fast = self.hf_burst_alpha
        if self.hf_use_burst_weight is False and self.alpha_fast == _DEFAULT_ALPHA_FAST:
            self.alpha_fast = 0.0

        # Old misc names.
        if self.hf_gripper_indices is not None and self.gripper_indices == _DEFAULT_GRIPPER_INDICES:
            self.gripper_indices = self.hf_gripper_indices
        if self.hf_base_loss is not None and self.base_loss == _DEFAULT_BASE_LOSS:
            self.base_loss = self.hf_base_loss
        if self.hf_eps is not None and self.eps == _DEFAULT_EPS:
            self.eps = self.hf_eps
