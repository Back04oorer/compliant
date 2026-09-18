#!/usr/bin/env python3
# ┌─────────────────────────────────────────────────────────────┐
# │      lerobot_policy_act_hf/modeling_act_hf.py  数据流       │
# └─────────────────────────────────────────────────────────────┘
#
#   batch {
#     observation.state (B,28)        当前关节状态
#     observation.image (B,3,H,W)     顶部相机
#     observation.left/right_wrist_image
#     action         (B,T,14)         训练标签
#     action_is_pad  (B,T)  可选      填充掩码
#     b_chunk/phase_label   可选      相位标签
#     episode_index/timestamp 可选    用于 CSV 标签
#   }
#          │
#          ▼
#   ACTPolicy.model(batch)   (基类 Transformer 骨干)
#   → actions_hat (B,T,14)
#   → (mu_hat, log_sigma_x2_hat)  [VAE模式]
#          │
#          ├─ _valid_action_mask()  →  valid (B,T,1)
#          │
#          ├─ _build_phase_labels()
#          │   ├─ 从 batch 读取预计算标签 (b_chunk 等)
#          │   ├─ 或从 CSV burst intervals 按时间戳计算
#          │   └─ 默认全零 (无加权)
#          │   → phase (B,T,1) ∈ [0,1]
#          │
#          ├─ _split_action_dims()
#          │   → a_cont (连续关节)  a_grip (夹爪)
#          │
#          ▼
#   ┌──────────────────────────────────────────────────────┐
#   │  损失计算 (每样本独立)                               │
#   │                                                      │
#   │  base_cont = MSE/L1(pred,gt)                        │
#   │              × (1 + alpha_fast × phase)             │
#   │  base_grip = MSE/L1(pred_grip, gt_grip)             │
#   │                                                      │
#   │  vel_ps  = ||(dp_pred - dp_gt)||²                   │
#   │            × (1 + alpha_fast × vel_phase)           │
#   │  acc_ps  = ||(ddp_pred - ddp_gt)||²                 │
#   │            × (1 + alpha_fast × acc_phase)           │
#   │  dct_ps  = freq_weight × (C_pred - C_gt)²          │
#   │            (仅全有效帧序列, DCT-II 频域)             │
#   │  kld_ps  = KL散度  [VAE模式]                        │
#   └──────────────────────────┬───────────────────────────┘
#                              │
#                              ▼
#   total = base + λ_vel·vel + λ_acc·acc + λ_dct·dct [+β·kld]
#   loss  = total.mean()
#                              │
#                              ▼
#   (loss, loss_dict)  → 训练框架反向传播 + 日志

from __future__ import annotations

import csv
import math
import re
from collections import deque
from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .configuration_act_hf import ACTHFConfig


class _DCTBasisCache:
    """Small cache for DCT-II basis matrices indexed by (T, device, dtype)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str, str], Tensor] = {}

    def get(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (T, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        n = torch.arange(T, device=device, dtype=dtype).view(1, T)
        k = torch.arange(T, device=device, dtype=dtype).view(T, 1)
        basis = torch.cos((math.pi / T) * (n + 0.5) * k)
        self._cache[key] = basis
        return basis


class _BurstIntervalCache:
    """Load and cache per-episode burst intervals from CSV files once."""

    def __init__(self, csv_dir: str, csv_glob: str, episode_regex: str) -> None:
        self.csv_dir = csv_dir.strip()
        self.csv_glob = csv_glob.strip() or "*.csv"
        self._regex = re.compile(episode_regex) if episode_regex.strip() else None
        self._loaded = False
        self._episode_to_intervals: dict[int, list[tuple[float, float]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.csv_dir)

    def _parse_episode_index(self, path: Path) -> int | None:
        if self._regex is not None:
            m = self._regex.search(path.stem)
            if m is not None:
                try:
                    return int(m.group(1))
                except (TypeError, ValueError):
                    return None
        if path.stem.strip().lower() == "episode":
            return 0
        if path.stem.isdigit():
            return int(path.stem)
        return None

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _read_intervals(self, path: Path) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [x.strip().lower() for x in (reader.fieldnames or []) if x]
            key_map = {k: k for k in fieldnames}
            start_key = (
                key_map.get("start_s")
                or key_map.get("start")
                or key_map.get("burst_start_s")
                or key_map.get("on_s")
            )
            end_key = (
                key_map.get("end_s")
                or key_map.get("end")
                or key_map.get("burst_end_s")
                or key_map.get("off_s")
            )
            if start_key is not None and end_key is not None:
                for row in reader:
                    row_l = {str(k).strip().lower(): v for k, v in row.items()}
                    start_s = self._to_float(row_l.get(start_key))
                    end_s = self._to_float(row_l.get(end_key))
                    if start_s is None or end_s is None:
                        continue
                    if end_s < start_s:
                        start_s, end_s = end_s, start_s
                    intervals.append((start_s, end_s))
                return intervals

        with path.open("r", encoding="utf-8", newline="") as f:
            reader2 = csv.reader(f)
            for row in reader2:
                if len(row) < 2:
                    continue
                start_s = self._to_float(row[0])
                end_s = self._to_float(row[1])
                if start_s is None or end_s is None:
                    continue
                if end_s < start_s:
                    start_s, end_s = end_s, start_s
                intervals.append((start_s, end_s))
        return intervals

    @staticmethod
    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        merged: list[tuple[float, float]] = []
        for start_s, end_s in sorted(intervals, key=lambda x: x[0]):
            if not merged:
                merged.append((start_s, end_s))
                continue
            prev_start, prev_end = merged[-1]
            if start_s <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end_s))
            else:
                merged.append((start_s, end_s))
        return merged

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            return

        root = Path(self.csv_dir).expanduser()
        if not root.is_dir():
            return

        for csv_path in sorted(root.rglob(self.csv_glob)):
            ep_idx = self._parse_episode_index(csv_path)
            if ep_idx is None:
                continue
            intervals = self._read_intervals(csv_path)
            if not intervals:
                continue
            self._episode_to_intervals.setdefault(ep_idx, []).extend(intervals)

        for ep_idx, intervals in list(self._episode_to_intervals.items()):
            self._episode_to_intervals[ep_idx] = self._merge_intervals(intervals)

    def get(self, episode_index: int) -> list[tuple[float, float]]:
        self._ensure_loaded()
        return self._episode_to_intervals.get(episode_index, [])


_DCT_CACHE = _DCTBasisCache()


class ACTHFPolicy(ACTPolicy):
    """ACT policy with phase-aware training losses for bursty motion."""

    config_class = ACTHFConfig
    name = "act_hf"

    def __init__(self, config: ACTHFConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if not hasattr(self, "_action_queue"):
            self._action_queue = deque([], maxlen=max(1, int(self.config.n_action_steps)))

        action_dim = 0
        if config.action_feature is not None and config.action_feature.shape:
            action_dim = int(config.action_feature.shape[0])

        self._gripper_idx = self._parse_gripper_indices(config.gripper_indices, action_dim)
        grip_set = set(self._gripper_idx)
        self._cont_idx = [idx for idx in range(action_dim) if idx not in grip_set]
        self._phase_label_keys = [x.strip() for x in config.phase_label_keys.split(",") if x.strip()]
        self._burst_cache = _BurstIntervalCache(
            csv_dir=config.burst_csv_dir,
            csv_glob=config.burst_csv_glob,
            episode_regex=config.burst_episode_regex,
        )

    def _parse_gripper_indices(self, raw: str, action_dim: int) -> list[int]:
        if not raw:
            return []
        parsed: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            idx = int(token)
            if idx < 0 or idx >= action_dim:
                raise ValueError(
                    f"gripper_indices contains out-of-range index {idx} for action dim {action_dim}"
                )
            if idx not in parsed:
                parsed.append(idx)
        return sorted(parsed)

    def _split_action_dims(self, a: Tensor) -> tuple[Tensor | None, Tensor | None]:
        a_cont = a[..., self._cont_idx] if self._cont_idx else None
        a_grip = a[..., self._gripper_idx] if self._gripper_idx else None
        return a_cont, a_grip

    def _valid_action_mask(self, batch: dict[str, Tensor], action: Tensor) -> Tensor:
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            return torch.ones((*action.shape[:2], 1), device=action.device, dtype=action.dtype)
        valid = (~is_pad) if is_pad.dtype == torch.bool else (is_pad <= 0)
        return valid.unsqueeze(-1).to(dtype=action.dtype)

    def _per_sample_weighted_mean(self, value: Tensor, weight: Tensor) -> Tensor:
        numerator = (value * weight).sum(dim=(1, 2))
        denom = weight.sum(dim=(1, 2)).clamp_min(self.config.eps)
        return numerator / denom

    def _broadcast_weight(self, value: Tensor, weight: Tensor) -> Tensor:
        if weight.shape == value.shape:
            return weight
        return weight.expand_as(value)

    def _masked_mean(self, value: Tensor, mask: Tensor) -> Tensor:
        mask_b = self._broadcast_weight(value, mask)
        numerator = (value * mask_b).sum()
        denom = mask_b.sum().clamp_min(self.config.eps)
        return numerator / denom

    def _as_bt1(
        self, value: Tensor, B: int, T: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        t = torch.as_tensor(value, device=device)
        if t.dtype == torch.bool:
            t = t.to(dtype=dtype)
        else:
            t = t.to(dtype=dtype)

        if t.ndim == 1:
            if t.shape[0] == T:
                t = t.unsqueeze(0).unsqueeze(-1)
            elif t.shape[0] == B:
                t = t.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1)
            else:
                return None
        elif t.ndim == 2:
            if t.shape == (B, T):
                t = t.unsqueeze(-1)
            elif t.shape == (T, 1) and B == 1:
                t = t.unsqueeze(0)
            elif t.shape == (B, 1):
                t = t.unsqueeze(1).expand(B, T, 1)
            else:
                return None
        elif t.ndim == 3:
            if t.shape == (B, T, 1):
                pass
            elif t.shape == (B, 1, T):
                t = t.transpose(1, 2)
            else:
                return None
        else:
            return None

        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, T, 1)
        if t.shape != (B, T, 1):
            return None
        return t.clamp(0.0, 1.0)

    def _phase_from_batch(self, batch: dict[str, Tensor], B: int, T: int, device, dtype) -> Tensor | None:
        for key in self._phase_label_keys:
            if key not in batch:
                continue
            parsed = self._as_bt1(batch[key], B, T, device, dtype)
            if parsed is not None:
                return parsed
        return None

    def _phase_from_csv_intervals(
        self, batch: dict[str, Tensor], B: int, T: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        if not self._burst_cache.enabled:
            return None
        if "episode_index" not in batch:
            return None

        if "timestamp" in batch:
            base_time = torch.as_tensor(batch["timestamp"], device=device, dtype=dtype).reshape(-1)
        elif "frame_index" in batch:
            frame_index = torch.as_tensor(batch["frame_index"], device=device, dtype=dtype).reshape(-1)
            base_time = frame_index / float(self.config.phase_hz)
        else:
            return None

        episode_index = torch.as_tensor(batch["episode_index"], device=device).reshape(-1)
        if base_time.numel() == 1 and B > 1:
            base_time = base_time.expand(B)
        if episode_index.numel() == 1 and B > 1:
            episode_index = episode_index.expand(B)
        if base_time.numel() != B or episode_index.numel() != B:
            return None

        dt = 1.0 / float(self.config.phase_hz)
        t_offsets = torch.arange(T, device=device, dtype=dtype) * dt
        labels = torch.zeros((B, T, 1), device=device, dtype=dtype)

        for b in range(B):
            ep_idx = int(round(float(episode_index[b].item())))
            intervals = self._burst_cache.get(ep_idx)
            if not intervals:
                continue
            ts = base_time[b] + t_offsets
            mask = torch.zeros(T, dtype=torch.bool, device=device)
            for start_s, end_s in intervals:
                mask |= (ts >= float(start_s)) & (ts <= float(end_s))
            labels[b, :, 0] = mask.to(dtype=dtype)
        return labels

    def _build_phase_labels(self, batch: dict[str, Tensor], action: Tensor, valid_mask: Tensor) -> Tensor:
        B, T, _ = action.shape
        device = action.device
        dtype = action.dtype
        phase = self._phase_from_batch(batch, B, T, device, dtype)
        if phase is None:
            phase = self._phase_from_csv_intervals(batch, B, T, device, dtype)
        if phase is None:
            phase = torch.zeros((B, T, 1), device=device, dtype=dtype)
        phase = phase.clamp(0.0, 1.0)
        return phase * valid_mask

    def _dct_time(self, a: Tensor) -> Tensor:
        _, T, _ = a.shape
        basis = _DCT_CACHE.get(T, a.device, a.dtype)
        return torch.einsum("kt,btd->bkd", basis, a)

    def _dct_loss_per_sample(self, a_pred: Tensor, a_gt: Tensor, valid_mask: Tensor) -> Tensor:
        B, T, _ = a_pred.shape
        out = torch.zeros(B, device=a_pred.device, dtype=a_pred.dtype)
        if T < 2:
            return out
        full_valid = (valid_mask.squeeze(-1) > 0.5).all(dim=1)
        if not torch.any(full_valid):
            return out

        cp = self._dct_time(a_pred[full_valid])
        cg = self._dct_time(a_gt[full_valid])

        k = torch.arange(T, device=a_pred.device, dtype=a_pred.dtype)
        denom = max(T - 1, 1)
        freq_weight = (k / denom).pow(self.config.dct_p)
        if self.config.dct_hi_start > 0:
            freq_weight[: self.config.dct_hi_start] = 0.0
        freq_weight = freq_weight.view(1, T, 1)

        out[full_valid] = (freq_weight * (cp - cg).pow(2)).mean(dim=(1, 2))
        return out

    def _derivative_losses_per_sample(
        self, a_pred: Tensor, a_gt: Tensor, valid_mask: Tensor, phase: Tensor
    ) -> tuple[Tensor, Tensor]:
        B, T, _ = a_pred.shape
        zero = torch.zeros(B, device=a_pred.device, dtype=a_pred.dtype)
        if T < 2:
            return zero, zero

        valid_bt = valid_mask.squeeze(-1) > 0.5
        phase_bt = phase.squeeze(-1).clamp(0.0, 1.0)

        dp = a_pred[:, 1:] - a_pred[:, :-1]
        dg = a_gt[:, 1:] - a_gt[:, :-1]
        vel_phase = torch.maximum(phase_bt[:, 1:], phase_bt[:, :-1]).unsqueeze(-1)
        vel_valid = (valid_bt[:, 1:] & valid_bt[:, :-1]).unsqueeze(-1).to(dtype=a_pred.dtype)
        vel_weight = vel_valid * (1.0 + self.config.alpha_fast * vel_phase)
        vel_weight = self._broadcast_weight(dp, vel_weight)
        vel_ps = self._per_sample_weighted_mean((dp - dg).pow(2), vel_weight)

        if T < 3:
            return vel_ps, zero

        ddp = dp[:, 1:] - dp[:, :-1]
        ddg = dg[:, 1:] - dg[:, :-1]
        acc_phase = torch.maximum(
            torch.maximum(phase_bt[:, 2:], phase_bt[:, 1:-1]), phase_bt[:, :-2]
        ).unsqueeze(-1)
        acc_valid = (valid_bt[:, 2:] & valid_bt[:, 1:-1] & valid_bt[:, :-2]).unsqueeze(-1).to(
            dtype=a_pred.dtype
        )
        acc_weight = acc_valid * (1.0 + self.config.alpha_fast * acc_phase)
        acc_weight = self._broadcast_weight(ddp, acc_weight)
        acc_ps = self._per_sample_weighted_mean((ddp - ddg).pow(2), acc_weight)
        return vel_ps, acc_ps

    def _base_error(self, pred: Tensor, gt: Tensor) -> Tensor:
        if self.config.base_loss == "l1":
            return torch.abs(pred - gt)
        return (pred - gt).pow(2)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        actions_gt = batch[ACTION]
        valid_mask = self._valid_action_mask(batch, actions_gt)
        phase = self._build_phase_labels(batch, actions_gt, valid_mask)

        pred_cont, pred_grip = self._split_action_dims(actions_hat)
        gt_cont, gt_grip = self._split_action_dims(actions_gt)

        B = actions_hat.shape[0]
        zero = torch.zeros(B, device=actions_hat.device, dtype=actions_hat.dtype)

        base_cont_ps = zero
        base_grip_ps = zero
        vel_ps = zero
        acc_ps = zero
        dct_ps = zero
        base_fast = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)
        base_slow = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)
        vel_fast = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)
        vel_slow = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)
        acc_fast = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)
        acc_slow = torch.zeros((), device=actions_hat.device, dtype=actions_hat.dtype)

        valid_sum = valid_mask.sum().clamp_min(self.config.eps)
        fast_ratio = phase.sum() / valid_sum
        weighted_ratio = torch.ones((), device=actions_hat.device, dtype=actions_hat.dtype)

        if pred_cont is not None:
            cont_phase_bin = (phase > 0.5).to(dtype=actions_hat.dtype)
            cont_fast_mask = valid_mask * cont_phase_bin
            cont_slow_mask = valid_mask * (1.0 - cont_phase_bin)

            cont_err = self._base_error(pred_cont, gt_cont)
            base_fast = self._masked_mean(cont_err, cont_fast_mask)
            base_slow = self._masked_mean(cont_err, cont_slow_mask)

            cont_weight_bt1 = valid_mask * (1.0 + self.config.alpha_fast * phase)
            weighted_ratio = cont_weight_bt1.sum() / valid_sum

            cont_weight = self._broadcast_weight(pred_cont, cont_weight_bt1)
            base_cont_ps = self._per_sample_weighted_mean(
                cont_err,
                cont_weight,
            )
            vel_ps, acc_ps = self._derivative_losses_per_sample(pred_cont, gt_cont, valid_mask, phase)

            T = pred_cont.shape[1]
            valid_bt = valid_mask.squeeze(-1) > 0.5
            phase_bt = phase.squeeze(-1).clamp(0.0, 1.0)
            if T >= 2:
                dp = pred_cont[:, 1:] - pred_cont[:, :-1]
                dg = gt_cont[:, 1:] - gt_cont[:, :-1]
                vel_err = (dp - dg).pow(2)
                vel_phase = torch.maximum(phase_bt[:, 1:], phase_bt[:, :-1]).unsqueeze(-1)
                vel_valid = (valid_bt[:, 1:] & valid_bt[:, :-1]).unsqueeze(-1).to(dtype=pred_cont.dtype)
                vel_fast_mask = vel_valid * (vel_phase > 0.5).to(dtype=pred_cont.dtype)
                vel_slow_mask = vel_valid * (vel_phase <= 0.5).to(dtype=pred_cont.dtype)
                vel_fast = self._masked_mean(vel_err, vel_fast_mask)
                vel_slow = self._masked_mean(vel_err, vel_slow_mask)

            if T >= 3:
                ddp = dp[:, 1:] - dp[:, :-1]
                ddg = dg[:, 1:] - dg[:, :-1]
                acc_err = (ddp - ddg).pow(2)
                acc_phase = torch.maximum(
                    torch.maximum(phase_bt[:, 2:], phase_bt[:, 1:-1]), phase_bt[:, :-2]
                ).unsqueeze(-1)
                acc_valid = (valid_bt[:, 2:] & valid_bt[:, 1:-1] & valid_bt[:, :-2]).unsqueeze(-1).to(
                    dtype=pred_cont.dtype
                )
                acc_fast_mask = acc_valid * (acc_phase > 0.5).to(dtype=pred_cont.dtype)
                acc_slow_mask = acc_valid * (acc_phase <= 0.5).to(dtype=pred_cont.dtype)
                acc_fast = self._masked_mean(acc_err, acc_fast_mask)
                acc_slow = self._masked_mean(acc_err, acc_slow_mask)

            if self.config.lambda_dct > 0.0:
                dct_ps = self._dct_loss_per_sample(pred_cont, gt_cont, valid_mask)

        if pred_grip is not None:
            grip_weight = self._broadcast_weight(pred_grip, valid_mask)
            base_grip_ps = self._per_sample_weighted_mean(
                self._base_error(pred_grip, gt_grip),
                grip_weight,
            )

        base_ps = base_cont_ps + base_grip_ps
        total_ps = (
            base_ps
            + self.config.lambda_vel * vel_ps
            + self.config.lambda_acc * acc_ps
            + self.config.lambda_dct * dct_ps
        )

        loss_dict = {
            "base_cont": float(base_cont_ps.mean().item()),
            "base_grip": float(base_grip_ps.mean().item()),
            "base_total": float(base_ps.mean().item()),
            "vel_loss": float(vel_ps.mean().item()),
            "acc_loss": float(acc_ps.mean().item()),
            "dct_loss": float(dct_ps.mean().item()),
            "base_loss_fast": float(base_fast.item()),
            "base_loss_slow": float(base_slow.item()),
            "vel_loss_fast": float(vel_fast.item()),
            "vel_loss_slow": float(vel_slow.item()),
            "acc_loss_fast": float(acc_fast.item()),
            "acc_loss_slow": float(acc_slow.item()),
            "fast_ratio": float(fast_ratio.item()),
            "weighted_ratio": float(weighted_ratio.item()),
            "phase_fast_ratio": float(fast_ratio.item()),
        }

        if self.config.use_vae:
            if mu_hat is None or log_sigma_x2_hat is None:
                raise RuntimeError("VAE is enabled but latent outputs are missing.")
            kld_ps = -0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
            kld_ps = kld_ps.sum(-1)
            total_ps = total_ps + self.config.kl_weight * kld_ps
            loss_dict["kld_loss"] = float(kld_ps.mean().item())

        total_mean = float(total_ps.mean().item())
        loss_dict["total"] = total_mean
        # Backward compatible logging keys.
        loss_dict["base_loss"] = loss_dict["base_total"]
        loss_dict["base_cont_loss"] = loss_dict["base_cont"]
        loss_dict["base_grip_loss"] = loss_dict["base_grip"]

        if reduction == "none":
            return total_ps, loss_dict
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")

        loss = total_ps.mean()
        loss_dict["loss"] = float(loss.item())
        return loss, loss_dict
