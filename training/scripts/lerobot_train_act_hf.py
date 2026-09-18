#!/usr/bin/env python3
# ┌──────────────────────────────────────────────────────────┐
# │           lerobot_train_act_hf.py  数据流                │
# └──────────────────────────────────────────────────────────┘
#
#   sys.argv  (CLI 参数)
#          │
#          ▼
#   _rewrite_alias_args()
#   --alpha-fast → --policy.alpha_fast  等简写展开
#          │
#          ▼
#   import lerobot_policy_act_hf
#   → 注册 ACTHFConfig / ACTHFPolicy 到 LeRobot 工厂
#          │
#          ▼
#   patch policy_factory.make_pre_post_processors
#          │
#          ├─ policy.type == "act_hf" + pretrained_path?
#          │   └─ 复用原处理器 + 替换 to_transition (保留元数据)
#          ├─ policy.type == "act_hf" 新建?
#          │   └─ make_act_hf_pre_post_processors()
#          └─ 其他类型 → 原始 LeRobot 处理路径
#          │
#          ▼
#   lerobot_train_main()  (LeRobot 原生训练主循环)
#          │
#          ▼  每个 batch:
#   preprocessor(batch)
#     → ACTHFPolicy.forward()
#         base_loss + λ_vel·vel + λ_acc·acc + λ_dct·dct [+β·kld]
#     → optimizer.step()
#          │
#          ▼
#   检查点写出  →  --output_dir/

"""Train entrypoint that registers the local `act_hf` policy plugin first."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Register local policy/config/plugin symbols before draccus parses --policy.type.
import lerobot_policy_act_hf  # noqa: F401,E402

from lerobot.policies import factory as policy_factory  # noqa: E402
from lerobot.processor.converters import transition_to_batch  # noqa: E402
from lerobot.scripts import lerobot_train as lerobot_train_module  # noqa: E402

from lerobot_policy_act_hf.configuration_act_hf import ACTHFConfig  # noqa: E402
from lerobot_policy_act_hf.processor_act_hf import (  # noqa: E402
    _batch_to_transition_with_meta,
    make_act_hf_pre_post_processors,
)


ALIASES = {
    "--alpha-fast": "--policy.alpha_fast",
    "--lambda-vel": "--policy.lambda_vel",
    "--lambda-acc": "--policy.lambda_acc",
    "--lambda-dct": "--policy.lambda_dct",
    "--dct-hi-start": "--policy.dct_hi_start",
    "--dct-p": "--policy.dct_p",
    "--burst-csv-dir": "--policy.burst_csv_dir",
    "--phase-hz": "--policy.phase_hz",
    "--phase-label-keys": "--policy.phase_label_keys",
}


_ORIG_MAKE_PRE_POST_PROCESSORS = policy_factory.make_pre_post_processors


def _make_pre_post_processors_for_act_hf(policy_cfg, pretrained_path: str | None = None, **kwargs: Any):
    # `ACTHFConfig` is a subclass of `ACTConfig`. Without this guard, LeRobot factory routes it
    # through the default ACT processor, which drops timestamp/frame_index needed for burst labels.
    if not isinstance(policy_cfg, ACTHFConfig) and getattr(policy_cfg, "type", None) != "act_hf":
        return _ORIG_MAKE_PRE_POST_PROCESSORS(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    if pretrained_path:
        preprocessor, postprocessor = _ORIG_MAKE_PRE_POST_PROCESSORS(
            policy_cfg,
            pretrained_path=pretrained_path,
            **kwargs,
        )
        # Ensure resumed runs keep timestamp/frame_index in complementary data.
        preprocessor.to_transition = _batch_to_transition_with_meta
        preprocessor.to_output = transition_to_batch
        return preprocessor, postprocessor

    return make_act_hf_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )


# Patch both call sites used in training to guarantee `act_hf` preprocessor routing.
policy_factory.make_pre_post_processors = _make_pre_post_processors_for_act_hf
lerobot_train_module.make_pre_post_processors = _make_pre_post_processors_for_act_hf
lerobot_train_main = lerobot_train_module.main


def _rewrite_alias_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        rewritten = False
        for alias, target in ALIASES.items():
            if token == alias:
                if i + 1 >= len(argv):
                    raise SystemExit(f"Missing value for {alias}")
                out.extend([target, argv[i + 1]])
                i += 2
                rewritten = True
                break
            if token.startswith(alias + "="):
                out.append(target + "=" + token.split("=", 1)[1])
                i += 1
                rewritten = True
                break
        if rewritten:
            continue
        out.append(token)
        i += 1
    return out


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_rewrite_alias_args(sys.argv[1:])]
    lerobot_train_main()
