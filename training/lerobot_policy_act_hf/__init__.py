# ┌──────────────────────────────────────────────────────────┐
# │        lerobot_policy_act_hf/__init__.py  数据流         │
# └──────────────────────────────────────────────────────────┘
#
#   import lerobot_policy_act_hf
#          │
#          ├─ from .configuration_act_hf import ACTHFConfig
#          │   └─ @PreTrainedConfig.register_subclass("act_hf")
#          │      "act_hf" 注册到 LeRobot 配置工厂
#          │
#          └─ from .modeling_act_hf import ACTHFPolicy
#              └─ ACTHFPolicy.name = "act_hf"
#                 注册到 LeRobot 策略工厂
#          │
#          ▼
#   LeRobot 可通过 --policy.type=act_hf 实例化此策略

"""High-frequency ACT plugin registration.

Importing this package registers the custom `act_hf` policy/config with LeRobot.
"""

from .configuration_act_hf import ACTHFConfig
from .modeling_act_hf import ACTHFPolicy

__all__ = ["ACTHFConfig", "ACTHFPolicy"]
