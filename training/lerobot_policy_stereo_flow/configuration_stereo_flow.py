from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_stereo_diffusion.configuration_stereo_diffusion import StereoDiffusionConfig


@PreTrainedConfig.register_subclass("stereo_flow")
@dataclass
class StereoFlowConfig(StereoDiffusionConfig):
    """Stereo fused perception with a compact Flow-Matching action expert.

    The visual contract deliberately inherits the deployed Stereo-Diffusion
    encoder settings. Diffusion-only fields remain serializable for checkpoint
    compatibility but are not used by the Flow-Matching action head.
    """

    # Spatial condition resampling. These are model-training ablations, unlike
    # RTC parameters which can be changed at deployment for one checkpoint.
    top_pool_latents: int = 32
    wrist_pool_latents: int = 4

    # Compact bidirectional action expert.
    flow_num_steps: int = 4
    flow_time_beta_alpha: float = 1.5
    flow_time_beta_beta: float = 1.0
    flow_time_min: float = 0.001
    action_expert_layers: int = 6
    action_expert_heads: int = 8
    action_expert_ffn_dim: int = 2048
    action_expert_dropout: float = 0.1

    # RTC is inference-only. Training checkpoints default to queue mode so the
    # same weights can be evaluated both with and without RTC.
    rtc_enabled: bool = False
    rtc_prefix_attention_schedule: str = "LINEAR"
    rtc_max_guidance_weight: float = 5.0
    rtc_execution_horizon: int = 10
    rtc_debug: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_stereo_flow()

    def _validate_stereo_flow(self) -> None:
        positive_ints = (
            "flow_num_steps",
            "action_expert_layers",
            "action_expert_heads",
            "action_expert_ffn_dim",
            "rtc_execution_horizon",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        positive_floats = (
            "flow_time_beta_alpha",
            "flow_time_beta_beta",
            "rtc_max_guidance_weight",
        )
        for name in positive_floats:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 < float(self.flow_time_min) < 1.0:
            raise ValueError("flow_time_min must be in (0, 1)")
        if not 0.0 <= float(self.action_expert_dropout) < 1.0:
            raise ValueError("action_expert_dropout must be in [0, 1)")
        if self.dim_model % self.action_expert_heads:
            raise ValueError("dim_model must be divisible by action_expert_heads")
        if self.rtc_prefix_attention_schedule not in {"ZEROS", "ONES", "LINEAR", "EXP"}:
            raise ValueError("Unsupported rtc_prefix_attention_schedule")
        if self.n_obs_steps != 1:
            raise ValueError("The first Stereo-Flow experiment fixes n_obs_steps=1")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")
