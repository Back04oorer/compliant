from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot_policy_stereo_flow.configuration_stereo_flow import StereoFlowConfig


@PreTrainedConfig.register_subclass("stereo_flow_v2")
@dataclass
class StereoFlowV2Config(StereoFlowConfig):
    """Full Stereo-ACT memory with a compact conditional Flow head.

    The memory length is determined by the dataset: one state token, the Top
    grid, and one wrist grid per recorded wrist camera.  This keeps the proven
    865-token single-arm contract while allowing a bimanual dataset to use
    both wrists (929 tokens for Top20x40 + two Wrist8x8 grids).
    """

    n_obs_steps: int = 1
    horizon: int = 32
    n_action_steps: int = 20

    # ACT condition encoder fields use the names expected by LeRobot ACTEncoder.
    n_encoder_layers: int = 4
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    pre_norm: bool = False

    flow_decoder_dim: int = 256
    flow_decoder_layers: int = 4
    flow_decoder_heads: int = 8
    flow_decoder_ffn_dim: int = 1024
    flow_decoder_dropout: float = 0.1
    flow_time_scale: float = 0.999
    flow_time_offset: float = 0.001
    wrist_modality_dropout: float = 0.10
    ema_decay: float = 0.999

    optimizer_lr: float = 1.0e-4
    optimizer_lr_backbone: float = 1.0e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1.0e-8
    optimizer_weight_decay: float = 1.0e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 80000
    scheduler_decay_lr: float = 1.0e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        self.normalization_mapping = dict(self.normalization_mapping)
        self.normalization_mapping.update(
            {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }
        )
        if self.n_obs_steps != 1:
            raise ValueError("Stereo-Flow V2 fixes n_obs_steps=1")
        if self.dim_model != 512:
            raise ValueError("Stereo-Flow V2 condition encoder fixes dim_model=512")
        if self.n_encoder_layers != 4 or self.n_heads != 8 or self.dim_feedforward != 3200:
            raise ValueError("Stereo-Flow V2 fixes the 4x512 ACT condition encoder")
        if self.flow_decoder_dim % self.flow_decoder_heads:
            raise ValueError("flow_decoder_dim must be divisible by flow_decoder_heads")
        if not 0.0 <= float(self.wrist_modality_dropout) < 1.0:
            raise ValueError("wrist_modality_dropout must be in [0, 1)")
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if abs(float(self.flow_time_scale) + float(self.flow_time_offset) - 1.0) > 1e-9:
            raise ValueError("flow_time_scale + flow_time_offset must equal 1")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
