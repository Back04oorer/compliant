from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("stereo_diffusion")
@dataclass
class StereoDiffusionConfig(DiffusionConfig):
    """LeRobot Diffusion Policy conditioned by Stereo-ACT visual tokens.

    ``cross`` and ``cross_relbias`` use both Top RGB/DINO and depth/DeFM.
    ``direct`` is reserved for a controlled single-Top-branch ablation: either
    DINO-only or DeFM-only.  The wrist/state encoders, token poolers and DP
    action head remain identical across all three experiments.
    """

    # At 30 FPS: horizon=32 covers 1.067 s; eight executed actions replan at
    # 3.75 Hz while the robot command queue is consumed at 30 Hz.
    n_obs_steps: int = 1
    horizon: int = 32
    n_action_steps: int = 8
    drop_n_last_frames: int = 0
    do_mask_loss_for_padding: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # A compact UNet is the initial real-time target. All dimensions remain
    # divisible by n_groups=8 and horizon is divisible by 2**len(down_dims).
    down_dims: tuple[int, ...] = (128, 256, 512)
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    num_inference_steps: int | None = 4
    optimizer_lr: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Shared visual encoder contract.
    fusion_mode: str = "cross_relbias"
    use_top_rgb: bool = True
    use_depth: bool = True
    top_rgb_feature_key: str = "observation.image"
    depth_feature_key: str = "observation.fastfs_depth_image"
    depth_feature_is_tokens: bool = False

    fastfs_valid_iters: int = 8
    fastfs_scale: float = 0.5161290322580645
    fastfs_input_width: int = 640
    fastfs_input_height: int = 320
    fastfs_max_disp: int = 64
    fastfs_max_depth_m: float = 2.5

    # Foundation models are external assets. Pass portable local paths at
    # train/load time; never serialize a developer-machine default here.
    dino_model_id: str = ""
    dino_feature_dim: int = 768
    dino_patch_size: int = 16
    dino_dtype: str = "bfloat16"
    dino_freeze: bool = True
    dino_train_last_n_blocks: int = 0
    dino_finetune_lr: float = 1e-6
    dino_finetune_dtype: str = "float32"

    defm_root: str = ""
    defm_model_name: str = "defm_vit_s14"
    defm_pretrained_path: str = ""
    defm_feature_dim: int = 384
    defm_patch_size: int = 14
    defm_freeze: bool = True
    defm_train_last_n_blocks: int = 0
    defm_finetune_lr: float = 1e-6
    defm_finetune_dtype: str = "float32"
    defm_disable_xformers: bool = True

    # These names match the Stereo-ACT encoder helpers exactly.
    dim_model: int = 512
    dropout: float = 0.1
    target_grid_h: int = 20
    target_grid_w: int = 40
    wrist_encoder: str = "resnet18"
    wrist_grid_h: int = 8
    wrist_grid_w: int = 8
    fusion_layers: int = 2
    fusion_heads: int = 8
    fusion_ffn_dim: int = 2048
    replace_final_stride_with_dilation: bool = False
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    optimizer_lr_backbone: float = 1e-5

    # Separate pooling prevents 800 Top tokens from overwhelming 64 wrist
    # tokens solely because of token count.
    top_pool_latents: int = 8
    wrist_pool_latents: int = 2
    pool_heads: int = 8
    top_embedding_dim: int = 512
    wrist_embedding_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_stereo_diffusion()

    def _validate_stereo_diffusion(self) -> None:
        if self.fusion_mode not in {"direct", "cross", "cross_relbias"}:
            raise ValueError("stereo_diffusion supports fusion_mode=direct, cross or cross_relbias")
        if not self.use_top_rgb and not self.use_depth:
            raise ValueError("stereo_diffusion requires at least one Top encoder")
        if self.fusion_mode in {"cross", "cross_relbias"} and not (
            self.use_top_rgb and self.use_depth
        ):
            raise ValueError(f"fusion_mode={self.fusion_mode!r} requires Top RGB and depth")
        if self.fusion_mode == "direct" and self.use_top_rgb == self.use_depth:
            raise ValueError(
                "stereo_diffusion direct mode is a single-branch ablation; "
                "enable exactly one of use_top_rgb/use_depth"
            )
        if self.wrist_encoder not in {"resnet18", "dino"}:
            raise ValueError("wrist_encoder must be resnet18 or dino")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError("n_action_steps exceeds the usable diffusion action horizon")
        if self.dino_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dino_dtype")
        if self.depth_feature_is_tokens and not self.defm_freeze:
            raise ValueError("Precomputed DeFM tokens require defm_freeze=true")
        if self.dino_freeze and self.dino_train_last_n_blocks:
            raise ValueError("dino_train_last_n_blocks requires dino_freeze=false")
        if self.defm_freeze and self.defm_train_last_n_blocks:
            raise ValueError("defm_train_last_n_blocks requires defm_freeze=false")
        positive = (
            "fastfs_valid_iters", "fastfs_scale", "fastfs_input_width", "fastfs_input_height",
            "fastfs_max_disp", "fastfs_max_depth_m", "dim_model", "target_grid_h", "target_grid_w",
            "wrist_grid_h", "wrist_grid_w", "fusion_layers", "fusion_heads", "fusion_ffn_dim",
            "top_pool_latents", "wrist_pool_latents", "pool_heads", "top_embedding_dim",
            "wrist_embedding_dim",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.dim_model % self.fusion_heads or self.dim_model % self.pool_heads:
            raise ValueError("dim_model must be divisible by fusion_heads and pool_heads")

    @property
    def stereo_observation_keys(self) -> list[str]:
        keys = [OBS_STATE]
        if self.use_top_rgb:
            keys.append(self.top_rgb_feature_key)
        if self.use_depth:
            keys.append(self.depth_feature_key)
        keys.extend(
            key
            for key in self.image_features
            if key not in {self.top_rgb_feature_key, self.depth_feature_key}
        )
        if self.env_state_feature is not None:
            keys.append("observation.environment_state")
        return keys

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError(f"{OBS_STATE!r} STATE feature is required")
        if self.action_feature is None:
            raise ValueError(f"{ACTION!r} ACTION feature is required")
        if self.use_top_rgb:
            if self.top_rgb_feature_key not in self.image_features:
                raise ValueError(f"Top RGB key {self.top_rgb_feature_key!r} must be a VISUAL feature")
            top_shape = self.input_features[self.top_rgb_feature_key].shape
            if len(top_shape) != 3 or top_shape[0] != 3:
                raise ValueError(f"Top RGB must be [3,H,W], got {top_shape}")
        if self.use_depth:
            if self.depth_feature_key not in (self.input_features or {}):
                raise ValueError(f"Depth key {self.depth_feature_key!r} is missing")
            depth_shape = self.input_features[self.depth_feature_key].shape
            if len(depth_shape) != 2:
                raise ValueError(f"Depth/tokens must be rank 2 per frame, got {depth_shape}")
        wrist_keys = [
            key for key in self.image_features
            if key not in {self.top_rgb_feature_key, self.depth_feature_key}
        ]
        if not wrist_keys:
            raise ValueError("The initial stereo_diffusion implementation requires a wrist camera")
        for key in wrist_keys:
            shape = self.input_features[key].shape
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"Wrist RGB {key!r} must be [3,H,W], got {shape}")
