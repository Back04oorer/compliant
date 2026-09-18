from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot_policy_act_hf.configuration_act_hf import ACTHFConfig


@PreTrainedConfig.register_subclass("stereo_act")
@dataclass
class StereoACTConfig(ACTHFConfig):
    """ACT with frozen DINOv3 RGB tokens and frozen DeFM depth tokens.

    fusion_mode:
      - direct: append top RGB, top depth, wrist RGB, state and z tokens directly.
      - cross: run region-aligned RGB/depth self-attention + RGB-query-depth cross-attention
        for the top camera before appending fused top tokens to ACT memory.
      - cross_relbias: cross plus learned factorized 2D relative-position bias.
    """

    fusion_mode: str = "direct"  # choices: direct | cross | cross_relbias
    use_top_rgb: bool = True
    use_depth: bool = True
    top_rgb_feature_key: str = "observation.image"
    depth_feature_key: str = "observation.fastfs_depth_image"
    depth_feature_is_tokens: bool = False

    # FastFS is external to the policy forward pass. These fields persist the
    # expected train/deploy depth-generation contract in every checkpoint.
    # Defaults preserve compatibility with checkpoints created before these
    # provenance fields existed. Input width/height=0 means that an older
    # checkpoint did not persist the scaled FastFS input resolution.
    fastfs_valid_iters: int = 12
    fastfs_scale: float = 1.0
    fastfs_input_width: int = 0
    fastfs_input_height: int = 0
    fastfs_max_disp: int = 192
    fastfs_max_depth_m: float = 2.5

    # This experiment is meant to isolate token organization, not the ACT-HF
    # auxiliary losses. Keep the default objective close to standard ACT.
    base_loss: str = "l1"
    alpha_fast: float = 0.0
    lambda_vel: float = 0.0
    lambda_acc: float = 0.0
    lambda_dct: float = 0.0
    gripper_indices: str = "auto"

    # External foundation models are deliberately not bundled. Pass local
    # paths explicitly at training time so checkpoints remain auditable.
    dino_model_id: str = ""
    dino_feature_dim: int = 768
    dino_patch_size: int = 16
    dino_dtype: str = "bfloat16"  # choices: float32 | float16 | bfloat16
    dino_freeze: bool = True
    # freeze=false,last_n=0 preserves full fine-tuning. last_n>0 trains only
    # the final transformer blocks and output norm.
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

    target_grid_h: int = 20
    target_grid_w: int = 40
    # Wrist images use the standard ACT ResNet by default for checkpoint
    # compatibility. The DINO option reuses the same frozen DINO model as the
    # top camera, but has an independent learned projection into ACT memory.
    wrist_encoder: str = "resnet18"  # choices: resnet18 | dino
    wrist_grid_h: int = 8
    wrist_grid_w: int = 8

    fusion_layers: int = 2
    fusion_heads: int = 8
    fusion_ffn_dim: int = 2048

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.gripper_indices).strip().lower() == "auto":
            self.gripper_indices = self._infer_gripper_indices()
        self.normalization_mapping = dict(self.normalization_mapping)
        self.normalization_mapping["VISUAL"] = NormalizationMode.IDENTITY
        self._validate_stereo_act()

    def _infer_gripper_indices(self) -> str:
        action_dim = 0
        if self.action_feature is not None and self.action_feature.shape:
            action_dim = int(self.action_feature.shape[0])
        if action_dim <= 0:
            # CLI parsing happens before dataset metadata supplies action_dim.
            # Keep the sentinel so StereoACTPolicy can infer it afterwards.
            return "auto"
        if action_dim % 7 == 0:
            return ",".join(str(i) for i in range(6, action_dim, 7))
        return str(action_dim - 1)

    def _validate_stereo_act(self) -> None:
        if self.fusion_mode not in {"direct", "cross", "cross_relbias"}:
            raise ValueError("fusion_mode must be one of: direct, cross, cross_relbias")
        if self.wrist_encoder not in {"resnet18", "dino"}:
            raise ValueError("wrist_encoder must be one of: resnet18, dino")
        if not self.use_top_rgb and not self.use_depth:
            raise ValueError("StereoACT requires at least one top encoder: use_top_rgb or use_depth")
        if self.fusion_mode in {"cross", "cross_relbias"} and not (self.use_top_rgb and self.use_depth):
            raise ValueError(
                f"fusion_mode={self.fusion_mode!r} requires use_top_rgb=True and use_depth=True"
            )
        if int(self.fastfs_valid_iters) <= 0:
            raise ValueError("fastfs_valid_iters must be > 0")
        if float(self.fastfs_scale) <= 0:
            raise ValueError("fastfs_scale must be > 0")
        if int(self.fastfs_input_width) < 0 or int(self.fastfs_input_height) < 0:
            raise ValueError("fastfs_input_width/height must be >= 0")
        if (int(self.fastfs_input_width) == 0) != (int(self.fastfs_input_height) == 0):
            raise ValueError("fastfs_input_width/height must either both be 0 or both be > 0")
        if int(self.fastfs_max_disp) <= 0:
            raise ValueError("fastfs_max_disp must be > 0")
        if float(self.fastfs_max_depth_m) <= 0:
            raise ValueError("fastfs_max_depth_m must be > 0")
        if self.dino_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dino_dtype must be one of: float32, float16, bfloat16")
        if self.depth_feature_is_tokens and not self.defm_freeze:
            raise ValueError("depth_feature_is_tokens=true requires defm_freeze=true")
        if int(self.dino_train_last_n_blocks) < 0 or int(self.defm_train_last_n_blocks) < 0:
            raise ValueError("foundation train_last_n_blocks values must be >= 0")
        if self.dino_freeze and int(self.dino_train_last_n_blocks) > 0:
            raise ValueError("dino_train_last_n_blocks requires dino_freeze=false")
        if self.defm_freeze and int(self.defm_train_last_n_blocks) > 0:
            raise ValueError("defm_train_last_n_blocks requires defm_freeze=false")
        if float(self.dino_finetune_lr) <= 0 or float(self.defm_finetune_lr) <= 0:
            raise ValueError("foundation fine-tune learning rates must be > 0")
        if self.dino_finetune_dtype != "float32" or self.defm_finetune_dtype != "float32":
            raise ValueError("Partial foundation fine-tuning requires float32 master parameters")
        if self.input_features:
            if self.use_top_rgb and self.top_rgb_feature_key not in self.input_features:
                raise ValueError(f"top_rgb_feature_key={self.top_rgb_feature_key!r} is not in input_features")
            if self.use_depth and self.depth_feature_key not in self.input_features:
                raise ValueError(f"depth_feature_key={self.depth_feature_key!r} is not in input_features")
        for name in (
            "dino_feature_dim",
            "dino_patch_size",
            "defm_feature_dim",
            "defm_patch_size",
            "target_grid_h",
            "target_grid_w",
            "wrist_grid_h",
            "wrist_grid_w",
            "fusion_layers",
            "fusion_heads",
            "fusion_ffn_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
