from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
from lerobot_policy_stereo_act.modeling_stereo_act import (
    StereoACT,
    StereoFusionBlock,
)


class StereoFusedObservationEncoder(nn.Module):
    """Standalone version of the proven Stereo-ACT visual path.

    The first implementation deliberately reuses the audited Stereo-ACT helper
    methods instead of changing the deployed Stereo-ACT module/state-dict
    layout.  It returns Top and wrist token groups separately so a downstream
    policy cannot accidentally let the 800 Top tokens swamp the 64 wrist
    tokens during pooling.
    """

    # Reuse foundation-model and tokenization behavior byte-for-byte.  These
    # helpers only require the attributes initialized below and do not build or
    # call the ACT transformer/action head.
    _init_dino = StereoACT._init_dino
    _init_defm = StereoACT._init_defm
    _defm_transformer_blocks = staticmethod(StereoACT._defm_transformer_blocks)
    _configure_foundation_trainability = staticmethod(StereoACT._configure_foundation_trainability)
    _promote_and_audit_partial_finetune = staticmethod(StereoACT._promote_and_audit_partial_finetune)
    _image_to_dino = StereoACT._image_to_dino
    _encode_dino_tokens = StereoACT._encode_dino_tokens
    _depth_to_defm = StereoACT._depth_to_defm
    _encode_defm_tokens = StereoACT._encode_defm_tokens
    _pos2d = StereoACT._pos2d
    _modality = StereoACT._modality
    _encode_wrist_resnet_memories = StereoACT._encode_wrist_resnet_memories
    _encode_wrist_dino_memories = StereoACT._encode_wrist_dino_memories
    _encode_stereo_memories = StereoACT._encode_stereo_memories

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        if config.fusion_mode not in {"direct", "cross", "cross_relbias"}:
            raise ValueError(
                "StereoFusedObservationEncoder requires fusion_mode=direct, cross or cross_relbias; "
                f"got {config.fusion_mode!r}"
            )
        if not config.use_top_rgb and not config.use_depth:
            raise ValueError("The visual encoder requires at least one Top branch")
        if config.fusion_mode in {"cross", "cross_relbias"} and not (
            config.use_top_rgb and config.use_depth
        ):
            raise ValueError(f"fusion_mode={config.fusion_mode!r} requires Top RGB and depth")
        if config.fusion_mode == "direct" and config.use_top_rgb == config.use_depth:
            raise ValueError("direct mode requires exactly one Top branch")
        if config.use_depth and config.depth_feature_key not in (config.input_features or {}):
            raise ValueError(f"Missing depth input feature {config.depth_feature_key!r}")

        excluded = {config.top_rgb_feature_key, config.depth_feature_key}
        self.wrist_image_keys = [key for key in config.image_features if key not in excluded]
        if config.use_top_rgb and config.top_rgb_feature_key not in config.image_features:
            raise ValueError(f"Missing Top RGB visual feature {config.top_rgb_feature_key!r}")

        self.dino_input_proj = (
            nn.Linear(config.dino_feature_dim, config.dim_model)
            if config.use_top_rgb
            else None
        )
        self.wrist_dino_input_proj = (
            nn.Linear(config.dino_feature_dim, config.dim_model)
            if config.wrist_encoder == "dino" and self.wrist_image_keys
            else None
        )
        self.defm_input_proj = (
            nn.Linear(config.defm_feature_dim, config.dim_model)
            if config.use_depth
            else None
        )
        self.top_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        self.wrist_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        self.modality_embed = nn.Embedding(8, config.dim_model)
        self.rgb_view_embed = nn.Embedding(max(1, len(self.wrist_image_keys)), config.dim_model)
        self.fusion_blocks = nn.ModuleList(
            [StereoFusionBlock(config) for _ in range(int(config.fusion_layers))]
            if config.use_top_rgb and config.use_depth
            else []
        )

        self.backbone = None
        self.encoder_img_feat_input_proj = None
        self.encoder_cam_feat_pos_embed = None
        if config.wrist_encoder == "resnet18" and self.wrist_image_keys:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    bool(config.replace_final_stride_with_dilation),
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.dino_model = None
        self._dino_train_blocks: list[nn.Module] = []
        self._dino_train_norms: list[nn.Module] = []
        self._dino_fully_frozen = True
        self.dino_num_prefix_tokens = 1
        self.defm_model = None
        self._defm_train_blocks: list[nn.Module] = []
        self._defm_train_norms: list[nn.Module] = []
        self._defm_fully_frozen = True

        if config.use_top_rgb or config.wrist_encoder == "dino":
            self._init_dino()
        if config.use_depth and not config.depth_feature_is_tokens:
            self._init_defm()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for model, fully_frozen, blocks, norms in (
            (self.dino_model, self._dino_fully_frozen, self._dino_train_blocks, self._dino_train_norms),
            (self.defm_model, self._defm_fully_frozen, self._defm_train_blocks, self._defm_train_norms),
        ):
            if model is None:
                continue
            if hasattr(model, "blocks"):
                all_blocks = self._defm_transformer_blocks(model)
            else:
                encoder_layers = getattr(getattr(model, "encoder", None), "layer", None)
                all_blocks = list(encoder_layers if encoder_layers is not None else model.layer)
            if fully_frozen or len(blocks) < len(all_blocks):
                model.eval()
                if mode and not fully_frozen:
                    for module in [*blocks, *norms]:
                        module.train()
        return self

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        first_key = (
            self.config.top_rgb_feature_key
            if self.config.use_top_rgb
            else self.config.depth_feature_key
        )
        first = batch[first_key]
        top_projection = self.dino_input_proj if self.dino_input_proj is not None else self.defm_input_proj
        if top_projection is None:
            raise RuntimeError("Stereo encoder has no Top projection")
        dtype = top_projection.weight.dtype
        device = top_projection.weight.device
        if first.device != device:
            raise RuntimeError(
                f"Stereo encoder input is on {first.device}, but encoder parameters are on {device}"
            )

        memories = self._encode_stereo_memories(batch, dtype=dtype, device=device)
        if not memories:
            raise RuntimeError("Stereo encoder returned no visual memories")

        top_tokens, top_pos = memories[0]
        wrist_memories = memories[1:]
        if wrist_memories:
            wrist_tokens = torch.cat([tokens for tokens, _ in wrist_memories], dim=1)
            wrist_pos = torch.cat([pos for _, pos in wrist_memories], dim=1)
        else:
            wrist_tokens = None
            wrist_pos = None
        return {
            "top_tokens": top_tokens,
            "top_pos": top_pos,
            "wrist_tokens": wrist_tokens,
            "wrist_pos": wrist_pos,
        }


class StereoTokenPooler(nn.Module):
    """Compress a variable token group into one fixed-size DP condition vector."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_latents: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.latents = nn.Parameter(torch.empty(1, num_latents, token_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_norm = nn.LayerNorm(token_dim)
        self.value_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(num_latents * token_dim),
            nn.Linear(num_latents * token_dim, output_dim),
            nn.Mish(),
        )

    def forward(self, tokens: Tensor, pos: Tensor) -> Tensor:
        if tokens.ndim != 3 or pos.shape != tokens.shape:
            raise ValueError(
                f"Expected matching [B,K,D] tokens/pos, got {tuple(tokens.shape)} and {tuple(pos.shape)}"
            )
        query = self.latents.expand(tokens.shape[0], -1, -1)
        attended = self.attention(
            self.query_norm(query),
            self.key_norm(tokens + pos),
            self.value_norm(tokens),
            need_weights=False,
        )[0]
        query = query + attended
        query = query + self.ffn(self.ffn_norm(query))
        return self.output(query.flatten(start_dim=1))
