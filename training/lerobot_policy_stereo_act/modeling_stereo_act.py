from __future__ import annotations

import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.act.modeling_act import ACT, ACTSinusoidalPositionEmbedding2d
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot_policy_act_hf.modeling_act_hf import ACTHFPolicy

from .configuration_stereo_act import StereoACTConfig


DINO_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGENET_STD = (0.229, 0.224, 0.225)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _normalize_imagenet_rgb(img: Tensor) -> Tensor:
    """Convert RGB input to [0,1] and apply ImageNet normalization."""
    x = img.float()
    if float(x.detach().amax().cpu()) > 2.0:
        x = x / 255.0
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"RGB image must be [B,3,H,W], got {tuple(x.shape)}")
    mean = torch.tensor(DINO_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DINO_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x.clamp(0.0, 1.0) - mean) / std


class FactorizedRelativePositionBias2d(nn.Module):
    """Shared factorized 2D relative bias for a fixed row-major token grid."""

    def __init__(self, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.row_bias = nn.Parameter(torch.zeros(2 * self.grid_h - 1))
        self.col_bias = nn.Parameter(torch.zeros(2 * self.grid_w - 1))

        rows = torch.arange(self.grid_h).repeat_interleave(self.grid_w)
        cols = torch.arange(self.grid_w).repeat(self.grid_h)
        row_index = rows[None, :] - rows[:, None] + self.grid_h - 1
        col_index = cols[None, :] - cols[:, None] + self.grid_w - 1
        self.register_buffer("row_index", row_index, persistent=False)
        self.register_buffer("col_index", col_index, persistent=False)

    def forward(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        row_index = self.row_index.to(device=device)
        col_index = self.col_index.to(device=device)
        return (self.row_bias[row_index] + self.col_bias[col_index]).to(device=device, dtype=dtype)


class StereoFusionBlock(nn.Module):
    """Region-aligned RGB/depth fusion block without learned gates."""

    def __init__(self, config: StereoACTConfig):
        super().__init__()
        d_model = int(config.dim_model)
        heads = int(config.fusion_heads)
        dropout = float(config.dropout)
        ffn_dim = int(config.fusion_ffn_dim)

        self.rgb_self_norm = nn.LayerNorm(d_model)
        self.depth_self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.rgb_self = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.depth_self = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_relative_bias = (
            FactorizedRelativePositionBias2d(
                int(config.target_grid_h),
                int(config.target_grid_w),
            )
            if config.fusion_mode == "cross_relbias"
            else None
        )
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, rgb: Tensor, depth: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        rgb_q = self.rgb_self_norm(rgb + pos)
        depth_q = self.depth_self_norm(depth + pos)
        rgb = rgb + self.dropout(self.rgb_self(rgb_q, rgb_q, rgb, need_weights=False)[0])
        depth = depth + self.dropout(self.depth_self(depth_q, depth_q, depth, need_weights=False)[0])

        q = self.cross_norm(rgb + pos)
        k = depth + pos
        attn_mask = None
        if self.cross_relative_bias is not None:
            attn_mask = self.cross_relative_bias(
                dtype=q.dtype,
                device=q.device,
            )
        rgb = rgb + self.dropout(
            self.cross(q, k, depth, attn_mask=attn_mask, need_weights=False)[0]
        )
        rgb = rgb + self.dropout(self.ffn(self.ffn_norm(rgb)))
        return rgb, depth


class StereoACT(ACT):
    def __init__(self, config: StereoACTConfig):
        super().__init__(config)
        self.config = config

        # ACT.__init__ creates the standard ResNet visual branch. Stereo-ACT uses
        # it for wrists by default, while the wrist-DINO experiment reuses the
        # same frozen DINO weights as the top camera.
        if config.use_depth and config.depth_feature_key not in (config.input_features or {}):
            raise ValueError(f"Stereo-ACT requires {config.depth_feature_key!r} in input_features")

        excluded = {config.top_rgb_feature_key}
        if config.use_depth:
            excluded.add(config.depth_feature_key)
        self.wrist_image_keys = [key for key in config.image_features.keys() if key not in excluded]
        if config.use_top_rgb and config.top_rgb_feature_key not in config.image_features:
            raise ValueError(
                f"top_rgb_feature_key={config.top_rgb_feature_key!r} must be a VISUAL feature "
                f"and cannot equal depth_feature_key. Visual keys={list(config.image_features.keys())}"
            )

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
        # A DINO-only ablation must not carry unused DeFM/fusion parameters in
        # the optimizer or checkpoint. Existing depth-enabled checkpoints keep
        # the exact same module structure as before.
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

        self.dino_model = None
        self._dino_train_blocks = []
        self._dino_train_norms = []
        self._dino_fully_frozen = True
        self.dino_num_prefix_tokens = 1
        self.defm_model = None
        self._defm_train_blocks = []
        self._defm_train_norms = []
        self._defm_fully_frozen = True
        if self.config.wrist_encoder == "dino" or not self.wrist_image_keys:
            # These standard ACT modules are not on the forward path when the
            # wrist uses DINO or when this is a Top-only dataset with no wrist
            # image keys. Keep them in the module layout for checkpoint/API
            # compatibility, but exclude the unused parameters from training.
            for module in (self.backbone, self.encoder_img_feat_input_proj):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if self.config.use_top_rgb or self.config.wrist_encoder == "dino":
            self._init_dino()
        if self.config.use_depth and not self.config.depth_feature_is_tokens:
            self._init_defm()

    def _init_dino(self) -> None:
        from transformers import AutoConfig, AutoModel

        if not str(self.config.dino_model_id).strip():
            raise ValueError(
                "DINO is enabled but dino_model_id is empty. Pass "
                "--dino-model /path/to/local/dinov3 when launching training."
            )
        dtype = _torch_dtype(self.config.dino_dtype)
        cfg = AutoConfig.from_pretrained(self.config.dino_model_id)
        self.dino_num_prefix_tokens = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        model = AutoModel.from_pretrained(self.config.dino_model_id, torch_dtype=dtype)
        model.eval()
        dino_layers = getattr(getattr(model, "encoder", None), "layer", None)
        if dino_layers is None:
            dino_layers = model.layer
        blocks = list(dino_layers)
        norms = [module for module in (getattr(model, "layernorm", None), getattr(model, "norm", None)) if module is not None]
        self._dino_train_blocks, self._dino_train_norms, self._dino_fully_frozen = self._configure_foundation_trainability(
            model, blocks, norms, bool(self.config.dino_freeze), int(self.config.dino_train_last_n_blocks), "DINO"
        )
        self._promote_and_audit_partial_finetune(
            model,
            blocks,
            self._dino_train_blocks,
            self._dino_train_norms,
            int(self.config.dino_train_last_n_blocks),
            _torch_dtype(self.config.dino_finetune_dtype),
            "DINO",
        )
        self.dino_model = model

    def _init_defm(self) -> None:
        if self.config.defm_disable_xformers:
            os.environ.setdefault("XFORMERS_DISABLED", "1")
        if not str(self.config.defm_root).strip():
            raise ValueError(
                "DeFM is enabled but defm_root is empty. Pass "
                "--defm-root /path/to/defm when launching training."
            )
        root = Path(self.config.defm_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DeFM repository not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from defm.model_factory import create_defm_model

        pretrained_path = self.config.defm_pretrained_path.strip() or None
        skip_pretrained_init = os.environ.get("LFV_DEFM_SKIP_PRETRAINED_INIT", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        model = create_defm_model(
            self.config.defm_model_name,
            pretrained=not skip_pretrained_init,
            pretrained_path=pretrained_path,
        )
        model.eval()
        blocks = self._defm_transformer_blocks(model)
        norms = [module for module in (getattr(model, "norm", None),) if module is not None]
        self._defm_train_blocks, self._defm_train_norms, self._defm_fully_frozen = self._configure_foundation_trainability(
            model, blocks, norms, bool(self.config.defm_freeze), int(self.config.defm_train_last_n_blocks), "DeFM"
        )
        self._promote_and_audit_partial_finetune(
            model,
            blocks,
            self._defm_train_blocks,
            self._defm_train_norms,
            int(self.config.defm_train_last_n_blocks),
            _torch_dtype(self.config.defm_finetune_dtype),
            "DeFM",
        )
        self.defm_model = model

    @staticmethod
    def _defm_transformer_blocks(model) -> list[nn.Module]:
        blocks = []
        for outer in model.blocks:
            if isinstance(outer, nn.ModuleList):
                blocks.extend(block for block in outer if not isinstance(block, nn.Identity))
            elif not isinstance(outer, nn.Identity):
                blocks.append(outer)
        return blocks

    @staticmethod
    def _configure_foundation_trainability(model, blocks, norms, freeze, last_n, label):
        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad = False
            print(f"[stereo-act] {label}: frozen, transformer_blocks={len(blocks)}")
            return [], [], True
        if last_n == 0:
            for parameter in model.parameters():
                parameter.requires_grad = True
            print(f"[stereo-act] {label}: full fine-tune, transformer_blocks={len(blocks)}")
            return blocks, norms, False
        if last_n > len(blocks):
            raise ValueError(f"{label} train_last_n_blocks={last_n} exceeds available blocks={len(blocks)}")
        for parameter in model.parameters():
            parameter.requires_grad = False
        selected = blocks[-last_n:]
        for module in [*selected, *norms]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"[stereo-act] {label}: partial fine-tune, transformer_blocks={len(blocks)} "
            f"train_last={last_n} final_norms={len(norms)} trainable_params={trainable}"
        )
        return selected, norms, False

    @staticmethod
    def _promote_and_audit_partial_finetune(
        model,
        all_blocks,
        selected_blocks,
        selected_norms,
        last_n,
        dtype,
        label,
    ) -> None:
        if last_n <= 0:
            return
        selected_modules = [*selected_blocks, *selected_norms]
        for module in selected_modules:
            module.to(dtype=dtype)

        allowed_parameter_ids = {
            id(parameter)
            for module in selected_modules
            for parameter in module.parameters()
        }
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        unexpected = [name for name, parameter in trainable if id(parameter) not in allowed_parameter_ids]
        if unexpected:
            raise RuntimeError(f"{label} has unexpectedly trainable parameters: {unexpected}")
        wrong_dtype = [name for name, parameter in trainable if parameter.dtype != dtype]
        if wrong_dtype:
            raise RuntimeError(f"{label} trainable parameters are not {dtype}: {wrong_dtype}")

        first_block = len(all_blocks) - last_n
        frozen_dtypes = sorted(
            {str(parameter.dtype) for block in all_blocks[:first_block] for parameter in block.parameters()}
        )
        module_names = {id(module): name or "<root>" for name, module in model.named_modules()}
        block_names = [module_names.get(id(module), "<unknown>") for module in selected_blocks]
        norm_names = [module_names.get(id(module), "<unknown>") for module in selected_norms]
        print(
            f"[stereo-act] {label} partial audit: actual_blocks={len(all_blocks)} "
            f"train_block_indices={list(range(first_block, len(all_blocks)))} "
            f"train_block_names={block_names} final_norm_names={norm_names} "
            f"master_dtype={dtype} frozen_block_dtypes={frozen_dtypes}"
        )
        for name, parameter in trainable:
            print(
                f"[stereo-act] {label} trainable_parameter={name} "
                f"dtype={parameter.dtype} shape={tuple(parameter.shape)}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
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

    def _image_to_dino(self, img: Tensor, grid_hw: tuple[int, int]) -> Tensor:
        if img.ndim != 4:
            raise ValueError(f"RGB image must be [B,C,H,W], got {tuple(img.shape)}")
        h = int(grid_hw[0]) * int(self.config.dino_patch_size)
        w = int(grid_hw[1]) * int(self.config.dino_patch_size)
        x = F.interpolate(img.float(), size=(h, w), mode="bilinear", align_corners=False)
        return _normalize_imagenet_rgb(x)

    def _encode_dino_tokens(self, img: Tensor, grid_hw: tuple[int, int], dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.dino_model is None:
            raise RuntimeError("DINO model is not initialized")
        model = self.dino_model.to(device)
        x = self._image_to_dino(img.to(device=device), grid_hw)
        model_dtype = next(model.parameters()).dtype
        context = torch.inference_mode() if self._dino_fully_frozen else nullcontext()
        autocast_dtype = _torch_dtype(self.config.dino_dtype)
        autocast_enabled = (
            device.type == "cuda"
            and not self._dino_fully_frozen
            and autocast_dtype in {torch.float16, torch.bfloat16}
        )
        with context, torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(pixel_values=x.to(dtype=model_dtype))
        tokens = outputs.last_hidden_state[:, self.dino_num_prefix_tokens :]
        expected = int(grid_hw[0]) * int(grid_hw[1])
        if tokens.shape[1] != expected:
            raise ValueError(f"DINO token mismatch: got {tokens.shape[1]}, expected {expected}")
        if tokens.shape[2] != int(self.config.dino_feature_dim):
            raise ValueError(f"DINO dim mismatch: got {tokens.shape[2]}, expected {self.config.dino_feature_dim}")
        tokens = tokens.to(device=device, dtype=dtype)
        return tokens.clone() if self._dino_fully_frozen else tokens

    def _depth_to_defm(self, depth: Tensor, device: torch.device) -> Tensor:
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(1)
        elif depth.ndim == 4 and depth.shape[1] != 1:
            depth = depth[:, :1]
        elif depth.ndim != 4:
            raise ValueError(f"Depth must be [B,H,W] or [B,C,H,W], got {tuple(depth.shape)}")
        target_hw = (
            int(self.config.target_grid_h) * int(self.config.defm_patch_size),
            int(self.config.target_grid_w) * int(self.config.defm_patch_size),
        )
        from defm.utils.utils import preprocess_depth_batch

        return preprocess_depth_batch(
            depth,
            target_size=target_hw,
            patch_size=int(self.config.defm_patch_size),
            device=device,
        )

    def _encode_defm_tokens(self, depth: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.config.depth_feature_is_tokens:
            tokens = depth.to(device=device)
            if tokens.ndim == 4 and tokens.shape[1] == 1:
                tokens = tokens[:, 0]
            if tokens.ndim != 3:
                raise ValueError(
                    f"Precomputed DeFM depth tokens must be [B,K,C], got {tuple(tokens.shape)}"
                )
            expected = int(self.config.target_grid_h) * int(self.config.target_grid_w)
            if tokens.shape[1] != expected:
                raise ValueError(f"DeFM token mismatch: got {tokens.shape[1]}, expected {expected}")
            if tokens.shape[2] != int(self.config.defm_feature_dim):
                raise ValueError(f"DeFM dim mismatch: got {tokens.shape[2]}, expected {self.config.defm_feature_dim}")
            return tokens.to(device=device, dtype=dtype)
        if self.defm_model is None:
            raise RuntimeError("DeFM model is not initialized")
        model = self.defm_model.to(device)
        x = self._depth_to_defm(depth, device=device)
        context = torch.inference_mode() if self._defm_fully_frozen else nullcontext()
        with context:
            out = model.get_intermediate_layers(x.float(), n=1, reshape=True, return_class_token=True)
        spatial, _cls = out[0]
        tokens = einops.rearrange(spatial, "b c h w -> b (h w) c")
        expected = int(self.config.target_grid_h) * int(self.config.target_grid_w)
        if tokens.shape[1] != expected:
            raise ValueError(f"DeFM token mismatch: got {tokens.shape[1]}, expected {expected}")
        if tokens.shape[2] != int(self.config.defm_feature_dim):
            raise ValueError(f"DeFM dim mismatch: got {tokens.shape[2]}, expected {self.config.defm_feature_dim}")
        tokens = tokens.to(device=device, dtype=dtype)
        return tokens.clone() if self._defm_fully_frozen else tokens

    def _pos2d(self, grid_hw: tuple[int, int], batch_size: int, dtype: torch.dtype, device: torch.device, top: bool) -> Tensor:
        h, w = int(grid_hw[0]), int(grid_hw[1])
        dummy = torch.zeros(batch_size, int(self.config.dim_model), h, w, dtype=dtype, device=device)
        pos_fn = self.top_pos_embed if top else self.wrist_pos_embed
        pos = pos_fn(dummy).to(dtype=dtype, device=device)
        if pos.shape[0] == 1 and batch_size != 1:
            pos = pos.expand(batch_size, -1, -1, -1)
        return einops.rearrange(pos, "b c h w -> b (h w) c")

    def _modality(self, idx: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.modality_embed.weight[idx].to(dtype=dtype, device=device).view(1, 1, -1)

    def _encode_vae_latent(self, batch: dict[str, Tensor], batch_size: int):
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            parts = [cls_embed]
            if self.config.robot_state_feature:
                parts.append(self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
            parts.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_encoder_input = torch.cat(parts, axis=1)
            pos_embed = self.vae_encoder_pos_enc[:, : vae_encoder_input.shape[1]].clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            return latent_sample, mu, log_sigma_x2

        device = batch[OBS_STATE].device
        latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=device)
        return latent_sample, None, None

    def _encode_wrist_resnet_memories(self, batch: dict[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
        memories: list[tuple[Tensor, Tensor]] = []
        for view_idx, key in enumerate(self.wrist_image_keys):
            img = batch[key]
            cam_features = self.backbone(_normalize_imagenet_rgb(img))["feature_map"]
            wrist_hw = (int(self.config.wrist_grid_h), int(self.config.wrist_grid_w))
            if tuple(cam_features.shape[-2:]) != wrist_hw:
                cam_features = F.interpolate(cam_features, size=wrist_hw, mode="bilinear", align_corners=False)
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            if cam_pos_embed.shape[0] == 1 and img.shape[0] != 1:
                cam_pos_embed = cam_pos_embed.expand(img.shape[0], -1, -1, -1)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            tokens = einops.rearrange(cam_features, "b c h w -> b (h w) c")
            pos = einops.rearrange(cam_pos_embed, "b c h w -> b (h w) c")
            view = self.rgb_view_embed.weight[view_idx].to(dtype=tokens.dtype, device=tokens.device).view(1, 1, -1)
            memories.append((tokens, pos + view + self._modality(3, tokens.dtype, tokens.device)))
        return memories

    def _encode_wrist_dino_memories(
        self,
        batch: dict[str, Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[tuple[Tensor, Tensor]]:
        if self.wrist_dino_input_proj is None:
            if self.wrist_image_keys:
                raise RuntimeError("wrist_encoder=dino but wrist_dino_input_proj is not initialized")
            return []
        memories: list[tuple[Tensor, Tensor]] = []
        wrist_hw = (int(self.config.wrist_grid_h), int(self.config.wrist_grid_w))
        for view_idx, key in enumerate(self.wrist_image_keys):
            tokens = self.wrist_dino_input_proj(
                self._encode_dino_tokens(batch[key], wrist_hw, dtype=dtype, device=device)
            )
            pos = self._pos2d(
                wrist_hw,
                tokens.shape[0],
                tokens.dtype,
                tokens.device,
                top=False,
            )
            view = self.rgb_view_embed.weight[view_idx].to(
                dtype=tokens.dtype, device=tokens.device
            ).view(1, 1, -1)
            memories.append(
                (tokens, pos + view + self._modality(3, tokens.dtype, tokens.device))
            )
        return memories

    def _encode_stereo_memories(self, batch: dict[str, Tensor], dtype: torch.dtype, device: torch.device) -> list[tuple[Tensor, Tensor]]:
        top_grid = (int(self.config.target_grid_h), int(self.config.target_grid_w))
        memories: list[tuple[Tensor, Tensor]] = []
        top_rgb = None
        top_pos = None
        if self.config.use_top_rgb:
            if self.dino_input_proj is None:
                raise RuntimeError("Top RGB is enabled but dino_input_proj is not initialized")
            top_rgb = self.dino_input_proj(
                self._encode_dino_tokens(
                    batch[self.config.top_rgb_feature_key], top_grid, dtype=dtype, device=device
                )
            )
            top_pos = self._pos2d(
                top_grid, top_rgb.shape[0], top_rgb.dtype, top_rgb.device, top=True
            ) + self._modality(0, top_rgb.dtype, top_rgb.device)

        depth = None
        depth_pos = None
        if self.config.use_depth:
            if self.defm_input_proj is None:
                raise RuntimeError("Depth is enabled but defm_input_proj is not initialized")
            depth = self.defm_input_proj(
                self._encode_defm_tokens(batch[self.config.depth_feature_key], dtype=dtype, device=device)
            )
            depth_pos = self._pos2d(
                top_grid,
                depth.shape[0],
                depth.dtype,
                depth.device,
                top=True,
            ) + self._modality(1, depth.dtype, depth.device)

        if top_rgb is not None and depth is not None:
            if self.config.fusion_mode == "direct":
                memories.append((top_rgb, top_pos))
                memories.append((depth, depth_pos))
            else:
                fused = top_rgb
                depth_work = depth
                aligned_pos = self._pos2d(top_grid, top_rgb.shape[0], top_rgb.dtype, top_rgb.device, top=True)
                for block in self.fusion_blocks:
                    fused, depth_work = block(fused, depth_work, aligned_pos)
                memories.append((fused, aligned_pos + self._modality(2, fused.dtype, fused.device)))
        elif top_rgb is not None:
            memories.append((top_rgb, top_pos))
        elif depth is not None:
            memories.append((depth, depth_pos))
        else:
            raise RuntimeError("StereoACT has neither Top RGB nor depth memories")

        if self.config.wrist_encoder == "dino":
            memories.extend(self._encode_wrist_dino_memories(batch, dtype=dtype, device=device))
        else:
            memories.extend(self._encode_wrist_resnet_memories(batch))
        return memories

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

        first_key = (
            self.config.top_rgb_feature_key
            if self.config.use_top_rgb
            else self.config.depth_feature_key
        )
        batch_size = batch[first_key].shape[0]
        latent_sample, mu, log_sigma_x2 = self._encode_vae_latent(batch, batch_size)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = [
            pos.view(1, 1, -1).expand(1, batch_size, -1)
            for pos in self.encoder_1d_feature_pos_embed.weight
        ]
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        dtype = encoder_in_tokens[0].dtype
        device = encoder_in_tokens[0].device
        for tokens, pos in self._encode_stereo_memories(batch, dtype=dtype, device=device):
            encoder_in_tokens.append(einops.rearrange(tokens, "b k d -> k b d"))
            encoder_in_pos_embed.append(einops.rearrange(pos, "b k d -> k b d"))

        encoder_in_tokens_t = torch.cat(
            [x.unsqueeze(0) if x.ndim == 2 else x for x in encoder_in_tokens], dim=0
        )
        encoder_in_pos_t = torch.cat(
            [x.unsqueeze(0) if x.ndim == 2 else x for x in encoder_in_pos_embed], dim=0
        )
        encoder_out = self.encoder(encoder_in_tokens_t, pos_embed=encoder_in_pos_t)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_t.dtype,
            device=encoder_in_pos_t.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_t,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        actions = self.action_head(decoder_out.transpose(0, 1))
        return actions, (mu, log_sigma_x2)


class StereoACTPolicy(ACTHFPolicy):
    config_class = StereoACTConfig
    name = "stereo_act"

    def __init__(self, config: StereoACTConfig, **kwargs):
        if str(config.gripper_indices).strip().lower() == "auto":
            config.gripper_indices = config._infer_gripper_indices()
        if str(config.gripper_indices).strip().lower() == "auto":
            raise ValueError("Cannot infer gripper indices before action feature metadata is available")
        super().__init__(config, **kwargs)
        self.config = config
        self.model = StereoACT(config)
        self.reset()
        self._action_queue = deque([], maxlen=max(1, int(self.config.n_action_steps)))
        # Debug-only cache of the most recent *raw* model prediction.  The ROS
        # runtime reads this after select_action(), so exposing a raw action
        # chunk does not require a second expensive model forward.
        self.last_predicted_action_chunk = None
        self.last_predicted_action_chunk_sequence = 0

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        actions = super().predict_action_chunk(batch)
        self.last_predicted_action_chunk = actions.detach().clone()
        self.last_predicted_action_chunk_sequence += 1
        return actions

    def get_optim_params(self):
        named = list(self.named_parameters())
        groups = []
        base = [
            p for n, p in named
            if p.requires_grad
            and not n.startswith("model.backbone")
            and not n.startswith("model.dino_model")
            and not n.startswith("model.defm_model")
        ]
        wrist = [p for n, p in named if p.requires_grad and n.startswith("model.backbone")]
        dino = [p for n, p in named if p.requires_grad and n.startswith("model.dino_model")]
        defm = [p for n, p in named if p.requires_grad and n.startswith("model.defm_model")]
        if base:
            groups.append({"params": base, "name": "stereo_act"})
        if wrist:
            groups.append({"params": wrist, "lr": self.config.optimizer_lr_backbone, "name": "wrist_resnet"})
        if dino:
            groups.append({"params": dino, "lr": self.config.dino_finetune_lr, "name": "dino"})
        if defm:
            groups.append({"params": defm, "lr": self.config.defm_finetune_lr, "name": "defm"})
        for group in groups:
            lr = group.get("lr", self.config.optimizer_lr)
            count = sum(parameter.numel() for parameter in group["params"])
            print(f"[stereo-act] optimizer_group={group['name']} lr={lr:.2e} params={count}")
        return groups
