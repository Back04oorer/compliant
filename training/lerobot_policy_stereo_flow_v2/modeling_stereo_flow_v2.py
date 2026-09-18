from __future__ import annotations

import math
import os
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot_policy_stereo_shared import (
    OneShotGradientAudit,
    StereoFusedObservationEncoder,
    TrainableParameterEMA,
)

from .configuration_stereo_flow_v2 import StereoFlowV2Config


class FullJacobianRTCProcessor(RTCProcessor):
    """RTC guidance with the denoiser VJP required by the RTC paper.

    LeRobot's current processor marks ``x_t`` as requiring gradients only after
    evaluating the denoiser.  That drops the denoiser Jacobian from the VJP and
    reduces the correction to an identity approximation.  Keep this fix local
    to Stereo-Flow V2 so the environment package remains untouched.
    """

    full_jacobian: bool = True

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        time: float | Tensor,
        original_denoise_step_partial,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.detach().clone()
        squeezed = x_t.ndim < 3
        if squeezed:
            x_t = x_t.unsqueeze(0)
        previous = prev_chunk_left_over.detach()
        if previous.ndim < 3:
            previous = previous.unsqueeze(0)
        previous = previous.to(device=x_t.device, dtype=x_t.dtype)

        prefix_horizon = (
            int(self.rtc_config.execution_horizon)
            if execution_horizon is None
            else int(execution_horizon)
        )
        prefix_horizon = min(prefix_horizon, int(previous.shape[1]))
        batch_size, chunk_size, action_dim = x_t.shape
        if int(previous.shape[0]) != batch_size:
            raise ValueError(
                f"RTC batch mismatch: previous={tuple(previous.shape)} current={tuple(x_t.shape)}"
            )
        if int(previous.shape[1]) > chunk_size or int(previous.shape[2]) > action_dim:
            raise ValueError(
                f"RTC previous chunk is larger than current chunk: previous={tuple(previous.shape)} "
                f"current={tuple(x_t.shape)}"
            )
        if tuple(previous.shape) != tuple(x_t.shape):
            padded = x_t.new_zeros((batch_size, chunk_size, action_dim))
            padded[:, : previous.shape[1], : previous.shape[2]] = previous
            previous = padded

        delay = max(0, min(int(inference_delay), chunk_size))
        weights = self.get_prefix_weights(delay, prefix_horizon, chunk_size).to(
            device=x_t.device,
            dtype=x_t.dtype,
        )[None, :, None]
        time_tensor = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)

        with torch.enable_grad():
            # This must happen before the denoiser forward; otherwise its
            # Jacobian with respect to x_t is absent from the autograd graph.
            x_t.requires_grad_(True)
            velocity = original_denoise_step_partial(x_t)
            predicted_clean_action = x_t - time_tensor * velocity
            error = (previous - predicted_clean_action) * weights
            correction = torch.autograd.grad(
                predicted_clean_action,
                x_t,
                grad_outputs=error.detach(),
                retain_graph=False,
                create_graph=False,
            )[0]

        tau = 1.0 - time_tensor
        max_weight = torch.as_tensor(
            float(self.rtc_config.max_guidance_weight),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        squared_one_minus_tau = (1.0 - tau).square()
        inverse_r2 = (squared_one_minus_tau + tau.square()) / squared_one_minus_tau
        coefficient = torch.nan_to_num((1.0 - tau) / tau, posinf=float(max_weight.item()))
        guidance_weight = torch.minimum(
            torch.nan_to_num(coefficient * inverse_r2, posinf=float(max_weight.item())),
            max_weight,
        )
        result = velocity - guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)
            predicted_clean_action = predicted_clean_action.squeeze(0)
            correction = correction.squeeze(0)
            error = error.squeeze(0)
        self.track(
            time=time,
            x1_t=predicted_clean_action,
            correction=correction,
            err=error,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=delay,
            execution_horizon=prefix_horizon,
            full_jacobian=True,
        )
        return result


def _sinusoidal_time_embedding(time: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = time.float().unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1])) if embedding.shape[-1] < dim else embedding


def _adaln(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _xavier_trainable(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            nn.init.xavier_uniform_(child.out_proj.weight)
            if child.out_proj.bias is not None:
                nn.init.zeros_(child.out_proj.bias)


class FlowDecoderBlock(nn.Module):
    def __init__(self, config: StereoFlowV2Config) -> None:
        super().__init__()
        dim = int(config.flow_decoder_dim)
        heads = int(config.flow_decoder_heads)
        dropout = float(config.flow_decoder_dropout)
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(config.flow_decoder_ffn_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config.flow_decoder_ffn_dim), dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def reset_parameters(self) -> None:
        _xavier_trainable(self)
        nn.init.zeros_(self.time_modulation[-1].weight)
        nn.init.zeros_(self.time_modulation[-1].bias)

    def forward(
        self,
        action_tokens: Tensor,
        memory: Tensor,
        time_embedding: Tensor,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        shift_self, scale_self, shift_cross, scale_cross, shift_ffn, scale_ffn = (
            self.time_modulation(time_embedding).chunk(6, dim=-1)
        )
        self_input = _adaln(self.self_norm(action_tokens), shift_self, scale_self)
        residual = self.self_attention(
            self_input,
            self_input,
            self_input,
            key_padding_mask=action_is_pad,
            need_weights=False,
        )[0]
        action_tokens = action_tokens + self.dropout(residual)
        residual = self.cross_attention(
            _adaln(self.cross_norm(action_tokens), shift_cross, scale_cross),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        action_tokens = action_tokens + self.dropout(residual)
        residual = self.ffn(_adaln(self.ffn_norm(action_tokens), shift_ffn, scale_ffn))
        return action_tokens + self.dropout(residual)


class ConditionalFlowDecoder(nn.Module):
    def __init__(self, config: StereoFlowV2Config) -> None:
        super().__init__()
        self.config = config
        dim = int(config.flow_decoder_dim)
        action_dim = int(config.action_feature.shape[0])
        self.action_input = nn.Linear(action_dim, dim)
        self.action_position = nn.Parameter(torch.empty(1, int(config.horizon), dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList([FlowDecoderBlock(config) for _ in range(config.flow_decoder_layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.velocity_head = nn.Linear(dim, action_dim)
        _xavier_trainable(self)
        nn.init.normal_(self.action_position, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.normal_(self.velocity_head.weight, std=0.01)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        noisy_actions: Tensor,
        flow_time: Tensor,
        memory: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        tokens = self.action_input(noisy_actions) + self.action_position
        time_embedding = self.time_mlp(
            _sinusoidal_time_embedding(flow_time, int(self.config.flow_decoder_dim)).to(tokens.dtype)
        )
        for block in self.blocks:
            tokens = block(tokens, memory, time_embedding, action_is_pad)
        return self.velocity_head(self.output_norm(tokens)).float()


class StereoFlowV2Model(nn.Module):
    def __init__(self, config: StereoFlowV2Config) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = StereoFusedObservationEncoder(config)
        self.state_projection = nn.Sequential(
            nn.Linear(config.robot_state_feature.shape[0], config.dim_model),
            nn.LayerNorm(config.dim_model),
        )
        self.state_position = nn.Parameter(torch.empty(1, 1, config.dim_model))
        self.missing_wrist = nn.Parameter(torch.empty(1, 1, config.dim_model))
        self.condition_encoder = ACTEncoder(config)
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(config.dim_model),
            nn.Linear(config.dim_model, config.flow_decoder_dim),
        )
        self.flow_decoder = ConditionalFlowDecoder(config)
        self.rtc_processor = self._make_rtc_processor() if config.rtc_enabled else None
        self._initialize_condition_path()

    def _make_rtc_processor(self) -> FullJacobianRTCProcessor:
        return FullJacobianRTCProcessor(
            RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule[
                    self.config.rtc_prefix_attention_schedule
                ],
                max_guidance_weight=float(self.config.rtc_max_guidance_weight),
                execution_horizon=int(self.config.rtc_execution_horizon),
                debug=bool(self.config.rtc_debug),
            )
        )

    def set_rtc_enabled(self, enabled: bool) -> None:
        self.config.rtc_enabled = bool(enabled)
        self.rtc_processor = self._make_rtc_processor() if enabled else None

    def _initialize_condition_path(self) -> None:
        encoder = self.observation_encoder
        modules = [
            encoder.dino_input_proj,
            encoder.defm_input_proj,
            encoder.wrist_dino_input_proj,
            encoder.encoder_img_feat_input_proj,
            encoder.fusion_blocks,
            self.state_projection,
            self.condition_encoder,
            self.memory_projection,
        ]
        for module in modules:
            if module is not None:
                _xavier_trainable(module)
        nn.init.xavier_uniform_(encoder.modality_embed.weight)
        nn.init.xavier_uniform_(encoder.rgb_view_embed.weight)
        nn.init.normal_(self.state_position, std=0.02)
        nn.init.normal_(self.missing_wrist, std=0.02)

    def encode_condition(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim != 3 or state.shape[1] != 1:
            raise ValueError(f"Stereo-Flow V2 expects state [B,1,D], got {tuple(state.shape)}")
        batch_size = state.shape[0]
        flat: dict[str, Tensor] = {}
        for key in self.config.stereo_observation_keys:
            if key in {OBS_STATE, OBS_ENV_STATE}:
                continue
            value = batch[key]
            if value.shape[:2] != (batch_size, 1):
                raise ValueError(f"{key} has incompatible observation axes: {tuple(value.shape)}")
            flat[key] = value[:, 0]
        encoded = self.observation_encoder(flat)
        top_tokens, top_pos = encoded["top_tokens"], encoded["top_pos"]
        wrist_tokens, wrist_pos = encoded["wrist_tokens"], encoded["wrist_pos"]
        if wrist_tokens is None or wrist_pos is None:
            raise RuntimeError("Stereo-Flow V2 requires wrist tokens")
        if self.training and float(self.config.wrist_modality_dropout) > 0.0:
            missing = torch.rand(batch_size, device=wrist_tokens.device) < float(
                self.config.wrist_modality_dropout
            )
            replacement = self.missing_wrist.to(dtype=wrist_tokens.dtype).expand_as(wrist_tokens)
            wrist_tokens = torch.where(missing[:, None, None], replacement, wrist_tokens)
        state_token = self.state_projection(state[:, 0]).unsqueeze(1).to(dtype=top_tokens.dtype)
        state_pos = self.state_position.to(dtype=top_tokens.dtype).expand(batch_size, -1, -1)
        tokens = torch.cat([state_token, top_tokens, wrist_tokens], dim=1)
        positions = torch.cat([state_pos, top_pos, wrist_pos], dim=1)
        wrist_count = len(self.observation_encoder.wrist_image_keys)
        expected_tokens = (
            1
            + int(self.config.target_grid_h) * int(self.config.target_grid_w)
            + wrist_count * int(self.config.wrist_grid_h) * int(self.config.wrist_grid_w)
        )
        expected_shape = (expected_tokens, int(self.config.dim_model))
        if tokens.shape[1:] != expected_shape:
            raise RuntimeError(
                "Unexpected condition token shape: "
                f"expected [B,{expected_tokens},{self.config.dim_model}] for "
                f"{wrist_count} wrist camera(s), got {tuple(tokens.shape)}"
            )
        memory = self.condition_encoder(
            tokens.transpose(0, 1), pos_embed=positions.transpose(0, 1)
        ).transpose(0, 1)
        return self.memory_projection(memory)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch[ACTION]
        action_is_pad = batch["action_is_pad"].bool()
        if actions.shape[1:] != (self.config.horizon, self.config.action_feature.shape[0]):
            raise ValueError(f"Invalid action shape: {tuple(actions.shape)}")
        memory = self.encode_condition(batch)
        noise = torch.randn_like(actions)
        beta = torch.distributions.Beta(
            torch.tensor(float(self.config.flow_time_beta_alpha), device=actions.device),
            torch.tensor(float(self.config.flow_time_beta_beta), device=actions.device),
        )
        flow_time = beta.sample((actions.shape[0],)).float()
        flow_time = flow_time * float(self.config.flow_time_scale) + float(self.config.flow_time_offset)
        expanded_time = flow_time[:, None, None].to(actions.dtype)
        noisy_actions = expanded_time * noise + (1.0 - expanded_time) * actions
        noisy_actions = noisy_actions.masked_fill(action_is_pad.unsqueeze(-1), 0.0)
        target_velocity = noise - actions
        prediction = self.flow_decoder(noisy_actions, flow_time, memory, action_is_pad)
        squared_error = F.mse_loss(prediction, target_velocity.float(), reduction="none")
        valid = (~action_is_pad).unsqueeze(-1).to(squared_error.dtype)
        denominator = valid.sum() * squared_error.shape[-1]
        return (squared_error * valid).sum() / denominator.clamp_min(1.0)

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
    ) -> Tensor:
        memory = self.encode_condition(batch)
        batch_size = batch[OBS_STATE].shape[0]
        shape = (batch_size, self.config.horizon, self.config.action_feature.shape[0])
        x_t = torch.randn(shape, device=memory.device, dtype=memory.dtype) if noise is None else noise.to(memory)
        if tuple(x_t.shape) != shape:
            raise ValueError(f"Invalid flow noise shape: {tuple(x_t.shape)}")
        steps = int(self.config.flow_num_steps)
        dt = -1.0 / steps
        for step in range(steps):
            flow_time = torch.full(
                (batch_size,), 1.0 + step * dt, device=x_t.device, dtype=torch.float32
            )
            def velocity_fn(input_actions: Tensor, current_time: Tensor = flow_time) -> Tensor:
                return self.flow_decoder(input_actions, current_time, memory)

            if self.rtc_processor is not None and prev_chunk_left_over is not None:
                velocity = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=int(inference_delay),
                    time=float(flow_time[0].item()),
                    original_denoise_step_partial=velocity_fn,
                    execution_horizon=execution_horizon,
                )
            else:
                velocity = velocity_fn(x_t)
            x_t = x_t + dt * velocity.to(dtype=x_t.dtype)
        return x_t[:, : int(self.config.n_action_steps)].detach()


class StereoFlowV2Policy(PreTrainedPolicy):
    config_class = StereoFlowV2Config
    name = "stereo_flow_v2"

    def __init__(self, config: StereoFlowV2Config, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.flow = StereoFlowV2Model(config)
        self.ema = TrainableParameterEMA(self, decay=config.ema_decay)
        self._gradient_audit = None
        if os.environ.get("STEREO_GRAD_AUDIT") == "1":
            self._gradient_audit = OneShotGradientAudit(
                self,
                categories={
                    "fusion_projection": lambda name: name.startswith("flow.observation_encoder.")
                    and not name.startswith(
                        ("flow.observation_encoder.dino_model", "flow.observation_encoder.defm_model", "flow.observation_encoder.backbone")
                    ),
                    "wrist": lambda name: name.startswith("flow.observation_encoder.backbone"),
                    "condition_encoder": lambda name: name.startswith(
                        ("flow.state_projection", "flow.condition_encoder", "flow.memory_projection")
                    ),
                    "flow_decoder": lambda name: name.startswith("flow.flow_decoder"),
                },
                frozen_prefixes=(
                    "flow.observation_encoder.dino_model",
                    "flow.observation_encoder.defm_model",
                ),
            )
        self.reset()

    @staticmethod
    def _no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
        lowered = name.lower()
        return (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in lowered
            or "position" in lowered
            or "embedding" in lowered
            or "missing_wrist" in lowered
        )

    def get_optim_params(self):
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            wrist_backbone = name.startswith("flow.observation_encoder.backbone")
            no_decay = self._no_weight_decay(name, parameter)
            groups.setdefault((wrist_backbone, no_decay), []).append(parameter)
        result = []
        for (wrist_backbone, no_decay), parameters in sorted(groups.items()):
            result.append(
                {
                    "params": parameters,
                    "lr": self.config.optimizer_lr_backbone if wrist_backbone else self.config.optimizer_lr,
                    "weight_decay": 0.0 if no_decay else self.config.optimizer_weight_decay,
                    "name": f"{'wrist' if wrist_backbone else 'main'}_{'nodecay' if no_decay else 'decay'}",
                }
            )
        return result

    @torch.no_grad()
    def update(self) -> None:
        self.ema.step(self)
        if self._gradient_audit is not None:
            self._gradient_audit.verify()

    def reset(self) -> None:
        self._queues = {key: deque(maxlen=1) for key in self.config.stereo_observation_keys}
        self._queues[ACTION] = deque(maxlen=self.config.n_action_steps)
        if self.flow.rtc_processor is not None:
            self.flow.rtc_processor.reset_tracker()

    def set_rtc_enabled(self, enabled: bool) -> None:
        self.flow.set_rtc_enabled(enabled)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
    ) -> Tensor:
        observations = {key: value for key, value in batch.items() if key != ACTION}
        self._queues = populate_queues(self._queues, observations)
        stacked = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in self.config.stereo_observation_keys
        }
        with self.ema.average_parameters(self):
            return self.flow.generate_actions(
                stacked,
                noise=noise,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if self.flow.rtc_processor is not None:
            raise RuntimeError("RTC requires predict_action_chunk with an asynchronous action executor")
        if not self._queues[ACTION]:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = dict(batch)
        for key in self.config.stereo_observation_keys:
            feature = self.config.input_features[key]
            if batch[key].ndim == len(feature.shape) + 1:
                batch[key] = batch[key].unsqueeze(1)
        return self.flow.compute_loss(batch), None
