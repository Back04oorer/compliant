from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from safetensors.torch import load_file

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot_policy_stereo_shared import StereoFusedObservationEncoder

from .configuration_stereo_flow import StereoFlowConfig


def _sinusoidal_time_embedding(time: Tensor, dim: int) -> Tensor:
    """Embed scalar flow time in [0, 1] without imposing a discrete schedule."""
    if time.ndim != 1:
        raise ValueError(f"Expected flow time [B], got {tuple(time.shape)}")
    half = dim // 2
    if half < 1:
        raise ValueError("Flow time embedding dimension must be >= 2")
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = time.float().unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class StereoTokenResampler(nn.Module):
    """Position-aware learned-query resampler returning tokens, not a flat vector."""

    def __init__(self, dim: int, num_latents: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(1, num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

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
        return self.output_norm(query)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FlowActionExpertBlock(nn.Module):
    """Bidirectional action attention followed by visual/state cross-attention."""

    def __init__(self, config: StereoFlowConfig) -> None:
        super().__init__()
        dim = int(config.dim_model)
        heads = int(config.action_expert_heads)
        dropout = float(config.action_expert_dropout)
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(config.action_expert_ffn_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config.action_expert_ffn_dim), dim),
        )
        self.dropout = nn.Dropout(dropout)
        # shift, scale and residual gate for self-attention, cross-attention and FFN.
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.time_modulation[-1].weight)
        nn.init.zeros_(self.time_modulation[-1].bias)

    def forward(
        self,
        action_tokens: Tensor,
        memory: Tensor,
        time_embedding: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        modulation = self.time_modulation(time_embedding).chunk(9, dim=-1)
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = modulation

        residual = self.self_attention(
            _modulate(self.self_norm(action_tokens), s1, c1),
            _modulate(self.self_norm(action_tokens), s1, c1),
            _modulate(self.self_norm(action_tokens), s1, c1),
            key_padding_mask=action_is_pad,
            need_weights=False,
        )[0]
        action_tokens = action_tokens + g1.unsqueeze(1) * self.dropout(residual)

        residual = self.cross_attention(
            _modulate(self.cross_norm(action_tokens), s2, c2),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        action_tokens = action_tokens + g2.unsqueeze(1) * self.dropout(residual)

        residual = self.ffn(_modulate(self.ffn_norm(action_tokens), s3, c3))
        return action_tokens + g3.unsqueeze(1) * self.dropout(residual)


class FlowActionExpert(nn.Module):
    def __init__(self, config: StereoFlowConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.dim_model)
        action_dim = int(config.action_feature.shape[0])
        self.action_input = nn.Linear(action_dim, dim)
        self.action_step_position = nn.Parameter(torch.empty(1, config.horizon, dim))
        nn.init.normal_(self.action_step_position, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.blocks = nn.ModuleList(
            [FlowActionExpertBlock(config) for _ in range(int(config.action_expert_layers))]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.action_output = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    def forward(
        self,
        noisy_actions: Tensor,
        flow_time: Tensor,
        memory: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1] != self.config.horizon:
            raise ValueError(
                f"Expected noisy actions [B,{self.config.horizon},A], got {tuple(noisy_actions.shape)}"
            )
        tokens = self.action_input(noisy_actions) + self.action_step_position
        time_embedding = self.time_mlp(
            _sinusoidal_time_embedding(flow_time, int(self.config.dim_model)).to(tokens.dtype)
        )
        for block in self.blocks:
            tokens = block(tokens, memory, time_embedding, action_is_pad=action_is_pad)
        return self.action_output(self.output_norm(tokens)).float()


class StereoFlowModel(nn.Module):
    def __init__(self, config: StereoFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = StereoFusedObservationEncoder(config)
        self.top_resampler = StereoTokenResampler(
            config.dim_model,
            config.top_pool_latents,
            config.pool_heads,
            config.dropout,
        )
        self.wrist_resampler = StereoTokenResampler(
            config.dim_model,
            config.wrist_pool_latents,
            config.pool_heads,
            config.dropout,
        )
        self.state_projection = nn.Sequential(
            nn.Linear(config.robot_state_feature.shape[0], config.dim_model),
            nn.LayerNorm(config.dim_model),
        )
        self.memory_type_embedding = nn.Embedding(3, config.dim_model)
        self.memory_norm = nn.LayerNorm(config.dim_model)
        self.action_expert = FlowActionExpert(config)
        self.rtc_processor = self._make_rtc_processor() if config.rtc_enabled else None

    def _make_rtc_processor(self) -> RTCProcessor:
        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule[
                self.config.rtc_prefix_attention_schedule
            ],
            max_guidance_weight=float(self.config.rtc_max_guidance_weight),
            execution_horizon=int(self.config.rtc_execution_horizon),
            debug=bool(self.config.rtc_debug),
        )
        return RTCProcessor(rtc_config)

    def set_rtc_enabled(self, enabled: bool) -> None:
        self.config.rtc_enabled = bool(enabled)
        self.rtc_processor = self._make_rtc_processor() if enabled else None

    def encode_condition(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim != 3:
            raise ValueError(f"Expected state [B,S,D], got {tuple(state.shape)}")
        batch_size, n_obs_steps = state.shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected n_obs_steps={self.config.n_obs_steps}, got {n_obs_steps}")

        flat: dict[str, Tensor] = {}
        for key in self.config.stereo_observation_keys:
            if key in {OBS_STATE, OBS_ENV_STATE}:
                continue
            value = batch[key]
            if value.shape[:2] != (batch_size, n_obs_steps):
                raise ValueError(f"{key} has incompatible observation axes: {tuple(value.shape)}")
            flat[key] = value.flatten(0, 1)

        encoded = self.observation_encoder(flat)
        top = self.top_resampler(encoded["top_tokens"], encoded["top_pos"])
        if encoded["wrist_tokens"] is None or encoded["wrist_pos"] is None:
            raise RuntimeError("Stereo-Flow requires wrist tokens")
        wrist = self.wrist_resampler(encoded["wrist_tokens"], encoded["wrist_pos"])
        state_token = self.state_projection(state.flatten(0, 1)).unsqueeze(1)

        type_embedding = self.memory_type_embedding.weight.to(dtype=top.dtype)
        top = top + type_embedding[0].view(1, 1, -1)
        wrist = wrist + type_embedding[1].view(1, 1, -1)
        state_token = state_token.to(dtype=top.dtype) + type_embedding[2].view(1, 1, -1)
        memory = torch.cat([top, wrist, state_token], dim=1)
        memory = memory.view(batch_size, n_obs_steps * memory.shape[1], memory.shape[2])
        return self.memory_norm(memory)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        if not {OBS_STATE, ACTION, "action_is_pad"}.issubset(batch):
            raise ValueError("Training batch is missing state, action, or action_is_pad")
        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[1] != self.config.horizon:
            raise ValueError(
                f"Expected action [B,{self.config.horizon},A], got {tuple(actions.shape)}"
            )
        action_is_pad = batch["action_is_pad"].bool()
        if action_is_pad.shape != actions.shape[:2]:
            raise ValueError(
                f"action_is_pad shape {tuple(action_is_pad.shape)} does not match {tuple(actions.shape[:2])}"
            )
        memory = self.encode_condition(batch)
        noise = torch.randn_like(actions)
        beta = torch.distributions.Beta(
            torch.tensor(float(self.config.flow_time_beta_alpha), device=actions.device),
            torch.tensor(float(self.config.flow_time_beta_beta), device=actions.device),
        )
        flow_time = beta.sample((actions.shape[0],)).to(dtype=torch.float32)
        minimum = float(self.config.flow_time_min)
        flow_time = flow_time * (1.0 - minimum) + minimum
        expanded_time = flow_time[:, None, None].to(dtype=actions.dtype)
        noisy_actions = expanded_time * noise + (1.0 - expanded_time) * actions
        target_velocity = noise - actions
        predicted_velocity = self.action_expert(
            noisy_actions,
            flow_time,
            memory,
            action_is_pad=action_is_pad,
        )
        squared_error = F.mse_loss(predicted_velocity, target_velocity.float(), reduction="none")
        valid = (~action_is_pad).unsqueeze(-1).to(dtype=squared_error.dtype)
        denominator = valid.sum() * squared_error.shape[-1]
        return (squared_error * valid).sum() / denominator.clamp_min(1.0)

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
    ) -> Tensor:
        memory = self.encode_condition(batch)
        batch_size = batch[OBS_STATE].shape[0]
        action_dim = int(self.config.action_feature.shape[0])
        if noise is None:
            x_t = torch.randn(
                batch_size,
                self.config.horizon,
                action_dim,
                device=memory.device,
                dtype=memory.dtype,
            )
        else:
            x_t = noise.to(device=memory.device, dtype=memory.dtype)
        if x_t.shape != (batch_size, self.config.horizon, action_dim):
            raise ValueError(f"Invalid flow noise shape: {tuple(x_t.shape)}")

        steps = int(self.config.flow_num_steps)
        dt = -1.0 / steps
        for step in range(steps):
            scalar_time = 1.0 + step * dt
            flow_time = torch.full(
                (batch_size,), scalar_time, device=x_t.device, dtype=torch.float32
            )

            def velocity_fn(input_actions: Tensor, current_time: Tensor = flow_time) -> Tensor:
                return self.action_expert(input_actions, current_time, memory)

            if self.rtc_processor is not None and prev_chunk_left_over is not None:
                velocity = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=int(inference_delay),
                    time=scalar_time,
                    original_denoise_step_partial=velocity_fn,
                    execution_horizon=execution_horizon,
                )
            else:
                velocity = velocity_fn(x_t)
            x_t = x_t + dt * velocity
        return x_t


class StereoFlowPolicy(PreTrainedPolicy):
    config_class = StereoFlowConfig
    name = "stereo_flow"

    def __init__(self, config: StereoFlowConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.flow = StereoFlowModel(config)
        self._maybe_initialize_visual_encoder()
        self.reset()

    def _maybe_initialize_visual_encoder(self) -> None:
        """Optionally warm-start only the proven Stereo-Diffusion visual path.

        The environment variable is intentionally not part of the serialized
        checkpoint config. A completed Stereo-Flow checkpoint must load without
        requiring its initialization source to remain present.
        """
        raw_path = os.environ.get("LFV_STEREO_FLOW_INIT_VISUAL_CHECKPOINT", "").strip()
        if not raw_path:
            return
        source = Path(raw_path).expanduser()
        if source.is_dir():
            source = source / "model.safetensors"
        if not source.is_file():
            raise FileNotFoundError(f"Stereo-Flow visual initialization checkpoint missing: {source}")
        source_state = load_file(str(source), device="cpu")
        target_state = self.state_dict()
        prefix = "diffusion.observation_encoder."
        loaded = {}
        skipped_shape = []
        for source_name, value in source_state.items():
            if not source_name.startswith(prefix):
                continue
            target_name = "flow.observation_encoder." + source_name[len(prefix) :]
            target_value = target_state.get(target_name)
            if target_value is None:
                continue
            if tuple(target_value.shape) != tuple(value.shape):
                skipped_shape.append((source_name, tuple(value.shape), tuple(target_value.shape)))
                continue
            loaded[target_name] = value
        if not loaded:
            raise RuntimeError(f"No compatible Stereo-Diffusion visual weights found in {source}")
        incompatible = self.load_state_dict(loaded, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key in loaded]
        if unexpected:
            raise RuntimeError(f"Unexpected visual initialization keys: {unexpected[:5]}")
        print(
            f"[stereo-flow] visual_init={source} tensors={len(loaded)} "
            f"shape_skips={len(skipped_shape)}"
        )

    def get_optim_params(self):
        named = list(self.named_parameters())
        groups = []
        base = [
            parameter
            for name, parameter in named
            if parameter.requires_grad
            and not name.startswith("flow.observation_encoder.backbone")
            and not name.startswith("flow.observation_encoder.dino_model")
            and not name.startswith("flow.observation_encoder.defm_model")
        ]
        wrist = [
            parameter
            for name, parameter in named
            if parameter.requires_grad and name.startswith("flow.observation_encoder.backbone")
        ]
        dino = [
            parameter
            for name, parameter in named
            if parameter.requires_grad and name.startswith("flow.observation_encoder.dino_model")
        ]
        defm = [
            parameter
            for name, parameter in named
            if parameter.requires_grad and name.startswith("flow.observation_encoder.defm_model")
        ]
        if base:
            groups.append({"params": base, "name": "stereo_flow"})
        if wrist:
            groups.append(
                {"params": wrist, "lr": self.config.optimizer_lr_backbone, "name": "wrist_resnet"}
            )
        if dino:
            groups.append({"params": dino, "lr": self.config.dino_finetune_lr, "name": "dino"})
        if defm:
            groups.append({"params": defm, "lr": self.config.defm_finetune_lr, "name": "defm"})
        return groups

    def reset(self) -> None:
        self._queues = {
            key: deque(maxlen=self.config.n_obs_steps)
            for key in self.config.stereo_observation_keys
        }
        self._queues[ACTION] = deque(maxlen=self.config.n_action_steps)
        if self.flow.rtc_processor is not None:
            self.flow.rtc_processor.reset_tracker()

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
            raise RuntimeError("RTC requires predict_action_chunk with an asynchronous ActionQueue")
        observations = {key: value for key, value in batch.items() if key != ACTION}
        self._queues = populate_queues(self._queues, observations)
        if not self._queues[ACTION]:
            stacked = {
                key: torch.stack(list(self._queues[key]), dim=1)
                for key in self.config.stereo_observation_keys
            }
            actions = self.flow.generate_actions(stacked, noise=noise)
            self._queues[ACTION].extend(actions[:, : self.config.n_action_steps].transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = dict(batch)
        for key in self.config.stereo_observation_keys:
            feature = self.config.input_features[key]
            if batch[key].ndim == len(feature.shape) + 1:
                batch[key] = batch[key].unsqueeze(1)
        return self.flow.compute_loss(batch), None
