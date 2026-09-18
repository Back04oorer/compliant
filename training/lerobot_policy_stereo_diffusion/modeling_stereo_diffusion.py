from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot_policy_stereo_shared import StereoFusedObservationEncoder, StereoTokenPooler

from .configuration_stereo_diffusion import StereoDiffusionConfig


class StereoDiffusionModel(nn.Module):
    def __init__(self, config: StereoDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = StereoFusedObservationEncoder(config)
        self.top_pooler = StereoTokenPooler(
            token_dim=config.dim_model,
            num_latents=config.top_pool_latents,
            output_dim=config.top_embedding_dim,
            num_heads=config.pool_heads,
            dropout=config.dropout,
        )
        self.wrist_pooler = StereoTokenPooler(
            token_dim=config.dim_model,
            num_latents=config.wrist_pool_latents,
            output_dim=config.wrist_embedding_dim,
            num_heads=config.pool_heads,
            dropout=config.dropout,
        )

        per_observation_dim = (
            config.robot_state_feature.shape[0]
            + config.top_embedding_dim
            + config.wrist_embedding_dim
        )
        if config.env_state_feature is not None:
            per_observation_dim += config.env_state_feature.shape[0]
        self.global_cond_dim = per_observation_dim * config.n_obs_steps
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=self.global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else int(config.num_inference_steps)
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
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
        top = self.top_pooler(encoded["top_tokens"], encoded["top_pos"])
        if encoded["wrist_tokens"] is None or encoded["wrist_pos"] is None:
            raise RuntimeError("Stereo-Diffusion checkpoint requires wrist tokens")
        wrist = self.wrist_pooler(encoded["wrist_tokens"], encoded["wrist_pos"])
        top = top.view(batch_size, n_obs_steps, -1)
        wrist = wrist.view(batch_size, n_obs_steps, -1)

        features = [state.to(dtype=top.dtype), top, wrist]
        if self.config.env_state_feature is not None:
            features.append(batch[OBS_ENV_STATE].to(dtype=top.dtype))
        condition = torch.cat(features, dim=-1).flatten(start_dim=1)
        if condition.shape[-1] != self.global_cond_dim:
            raise RuntimeError(
                f"global condition dim mismatch: got {condition.shape[-1]}, expected {self.global_cond_dim}"
            )
        return condition

    def conditional_sample(
        self,
        batch_size: int,
        *,
        global_cond: Tensor,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self.unet)
        sample = noise
        if sample is None:
            sample = torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                device=device,
                dtype=dtype,
                generator=generator,
            )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full((batch_size,), timestep, device=device, dtype=torch.long),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(
                model_output, timestep, sample, generator=generator
            ).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        global_cond = self._prepare_global_conditioning(batch)
        trajectory = self.conditional_sample(
            batch[OBS_STATE].shape[0], global_cond=global_cond, noise=noise
        )
        start = self.config.n_obs_steps - 1
        # Keep the complete usable denoised horizon for passive ROS debugging.
        # Execution semantics remain unchanged: select_action still enqueues only
        # n_action_steps below.  This lets an execute-N experiment distinguish
        # "replanned too early" from "the model itself predicted a stop" without
        # running a second stochastic diffusion forward.
        self.last_generated_action_horizon = trajectory[:, start:]
        return self.last_generated_action_horizon[:, : self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        if not {OBS_STATE, ACTION, "action_is_pad"}.issubset(batch):
            raise ValueError("Training batch is missing state, action, or action_is_pad")
        trajectory = batch[ACTION]
        if trajectory.ndim != 3 or trajectory.shape[1] != self.config.horizon:
            raise ValueError(
                f"Expected action [B,{self.config.horizon},D], got {tuple(trajectory.shape)}"
            )
        global_cond = self._prepare_global_conditioning(batch)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        prediction = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(prediction, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)
        return loss.mean()


class StereoDiffusionPolicy(PreTrainedPolicy):
    config_class = StereoDiffusionConfig
    name = "stereo_diffusion"

    def __init__(self, config: StereoDiffusionConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion = StereoDiffusionModel(config)
        self.reset()

    def get_optim_params(self):
        named = list(self.named_parameters())
        groups = []
        base = [
            parameter
            for name, parameter in named
            if parameter.requires_grad
            and not name.startswith("diffusion.observation_encoder.backbone")
            and not name.startswith("diffusion.observation_encoder.dino_model")
            and not name.startswith("diffusion.observation_encoder.defm_model")
        ]
        wrist = [
            parameter for name, parameter in named
            if parameter.requires_grad and name.startswith("diffusion.observation_encoder.backbone")
        ]
        dino = [
            parameter for name, parameter in named
            if parameter.requires_grad and name.startswith("diffusion.observation_encoder.dino_model")
        ]
        defm = [
            parameter for name, parameter in named
            if parameter.requires_grad and name.startswith("diffusion.observation_encoder.defm_model")
        ]
        if base:
            groups.append({"params": base, "name": "stereo_diffusion"})
        if wrist:
            groups.append({"params": wrist, "lr": self.config.optimizer_lr_backbone, "name": "wrist_resnet"})
        if dino:
            groups.append({"params": dino, "lr": self.config.dino_finetune_lr, "name": "dino"})
        if defm:
            groups.append({"params": defm, "lr": self.config.defm_finetune_lr, "name": "defm"})
        for group in groups:
            count = sum(parameter.numel() for parameter in group["params"])
            lr = group.get("lr", self.config.optimizer_lr)
            print(f"[stereo-diffusion] optimizer_group={group['name']} lr={lr:.2e} params={count}")
        return groups

    def reset(self) -> None:
        previous_sequence = int(getattr(self, "last_predicted_action_chunk_sequence", 0))
        self._queues = {
            key: deque(maxlen=self.config.n_obs_steps)
            for key in self.config.stereo_observation_keys
        }
        self._queues[ACTION] = deque(maxlen=self.config.n_action_steps)
        self.last_predicted_action_chunk = None
        self.last_predicted_action_chunk_sequence = previous_sequence

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        update_observation_queue: bool = True,
    ) -> Tensor:
        if update_observation_queue:
            observations = {key: value for key, value in batch.items() if key != ACTION}
            self._queues = populate_queues(self._queues, observations)
        missing = [key for key in self.config.stereo_observation_keys if not self._queues[key]]
        if missing:
            raise ValueError(f"Missing runtime observations: {missing}")
        stacked = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in self.config.stereo_observation_keys
        }
        return self.diffusion.generate_actions(stacked, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        observations = {key: value for key, value in batch.items() if key != ACTION}
        self._queues = populate_queues(self._queues, observations)
        if not self._queues[ACTION]:
            actions = self.predict_action_chunk(
                batch,
                noise=noise,
                update_observation_queue=False,
            )
            # Cache the full horizon produced by this exact DDIM sample.  Only
            # `actions` (the first n_action_steps) enters the execution queue.
            # The ROS debug publisher can therefore expose both the executed
            # prefix and the unexecuted continuation without a second forward.
            debug_horizon = getattr(self.diffusion, "last_generated_action_horizon", actions)
            self.last_predicted_action_chunk = debug_horizon.detach().clone()
            self.last_predicted_action_chunk_sequence += 1
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = dict(batch)
        for key in self.config.stereo_observation_keys:
            feature = self.config.input_features[key]
            if batch[key].ndim == len(feature.shape) + 1:
                batch[key] = batch[key].unsqueeze(1)
        return self.diffusion.compute_loss(batch), None
