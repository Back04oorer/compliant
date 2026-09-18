from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn


class TrainableParameterEMA(nn.Module):
    """Checkpointable EMA for trainable parameters only.

    Foundation encoders in the stereo policies are frozen and large.  Keeping a
    deepcopy of the complete policy would duplicate DINO and DeFM on the GPU.
    This module stores shadows only for parameters with ``requires_grad=True``
    and temporarily swaps them into the live policy for rollout.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        decay: float | None = None,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        super().__init__()
        if decay is not None and not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.fixed_decay = None if decay is None else float(decay)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.parameter_names = [name for name, value in module.named_parameters() if value.requires_grad]
        if not self.parameter_names:
            raise ValueError("EMA requires at least one trainable parameter")
        parameters = dict(module.named_parameters())
        for index, name in enumerate(self.parameter_names):
            self.register_buffer(self._shadow_name(index), parameters[name].detach().clone(), persistent=True)
        self.register_buffer("optimization_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("last_decay", torch.zeros((), dtype=torch.float32), persistent=True)

    @staticmethod
    def _shadow_name(index: int) -> str:
        return f"shadow_{index:05d}"

    def _decay(self) -> float:
        if self.fixed_decay is not None:
            return self.fixed_decay
        step = max(0, int(self.optimization_step.item()) - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    def _resolve(self, module: nn.Module) -> list[Tensor]:
        parameters = dict(module.named_parameters())
        missing = [name for name in self.parameter_names if name not in parameters]
        if missing:
            raise RuntimeError(f"EMA parameter layout changed; missing={missing[:3]}")
        return [parameters[name] for name in self.parameter_names]

    @torch.no_grad()
    def step(self, module: nn.Module) -> None:
        decay = self._decay()
        for index, parameter in enumerate(self._resolve(module)):
            shadow = getattr(self, self._shadow_name(index))
            shadow.mul_(decay).add_(parameter.detach().to(dtype=shadow.dtype), alpha=1.0 - decay)
        self.last_decay.fill_(decay)
        self.optimization_step.add_(1)

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, module: nn.Module) -> Iterator[None]:
        parameters = self._resolve(module)
        originals = [parameter.detach().clone() for parameter in parameters]
        try:
            for index, parameter in enumerate(parameters):
                shadow = getattr(self, self._shadow_name(index))
                parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
            yield
        finally:
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)

    def extra_repr(self) -> str:
        schedule = f"fixed={self.fixed_decay}" if self.fixed_decay is not None else (
            f"power={self.power}, max={self.max_value}"
        )
        return f"parameters={len(self.parameter_names)}, {schedule}"
