from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class OneShotGradientAudit:
    """Assert one real backward pass reaches every intended trainable branch.

    This helper is only instantiated by smoke tests when
    ``STEREO_GRAD_AUDIT=1``.  It is deliberately not an ``nn.Module`` so it
    never changes checkpoint state dictionaries.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        categories: dict[str, Callable[[str], bool]],
        frozen_prefixes: tuple[str, ...],
    ) -> None:
        named_parameters = dict(module.named_parameters())
        incorrectly_trainable = [
            name
            for name, parameter in named_parameters.items()
            if name.startswith(frozen_prefixes) and parameter.requires_grad
        ]
        if incorrectly_trainable:
            raise RuntimeError(
                "Gradient audit: frozen foundation parameters are trainable: "
                f"{incorrectly_trainable[:3]}"
            )
        self._seen = {category: 0.0 for category in categories}
        self._handles = []
        matched = {category: 0 for category in categories}
        for name, parameter in named_parameters.items():
            if not parameter.requires_grad:
                continue
            for category, predicate in categories.items():
                if predicate(name):
                    matched[category] += 1
                    self._handles.append(parameter.register_hook(self._hook(category)))
                    break
        missing = [category for category, count in matched.items() if count == 0]
        if missing:
            raise RuntimeError(f"Gradient audit categories matched no parameters: {missing}")
        self._done = False

    def _hook(self, category: str):
        def record(gradient: Tensor) -> Tensor:
            norm = float(gradient.detach().float().norm().item())
            if not torch.isfinite(torch.tensor(norm)):
                raise RuntimeError(f"Gradient audit: non-finite gradient in {category}")
            self._seen[category] += norm
            return gradient

        return record

    def verify(self) -> None:
        if self._done:
            return
        zero = [category for category, norm in self._seen.items() if not norm > 0.0]
        if zero:
            raise RuntimeError(f"Gradient audit: no non-zero gradient reached {zero}")
        summary = " ".join(f"{key}={value:.3e}" for key, value in self._seen.items())
        print(f"[PASS] gradient audit: frozen DINO/DeFM; {summary}", flush=True)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._done = True
