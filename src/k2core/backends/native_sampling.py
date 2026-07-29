"""Deterministic clean Krea2 Turbo sampling primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


KREA2_FLOW_SHIFT = 1.15
KREA2_TRAINING_TIMESTEPS = 10_000


def flux_time_shift(shift: float, timestep: float) -> float:
    if not 0.0 < timestep <= 1.0:
        raise ValueError("normalized flow timestep must be in (0, 1]")
    numerator = math.exp(shift)
    return numerator / (numerator + (1.0 / timestep - 1.0))


def simple_sigmas(
    steps: int,
    *,
    shift: float = KREA2_FLOW_SHIFT,
    training_timesteps: int = KREA2_TRAINING_TIMESTEPS,
) -> tuple[float, ...]:
    """Reproduce ComfyUI's simple scheduler over the Krea2 Flux-style schedule."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if training_timesteps < steps:
        raise ValueError("training_timesteps must be at least steps")
    schedule = tuple(
        flux_time_shift(shift, index / training_timesteps)
        for index in range(1, training_timesteps + 1)
    )
    stride = len(schedule) / steps
    selected = tuple(
        schedule[-(1 + int(index * stride))]
        for index in range(steps)
    )
    return (*selected, 0.0)


@dataclass(frozen=True, slots=True)
class DenoisingCheckpoint:
    step: int
    sigma: float
    sigma_next: float
    latent: Any
    model_output: Any


def euler_flow_sample(
    model: Callable[[Any, Any], Any],
    latent: Any,
    sigmas: Sequence[float],
    *,
    checkpoint: Callable[[DenoisingCheckpoint], None] | None = None,
) -> Any:
    """Euler integration for CONST/flow prediction with CFG fixed at one."""

    if len(sigmas) < 2:
        raise ValueError("Euler sampling requires at least two sigma values")
    current = latent
    for step, (sigma, sigma_next) in enumerate(zip(sigmas, sigmas[1:])):
        model_output = model(current, sigma)
        if checkpoint is not None:
            checkpoint(
                DenoisingCheckpoint(
                    step=step,
                    sigma=float(sigma),
                    sigma_next=float(sigma_next),
                    latent=current,
                    model_output=model_output,
                )
            )
        current = current + model_output * (sigma_next - sigma)
    return current


def prepare_noise(shape, seed: int, *, device: str, dtype):
    """Match the reference's seeded Torch generator and randn call."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("native noise generation requires PyTorch") from error
    generator = torch.manual_seed(seed)
    return torch.randn(
        shape,
        dtype=dtype,
        layout=torch.strided,
        generator=generator,
        device=device,
    )


__all__ = [
    "DenoisingCheckpoint",
    "KREA2_FLOW_SHIFT",
    "KREA2_TRAINING_TIMESTEPS",
    "euler_flow_sample",
    "flux_time_shift",
    "prepare_noise",
    "simple_sigmas",
]
