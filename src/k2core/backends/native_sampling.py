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
    selected = tuple(schedule[-(1 + int(index * stride))] for index in range(steps))
    return (*selected, 0.0)


def partial_denoise_sigmas(
    steps: int,
    denoise: float,
) -> tuple[float, ...]:
    """Match ComfyUI's KSampler schedule truncation for img2img strength."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be between zero and one")
    if denoise == 0.0:
        return (0.0,)
    if denoise > 0.9999:
        return simple_sigmas(steps)
    expanded_steps = int(steps / denoise)
    return simple_sigmas(expanded_steps)[-(steps + 1) :]


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


def euler_flow_image_sample(
    model: Callable[[Any, Any], Any],
    source_latent: Any,
    noise: Any,
    sigmas: Sequence[float],
    denoise_mask: Any,
    *,
    checkpoint: Callable[[DenoisingCheckpoint], None] | None = None,
) -> Any:
    """Euler flow img2img with Comfy-equivalent source and soft-mask handling."""

    if not sigmas:
        raise ValueError("image sampling requires a sigma schedule")
    if len(sigmas) == 1:
        if float(sigmas[0]) != 0.0:
            raise ValueError("a one-value image schedule must contain only zero")
        return source_latent
    sigma_start = float(sigmas[0])
    current = noise * sigma_start + source_latent * (1.0 - sigma_start)
    inverse_mask = 1.0 - denoise_mask
    for step, (sigma, sigma_next) in enumerate(zip(sigmas, sigmas[1:])):
        sigma = float(sigma)
        sigma_next = float(sigma_next)
        if sigma <= 0.0:
            raise ValueError("non-terminal image sampling sigmas must be positive")
        noisy_source = noise * sigma + source_latent * (1.0 - sigma)
        model_input = current * denoise_mask + noisy_source * inverse_mask
        velocity = model(model_input, sigma)
        denoised = model_input - velocity * sigma
        masked_denoised = denoised * denoise_mask + source_latent * inverse_mask
        derivative = (model_input - masked_denoised) / sigma
        if checkpoint is not None:
            checkpoint(
                DenoisingCheckpoint(
                    step=step,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    latent=model_input,
                    model_output=velocity,
                )
            )
        current = model_input + derivative * (sigma_next - sigma)
    return current


def prepare_noise(shape, seed: int, *, device: str, dtype):
    """Match the reference's seeded Torch generator and randn call."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("native noise generation requires PyTorch") from error
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
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
    "euler_flow_image_sample",
    "euler_flow_sample",
    "flux_time_shift",
    "partial_denoise_sigmas",
    "prepare_noise",
    "simple_sigmas",
]
