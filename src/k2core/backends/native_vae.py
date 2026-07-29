"""Strict executable mapping and decode support for the Krea2 Qwen Image VAE."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Mapping

from k2core.backends.native_loading import NativeComponent
from k2core.inference.errors import ConfigurationError, WeightMappingError
from k2core.model import ArtifactKind


KREA2_VAE_LATENT_CHANNELS = 16
KREA2_VAE_SCALE_FACTOR = 8


@dataclass(frozen=True, slots=True)
class Krea2VAELoadReport:
    source_tensor_count: int
    mapped_tensor_count: int
    parameter_count: int


class NativeKrea2VAE:
    """Own the executable upstream Qwen Image autoencoder."""

    def __init__(self, *, model: Any, report: Krea2VAELoadReport) -> None:
        self.model = model
        self.report = report

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def decode(
        self,
        latent: Any,
        *,
        device: str = "cuda",
        output_range: str = "zero_one",
    ):
        """Decode normalized Krea2 latents into BCHW RGB tensors."""

        if self.model is None:
            raise RuntimeError("native Krea2 VAE has been unloaded")
        torch = _import_runtime()
        resolved_device = _resolve_execution_device(torch, device)
        if latent.ndim != 5:
            raise ValueError("Krea2 VAE latent must have shape (batch, 16, time, height, width)")
        if latent.shape[1] != KREA2_VAE_LATENT_CHANNELS:
            raise ValueError(
                "Krea2 VAE latent must have "
                f"{KREA2_VAE_LATENT_CHANNELS} channels, got {latent.shape[1]}"
            )
        if latent.shape[2] != 1:
            raise ValueError("clean Krea2 VAE decode currently requires one latent frame")
        if output_range not in {"minus_one_one", "zero_one"}:
            raise ValueError("output_range must be either 'minus_one_one' or 'zero_one'")

        self.model.to(device=resolved_device)
        dtype = self.model.post_quant_conv.weight.dtype
        prepared = latent.to(device=resolved_device, dtype=dtype)
        mean, std = self._latent_statistics(
            torch,
            device=resolved_device,
            dtype=dtype,
        )
        prepared = prepared * std + mean

        with torch.inference_mode():
            decoded = self.model.decode(prepared, return_dict=False)[0][:, :, 0]
            if output_range == "zero_one":
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
        return decoded

    def encode(
        self,
        pixels: Any,
        *,
        device: str = "cuda",
    ):
        """Encode BCHW RGB pixels in [0, 1] into normalized Krea2 latents."""

        if self.model is None:
            raise RuntimeError("native Krea2 VAE has been unloaded")
        torch = _import_runtime()
        resolved_device = _resolve_execution_device(torch, device)
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Krea2 VAE pixels must have shape (batch, 3, height, width)")
        if pixels.shape[-2] % KREA2_VAE_SCALE_FACTOR or (pixels.shape[-1] % KREA2_VAE_SCALE_FACTOR):
            raise ValueError("Krea2 VAE image dimensions must be multiples of eight")

        self.model.to(device=resolved_device)
        dtype = self.model.quant_conv.weight.dtype
        prepared = pixels.to(device=resolved_device, dtype=dtype).mul(2.0).sub(1.0).unsqueeze(2)
        with torch.inference_mode():
            posterior = self.model.encode(prepared, return_dict=False)[0]
            encoded = posterior.mode()
        mean, std = self._latent_statistics(
            torch,
            device=resolved_device,
            dtype=dtype,
        )
        return (encoded - mean) / std

    def _latent_statistics(self, torch, *, device: Any, dtype: Any):
        mean = torch.tensor(
            self.model.config.latents_mean,
            device=device,
            dtype=dtype,
        ).view(1, KREA2_VAE_LATENT_CHANNELS, 1, 1, 1)
        std = torch.tensor(
            self.model.config.latents_std,
            device=device,
            dtype=dtype,
        ).view(1, KREA2_VAE_LATENT_CHANNELS, 1, 1, 1)
        return mean, std

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            torch = _import_runtime()
        except ConfigurationError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_krea2_vae(component: NativeComponent) -> NativeKrea2VAE:
    """Map the exact reviewed VAE checkpoint onto Diffusers' upstream graph."""

    if component.config.role != ArtifactKind.VAE:
        raise WeightMappingError(
            "native Krea2 VAE builder received the wrong component role",
            technical_detail=f"got {component.config.role.value}",
            backend_name="native",
            phase="vae",
        )
    if not component.loaded:
        raise RuntimeError("native VAE component has been unloaded")

    torch = _import_runtime()
    try:
        from diffusers import AutoencoderKLQwenImage
    except (ImportError, RuntimeError) as error:
        raise ConfigurationError(
            "Native Krea2 VAE execution requires Diffusers with Qwen Image support.",
            technical_detail=str(error),
            backend_name="native",
            phase="vae",
            remediation="Install Diffusers 0.39 or newer in the worker environment.",
        ) from error

    with torch.device("meta"):
        model = AutoencoderKLQwenImage()
    state = _vae_state_dict(component.tensors)
    expected = model.state_dict()
    _validate_state_dict_shapes(expected, state)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise WeightMappingError(
            "Krea2 VAE executable state mapping was not strict",
            technical_detail=(
                f"missing={incompatible.missing_keys}; unexpected={incompatible.unexpected_keys}"
            ),
            backend_name="native",
            phase="vae",
        )
    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise WeightMappingError(
            "Krea2 VAE executable graph contains unmaterialized tensors",
            technical_detail=", ".join(meta_tensors[:20]),
            backend_name="native",
            phase="vae",
        )
    model.requires_grad_(False)
    model.eval()
    return NativeKrea2VAE(
        model=model,
        report=Krea2VAELoadReport(
            source_tensor_count=len(component.tensors),
            mapped_tensor_count=len(state),
            parameter_count=sum(int(tensor.numel()) for tensor in state.values()),
        ),
    )


def _import_runtime():
    try:
        import torch
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 VAE execution requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="vae",
            remediation="Install K2Lab's model dependencies in the worker environment.",
        ) from error
    return torch


def _resolve_execution_device(torch, requested: str):
    normalized = requested.strip().casefold()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigurationError(
            "A CUDA/ROCm VAE device was requested but is not available.",
            technical_detail=f"requested device: {requested}",
            backend_name="native",
            phase="vae",
            remediation="Select CPU or run on a worker with a supported accelerator.",
        )
    try:
        return torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ConfigurationError(
            "The native VAE device is invalid.",
            technical_detail=str(error),
            backend_name="native",
            phase="vae",
        ) from error


def _vae_state_dict(tensors: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for source_key, tensor in tensors.items():
        target_key = _map_vae_key(source_key)
        if target_key in state:
            raise WeightMappingError(
                f"duplicate mapped Krea2 VAE key: {target_key}",
                technical_detail=f"source key: {source_key}",
                backend_name="native",
                phase="vae",
            )
        state[target_key] = tensor
    return state


def _map_vae_key(key: str) -> str:
    direct = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
        "encoder.conv1.weight": "encoder.conv_in.weight",
        "encoder.conv1.bias": "encoder.conv_in.bias",
        "decoder.conv1.weight": "decoder.conv_in.weight",
        "decoder.conv1.bias": "decoder.conv_in.bias",
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.weight": "decoder.conv_out.weight",
        "decoder.head.2.bias": "decoder.conv_out.bias",
    }
    if key in direct:
        return direct[key]

    middle = _map_middle_key(key)
    if middle is not None:
        return middle
    if key.startswith("encoder.downsamples."):
        return _map_encoder_downsample(key)
    if key.startswith("decoder.upsamples."):
        return _map_decoder_upsample(key)
    raise WeightMappingError(
        f"unmapped Krea2 VAE source key: {key}",
        backend_name="native",
        phase="vae",
    )


def _map_middle_key(key: str) -> str | None:
    parts = key.split(".")
    if len(parts) < 5 or parts[0] not in {"encoder", "decoder"}:
        return None
    if parts[1] != "middle":
        return None
    side, block = parts[0], parts[2]
    if block in {"0", "2"} and parts[3] == "residual":
        resnet = "0" if block == "0" else "1"
        suffix = _map_residual_suffix(".".join(parts[4:]))
        return f"{side}.mid_block.resnets.{resnet}.{suffix}"
    if block == "1":
        attention_suffixes = {
            "norm.gamma": "norm.gamma",
            "to_qkv.weight": "to_qkv.weight",
            "to_qkv.bias": "to_qkv.bias",
            "proj.weight": "proj.weight",
            "proj.bias": "proj.bias",
        }
        source_suffix = ".".join(parts[3:])
        target_suffix = attention_suffixes.get(source_suffix)
        if target_suffix is not None:
            return f"{side}.mid_block.attentions.0.{target_suffix}"
    return None


def _map_encoder_downsample(key: str) -> str:
    target = key.replace("encoder.downsamples.", "encoder.down_blocks.", 1)
    target = _replace_residual_names(target)
    return target.replace(".shortcut.", ".conv_shortcut.")


def _map_decoder_upsample(key: str) -> str:
    parts = key.split(".")
    try:
        source_block = int(parts[2])
    except (IndexError, ValueError) as error:
        raise WeightMappingError(
            f"invalid Krea2 VAE decoder key: {key}",
            backend_name="native",
            phase="vae",
        ) from error

    if source_block in {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}:
        target_block = source_block // 4
        target_resnet = source_block % 4
        prefix = f"decoder.up_blocks.{target_block}.resnets.{target_resnet}"
        suffix = ".".join(parts[3:])
        if suffix.startswith("residual."):
            return f"{prefix}.{_map_residual_suffix(suffix.removeprefix('residual.'))}"
        if suffix.startswith("shortcut."):
            return f"{prefix}.conv_shortcut.{suffix.removeprefix('shortcut.')}"
    if source_block in {3, 7, 11}:
        target_block = source_block // 4
        suffix = ".".join(parts[3:])
        return f"decoder.up_blocks.{target_block}.upsamplers.0.{suffix}"
    raise WeightMappingError(
        f"unmapped Krea2 VAE decoder key: {key}",
        backend_name="native",
        phase="vae",
    )


def _replace_residual_names(key: str) -> str:
    if ".residual." not in key:
        return key
    prefix, suffix = key.split(".residual.", 1)
    return f"{prefix}.{_map_residual_suffix(suffix)}"


def _map_residual_suffix(suffix: str) -> str:
    mapping = {
        "0.gamma": "norm1.gamma",
        "2.weight": "conv1.weight",
        "2.bias": "conv1.bias",
        "3.gamma": "norm2.gamma",
        "6.weight": "conv2.weight",
        "6.bias": "conv2.bias",
    }
    try:
        return mapping[suffix]
    except KeyError as error:
        raise WeightMappingError(
            f"unmapped Krea2 VAE residual key suffix: {suffix}",
            backend_name="native",
            phase="vae",
        ) from error


def _validate_state_dict_shapes(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        (
            key,
            tuple(expected[key].shape),
            tuple(actual[key].shape),
        )
        for key in set(expected) & set(actual)
        if expected[key].shape != actual[key].shape
    )
    if missing or unexpected or mismatched:
        raise WeightMappingError(
            "Krea2 VAE state-dict keys or shapes do not match the executable graph",
            technical_detail=(
                f"missing={missing[:20]}; unexpected={unexpected[:20]}; "
                f"shape_mismatches={mismatched[:20]}"
            ),
            backend_name="native",
            phase="vae",
        )
