from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from k2core.depth.checkpoint import (
    KREA2_DEPTH_PUBLIC_SHA256,
    DepthCheckpointInfo,
    inspect_depth_checkpoint,
    lora_pairs,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # Lightweight control-plane environments.
    torch = None
    nn = None
    functional = None


DEPTH_ATTACHMENT_KEY = "k2_krea_depth_control_lora"
DEPTH_LATENT_KEY = "k2_krea_depth_control_latent"
DEPTH_TOKEN_STRENGTH_KEY = "k2_krea_depth_token_strength"
DEPTH_ADAPTER_FORMAT = "krea2-depth-control-lora-v1"


class KreaDepthAdapterError(RuntimeError):
    def __init__(self, code: str, safe_message: str, *, private_detail: str = "") -> None:
        super().__init__(private_detail or safe_message)
        self.code = code
        self.safe_message = safe_message
        self.private_detail = private_detail or safe_message


@dataclass(frozen=True, slots=True)
class DepthControlLatent:
    value: Any
    source_sha256: str
    encode_seconds: float


@dataclass(frozen=True, slots=True)
class DepthAdapterRuntimeReport:
    checkpoint_sha256: str
    format_id: str
    loaded_lora_keys: int
    patched_model_keys: int
    denoiser_calls: int
    control_latent_shape: tuple[int, ...]

    def document(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "format_id": self.format_id,
            "loaded_lora_keys": self.loaded_lora_keys,
            "patched_model_keys": self.patched_model_keys,
            "denoiser_calls": self.denoiser_calls,
            "control_latent_shape": list(self.control_latent_shape),
        }


if nn is not None:

    class KreaDepthInputProjection(nn.Module):
        """Preserve Krea's native image projection and add depth-token influence."""

        def __init__(
            self,
            weight,
            *,
            image_features: int,
            original_first: Any,
        ) -> None:
            super().__init__()
            if weight.ndim != 2 or weight.shape[1] != image_features * 2:
                raise KreaDepthAdapterError(
                    "depth_projection_incompatible",
                    "The selected depth adapter has an incompatible input projection.",
                )
            self.image_features = int(image_features)
            self.control_features = int(image_features)
            self.weight = nn.Parameter(weight.detach().cpu().clone(), requires_grad=False)
            self.control_tokens = None
            self.token_strength = None
            object.__setattr__(self, "_original_first", original_first)

        @property
        def original_first(self):
            return object.__getattribute__(self, "_original_first")

        def set_original_first(self, value: Any) -> None:
            object.__setattr__(self, "_original_first", value)

        def forward(self, image_tokens):
            if image_tokens.shape[-1] != self.image_features:
                raise KreaDepthAdapterError(
                    "depth_latent_shape_invalid",
                    "The depth adapter received an incompatible Krea latent shape.",
                )
            if self.control_tokens is None:
                raise KreaDepthAdapterError(
                    "depth_control_missing",
                    "No depth control is attached to the active generation.",
                )
            control = self.control_tokens.to(
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )
            if control.shape[1] != image_tokens.shape[1]:
                raise KreaDepthAdapterError(
                    "depth_latent_shape_invalid",
                    "The depth token count does not match the Krea image tokens.",
                )
            import comfy.model_management
            import comfy.utils

            control = comfy.utils.repeat_to_batch_size(control, image_tokens.shape[0])
            control_weight = comfy.model_management.cast_to_device(
                self.weight[:, self.image_features :],
                image_tokens.device,
                image_tokens.dtype,
            )
            contribution = functional.linear(control, control_weight, None)
            if self.token_strength is not None:
                token_strength = self.token_strength.to(
                    device=image_tokens.device,
                    dtype=image_tokens.dtype,
                )
                if token_strength.ndim == 1:
                    token_strength = token_strength.view(1, -1, 1)
                elif token_strength.ndim == 2:
                    token_strength = token_strength.unsqueeze(-1)
                if (
                    token_strength.ndim != 3
                    or token_strength.shape[1] != image_tokens.shape[1]
                ):
                    raise KreaDepthAdapterError(
                        "depth_strength_shape_invalid",
                        "The regional depth-strength field does not match Krea image tokens.",
                    )
                contribution = contribution * token_strength
            return self.original_first(image_tokens) + contribution

else:

    class KreaDepthInputProjection:  # pragma: no cover
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Torch is required for Krea depth-control inference")


def process_depth_latent_for_model(model_patcher, latent):
    if torch is None or not torch.is_tensor(latent) or latent.ndim not in (4, 5):
        raise KreaDepthAdapterError(
            "depth_latent_shape_invalid",
            "The depth adapter control latent has an invalid shape.",
        )
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception as error:
        raise KreaDepthAdapterError(
            "depth_model_incompatible",
            "The selected model does not expose native Krea latent formatting.",
        ) from error
    expected = getattr(latent_format, "latent_channels", None)
    if expected is not None and int(latent.shape[1]) != int(expected):
        raise KreaDepthAdapterError(
            "depth_vae_incompatible",
            "Select the Krea/Qwen image VAE for depth control.",
        )
    added_time = getattr(latent_format, "latent_dimensions", 2) == 3 and latent.ndim == 4
    processed = latent.unsqueeze(2) if added_time else latent
    processed = model_patcher.model.process_latent_in(processed)
    if added_time and processed.ndim == 5 and processed.shape[2] == 1:
        processed = processed[:, :, 0]
    return processed


def encode_depth_control(vae, model_patcher, depth_values: np.ndarray) -> DepthControlLatent:
    if torch is None:
        raise KreaDepthAdapterError(
            "depth_encode_failed",
            "Torch is unavailable for depth-control preprocessing.",
        )
    values = np.asarray(depth_values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("depth control values must be a finite two-dimensional field")
    started = time.monotonic()
    rgb = np.repeat(values[..., None], 3, axis=-1)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(dtype=torch.float32)
    try:
        with torch.inference_mode():
            latent = vae.encode(image.unsqueeze(0))
    except Exception as error:
        raise KreaDepthAdapterError(
            "depth_encode_failed",
            "The Krea/Qwen VAE could not encode the depth image.",
            private_detail=f"depth VAE encode failed: {type(error).__name__}: {error}",
        ) from error
    processed = process_depth_latent_for_model(model_patcher, latent)
    return DepthControlLatent(
        value=processed,
        source_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        encode_seconds=time.monotonic() - started,
    )


def _strip_prefixes(base: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in (
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                changed = True
    return base


def _shape_from_weight(weight) -> tuple[int, ...] | None:
    tensor_shape = getattr(weight, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(int(value) for value in tensor_shape)
    data = getattr(weight, "data", None)
    tensor_shape = getattr(data, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(int(value) for value in tensor_shape)
    shape = getattr(weight, "shape", None)
    return tuple(int(value) for value in shape) if shape is not None else None


def _nested_attr(root, path: str):
    current = root
    for component in path.split("."):
        if component.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


def _target_key(base: str) -> str | None:
    normalized = _strip_prefixes(base)
    return f"diffusion_model.{normalized}.weight" if normalized.startswith("blocks.") else None


def _model_target_shape(model_patcher, target: str) -> tuple[int, ...] | None:
    try:
        return _shape_from_weight(_nested_attr(model_patcher.model, target))
    except Exception:
        value = model_patcher.model.state_dict().get(target)
        return _shape_from_weight(value) if value is not None else None


def _runtime_lora_patches(state_dict: Mapping[str, Any], model_patcher):
    from comfy.weight_adapter.lora import LoRAAdapter

    patches: dict[str, Any] = {}
    loaded: set[str] = set()
    skipped: dict[str, str] = {}
    for base, down_key, up_key in lora_pairs(list(state_dict)):
        target = _target_key(base)
        if target is None:
            continue
        shape = _model_target_shape(model_patcher, target)
        if shape is None or len(shape) < 2:
            skipped[down_key] = f"target {target!r} is unavailable"
            continue
        down, up = state_dict[down_key], state_dict[up_key]
        if (
            torch is None
            or not torch.is_tensor(down)
            or not torch.is_tensor(up)
            or down.ndim != 2
            or up.ndim != 2
        ):
            skipped[down_key] = "adapter tensors are not 2D"
            continue
        out_features, in_features = shape[:2]
        if (
            up.shape[0] == out_features
            and down.shape[1] == in_features
            and up.shape[1] == down.shape[0]
        ):
            rank = int(down.shape[0])
        elif (
            down.shape[0] == in_features
            and up.shape[1] == out_features
            and down.shape[1] == up.shape[0]
        ):
            down = down.t().contiguous()
            up = up.t().contiguous()
            rank = int(down.shape[0])
        else:
            skipped[down_key] = f"adapter does not match live target shape {shape}"
            continue
        alpha: float = rank
        alpha_key = None
        for suffix in (".alpha", ".network_alpha", ".scale"):
            candidate = base + suffix
            if candidate in state_dict:
                alpha_key = candidate
                alpha = float(state_dict[candidate].detach().cpu().reshape(-1)[0])
                break
        keys = {down_key, up_key}
        if alpha_key is not None:
            keys.add(alpha_key)
        patches[target] = LoRAAdapter(keys, (up, down, alpha, None, None, None))
        loaded.update(keys)
    return patches, loaded, skipped


def _first_module(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model.first")
    except Exception as error:
        raise KreaDepthAdapterError(
            "depth_model_incompatible",
            "Select a native ComfyUI Krea 2 model for depth control.",
        ) from error


def _runtime_projection(
    model_patcher,
    state_dict: Mapping[str, Any],
    info: DepthCheckpointInfo,
) -> KreaDepthInputProjection:
    first = _first_module(model_patcher)
    shape = _shape_from_weight(getattr(first, "weight", None))
    if shape is None or len(shape) != 2:
        raise KreaDepthAdapterError(
            "depth_projection_missing",
            "The selected Krea model has no compatible native input projection.",
        )
    weight = state_dict.get(info.expanded_projection_key)
    if (
        torch is None
        or not torch.is_tensor(weight)
        or weight.ndim != 2
        or tuple(weight.shape) != (shape[0], shape[1] * 2)
    ):
        raise KreaDepthAdapterError(
            "depth_checkpoint_incompatible",
            "The depth adapter input projection does not match the selected Krea model.",
            private_detail=(
                f"checkpoint projection {getattr(weight, 'shape', None)} does not match "
                f"live projection {shape}"
            ),
        )
    return KreaDepthInputProjection(
        weight,
        image_features=shape[1],
        original_first=first,
    )


def _flatten_control_latent(latent):
    if latent.ndim == 4:
        return latent
    if latent.ndim == 5:
        batch, channels, frames, height, width = latent.shape
        return latent.reshape(batch * frames, channels, height, width)
    raise KreaDepthAdapterError(
        "depth_latent_shape_invalid",
        "The depth adapter control latent has an invalid shape.",
    )


def _control_tokens(latent, x, patch: int, expected_features: int):
    import comfy.ldm.common_dit
    import comfy.model_management
    import comfy.utils

    target_batch = int(x.shape[0] * x.shape[2]) if x.ndim == 5 else int(x.shape[0])
    control = comfy.utils.repeat_to_batch_size(_flatten_control_latent(latent), target_batch)
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)
    target_height, target_width = int(x.shape[-2]), int(x.shape[-1])
    if tuple(control.shape[-2:]) != (target_height, target_width):
        control = comfy.utils.common_upscale(
            control,
            target_width,
            target_height,
            "bilinear",
            "disabled",
        )
    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch, patch))
    batch, channels, height, width = control.shape
    features = channels * patch * patch
    if features != expected_features or height % patch or width % patch:
        raise KreaDepthAdapterError(
            "depth_latent_shape_invalid",
            "The depth control latent does not match the Krea image-token layout.",
        )
    return (
        control.reshape(
            batch,
            channels,
            height // patch,
            patch,
            width // patch,
            patch,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, (height // patch) * (width // patch), features)
    )


def _transformer_options(args, kwargs) -> dict[str, Any] | None:
    candidate = kwargs.get("transformer_options")
    if isinstance(candidate, dict):
        return candidate
    if len(args) >= 5 and isinstance(args[4], dict):
        return args[4]
    if args and isinstance(args[-1], dict):
        return args[-1]
    return None


def _restore_projection(diffusion_model, projection: KreaDepthInputProjection) -> None:
    projection.control_tokens = None
    projection.token_strength = None
    if getattr(diffusion_model, "first", None) is projection:
        diffusion_model.first = projection.original_first


def _depth_wrapper(projection: KreaDepthInputProjection, calls: dict[str, int]):
    def wrapper(executor, *args, **kwargs):
        options = _transformer_options(args, kwargs)
        if options is None:
            raise KreaDepthAdapterError(
                "depth_sampler_incompatible",
                "The current sampler cannot provide Krea depth-control options.",
            )
        latent = options.get(DEPTH_LATENT_KEY)
        if latent is None:
            raise KreaDepthAdapterError(
                "depth_control_missing",
                "Depth control was not attached to this generation.",
            )
        diffusion_model = executor.class_obj
        previous_first = getattr(diffusion_model, "first", None)
        previous_tokens = projection.control_tokens
        previous_strength = projection.token_strength
        try:
            projection.control_tokens = _control_tokens(
                latent,
                args[0],
                int(diffusion_model.patch),
                projection.control_features,
            )
            token_strength = options.get(DEPTH_TOKEN_STRENGTH_KEY)
            if callable(getattr(token_strength, "current_values", None)):
                token_strength = token_strength.current_values()
            if token_strength is not None:
                projection.token_strength = (
                    token_strength
                    if torch is not None and torch.is_tensor(token_strength)
                    else torch.tensor(token_strength)
                )
            diffusion_model.first = projection
            calls["denoiser"] = calls.get("denoiser", 0) + 1
            return executor(*args, **kwargs)
        finally:
            projection.control_tokens = previous_tokens
            projection.token_strength = previous_strength
            if getattr(diffusion_model, "first", None) is projection:
                diffusion_model.first = projection.original_first or previous_first

    return wrapper


def _projection_injections(projection: KreaDepthInputProjection):
    import comfy.patcher_extension

    def inject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is None:
            return
        current = getattr(diffusion_model, "first", None)
        if current is not None and current is not projection:
            projection.set_original_first(current)
        diffusion_model.first = projection.original_first
        projection.control_tokens = None
        projection.token_strength = None

    def eject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is not None:
            _restore_projection(diffusion_model, projection)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _projection_cleanup(model_patcher, *_args) -> None:
    attachment = model_patcher.get_attachment(DEPTH_ATTACHMENT_KEY)
    if not isinstance(attachment, Mapping):
        return
    projection = attachment.get("projection")
    diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
    if (
        projection is not None
        and hasattr(projection, "original_first")
        and diffusion_model is not None
    ):
        _restore_projection(diffusion_model, projection)


def install_depth_adapter(
    model_patcher,
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = KREA2_DEPTH_PUBLIC_SHA256,
):
    if torch is None:
        raise KreaDepthAdapterError(
            "depth_model_incompatible",
            "Torch is unavailable for Krea depth-control inference.",
        )
    compatibility = inspect_depth_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    if not compatibility.compatible or compatibility.checkpoint is None:
        raise KreaDepthAdapterError(
            "depth_checkpoint_incompatible",
            "The selected Krea depth adapter is incompatible with this runtime.",
            private_detail="; ".join(compatibility.errors),
        )
    if model_patcher.get_attachment(DEPTH_ATTACHMENT_KEY) is not None:
        raise KreaDepthAdapterError(
            "depth_adapter_conflict",
            "Only one Krea depth adapter can be active in a generation.",
        )
    import comfy.patcher_extension
    import comfy.utils

    try:
        state_dict = comfy.utils.load_torch_file(str(checkpoint_path), safe_load=True)
    except Exception as error:
        raise KreaDepthAdapterError(
            "depth_checkpoint_invalid",
            "The selected Krea depth adapter could not be read safely.",
        ) from error
    generation_model = model_patcher.clone()
    projection = _runtime_projection(
        generation_model,
        state_dict,
        compatibility.checkpoint,
    )
    patches, loaded_keys, skipped = _runtime_lora_patches(state_dict, generation_model)
    if not patches:
        raise KreaDepthAdapterError(
            "depth_weights_missing",
            "The selected depth adapter has no compatible Krea block weights.",
            private_detail=f"skipped depth Control-LoRA targets: {skipped}",
        )
    patched_keys = generation_model.add_patches(
        patches,
        strength_patch=1.0,
        strength_model=1.0,
    )
    if not patched_keys:
        raise KreaDepthAdapterError(
            "depth_weights_missing",
            "The selected Krea model rejected every depth-adapter block weight.",
        )
    calls: dict[str, int] = {}
    generation_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        DEPTH_ATTACHMENT_KEY,
        _depth_wrapper(projection, calls),
    )
    generation_model.set_injections(
        DEPTH_ATTACHMENT_KEY,
        _projection_injections(projection),
    )
    generation_model.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_DETACH,
        DEPTH_ATTACHMENT_KEY,
        _projection_cleanup,
    )
    generation_model.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        DEPTH_ATTACHMENT_KEY,
        _projection_cleanup,
    )
    generation_model.set_attachments(
        DEPTH_ATTACHMENT_KEY,
        {
            "checkpoint": compatibility.checkpoint,
            "format_id": DEPTH_ADAPTER_FORMAT,
            "loaded_lora_keys": len(loaded_keys),
            "patched_model_keys": len(patched_keys),
            "skipped_targets": skipped,
            "projection": projection,
            "calls": calls,
        },
    )
    return generation_model, compatibility


def attach_depth_control(
    model_patcher,
    latent: DepthControlLatent,
    *,
    token_strength: Any,
):
    if model_patcher.get_attachment(DEPTH_ATTACHMENT_KEY) is None:
        raise KreaDepthAdapterError(
            "depth_model_incompatible",
            "The Krea depth adapter must be installed before control is attached.",
        )
    generation_model = model_patcher.clone()
    options = generation_model.model_options.setdefault("transformer_options", {})
    options[DEPTH_LATENT_KEY] = latent.value
    options[DEPTH_TOKEN_STRENGTH_KEY] = token_strength
    return generation_model


def clear_depth_control(model_patcher) -> None:
    options = model_patcher.model_options.get("transformer_options", {})
    options.pop(DEPTH_LATENT_KEY, None)
    options.pop(DEPTH_TOKEN_STRENGTH_KEY, None)
    _projection_cleanup(model_patcher)
    try:
        model_patcher.remove_injections(DEPTH_ATTACHMENT_KEY)
    except (AttributeError, KeyError):
        pass


def depth_adapter_runtime_report(model_patcher) -> DepthAdapterRuntimeReport:
    attachment = model_patcher.get_attachment(DEPTH_ATTACHMENT_KEY)
    if not isinstance(attachment, Mapping):
        raise KreaDepthAdapterError(
            "depth_model_incompatible",
            "The generation model has no active Krea depth adapter.",
        )
    checkpoint = attachment["checkpoint"]
    options = model_patcher.model_options.get("transformer_options", {})
    latent = options.get(DEPTH_LATENT_KEY)
    if latent is None:
        raise KreaDepthAdapterError(
            "depth_control_missing",
            "The generation model has no active depth control.",
        )
    return DepthAdapterRuntimeReport(
        checkpoint_sha256=checkpoint.sha256,
        format_id=str(attachment["format_id"]),
        loaded_lora_keys=int(attachment["loaded_lora_keys"]),
        patched_model_keys=int(attachment["patched_model_keys"]),
        denoiser_calls=int(attachment["calls"].get("denoiser", 0)),
        control_latent_shape=tuple(int(value) for value in latent.shape),
    )
