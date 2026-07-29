"""Strict executable mapping for the upstream Diffusers Krea2 transformer."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from k2core.backends.native_loading import NativeComponent
from k2core.backends.native_quant import (
    apply_compute_dtype,
    replace_submodule,
    scaled_fp8_linear_class,
)
from k2core.inference.errors import (
    ConfigurationError,
    WeightMappingError,
)
from k2core.inference.schemas import DTypePolicy
from k2core.model import ArtifactKind


KREA2_LATENT_CHANNELS = 16
KREA2_PATCH_SIZE = 2
KREA2_TEXT_LAYERS = 12
KREA2_TEXT_DIM = 2560
KREA2_CONDITIONING_DIM = KREA2_TEXT_LAYERS * KREA2_TEXT_DIM
KREA2_QUANTIZED_LINEAR_COUNT = 256


@dataclass(frozen=True, slots=True)
class Krea2TransformerLoadReport:
    source_tensor_count: int
    mapped_tensor_count: int
    parameter_count: int
    quantized_linear_count: int
    full_precision_matrix_mult_count: int
    checkpoint_optimized_matrix_mult_count: int


class NativeKrea2Transformer:
    """Own the executable upstream Krea2 transformer graph."""

    def __init__(self, *, model: Any, report: Krea2TransformerLoadReport) -> None:
        self.model = model
        self.report = report

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict_velocity(
        self,
        latent: Any,
        conditioning: Any,
        sigma: float | Any,
        *,
        attention_mask: Any | None = None,
        device: str = "cuda",
    ):
        if self.model is None:
            raise RuntimeError("native Krea2 transformer has been unloaded")

        torch, functional, _ = _import_runtime()
        resolved_device = _resolve_execution_device(torch, device)
        self.model.to(device=resolved_device)
        dtype = self.model.img_in.weight.dtype

        if latent.ndim != 5:
            raise ValueError("Krea2 latent must have shape (batch, 16, time, height, width)")
        batch, channels, frames, original_height, original_width = latent.shape
        if channels != KREA2_LATENT_CHANNELS:
            raise ValueError(
                f"Krea2 latent must have {KREA2_LATENT_CHANNELS} channels, got {channels}"
            )
        if frames != 1:
            raise ValueError("clean Krea2 generation currently requires one latent frame")
        if conditioning.ndim != 3 or conditioning.shape[-1] != KREA2_CONDITIONING_DIM:
            raise ValueError(
                f"Krea2 conditioning must have shape (batch, sequence, {KREA2_CONDITIONING_DIM})"
            )
        if conditioning.shape[0] != batch:
            raise ValueError("latent and conditioning batch sizes do not match")

        image = latent[:, :, 0].to(device=resolved_device, dtype=dtype)
        pad_height = (-original_height) % KREA2_PATCH_SIZE
        pad_width = (-original_width) % KREA2_PATCH_SIZE
        if pad_height or pad_width:
            image = functional.pad(image, (0, pad_width, 0, pad_height))
        height, width = image.shape[-2:]
        packed = _pack_latent(image, KREA2_PATCH_SIZE)

        text_sequence = conditioning.shape[1]
        encoder_hidden_states = conditioning.reshape(
            batch,
            text_sequence,
            KREA2_TEXT_LAYERS,
            KREA2_TEXT_DIM,
        ).to(device=resolved_device, dtype=dtype)
        prepared_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, text_sequence):
                raise ValueError("attention mask must match the conditioning batch and sequence")
            candidate_mask = attention_mask.to(
                device=resolved_device,
                dtype=torch.bool,
            )
            # Comfy drops its attention mask when every token is valid.
            if not bool(candidate_mask.all()):
                prepared_mask = candidate_mask
        timestep = torch.as_tensor(sigma, device=resolved_device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        elif timestep.shape != (batch,):
            raise ValueError(f"sigma must be scalar or have shape ({batch},)")
        position_ids = _position_ids(
            torch,
            text_sequence,
            height // KREA2_PATCH_SIZE,
            width // KREA2_PATCH_SIZE,
            resolved_device,
        )

        with torch.inference_mode():
            output = self.model(
                hidden_states=packed,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                position_ids=position_ids,
                encoder_attention_mask=prepared_mask,
                return_dict=False,
            )[0]
            velocity = _unpack_latent(
                output,
                KREA2_LATENT_CHANNELS,
                height,
                width,
                KREA2_PATCH_SIZE,
            )
            velocity = velocity[:, :, :original_height, :original_width].unsqueeze(2)
        return velocity

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            torch, _, _ = _import_runtime()
        except ConfigurationError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_krea2_transformer(
    component: NativeComponent,
    *,
    spatial_attention: Any | None = None,
    compute_dtype: DTypePolicy = DTypePolicy.AUTO,
) -> NativeKrea2Transformer:
    """Map the exact reviewed Krea2 checkpoint onto Diffusers' upstream graph."""

    if component.config.role != ArtifactKind.TRANSFORMER:
        raise WeightMappingError(
            "native Krea2 builder received the wrong component role",
            technical_detail=f"got {component.config.role.value}",
            backend_name="native",
            phase="transformer",
        )
    if not component.loaded:
        raise RuntimeError("native transformer component has been unloaded")

    torch, functional, nn = _import_runtime()
    try:
        from diffusers import Krea2Transformer2DModel
        from diffusers.models.embeddings import apply_rotary_emb
    except (ImportError, RuntimeError) as error:
        raise ConfigurationError(
            "Native Krea2 execution requires Diffusers with Krea2 support.",
            technical_detail=str(error),
            backend_name="native",
            phase="transformer",
            remediation="Install Diffusers 0.39 or newer in the worker environment.",
        ) from error

    quantization = _validate_quantization_metadata(
        component.metadata,
        component.tensors,
    )
    with torch.device("meta"):
        model = Krea2Transformer2DModel()

    scaled_linear = scaled_fp8_linear_class(torch, nn, functional)
    for module_path in sorted(quantization):
        original = model.get_submodule(module_path)
        if not isinstance(original, nn.Linear):
            raise WeightMappingError(
                f"quantized Krea2 target is not a linear layer: {module_path}",
                backend_name="native",
                phase="transformer",
            )
        replace_submodule(
            model,
            module_path,
            scaled_linear(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                device="meta",
                quantize_input=not quantization[module_path].get(
                    "full_precision_matrix_mult",
                    False,
                ),
            ),
        )

    state = _transformer_state_dict(component.tensors)
    expected = model.state_dict()
    _validate_state_dict_shapes(expected, state)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise WeightMappingError(
            "Krea2 executable state mapping was not strict",
            technical_detail=(
                f"missing={incompatible.missing_keys}; unexpected={incompatible.unexpected_keys}"
            ),
            backend_name="native",
            phase="transformer",
        )
    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise WeightMappingError(
            "Krea2 executable graph contains unmaterialized tensors",
            technical_detail=", ".join(meta_tensors[:20]),
            backend_name="native",
            phase="transformer",
        )
    # Diffusers marks every Krea2 RMSNorm as keep-in-FP32. A direct assign
    # preserves the checkpoint's BF16 storage dtype, so materialize these small
    # scales explicitly to match both the upstream contract and Comfy's forward.
    for module in model.modules():
        if module.__class__.__name__ == "Krea2RMSNorm":
            module.weight = nn.Parameter(
                module.weight.float(),
                requires_grad=False,
            )
    apply_compute_dtype(model, torch, compute_dtype)
    model.requires_grad_(False)
    model.eval()
    attention_processor = _repeated_gqa_processor(
        functional,
        apply_rotary_emb,
        spatial_attention=spatial_attention,
    )
    for module in model.modules():
        if module.__class__.__name__ == "Krea2Attention":
            module.set_processor(attention_processor())

    full_precision_count = sum(
        bool(config.get("full_precision_matrix_mult", False)) for config in quantization.values()
    )
    report = Krea2TransformerLoadReport(
        source_tensor_count=len(component.tensors),
        mapped_tensor_count=len(state),
        parameter_count=sum(
            int(tensor.numel())
            for name, tensor in state.items()
            if not name.endswith(".weight_scale")
        ),
        quantized_linear_count=len(quantization),
        full_precision_matrix_mult_count=full_precision_count,
        checkpoint_optimized_matrix_mult_count=len(quantization) - full_precision_count,
    )
    return NativeKrea2Transformer(model=model, report=report)


def _import_runtime():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 transformer execution requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="transformer",
        ) from error
    return torch, functional, nn


def _repeated_gqa_processor(
    functional,
    apply_rotary_emb,
    *,
    spatial_attention=None,
):
    class RepeatedGQAAttentionProcessor:
        def __call__(
            self,
            attn,
            hidden_states,
            attention_mask=None,
            image_rotary_emb=None,
        ):
            query = attn.to_q(hidden_states).unflatten(
                -1,
                (attn.num_heads, attn.head_dim),
            )
            key = attn.to_k(hidden_states).unflatten(
                -1,
                (attn.num_kv_heads, attn.head_dim),
            )
            value = attn.to_v(hidden_states).unflatten(
                -1,
                (attn.num_kv_heads, attn.head_dim),
            )
            gate = attn.to_gate(hidden_states)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(
                    query,
                    image_rotary_emb,
                    sequence_dim=1,
                )
                key = apply_rotary_emb(
                    key,
                    image_rotary_emb,
                    sequence_dim=1,
                )

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if attn.num_kv_heads != attn.num_heads:
                repeats = attn.num_heads // attn.num_kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            regional_stream = _regional_attention_stream(
                spatial_attention,
                query,
                key,
            )
            if regional_stream is not None:
                if attention_mask is not None:
                    raise RuntimeError("native regional attention requires an unmasked stream")
                output = spatial_attention.attend(
                    query,
                    key,
                    value,
                    scale=attn.head_dim**-0.5,
                    main_stream=regional_stream,
                )
            else:
                output = functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            output = output.transpose(1, 2).flatten(2, 3)
            output = output * functional.sigmoid(gate)
            return attn.to_out[0](output)

    return RepeatedGQAAttentionProcessor


def _regional_attention_stream(spatial_attention, query, key) -> bool | None:
    if spatial_attention is None:
        return None
    query_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    if (
        query_length == spatial_attention.expected_sequence_length
        and key_length == spatial_attention.expected_sequence_length
    ):
        return True
    folded_layerwise_text = (
        query_length == 12
        and query_length == spatial_attention.plan.text_token_count
        and int(query.shape[0]) >= spatial_attention.plan.text_token_count
        and int(query.shape[0]) % spatial_attention.plan.text_token_count == 0
    )
    if (
        query_length == spatial_attention.plan.text_token_count
        and key_length == spatial_attention.plan.text_token_count
        and not folded_layerwise_text
    ):
        return False
    return None


def _resolve_execution_device(torch, requested: str):
    normalized = requested.strip().casefold()
    if normalized in {"auto", "rocm"}:
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigurationError(
            f"requested transformer device is unavailable: {requested}",
            backend_name="native",
            phase="transformer",
        )
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConfigurationError(
            f"invalid transformer device: {requested}",
            technical_detail=str(error),
            backend_name="native",
            phase="transformer",
        ) from error


def _validate_quantization_metadata(
    metadata: Mapping[str, str],
    tensors: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    encoded = metadata.get("_quantization_metadata")
    if encoded is None:
        raise WeightMappingError(
            "Krea2 checkpoint is missing scaled-FP8 metadata",
            backend_name="native",
            phase="transformer",
        )
    try:
        document = json.loads(encoded)
        layers = document["layers"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise WeightMappingError(
            "Krea2 checkpoint has invalid quantization metadata",
            technical_detail=str(error),
            backend_name="native",
            phase="transformer",
        ) from error
    if not isinstance(layers, dict):
        raise WeightMappingError(
            "Krea2 quantization layer metadata must be a mapping",
            backend_name="native",
            phase="transformer",
        )

    source_scaled = {
        key.removesuffix(".weight_scale") for key in tensors if key.endswith(".weight_scale")
    }
    if set(layers) != source_scaled:
        raise WeightMappingError(
            "Krea2 FP8 metadata and scale tensors do not target the same layers",
            technical_detail=(
                f"metadata-only={sorted(set(layers) - source_scaled)[:20]}; "
                f"scale-only={sorted(source_scaled - set(layers))[:20]}"
            ),
            backend_name="native",
            phase="transformer",
        )
    if len(layers) != KREA2_QUANTIZED_LINEAR_COUNT:
        raise WeightMappingError(
            "Krea2 checkpoint has an unexpected quantized layer count",
            technical_detail=(f"expected {KREA2_QUANTIZED_LINEAR_COUNT}, got {len(layers)}"),
            backend_name="native",
            phase="transformer",
        )

    mapped: dict[str, dict[str, Any]] = {}
    for source_module, raw_config in layers.items():
        if not isinstance(raw_config, dict):
            raise WeightMappingError(
                f"invalid Krea2 quantization entry: {source_module}",
                backend_name="native",
                phase="transformer",
            )
        allowed_keys = {"format", "full_precision_matrix_mult"}
        if (
            set(raw_config) - allowed_keys
            or raw_config.get("format") != "float8_e4m3fn"
            or not isinstance(
                raw_config.get("full_precision_matrix_mult", False),
                bool,
            )
        ):
            raise WeightMappingError(
                f"unsupported Krea2 quantization contract: {source_module}",
                technical_detail=str(raw_config),
                backend_name="native",
                phase="transformer",
            )
        target_module = _map_linear_module(source_module)
        if target_module in mapped:
            raise WeightMappingError(
                f"duplicate mapped Krea2 quantization target: {target_module}",
                backend_name="native",
                phase="transformer",
            )
        mapped[target_module] = dict(raw_config)
    return mapped


def _map_linear_module(source: str) -> str:
    standalone = {
        "first": "img_in",
        "last.linear": "final_layer.linear",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "txtfusion.projector": "text_fusion.projector",
    }
    if source in standalone:
        return standalone[source]

    block_match = re.fullmatch(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)", source)
    if block_match:
        index, kind, name = block_match.groups()
        suffix = _mapped_attn_or_mlp(kind, name)
        return f"transformer_blocks.{index}.{suffix}"

    text_match = re.fullmatch(
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)"
        r"\.(attn|mlp)\.(\w+)",
        source,
    )
    if text_match:
        group, index, kind, name = text_match.groups()
        suffix = _mapped_attn_or_mlp(kind, name)
        return f"text_fusion.{group}.{index}.{suffix}"
    raise WeightMappingError(
        f"unmapped Krea2 linear module: {source}",
        backend_name="native",
        phase="transformer",
    )


def _mapped_attn_or_mlp(kind: str, name: str) -> str:
    if kind == "attn":
        mapped = {
            "wq": "attn.to_q",
            "wk": "attn.to_k",
            "wv": "attn.to_v",
            "wo": "attn.to_out.0",
            "gate": "attn.to_gate",
        }.get(name)
    else:
        mapped = {
            "gate": "ff.gate",
            "up": "ff.up",
            "down": "ff.down",
        }.get(name)
    if mapped is None:
        raise WeightMappingError(
            f"unmapped Krea2 {kind} module: {name}",
            backend_name="native",
            phase="transformer",
        )
    return mapped


def _map_transformer_key(source: str) -> str:
    standalone_parameters = {
        "last.modulation.lin": "final_layer.scale_shift_table",
        "last.norm.scale": "final_layer.norm.weight",
        "txtmlp.0.scale": "txt_in.norm.weight",
    }
    if source in standalone_parameters:
        return standalone_parameters[source]

    block_mod = re.fullmatch(r"blocks\.(\d+)\.mod\.lin", source)
    if block_mod:
        return f"transformer_blocks.{block_mod.group(1)}.scale_shift_table"

    block_norm = re.fullmatch(
        r"blocks\.(\d+)\.(prenorm|postnorm)\.scale",
        source,
    )
    if block_norm:
        index, name = block_norm.groups()
        norm = "norm1" if name == "prenorm" else "norm2"
        return f"transformer_blocks.{index}.{norm}.weight"

    block_qk = re.fullmatch(
        r"blocks\.(\d+)\.attn\.qknorm\.(qnorm|knorm)\.scale",
        source,
    )
    if block_qk:
        index, name = block_qk.groups()
        norm = "norm_q" if name == "qnorm" else "norm_k"
        return f"transformer_blocks.{index}.attn.{norm}.weight"

    text_norm = re.fullmatch(
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)"
        r"\.(prenorm|postnorm)\.scale",
        source,
    )
    if text_norm:
        group, index, name = text_norm.groups()
        norm = "norm1" if name == "prenorm" else "norm2"
        return f"text_fusion.{group}.{index}.{norm}.weight"

    text_qk = re.fullmatch(
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)"
        r"\.attn\.qknorm\.(qnorm|knorm)\.scale",
        source,
    )
    if text_qk:
        group, index, name = text_qk.groups()
        norm = "norm_q" if name == "qnorm" else "norm_k"
        return f"text_fusion.{group}.{index}.attn.{norm}.weight"

    for suffix in (".weight_scale", ".weight", ".bias"):
        if source.endswith(suffix):
            module = source.removesuffix(suffix)
            return f"{_map_linear_module(module)}{suffix}"
    raise WeightMappingError(
        f"unmapped Krea2 checkpoint key: {source}",
        backend_name="native",
        phase="transformer",
    )


def _transformer_state_dict(tensors: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for source, tensor in tensors.items():
        target = _map_transformer_key(source)
        if re.fullmatch(r"blocks\.\d+\.mod\.lin", source):
            tensor = tensor.reshape(6, -1)
        if target in state:
            raise WeightMappingError(
                f"duplicate Krea2 target key: {target}",
                backend_name="native",
                phase="transformer",
            )
        state[target] = tensor
    return state


def _validate_state_dict_shapes(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(observed)
        if tuple(expected[name].shape) != tuple(observed[name].shape)
    )
    if missing or unexpected or mismatched:
        raise WeightMappingError(
            "Krea2 checkpoint does not match the upstream executable graph",
            technical_detail=(
                f"missing={missing[:20]}; unexpected={unexpected[:20]}; "
                f"shape_mismatch={mismatched[:20]}"
            ),
            backend_name="native",
            phase="transformer",
        )


def _pack_latent(image: Any, patch_size: int):
    batch, channels, height, width = image.shape
    return (
        image.view(
            batch,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(
            batch,
            (height // patch_size) * (width // patch_size),
            channels * patch_size * patch_size,
        )
    )


def _unpack_latent(
    packed: Any,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
):
    batch = packed.shape[0]
    return (
        packed.view(
            batch,
            height // patch_size,
            width // patch_size,
            channels,
            patch_size,
            patch_size,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


def _position_ids(
    torch,
    text_sequence: int,
    grid_height: int,
    grid_width: int,
    device: Any,
):
    text_ids = torch.zeros(text_sequence, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    return torch.cat(
        (text_ids, image_ids.reshape(grid_height * grid_width, 3)),
        dim=0,
    )


__all__ = [
    "KREA2_CONDITIONING_DIM",
    "KREA2_LATENT_CHANNELS",
    "KREA2_PATCH_SIZE",
    "Krea2TransformerLoadReport",
    "NativeKrea2Transformer",
    "build_krea2_transformer",
]
