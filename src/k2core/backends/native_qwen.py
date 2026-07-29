"""Executable ComfyUI-independent Qwen3-VL text conditioning for Krea2."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from typing import Any, Mapping

from k2core.backends.native_loading import NativeComponent
from k2core.backends.native_quant import replace_submodule, scaled_fp8_linear_class
from k2core.backends.native_text import KreaPromptTokens, load_tokenizer, tokenize_prompt
from k2core.inference.errors import (
    ConfigurationError,
    ModelCompatibilityError,
    WeightMappingError,
)
from k2core.model import ArtifactKind, TokenizerReference


KREA2_QWEN_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
QWEN3_VL_TEXT_HIDDEN_SIZE = 2560
KREA2_TEXT_FEATURE_SIZE = len(KREA2_QWEN_TAP_LAYERS) * QWEN3_VL_TEXT_HIDDEN_SIZE
SUPPORTED_QUANT_FORMAT = "float8_e4m3fn"


@dataclass(frozen=True, slots=True)
class QwenExecutableLoadReport:
    source_tensor_count: int
    mapped_tensor_count: int
    text_parameter_count: int
    quantized_linear_count: int
    ignored_vision_tensor_count: int
    tap_layers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KreaTextEncoding:
    conditioning: Any
    attention_mask: Any
    tokens: KreaPromptTokens
    tap_layers: tuple[int, ...] = KREA2_QWEN_TAP_LAYERS


class NativeQwenTextEncoder:
    """Own an executable Qwen3-VL-4B text model and its verified tokenizer."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        report: QwenExecutableLoadReport,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.report = report

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def encode(
        self,
        prompt: str,
        *,
        device: str = "cuda",
    ) -> KreaTextEncoding:
        if self.model is None:
            raise RuntimeError("native Qwen text encoder has been unloaded")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        torch, _, _ = _import_runtime()
        resolved_device = _resolve_execution_device(torch, device)
        self.model.to(device=resolved_device)
        tokens = tokenize_prompt(prompt, self.tokenizer)
        input_ids = torch.tensor(
            (tokens.input_ids,),
            dtype=torch.long,
            device=resolved_device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            # Comfy's Krea2 SDClipModel requests float32 Qwen activations even
            # though the stored embedding and norm parameters are BF16.
            inputs_embeds = self.model.embed_tokens(input_ids).to(torch.float32)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) != 37:
                observed = 0 if hidden_states is None else len(hidden_states)
                raise ModelCompatibilityError(
                    "Qwen3-VL did not return the required layer states.",
                    technical_detail=f"expected 37 hidden states, got {observed}",
                    backend_name="native",
                    phase="text_encoding",
                )

            # Transformers hidden_states[k] is the input to decoder layer k. Krea2
            # taps that same state and deliberately disables the optional final
            # normalization of intermediate states.
            tapped = tuple(hidden_states[layer] for layer in KREA2_QWEN_TAP_LAYERS)
            stacked = torch.stack(tapped, dim=2)
            conditioned = stacked[:, tokens.output_start :, :, :].reshape(
                input_ids.shape[0],
                len(tokens.conditioned_ids),
                KREA2_TEXT_FEATURE_SIZE,
            )
            conditioned_mask = attention_mask[:, tokens.output_start :]

        return KreaTextEncoding(
            conditioning=conditioned,
            attention_mask=conditioned_mask,
            tokens=tokens,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            torch, _, _ = _import_runtime()
        except ConfigurationError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_qwen_text_encoder(
    component: NativeComponent,
    tokenizer_reference: TokenizerReference,
) -> NativeQwenTextEncoder:
    """Build the reviewed text-only Qwen3-VL graph from a strict component."""

    if component.config.role != ArtifactKind.TEXT_ENCODER:
        raise WeightMappingError(
            "native Qwen builder received the wrong component role",
            technical_detail=f"got {component.config.role.value}",
            backend_name="native",
            phase="text_encoding",
        )
    if not component.loaded:
        raise RuntimeError("native text encoder component has been unloaded")

    torch, nn, functional = _import_runtime()
    try:
        from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 text encoding requires Qwen3-VL support in Transformers.",
            technical_detail=str(error),
            backend_name="native",
            phase="text_encoding",
            remediation="Install the worker's pinned model dependencies.",
        ) from error

    config = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=QWEN3_VL_TEXT_HIDDEN_SIZE,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)

    quantized_paths = _validate_quantization_markers(component.tensors)
    scaled_linear = scaled_fp8_linear_class(torch, nn, functional)
    for module_path in sorted(quantized_paths):
        original = model.get_submodule(module_path)
        if not isinstance(original, nn.Linear):
            raise WeightMappingError(
                f"quantized Qwen target is not a linear layer: {module_path}",
                backend_name="native",
                phase="text_encoding",
            )
        replacement = scaled_linear(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
            device="meta",
        )
        replace_submodule(model, module_path, replacement)

    state, ignored_vision = _text_state_dict(component.tensors)
    expected = model.state_dict()
    _validate_state_dict_shapes(expected, state)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise WeightMappingError(
            "Qwen3-VL executable state mapping was not strict",
            technical_detail=(
                f"missing={incompatible.missing_keys}; "
                f"unexpected={incompatible.unexpected_keys}"
            ),
            backend_name="native",
            phase="text_encoding",
        )
    # The rotary frequencies are a derived, non-persistent buffer, so they are not
    # present in the checkpoint and remain on meta after strict parameter loading.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    model.requires_grad_(False)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_reference)
    report = QwenExecutableLoadReport(
        source_tensor_count=len(component.tensors),
        mapped_tensor_count=len(state),
        text_parameter_count=sum(
            int(tensor.numel())
            for name, tensor in state.items()
            if not name.endswith(".weight_scale")
        ),
        quantized_linear_count=len(quantized_paths),
        ignored_vision_tensor_count=ignored_vision,
        tap_layers=KREA2_QWEN_TAP_LAYERS,
    )
    return NativeQwenTextEncoder(model=model, tokenizer=tokenizer, report=report)


def _import_runtime():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 text encoding requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="text_encoding",
            remediation="Install K2Lab's model dependencies in the worker environment.",
        ) from error
    return torch, nn, functional


def _resolve_execution_device(torch, requested: str):
    normalized = requested.strip().casefold()
    if normalized in {"auto", "rocm"}:
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigurationError(
            f"requested text encoder device is unavailable: {requested}",
            backend_name="native",
            phase="text_encoding",
        )
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConfigurationError(
            f"invalid text encoder device: {requested}",
            technical_detail=str(error),
            backend_name="native",
            phase="text_encoding",
        ) from error


def _validate_quantization_markers(tensors: Mapping[str, Any]) -> frozenset[str]:
    markers = {
        key.removeprefix("model.").removesuffix(".comfy_quant"): tensor
        for key, tensor in tensors.items()
        if key.startswith("model.")
        and not key.startswith("model.visual.")
        and key.endswith(".comfy_quant")
    }
    scales = {
        key.removeprefix("model.").removesuffix(".weight_scale")
        for key in tensors
        if key.startswith("model.")
        and not key.startswith("model.visual.")
        and key.endswith(".weight_scale")
    }
    if set(markers) != scales:
        raise WeightMappingError(
            "Qwen3-VL FP8 markers and scales do not target the same layers",
            technical_detail=(
                f"marker-only={sorted(set(markers) - scales)}; "
                f"scale-only={sorted(scales - set(markers))}"
            ),
            backend_name="native",
            phase="text_encoding",
        )

    for module_path, marker in markers.items():
        try:
            payload = json.loads(bytes(marker.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WeightMappingError(
                f"invalid Qwen quantization marker: {module_path}",
                technical_detail=str(error),
                backend_name="native",
                phase="text_encoding",
            ) from error
        expected = {
            "format": SUPPORTED_QUANT_FORMAT,
            "full_precision_matrix_mult": False,
        }
        if payload != expected:
            raise WeightMappingError(
                f"unsupported Qwen quantization contract: {module_path}",
                technical_detail=f"expected {expected}, got {payload}",
                backend_name="native",
                phase="text_encoding",
            )
    return frozenset(markers)


def _text_state_dict(
    tensors: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    state: dict[str, Any] = {}
    ignored_vision = 0
    for source_name, tensor in tensors.items():
        if source_name.startswith("model.visual."):
            ignored_vision += 1
            continue
        if not source_name.startswith("model."):
            raise WeightMappingError(
                f"unrecognized Qwen checkpoint namespace: {source_name}",
                backend_name="native",
                phase="text_encoding",
            )
        if source_name.endswith(".comfy_quant"):
            continue
        target_name = source_name.removeprefix("model.")
        if target_name in state:
            raise WeightMappingError(
                f"duplicate Qwen target key: {target_name}",
                backend_name="native",
                phase="text_encoding",
            )
        state[target_name] = tensor
    return state, ignored_vision


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
        detail = (
            f"missing={missing[:20]}; unexpected={unexpected[:20]}; "
            f"shape_mismatch={mismatched[:20]}"
        )
        raise WeightMappingError(
            "Qwen3-VL text checkpoint does not match the reviewed executable graph",
            technical_detail=detail,
            backend_name="native",
            phase="text_encoding",
            remediation="Review the complete text-only state-dict mapping.",
        )


__all__ = [
    "KREA2_QWEN_TAP_LAYERS",
    "KREA2_TEXT_FEATURE_SIZE",
    "KreaTextEncoding",
    "NativeQwenTextEncoder",
    "QwenExecutableLoadReport",
    "build_qwen_text_encoder",
]
