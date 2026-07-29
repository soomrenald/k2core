"""Strict ComfyUI-independent safetensors loading for native Krea2 components."""

from __future__ import annotations

import gc
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from k2core.inference.errors import (
    ConfigurationError,
    ModelCompatibilityError,
    WeightMappingError,
)
from k2core.inference.schemas import DTypePolicy, DevicePolicy
from k2core.model import (
    ArtifactKind,
    ComponentReference,
    ModelRegistry,
    RegisteredModel,
    validate_model_registry,
)


SUPPORTED_KREA2_COMPONENT_HASHES: Mapping[ArtifactKind, frozenset[str]] = {
    ArtifactKind.TRANSFORMER: frozenset(
        {"eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1"}
    ),
    ArtifactKind.TEXT_ENCODER: frozenset(
        {"54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"}
    ),
    ArtifactKind.VAE: frozenset(
        {"a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"}
    ),
}


@dataclass(frozen=True, slots=True)
class NativeComponentConfig:
    architecture: str
    role: ArtifactKind
    allowed_prefixes: tuple[str, ...]
    layer_count: int | None = None
    hidden_size: int | None = None


KREA2_COMPONENT_CONFIGS: Mapping[ArtifactKind, NativeComponentConfig] = {
    ArtifactKind.TRANSFORMER: NativeComponentConfig(
        architecture="krea2_single_stream_dit",
        role=ArtifactKind.TRANSFORMER,
        allowed_prefixes=(
            "first.",
            "blocks.",
            "tmlp.",
            "txtfusion.",
            "txtmlp.",
            "last.",
            "tproj.",
        ),
        layer_count=28,
        hidden_size=6144,
    ),
    ArtifactKind.TEXT_ENCODER: NativeComponentConfig(
        architecture="qwen3_vl_4b",
        role=ArtifactKind.TEXT_ENCODER,
        allowed_prefixes=("model.",),
        layer_count=36,
        hidden_size=2560,
    ),
    ArtifactKind.VAE: NativeComponentConfig(
        architecture="qwen_image_vae",
        role=ArtifactKind.VAE,
        allowed_prefixes=("encoder.", "decoder.", "conv1.", "conv2."),
    ),
}


@dataclass(frozen=True, slots=True)
class ComponentLoadReport:
    kind: ArtifactKind
    architecture: str
    path: Path
    sha256: str
    device: str
    requested_dtype: str
    tensor_count: int
    parameter_count: int
    storage_bytes: int
    source_dtypes: tuple[tuple[str, int], ...]
    loaded_dtypes: tuple[tuple[str, int], ...]
    metadata_keys: tuple[str, ...]
    mapped_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    load_seconds: float

    @property
    def strict_match(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["path"] = str(self.path)
        return payload


@dataclass(slots=True)
class NativeComponent:
    config: NativeComponentConfig
    tensors: dict[str, Any]
    metadata: dict[str, str]
    report: ComponentLoadReport

    @property
    def loaded(self) -> bool:
        return bool(self.tensors)

    def unload(self) -> None:
        self.tensors.clear()
        self.metadata.clear()


@dataclass(slots=True)
class NativePipelineState:
    model_name: str
    transformer: NativeComponent
    text_encoder: NativeComponent
    vae: NativeComponent

    @property
    def loaded(self) -> bool:
        return all(
            component.loaded
            for component in (self.transformer, self.text_encoder, self.vae)
        )

    def reports(self) -> tuple[ComponentLoadReport, ...]:
        return (
            self.transformer.report,
            self.text_encoder.report,
            self.vae.report,
        )

    def unload(self) -> None:
        self.transformer.unload()
        self.text_encoder.unload()
        self.vae.unload()


class NativeModelLoader:
    """Load verified state dictionaries without importing ComfyUI."""

    def __init__(
        self,
        *,
        supported_hashes: Mapping[ArtifactKind, frozenset[str]] | None = None,
    ) -> None:
        self.supported_hashes = dict(
            SUPPORTED_KREA2_COMPONENT_HASHES
            if supported_hashes is None
            else supported_hashes
        )

    def load(
        self,
        model: RegisteredModel,
        *,
        device_policy: DevicePolicy = DevicePolicy(),
        strict: bool = True,
    ) -> NativePipelineState:
        validation = validate_model_registry(ModelRegistry(models=(model,)))
        if not validation.valid:
            detail = "; ".join(
                error
                for validated_model in validation.models
                for error in (
                    *validated_model.errors,
                    *(
                        component_error
                        for component in validated_model.components
                        for component_error in component.errors
                    ),
                )
            )
            raise ModelCompatibilityError(
                f"registered model {model.name!r} failed strict validation",
                technical_detail=detail,
                backend_name="native",
                phase="model_loading",
                remediation="Regenerate or correct the model registry before loading.",
            )

        loaded: list[NativeComponent] = []
        try:
            components = dict(model.components())
            if strict:
                self._validate_approved_identities(components)
            transformer = self._load_component(
                ArtifactKind.TRANSFORMER,
                components[ArtifactKind.TRANSFORMER],
                device=device_policy.transformer_device,
                dtype_policy=device_policy.weight_dtype,
                strict=strict,
            )
            loaded.append(transformer)
            text_encoder = self._load_component(
                ArtifactKind.TEXT_ENCODER,
                components[ArtifactKind.TEXT_ENCODER],
                device=device_policy.text_encoder_device,
                dtype_policy=device_policy.weight_dtype,
                strict=strict,
            )
            loaded.append(text_encoder)
            vae = self._load_component(
                ArtifactKind.VAE,
                components[ArtifactKind.VAE],
                device=device_policy.vae_device,
                dtype_policy=device_policy.weight_dtype,
                strict=strict,
            )
            loaded.append(vae)
        except BaseException:
            for component in loaded:
                component.unload()
            self.release_memory()
            raise
        return NativePipelineState(
            model_name=model.name,
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
        )

    def _validate_approved_identities(
        self,
        components: Mapping[ArtifactKind, ComponentReference],
    ) -> None:
        for kind, component in components.items():
            approved = self.supported_hashes.get(kind, frozenset())
            if component.sha256 not in approved:
                raise WeightMappingError(
                    f"{kind.value} has no approved strict state-dict mapping",
                    technical_detail=f"unrecognized SHA-256: {component.sha256}",
                    backend_name="native",
                    phase="model_loading",
                    remediation=(
                        "Review the complete key/shape manifest and add its hash to the "
                        "approved native component identities."
                    ),
                )

    def unload(self, pipeline: NativePipelineState) -> None:
        pipeline.unload()
        self.release_memory()

    def release_memory(self) -> None:
        gc.collect()
        try:
            torch = _import_torch()
        except ConfigurationError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_component(
        self,
        kind: ArtifactKind,
        component: ComponentReference,
        *,
        device: str,
        dtype_policy: DTypePolicy,
        strict: bool,
    ) -> NativeComponent:
        torch = _import_torch()
        safe_open = _import_safe_open()
        resolved_device = _resolve_device(torch, device)
        target_dtype = _resolve_dtype(torch, dtype_policy)
        config = KREA2_COMPONENT_CONFIGS[kind]
        resolved_path = component.path.expanduser().resolve(strict=True)
        started = time.perf_counter()
        tensors: dict[str, Any] = {}
        source_dtypes: Counter[str] = Counter()
        metadata: dict[str, str] = {}
        unexpected: tuple[str, ...] = ()
        try:
            with safe_open(
                str(resolved_path),
                framework="pt",
                device=resolved_device,
            ) as source:
                keys = tuple(source.keys())
                metadata = dict(source.metadata() or {})
                unexpected = tuple(
                    key
                    for key in keys
                    if not key.startswith(config.allowed_prefixes)
                )
                if strict and unexpected:
                    raise WeightMappingError(
                        f"{kind.value} contains unmapped state-dict keys",
                        technical_detail=", ".join(unexpected[:20]),
                        backend_name="native",
                        phase="model_loading",
                        remediation="Add an explicit reviewed mapping before loading this file.",
                    )
                for source_key in keys:
                    if source_key in tensors:
                        raise WeightMappingError(
                            f"duplicate mapped key for {kind.value}: {source_key}",
                            backend_name="native",
                            phase="model_loading",
                        )
                    tensor = source.get_tensor(source_key)
                    source_dtypes[str(tensor.dtype)] += 1
                    if target_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=target_dtype)
                    tensors[source_key] = tensor
        except BaseException:
            tensors.clear()
            raise

        loaded_dtypes = Counter(str(tensor.dtype) for tensor in tensors.values())
        parameter_count = sum(int(tensor.numel()) for tensor in tensors.values())
        storage_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in tensors.values()
        )
        report = ComponentLoadReport(
            kind=kind,
            architecture=config.architecture,
            path=resolved_path,
            sha256=component.sha256,
            device=resolved_device,
            requested_dtype=dtype_policy.value,
            tensor_count=len(tensors),
            parameter_count=parameter_count,
            storage_bytes=storage_bytes,
            source_dtypes=tuple(sorted(source_dtypes.items())),
            loaded_dtypes=tuple(sorted(loaded_dtypes.items())),
            metadata_keys=tuple(sorted(metadata)),
            mapped_key_count=len(tensors),
            missing_keys=(),
            unexpected_keys=unexpected,
            load_seconds=time.perf_counter() - started,
        )
        return NativeComponent(
            config=config,
            tensors=tensors,
            metadata=metadata,
            report=report,
        )


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise ConfigurationError(
            "Native model loading requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="model_loading",
            remediation="Install K2Lab's model dependencies in the selected worker environment.",
        ) from error
    return torch


def _import_safe_open():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ConfigurationError(
            "Native model loading requires safetensors.",
            technical_detail=str(error),
            backend_name="native",
            phase="model_loading",
            remediation="Install K2Lab's model dependencies in the selected worker environment.",
        ) from error
    return safe_open


def _resolve_device(torch, requested: str) -> str:
    normalized = requested.strip().casefold()
    if normalized == "auto":
        # Stage weights on CPU by default. Phase-specific placement can move
        # executable modules without forcing all three components into VRAM.
        return "cpu"
    if normalized == "rocm":
        normalized = "cuda"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigurationError(
            f"requested accelerator device is unavailable: {requested}",
            backend_name="native",
            phase="model_loading",
            remediation="Choose CPU/auto or make the CUDA/ROCm device visible.",
        )
    try:
        return str(torch.device(normalized))
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConfigurationError(
            f"invalid native model device: {requested}",
            technical_detail=str(error),
            backend_name="native",
            phase="model_loading",
        ) from error


def _resolve_dtype(torch, policy: DTypePolicy):
    if policy == DTypePolicy.AUTO:
        return None
    names = {
        DTypePolicy.BFLOAT16: "bfloat16",
        DTypePolicy.FLOAT16: "float16",
        DTypePolicy.FLOAT32: "float32",
        DTypePolicy.FLOAT8_E4M3FN: "float8_e4m3fn",
    }
    name = names[policy]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ConfigurationError(
            f"this PyTorch build does not support {policy.value}",
            backend_name="native",
            phase="model_loading",
        )
    return dtype


__all__ = [
    "ComponentLoadReport",
    "KREA2_COMPONENT_CONFIGS",
    "NativeComponent",
    "NativeComponentConfig",
    "NativeModelLoader",
    "NativePipelineState",
    "SUPPORTED_KREA2_COMPONENT_HASHES",
]
