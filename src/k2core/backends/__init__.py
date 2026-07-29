"""Typed backend contracts shared by K2 and Wan consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from k2core.face_detail import DetectedFace


ProgressCallback = Callable[[str, float | None, Mapping[str, Any]], None]


@runtime_checkable
class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_id: str
    modes: frozenset[str]
    accelerator_vendors: frozenset[str]
    parameters: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendResult:
    asset_paths: tuple[Path, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FaceDetectionResult:
    faces: tuple[DetectedFace, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageGeneratorBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def validate_image_request(self, request: Mapping[str, Any]) -> tuple[str, ...]: ...

    def generate_image(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def release(self) -> None: ...


@runtime_checkable
class FrameEditorBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def validate_edit_request(self, request: Mapping[str, Any]) -> tuple[str, ...]: ...

    def edit_frame(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def detect_faces(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> FaceDetectionResult: ...

    def refine_faces(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def release(self) -> None: ...


@runtime_checkable
class VideoGeneratorBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def validate_segment_request(self, request: Mapping[str, Any]) -> tuple[str, ...]: ...

    def generate_segment(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def release(self) -> None: ...


__all__ = [
    "BackendCapabilities",
    "BackendResult",
    "CancellationToken",
    "FaceDetectionResult",
    "FrameEditorBackend",
    "ImageGeneratorBackend",
    "ProgressCallback",
    "VideoGeneratorBackend",
]

from k2core.backends.comfy_krea import ComfyKreaBackend  # noqa: E402
from k2core.backends.comfyui import ComfyUIBackend  # noqa: E402
from k2core.backends.native import NativeK2Backend  # noqa: E402
from k2core.backends.native_loading import (  # noqa: E402
    ComponentLoadReport,
    NativeComponent,
    NativeComponentConfig,
    NativeModelLoader,
    NativePipelineState,
)
from k2core.backends.native_lora import (  # noqa: E402
    NativeLoraReport,
    NativeLoraTarget,
    apply_native_loras,
)
from k2core.backends.native_qwen import (  # noqa: E402
    KREA2_QWEN_TAP_LAYERS,
    KREA2_TEXT_FEATURE_SIZE,
    KreaTextEncoding,
    NativeQwenTextEncoder,
    QwenExecutableLoadReport,
    build_qwen_text_encoder,
)
from k2core.backends.native_sampling import (  # noqa: E402
    DenoisingCheckpoint,
    euler_flow_sample,
    flux_time_shift,
    prepare_noise,
    simple_sigmas,
)
from k2core.backends.native_text import (  # noqa: E402
    KREA2_TEMPLATE,
    KreaPromptTokens,
    load_tokenizer,
    tokenize_prompt,
)
from k2core.backends.native_transformer import (  # noqa: E402
    KREA2_CONDITIONING_DIM,
    KREA2_LATENT_CHANNELS,
    KREA2_PATCH_SIZE,
    Krea2TransformerLoadReport,
    NativeKrea2Transformer,
    build_krea2_transformer,
)
from k2core.backends.native_vae import (  # noqa: E402
    KREA2_VAE_LATENT_CHANNELS,
    KREA2_VAE_SCALE_FACTOR,
    Krea2VAELoadReport,
    NativeKrea2VAE,
    build_krea2_vae,
)

__all__ += [
    "ComfyKreaBackend",
    "ComfyUIBackend",
    "ComponentLoadReport",
    "DenoisingCheckpoint",
    "KREA2_QWEN_TAP_LAYERS",
    "KREA2_CONDITIONING_DIM",
    "KREA2_LATENT_CHANNELS",
    "KREA2_PATCH_SIZE",
    "KREA2_TEMPLATE",
    "KREA2_TEXT_FEATURE_SIZE",
    "KREA2_VAE_LATENT_CHANNELS",
    "KREA2_VAE_SCALE_FACTOR",
    "KreaPromptTokens",
    "KreaTextEncoding",
    "Krea2TransformerLoadReport",
    "Krea2VAELoadReport",
    "NativeComponent",
    "NativeComponentConfig",
    "NativeK2Backend",
    "NativeLoraReport",
    "NativeLoraTarget",
    "NativeModelLoader",
    "NativePipelineState",
    "NativeQwenTextEncoder",
    "NativeKrea2Transformer",
    "NativeKrea2VAE",
    "QwenExecutableLoadReport",
    "build_qwen_text_encoder",
    "build_krea2_transformer",
    "build_krea2_vae",
    "apply_native_loras",
    "euler_flow_sample",
    "flux_time_shift",
    "load_tokenizer",
    "prepare_noise",
    "simple_sigmas",
    "tokenize_prompt",
]
