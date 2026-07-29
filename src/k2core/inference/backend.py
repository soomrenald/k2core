"""Backend-neutral inference lifecycle contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from k2core.inference.schemas import (
    GenerationResult,
    ImageEncodeRequest,
    ImageResult,
    InferenceRequest,
    LatentDecodeRequest,
    LatentResult,
    LoadedPipeline,
    PipelineConfig,
    ProgressEvent,
)

if TYPE_CHECKING:
    from k2core.backends import BackendCapabilities


ProgressCallback = Callable[[ProgressEvent], None]
DiagnosticCallback = Callable[[str, dict[str, Any]], None]


@runtime_checkable
class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


class NullCancellationToken:
    cancelled = False

    def raise_if_cancelled(self) -> None:
        return None


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def load(self, config: PipelineConfig) -> LoadedPipeline: ...

    def generate(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
        diagnostic: DiagnosticCallback | None = None,
    ) -> GenerationResult: ...

    def encode_image(self, request: ImageEncodeRequest) -> LatentResult: ...

    def decode_latents(self, request: LatentDecodeRequest) -> ImageResult: ...

    def unload(self) -> None: ...

    def capabilities(self) -> "BackendCapabilities": ...


__all__ = [
    "CancellationToken",
    "DiagnosticCallback",
    "InferenceBackend",
    "NullCancellationToken",
    "ProgressCallback",
]
