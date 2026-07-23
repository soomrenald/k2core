"""Typed backend contracts shared by K2 and Wan consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


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
    "FrameEditorBackend",
    "ImageGeneratorBackend",
    "ProgressCallback",
    "VideoGeneratorBackend",
]

