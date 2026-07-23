"""Application-neutral asset records and storage contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """Immutable identity and provenance-facing metadata for one asset."""

    asset_id: str
    relative_path: str
    sha256: str
    media_type: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    parent_asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be empty")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("asset path must be project-relative and contained")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if self.byte_size < 0:
            raise ValueError("byte_size must not be negative")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")


@runtime_checkable
class AssetStore(Protocol):
    """Storage boundary implemented by local and workspace-backed adapters."""

    def register_imported(
        self,
        source: Path,
        *,
        media_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> AssetRecord: ...

    def create_derived(
        self,
        source: Path,
        *,
        parent_asset_ids: tuple[str, ...],
        media_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> AssetRecord: ...

    def resolve(self, asset: AssetRecord) -> Path: ...

    def verify(self, asset: AssetRecord) -> bool: ...


__all__ = ["AssetRecord", "AssetStore"]

