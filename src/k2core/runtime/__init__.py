"""Accelerator-neutral model lifecycle and memory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from k2core.memory import MemoryPolicy


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AcceleratorDevice:
    device_id: str
    name: str
    vendor: AcceleratorVendor
    total_memory_bytes: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ResidencyRecord:
    backend_id: str
    model_id: str
    residency_group: str
    device_id: str | None
    loaded: bool


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    devices: tuple[AcceleratorDevice, ...]
    residency: tuple[ResidencyRecord, ...]
    memory: dict[str, Any]
    warnings: tuple[str, ...] = ()


@runtime_checkable
class ModelRuntime(Protocol):
    """Model lifecycle interface shared by image and video backends."""

    def devices(self) -> tuple[AcceleratorDevice, ...]: ...

    def load_model(
        self,
        backend_id: str,
        model_id: str,
        *,
        residency_group: str,
        options: dict[str, Any] | None = None,
    ) -> ResidencyRecord: ...

    def release_model(self, backend_id: str, model_id: str) -> None: ...

    def release_group(self, residency_group: str) -> None: ...

    def release_all(self) -> None: ...

    def set_memory_policy(self, policy: MemoryPolicy) -> None: ...

    def status(self) -> RuntimeStatus: ...


__all__ = [
    "AcceleratorDevice",
    "AcceleratorVendor",
    "ModelRuntime",
    "ResidencyRecord",
    "RuntimeStatus",
]

