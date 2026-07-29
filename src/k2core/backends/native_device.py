"""K2-owned device planning, memory telemetry, and cleanup for native inference."""

from __future__ import annotations

import gc
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from k2core.inference.errors import ConfigurationError, OutOfMemoryError
from k2core.inference.schemas import DTypePolicy, DevicePolicy


@dataclass(frozen=True, slots=True)
class NativeDevicePlan:
    accelerator_available: bool
    accelerator_backend: str
    accelerator_name: str
    transformer_device: str
    text_encoder_device: str
    vae_device: str
    compute_dtype: str
    weight_dtype: str
    cpu_offload: bool
    sequential_components: bool
    vae_tiling: bool
    reserve_vram_bytes: int
    minimum_system_ram_bytes: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class NativeDeviceManager:
    """Resolve explicit placement and own allocator observations and cleanup."""

    def __init__(
        self,
        policy: DevicePolicy,
        *,
        reserve_vram_gb: float = 0.0,
        minimum_system_ram_gb: float = 0.0,
        cpu_vae: bool = False,
        torch: Any | None = None,
        memory_reader: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self.policy = policy
        self.torch = torch or _import_torch()
        self._memory_reader = memory_reader or _system_memory
        self._accelerator_available = bool(self.torch.cuda.is_available())
        self._accelerator_backend = self._detect_accelerator_backend()
        self._accelerator_name = self._detect_accelerator_name()
        self._cpu_vae = bool(cpu_vae)
        self._reserve_vram_bytes = round(float(reserve_vram_gb) * 1024**3)
        self._minimum_system_ram_bytes = round(float(minimum_system_ram_gb) * 1024**3)
        if self._reserve_vram_bytes < 0 or self._minimum_system_ram_bytes < 0:
            raise ValueError("native memory thresholds must not be negative")
        self._validate_dtype(policy.compute_dtype, usage="compute")
        self._validate_dtype(policy.weight_dtype, usage="weight")
        self._retry_available = False
        self._recovery_events: list[dict[str, str]] = []
        vae_request = "cpu" if cpu_vae else policy.vae_device
        self.plan = NativeDevicePlan(
            accelerator_available=self._accelerator_available,
            accelerator_backend=self._accelerator_backend,
            accelerator_name=self._accelerator_name,
            transformer_device=self._resolve_execution_device(
                policy.transformer_device,
                phase="transformer",
            ),
            text_encoder_device=self._resolve_execution_device(
                policy.text_encoder_device,
                phase="text_encoding",
            ),
            vae_device=self._resolve_execution_device(
                vae_request,
                phase="vae",
            ),
            compute_dtype=policy.compute_dtype.value,
            weight_dtype=policy.weight_dtype.value,
            cpu_offload=bool(policy.cpu_offload),
            sequential_components=True,
            vae_tiling=bool(policy.vae_tiling),
            reserve_vram_bytes=self._reserve_vram_bytes,
            minimum_system_ram_bytes=self._minimum_system_ram_bytes,
        )

    def staging_policy(self) -> DevicePolicy:
        """Keep auto/offloaded component tensors on CPU until their execution phase."""

        def staging_device(requested: str) -> str:
            if self.policy.cpu_offload or requested.strip().casefold() == "auto":
                return "cpu"
            return requested

        return replace(
            self.policy,
            transformer_device=staging_device(self.policy.transformer_device),
            text_encoder_device=staging_device(self.policy.text_encoder_device),
            vae_device=("cpu" if self._cpu_vae else staging_device(self.policy.vae_device)),
        )

    def preflight(self, stage: str) -> dict[str, Any]:
        snapshot = self.snapshot(stage)
        available_ram = int(snapshot["ram_available_bytes"])
        if available_ram < self._minimum_system_ram_bytes:
            raise OutOfMemoryError(
                "Available system RAM is below the native device-plan minimum.",
                technical_detail=(
                    f"available={available_ram}; "
                    f"minimum={self._minimum_system_ram_bytes}; stage={stage}"
                ),
                backend_name="native",
                phase=stage,
                remediation="Close other workloads or choose a lower-memory device policy.",
                retry_safe=True,
            )
        if self._accelerator_available:
            usable = int(snapshot["gpu_free_bytes"]) - self._reserve_vram_bytes
            if usable <= 0:
                raise OutOfMemoryError(
                    "Available accelerator memory is below the native reserve.",
                    technical_detail=(
                        f"free={snapshot['gpu_free_bytes']}; "
                        f"reserve={self._reserve_vram_bytes}; stage={stage}"
                    ),
                    backend_name="native",
                    phase=stage,
                    remediation="Close other GPU workloads or reduce the configured reserve.",
                    retry_safe=True,
                )
        return snapshot

    def snapshot(self, stage: str) -> dict[str, Any]:
        ram_available, ram_total = self._memory_reader()
        payload: dict[str, Any] = {
            "stage": stage,
            "accelerator_available": self._accelerator_available,
            "accelerator_backend": self._accelerator_backend,
            "accelerator_name": self._accelerator_name,
            "gpu_free_bytes": 0,
            "gpu_total_bytes": 0,
            "gpu_allocated_bytes": 0,
            "gpu_reserved_bytes": 0,
            "gpu_peak_allocated_bytes": 0,
            "gpu_peak_reserved_bytes": 0,
            "ram_available_bytes": int(ram_available),
            "ram_total_bytes": int(ram_total),
            "reserve_vram_bytes": self._reserve_vram_bytes,
            "minimum_system_ram_bytes": self._minimum_system_ram_bytes,
        }
        if not self._accelerator_available:
            return payload
        try:
            free, total = self.torch.cuda.mem_get_info()
            payload.update(
                {
                    "gpu_free_bytes": int(free),
                    "gpu_total_bytes": int(total),
                    "gpu_allocated_bytes": int(self.torch.cuda.memory_allocated()),
                    "gpu_reserved_bytes": int(self.torch.cuda.memory_reserved()),
                    "gpu_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
                    "gpu_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
                }
            )
        except (RuntimeError, TypeError, ValueError) as error:
            payload["telemetry_error"] = f"{type(error).__name__}: {error}"
        return payload

    def reset_peak_stats(self) -> None:
        if self._accelerator_available:
            self.torch.cuda.reset_peak_memory_stats()

    def begin_request(self, *, allow_oom_retry: bool) -> None:
        self._retry_available = bool(allow_oom_retry)
        self._recovery_events.clear()
        self.reset_peak_stats()

    def claim_oom_retry(self, phase: str, fallback: str) -> bool:
        if not self._retry_available:
            return False
        self._retry_available = False
        self._recovery_events.append(
            {
                "phase": phase,
                "fallback": fallback,
            }
        )
        return True

    def recovery_summary(self) -> dict[str, Any]:
        return {
            "retry_used": bool(self._recovery_events),
            "retry_remaining": self._retry_available,
            "events": tuple(dict(event) for event in self._recovery_events),
        }

    def release(self) -> dict[str, Any]:
        gc.collect()
        if self._accelerator_available:
            core = getattr(self.torch, "_C", None)
            clear_workspaces = getattr(core, "_cuda_clearCublasWorkspaces", None)
            if callable(clear_workspaces):
                clear_workspaces()
            self.torch.cuda.empty_cache()
        return self.snapshot("native cleanup complete")

    def cleanup_after_failure(self, phase: str) -> dict[str, Any]:
        snapshot = self.release()
        snapshot["failed_phase"] = phase
        return snapshot

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        return isinstance(error, (MemoryError, OutOfMemoryError)) or (
            "out of memory" in str(error).casefold()
        )

    def _resolve_execution_device(self, requested: str, *, phase: str) -> str:
        normalized = requested.strip().casefold()
        if normalized == "auto":
            if self._accelerator_available:
                return "cuda"
            raise ConfigurationError(
                "Native auto placement requires an accelerator; explicitly select CPU "
                "to allow CPU execution.",
                backend_name="native",
                phase=phase,
                remediation=(
                    "Run on a supported accelerator or explicitly select CPU for this component."
                ),
            )
        if normalized == "rocm":
            normalized = "cuda"
        if normalized.startswith("cuda") and not self._accelerator_available:
            raise ConfigurationError(
                f"requested accelerator device is unavailable: {requested}",
                backend_name="native",
                phase=phase,
                remediation="Make the CUDA/ROCm device visible or explicitly select CPU.",
            )
        try:
            return str(self.torch.device(normalized))
        except (RuntimeError, TypeError, ValueError) as error:
            raise ConfigurationError(
                f"invalid native execution device: {requested}",
                technical_detail=str(error),
                backend_name="native",
                phase=phase,
            ) from error

    def _validate_dtype(self, policy: DTypePolicy, *, usage: str) -> None:
        if policy == DTypePolicy.AUTO:
            return
        if usage == "compute" and policy == DTypePolicy.FLOAT8_E4M3FN:
            raise ConfigurationError(
                "FP8 is supported for native checkpoint weights, not as a compute dtype.",
                backend_name="native",
                phase="device_planning",
                remediation="Choose auto, bfloat16, float16, or float32 compute.",
            )
        attribute = {
            DTypePolicy.BFLOAT16: "bfloat16",
            DTypePolicy.FLOAT16: "float16",
            DTypePolicy.FLOAT32: "float32",
            DTypePolicy.FLOAT8_E4M3FN: "float8_e4m3fn",
        }[policy]
        if getattr(self.torch, attribute, None) is None:
            raise ConfigurationError(
                f"this PyTorch build does not support {policy.value} {usage}",
                backend_name="native",
                phase="device_planning",
            )

    def _detect_accelerator_backend(self) -> str:
        if not self._accelerator_available:
            return "cpu"
        version = getattr(self.torch, "version", None)
        return "rocm" if getattr(version, "hip", None) else "cuda"

    def _detect_accelerator_name(self) -> str:
        if not self._accelerator_available:
            return "none"
        try:
            return str(self.torch.cuda.get_device_name())
        except (RuntimeError, TypeError, ValueError):
            return "unknown"


def _system_memory() -> tuple[int, int]:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                key, value = line.split(":", 1)
                if key in {"MemAvailable", "MemTotal"}:
                    values[key] = int(value.strip().split()[0]) * 1024
        if {"MemAvailable", "MemTotal"} <= values.keys():
            return values["MemAvailable"], values["MemTotal"]
    except (OSError, ValueError):
        pass
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
    available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
    return available, total


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise ConfigurationError(
            "Native device management requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="device_planning",
            remediation="Install K2Lab's model dependencies in the selected worker environment.",
        ) from error
    return torch


__all__ = [
    "NativeDeviceManager",
    "NativeDevicePlan",
]
