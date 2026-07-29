"""Structured errors shared by inference backends and product transports."""

from __future__ import annotations

from typing import Any


class K2InferenceError(RuntimeError):
    """Base error carrying stable application-facing failure context."""

    category = "GenerationError"

    def __init__(
        self,
        summary: str,
        *,
        technical_detail: str = "",
        backend_name: str = "unknown",
        phase: str = "unknown",
        remediation: str = "",
        retry_safe: bool = False,
        gpu_work_started: bool = False,
        correlation_id: str = "",
    ) -> None:
        super().__init__(summary)
        self.summary = summary
        self.technical_detail = technical_detail
        self.backend_name = backend_name
        self.phase = phase
        self.remediation = remediation
        self.retry_safe = bool(retry_safe)
        self.gpu_work_started = bool(gpu_work_started)
        self.correlation_id = correlation_id

    def to_payload(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "summary": self.summary,
            "technical_detail": self.technical_detail,
            "backend_name": self.backend_name,
            "phase": self.phase,
            "remediation": self.remediation,
            "retry_safe": self.retry_safe,
            "gpu_work_started": self.gpu_work_started,
            "correlation_id": self.correlation_id,
        }


class ConfigurationError(K2InferenceError):
    category = "ConfigurationError"


class ModelNotFoundError(K2InferenceError):
    category = "ModelNotFoundError"


class ModelCompatibilityError(K2InferenceError):
    category = "ModelCompatibilityError"


class WeightMappingError(K2InferenceError):
    category = "WeightMappingError"


class UnsupportedFeatureError(K2InferenceError):
    category = "UnsupportedFeatureError"


class InvalidRequestError(K2InferenceError):
    category = "InvalidRequestError"


class OutOfMemoryError(K2InferenceError):
    category = "OutOfMemoryError"


class CancellationError(K2InferenceError):
    category = "CancellationError"


class BackendInitializationError(K2InferenceError):
    category = "BackendInitializationError"


class GenerationError(K2InferenceError):
    category = "GenerationError"


class DecodeError(K2InferenceError):
    category = "DecodeError"


class InfrastructureTimeoutError(K2InferenceError):
    category = "InfrastructureTimeoutError"


class WorkerDisconnectedError(K2InferenceError):
    category = "WorkerDisconnectedError"


def convert_error(
    error: BaseException,
    *,
    backend_name: str,
    phase: str,
    correlation_id: str = "",
    gpu_work_started: bool = False,
) -> K2InferenceError:
    """Preserve known structured errors and classify common legacy failures."""

    if isinstance(error, K2InferenceError):
        return error
    details = f"{type(error).__name__}: {error}"
    common = {
        "technical_detail": details,
        "backend_name": backend_name,
        "phase": phase,
        "correlation_id": correlation_id,
        "gpu_work_started": gpu_work_started,
    }
    if isinstance(error, FileNotFoundError):
        return ModelNotFoundError(
            str(error) or "A required model file was not found.",
            remediation="Verify the configured model paths and retry.",
            retry_safe=True,
            **common,
        )
    if isinstance(error, MemoryError) or "out of memory" in str(error).casefold():
        return OutOfMemoryError(
            "The inference backend ran out of memory.",
            remediation="Reduce the workload or choose a safer device policy before retrying.",
            retry_safe=True,
            **common,
        )
    if isinstance(error, (InterruptedError, KeyboardInterrupt)):
        return CancellationError(
            "The inference request was cancelled.",
            retry_safe=True,
            **common,
        )
    if isinstance(error, (TypeError, ValueError)):
        return InvalidRequestError(
            str(error) or "The inference request is invalid.",
            retry_safe=False,
            **common,
        )
    if phase in {"backend_initialization", "model_loading"}:
        return BackendInitializationError(
            str(error) or "The inference backend could not be initialized.",
            remediation=(
                "Verify the selected backend, accelerator runtime, and model configuration."
            ),
            retry_safe=True,
            **common,
        )
    return GenerationError(
        str(error) or "Inference failed.",
        retry_safe=False,
        **common,
    )


__all__ = [
    "BackendInitializationError",
    "CancellationError",
    "ConfigurationError",
    "DecodeError",
    "GenerationError",
    "InfrastructureTimeoutError",
    "InvalidRequestError",
    "K2InferenceError",
    "ModelCompatibilityError",
    "ModelNotFoundError",
    "OutOfMemoryError",
    "UnsupportedFeatureError",
    "WeightMappingError",
    "WorkerDisconnectedError",
    "convert_error",
]
