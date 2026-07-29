"""Explicit inference backend selection with a stable ComfyUI default."""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import Mapping, TypeVar

from k2core.inference.backend import InferenceBackend
from k2core.inference.errors import ConfigurationError, UnsupportedFeatureError


BACKEND_ENVIRONMENT_NAME = "K2LAB_BACKEND"


class BackendName(StrEnum):
    COMFYUI = "comfyui"
    NATIVE = "native"


def configured_backend_name(
    environment: Mapping[str, str] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> BackendName:
    source = os.environ if environment is None else environment
    supplied = str(source.get(BACKEND_ENVIRONMENT_NAME, "")).strip()
    raw = supplied or BackendName.COMFYUI.value
    normalized = str(raw).strip().casefold() or BackendName.COMFYUI.value
    try:
        selected = BackendName(normalized)
    except ValueError as error:
        raise ConfigurationError(
            f"Unsupported K2Lab backend {raw!r}.",
            technical_detail=(
                f"{BACKEND_ENVIRONMENT_NAME} must be one of: "
                + ", ".join(item.value for item in BackendName)
            ),
            backend_name=normalized,
            phase="backend_selection",
            remediation=f"Set {BACKEND_ENVIRONMENT_NAME}=comfyui to restore the reference backend.",
            retry_safe=True,
        ) from error
    (logger or logging.getLogger(__name__)).info(
        "selected inference backend=%s source=%s",
        selected.value,
        BACKEND_ENVIRONMENT_NAME if supplied else "default",
    )
    return selected


_BackendT = TypeVar("_BackendT", bound=InferenceBackend)


def select_backend(
    comfyui: _BackendT,
    *,
    native: _BackendT | None = None,
    environment: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> _BackendT:
    selected = configured_backend_name(environment, logger=logger)
    if selected is BackendName.COMFYUI:
        return comfyui
    if native is None:
        raise UnsupportedFeatureError(
            "The native K2 backend is not implemented.",
            backend_name=BackendName.NATIVE.value,
            phase="backend_selection",
            remediation=f"Set {BACKEND_ENVIRONMENT_NAME}=comfyui.",
            retry_safe=True,
        )
    return native


__all__ = [
    "BACKEND_ENVIRONMENT_NAME",
    "BackendName",
    "configured_backend_name",
    "select_backend",
]
