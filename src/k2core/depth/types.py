from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class DepthNormalizationMode(StrEnum):
    NONE = "none"
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    CAMERA_RANGE = "camera_range"
    CHECKPOINT_REFERENCE = "checkpoint_reference"


class DepthInvalidValuePolicy(StrEnum):
    FAR = "far"
    NEAR = "near"
    ERROR = "error"


class DepthRegionMode(StrEnum):
    INHERIT = "inherit"
    EMPHASIZE = "emphasize"
    RELAX = "relax"
    IGNORE = "ignore"
    OVERRIDE = "override"


@dataclass(frozen=True, slots=True)
class DepthImageInfo:
    path: Path
    format: str
    width: int
    height: int
    dtype: str
    bit_depth: int
    minimum: float
    maximum: float
    mode: str

    def document(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "bit_depth": self.bit_depth,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class DepthImage:
    values: np.ndarray
    info: DepthImageInfo

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("depth image must contain exactly one channel")
        if values.shape != (self.info.height, self.info.width):
            raise ValueError("depth pixels do not match the declared dimensions")
        if values.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError("depth image pixels must be uint8 or uint16")
        immutable = np.ascontiguousarray(values)
        immutable.setflags(write=False)
        object.__setattr__(self, "values", immutable)


@dataclass(frozen=True, slots=True)
class DepthPreprocessReport:
    normalization: DepthNormalizationMode
    invert: bool
    gamma: float
    clamp: bool
    source_minimum: float
    source_maximum: float
    normalization_low: float
    normalization_high: float
    normalized_minimum: float
    normalized_maximum: float
    invalid_count: int
    invalid_value_policy: DepthInvalidValuePolicy

    def document(self) -> dict[str, Any]:
        return {
            "normalization": self.normalization.value,
            "invert": self.invert,
            "gamma": self.gamma,
            "clamp": self.clamp,
            "source_minimum": self.source_minimum,
            "source_maximum": self.source_maximum,
            "normalization_low": self.normalization_low,
            "normalization_high": self.normalization_high,
            "normalized_minimum": self.normalized_minimum,
            "normalized_maximum": self.normalized_maximum,
            "invalid_count": self.invalid_count,
            "invalid_value_policy": self.invalid_value_policy.value,
        }


@dataclass(frozen=True, slots=True)
class NormalizedDepth:
    values: np.ndarray
    report: DepthPreprocessReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.dtype != np.dtype(np.float32):
            raise ValueError("normalized depth must be a 2D float32 array")
        if not np.isfinite(values).all():
            raise ValueError("normalized depth must contain only finite values")
        immutable = np.ascontiguousarray(values)
        immutable.setflags(write=False)
        object.__setattr__(self, "values", immutable)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
