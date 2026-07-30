from __future__ import annotations

from typing import Any

import numpy as np

from k2core.depth.config import DepthNormalizationSettings
from k2core.depth.types import (
    DepthImage,
    DepthInvalidValuePolicy,
    DepthNormalizationMode,
    DepthPreprocessReport,
    NormalizedDepth,
)


def _none_range(depth: DepthImage | np.ndarray, values: np.ndarray) -> tuple[float, float]:
    if isinstance(depth, DepthImage):
        return 0.0, float((1 << depth.info.bit_depth) - 1)
    if np.issubdtype(np.asarray(depth).dtype, np.integer):
        return 0.0, float(np.iinfo(np.asarray(depth).dtype).max)
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("unnormalized floating-point depth must already be in [0, 1]")
    return 0.0, 1.0


def normalize_depth(
    depth: DepthImage | np.ndarray,
    settings: DepthNormalizationSettings | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> NormalizedDepth:
    config = settings or DepthNormalizationSettings()
    source = depth.values if isinstance(depth, DepthImage) else np.asarray(depth)
    if source.ndim != 2:
        raise ValueError("depth normalization requires a 2D array")
    values = np.asarray(source, dtype=np.float64)
    finite = np.isfinite(values)
    invalid_count = int(values.size - finite.sum())
    if invalid_count and config.invalid_value_policy == DepthInvalidValuePolicy.ERROR:
        raise ValueError("depth image contains NaN or infinite values")
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("depth image contains no finite values")
    source_low = float(finite_values.min())
    source_high = float(finite_values.max())
    if source_high <= source_low:
        raise ValueError("depth image is fully constant")

    if config.mode == DepthNormalizationMode.NONE:
        low, high = _none_range(depth, finite_values)
    elif config.mode == DepthNormalizationMode.MINMAX:
        low, high = source_low, source_high
    elif config.mode == DepthNormalizationMode.PERCENTILE:
        low, high = (
            float(np.percentile(finite_values, config.near_percentile)),
            float(np.percentile(finite_values, config.far_percentile)),
        )
    elif config.mode == DepthNormalizationMode.CAMERA_RANGE:
        low, high = float(config.camera_near), float(config.camera_far)
    elif config.mode == DepthNormalizationMode.CHECKPOINT_REFERENCE:
        low, high = (
            float(config.checkpoint_minimum),
            float(config.checkpoint_maximum),
        )
    else:  # pragma: no cover - exhaustive enum guard.
        raise ValueError(f"unsupported depth normalization mode: {config.mode!r}")
    if not np.isfinite((low, high)).all() or high <= low:
        raise ValueError("depth normalization produced an invalid range")

    normalized = (values - low) / (high - low)
    if config.clamp:
        normalized = np.clip(normalized, 0.0, 1.0)
    elif np.any((normalized[finite] < 0.0) | (normalized[finite] > 1.0)):
        raise ValueError("depth values fall outside the selected unclamped range")
    if config.invert:
        normalized = 1.0 - normalized
    normalized = np.power(normalized, config.gamma)
    if invalid_count:
        replacement = (
            0.0
            if config.invalid_value_policy == DepthInvalidValuePolicy.FAR
            else 1.0
        )
        normalized[~finite] = replacement
    output = np.asarray(normalized, dtype=np.float32)
    report = DepthPreprocessReport(
        normalization=config.mode,
        invert=config.invert,
        gamma=config.gamma,
        clamp=config.clamp,
        source_minimum=source_low,
        source_maximum=source_high,
        normalization_low=low,
        normalization_high=high,
        normalized_minimum=float(output.min()),
        normalized_maximum=float(output.max()),
        invalid_count=invalid_count,
        invalid_value_policy=config.invalid_value_policy,
    )
    return NormalizedDepth(values=output, report=report, metadata=metadata or {})
