from __future__ import annotations

from typing import Any

import numpy as np

from k2core.depth.types import DepthImage, NormalizedDepth


def depth_histogram(
    values: DepthImage | NormalizedDepth | np.ndarray,
    *,
    bins: int = 64,
) -> dict[str, Any]:
    if bins <= 0 or bins > 4096:
        raise ValueError("depth histogram bins must be between 1 and 4096")
    source = values.values if isinstance(values, (DepthImage, NormalizedDepth)) else values
    array = np.asarray(source)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("depth histogram requires at least one finite value")
    counts, edges = np.histogram(
        finite,
        bins=bins,
        range=(float(finite.min()), float(finite.max())),
    )
    return {
        "bins": bins,
        "counts": [int(value) for value in counts],
        "edges": [float(value) for value in edges],
    }


def depth_summary(
    values: DepthImage | NormalizedDepth | np.ndarray,
) -> dict[str, Any]:
    source = values.values if isinstance(values, (DepthImage, NormalizedDepth)) else values
    array = np.asarray(source, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("depth summary requires at least one finite value")
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(np.asarray(source).dtype),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std()),
        "finite_count": int(finite.size),
        "invalid_count": int(array.size - finite.size),
    }
