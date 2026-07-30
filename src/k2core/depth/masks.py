from __future__ import annotations

import numpy as np
from PIL import Image

from k2core.regions import PixelBox


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("depth dimensions must be positive")


def resize_depth(values: np.ndarray, width: int, height: int) -> np.ndarray:
    _validate_dimensions(width, height)
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("depth resize requires a finite 2D array")
    if source.shape == (height, width):
        return source.copy()
    image = Image.fromarray(source, mode="F")
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def resize_mask(values: np.ndarray, width: int, height: int) -> np.ndarray:
    _validate_dimensions(width, height)
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("mask resize requires a finite 2D array")
    if np.any((source < 0.0) | (source > 1.0)):
        raise ValueError("mask values must be in [0, 1]")
    if source.shape == (height, width):
        return source.copy()
    image = Image.fromarray(source, mode="F")
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def feathered_box_mask(
    width: int,
    height: int,
    box: PixelBox,
    feather_pixels: float,
) -> np.ndarray:
    _validate_dimensions(width, height)
    if not 0.0 <= feather_pixels <= 2048.0:
        raise ValueError("depth feathering must be between 0 and 2048 pixels")
    clipped = box.clipped(width, height)
    x = np.arange(width, dtype=np.float64) + 0.5
    y = np.arange(height, dtype=np.float64) + 0.5
    dx = np.maximum.reduce(
        (
            clipped.x0 - x,
            np.zeros_like(x),
            x - clipped.x1,
        )
    )
    dy = np.maximum.reduce(
        (
            clipped.y0 - y,
            np.zeros_like(y),
            y - clipped.y1,
        )
    )
    distance = np.hypot(dy[:, None], dx[None, :])
    if feather_pixels == 0.0:
        return (distance == 0.0).astype(np.float32)
    u = np.clip(1.0 - distance / feather_pixels, 0.0, 1.0)
    smooth = u * u * (3.0 - 2.0 * u)
    return smooth.astype(np.float32)


def block_average_mask(values: np.ndarray, factor: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("block averaging requires a 2D mask")
    if factor <= 0:
        raise ValueError("block averaging factor must be positive")
    height, width = source.shape
    if height % factor or width % factor:
        raise ValueError("mask dimensions must be divisible by the block factor")
    return source.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3), dtype=np.float32)
