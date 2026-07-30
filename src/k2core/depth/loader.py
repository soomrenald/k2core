from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from k2core.depth.types import DepthImage, DepthImageInfo


_SUPPORTED_FORMATS = {"PNG", "TIFF"}
_GRAYSCALE_MODES = {"L", "I", "I;16", "I;16L", "I;16B"}


def _tiff_bits(image: Image.Image) -> int | None:
    tag = getattr(image, "tag_v2", None)
    if tag is None:
        return None
    value = tag.get(258)
    if isinstance(value, tuple):
        return int(value[0]) if len(value) == 1 else None
    return int(value) if value is not None else None


def load_depth_image(path: Path) -> DepthImage:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"depth image is not a readable file: {resolved}")
    try:
        with Image.open(resolved) as image:
            image_format = str(image.format or "").upper()
            mode = image.mode
            if image_format not in _SUPPORTED_FORMATS:
                raise ValueError("depth image must be PNG or TIFF")
            if mode not in _GRAYSCALE_MODES:
                raise ValueError(
                    f"depth image must be single-channel grayscale; received mode {mode!r}"
                )
            values = np.asarray(image)
            if values.ndim != 2:
                raise ValueError("depth image must contain exactly one channel")
            if mode == "L":
                bit_depth = 8
                pixels = np.asarray(values, dtype=np.uint8)
            else:
                minimum = int(values.min(initial=0))
                maximum = int(values.max(initial=0))
                if minimum < 0 or maximum > 65535:
                    raise ValueError("depth image integer values must fit unsigned 16-bit")
                bit_depth = _tiff_bits(image) or 16
                if bit_depth != 16:
                    raise ValueError(
                        f"TIFF depth image must be 16-bit grayscale; received {bit_depth}-bit"
                    )
                pixels = np.asarray(values, dtype=np.uint16)
            minimum = float(pixels.min())
            maximum = float(pixels.max())
            info = DepthImageInfo(
                path=resolved,
                format=image_format,
                width=int(image.width),
                height=int(image.height),
                dtype=str(pixels.dtype),
                bit_depth=bit_depth,
                minimum=minimum,
                maximum=maximum,
                mode=mode,
            )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"depth image could not be decoded safely: {type(error).__name__}"
        ) from error
    return DepthImage(values=pixels, info=info)


def depth_preview(depth: DepthImage) -> Image.Image:
    values = depth.values.astype(np.float64)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        preview = np.zeros(values.shape, dtype=np.uint8)
    else:
        preview = np.rint(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0)
        preview = preview.astype(np.uint8)
    return Image.fromarray(preview, mode="L")
