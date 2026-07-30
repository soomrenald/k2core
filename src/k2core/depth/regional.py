from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from k2core.depth.config import DepthControlSettings, DepthRegionSettings
from k2core.depth.masks import block_average_mask, feathered_box_mask
from k2core.depth.types import DepthRegionMode
from k2core.regions import PixelBox


@dataclass(frozen=True, slots=True)
class DepthRegion:
    settings: DepthRegionSettings
    box: PixelBox
    priority: int = 0


@dataclass(frozen=True, slots=True)
class EffectiveDepthField:
    pixel_values: np.ndarray
    image_token_values: np.ndarray
    region_multipliers: Mapping[str, float]
    overlap_policy: str = "priority-weighted-source-over-v1"

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixel_values, dtype=np.float32)
        tokens = np.asarray(self.image_token_values, dtype=np.float32)
        if pixels.ndim != 2 or tokens.ndim != 2:
            raise ValueError("effective depth fields must be two-dimensional")
        if not np.isfinite(pixels).all() or not np.isfinite(tokens).all():
            raise ValueError("effective depth fields must contain finite values")
        pixels = np.ascontiguousarray(pixels)
        tokens = np.ascontiguousarray(tokens)
        pixels.setflags(write=False)
        tokens.setflags(write=False)
        object.__setattr__(self, "pixel_values", pixels)
        object.__setattr__(self, "image_token_values", tokens)
        object.__setattr__(
            self,
            "region_multipliers",
            MappingProxyType(dict(self.region_multipliers)),
        )

    def document(self) -> dict[str, Any]:
        return {
            "overlap_policy": self.overlap_policy,
            "pixel_shape": list(self.pixel_values.shape),
            "image_token_shape": list(self.image_token_values.shape),
            "minimum": float(self.pixel_values.min()),
            "maximum": float(self.pixel_values.max()),
            "mean": float(self.pixel_values.mean()),
            "region_multipliers": dict(self.region_multipliers),
        }


def _active_multiplier(region: DepthRegionSettings, progress: float) -> float:
    if not region.start_percent <= progress <= region.end_percent:
        return 1.0
    return region.multiplier


def compose_effective_depth_field(
    settings: DepthControlSettings,
    regions: tuple[DepthRegion, ...],
    *,
    width: int,
    height: int,
    progress: float = 0.0,
    image_token_size: int = 16,
    allow_override: bool = False,
) -> EffectiveDepthField:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("depth progress must be between zero and one")
    if width <= 0 or height <= 0 or width % image_token_size or height % image_token_size:
        raise ValueError("depth canvas must be divisible by the image-token size")
    configured = {region.region_id: region for region in settings.regions}
    supplied_ids = [region.settings.region_id for region in regions]
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("depth regions must have unique IDs")
    if set(supplied_ids) != set(configured):
        missing = sorted(set(configured) - set(supplied_ids))
        unknown = sorted(set(supplied_ids) - set(configured))
        raise ValueError(f"depth region geometry mismatch; missing={missing}, unknown={unknown}")
    global_active = settings.start_percent <= progress <= settings.end_percent
    base = settings.global_strength if global_active else 0.0
    multiplier_field = np.ones((height, width), dtype=np.float32)
    multipliers: dict[str, float] = {}

    # Lower-priority and later equal-priority regions are applied first. A
    # higher-priority or earlier equal-priority region therefore wins through
    # the same source-over blend used for every overlap.
    indexed = list(enumerate(regions))
    ordered = sorted(indexed, key=lambda item: (item[1].priority, -item[0]))
    for _index, region in ordered:
        config = configured[region.settings.region_id]
        if config != region.settings:
            raise ValueError(f"depth settings disagree for region {region.settings.region_id!r}")
        if config.mode == DepthRegionMode.OVERRIDE and not allow_override:
            raise ValueError("override depth mode is disabled by the active feature flags")
        multiplier = _active_multiplier(config, progress)
        multipliers[config.region_id] = multiplier
        alpha = feathered_box_mask(
            width,
            height,
            region.box,
            settings.feather_pixels,
        )
        multiplier_field = multiplier_field * (1.0 - alpha) + multiplier * alpha

    effective = np.clip(
        base * multiplier_field,
        settings.minimum_effective_strength,
        settings.maximum_effective_strength,
    ).astype(np.float32)
    tokens = block_average_mask(effective, image_token_size)
    return EffectiveDepthField(
        pixel_values=effective,
        image_token_values=tokens,
        region_multipliers=multipliers,
    )
