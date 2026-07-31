from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from k2core.depth.config import DepthControlSettings
from k2core.depth.diagnostics import depth_histogram
from k2core.depth.loader import load_depth_image
from k2core.depth.masks import resize_depth
from k2core.depth.preprocess import normalize_depth
from k2core.depth.regional import (
    DepthRegion,
    EffectiveDepthField,
    compose_effective_depth_field,
    compose_override_depth,
)
from k2core.depth.types import NormalizedDepth
from k2core.regions import PixelBox, RegionDefinition


class DepthScheduleController:
    """Own exactly one immutable regional-strength field per denoising step."""

    def __init__(self, fields: tuple[EffectiveDepthField, ...]) -> None:
        if not fields:
            raise ValueError("depth schedule requires at least one field")
        shape = fields[0].image_token_values.shape
        if any(field.image_token_values.shape != shape for field in fields):
            raise ValueError("depth schedule fields must share one token layout")
        self._fields = fields
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def transition_count(self) -> int:
        return len(self._fields)

    def current_values(self) -> np.ndarray:
        return self._fields[self._step].image_token_values.reshape(-1)

    def advance_after(self, completed_step: int, total_steps: int) -> None:
        if total_steps != len(self._fields):
            raise RuntimeError("depth schedule and sampler step counts diverged")
        if completed_step != self._step:
            raise RuntimeError("depth schedule callback and sampler step diverged")
        self._step = min(completed_step + 1, len(self._fields) - 1)

    def reset(self) -> None:
        self._step = 0

    def document(self) -> dict[str, Any]:
        return {
            "transition_count": len(self._fields),
            "token_shape": list(self._fields[0].image_token_values.shape),
            "overlap_policy": self._fields[0].overlap_policy,
            "steps": [field.document() for field in self._fields],
        }


@dataclass(frozen=True, slots=True)
class DepthControlPreparation:
    checkpoint_path: Path
    normalized: NormalizedDepth
    resized_values: np.ndarray
    schedule: DepthScheduleController
    source_histogram: Mapping[str, Any]
    normalized_histogram: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    preprocess_seconds: float

    def document(self) -> dict[str, Any]:
        return {
            "source": self.normalized.metadata.get("source"),
            "preprocessing": self.normalized.report.document(),
            "source_histogram": dict(self.source_histogram),
            "normalized_histogram": dict(self.normalized_histogram),
            "schedule": self.schedule.document(),
            "checkpoint": dict(self.checkpoint),
            "preprocess_seconds": self.preprocess_seconds,
        }


def _depth_regions(
    settings: DepthControlSettings,
    regions: tuple[RegionDefinition, ...],
) -> tuple[DepthRegion, ...]:
    geometry = {region.region_id: region for region in regions}
    missing = sorted(
        item.region_id for item in settings.regions if item.region_id not in geometry
    )
    if missing:
        raise ValueError("depth settings reference missing regions: " + ", ".join(missing))
    return tuple(
        DepthRegion(
            settings=item,
            box=PixelBox(
                geometry[item.region_id].box.x0,
                geometry[item.region_id].box.y0,
                geometry[item.region_id].box.x1,
                geometry[item.region_id].box.y1,
            ),
            priority=geometry[item.region_id].priority,
        )
        for item in settings.regions
    )


def prepare_depth_control(
    settings: DepthControlSettings,
    *,
    regions: tuple[RegionDefinition, ...],
    width: int,
    height: int,
    steps: int,
    allow_override: bool = False,
    checkpoint_inspector=None,
) -> DepthControlPreparation:
    from k2core.depth.checkpoint import inspect_depth_checkpoint

    if not settings.enabled or settings.depth_image is None or settings.checkpoint is None:
        raise ValueError("depth preparation requires enabled depth settings")
    if steps <= 0:
        raise ValueError("depth preparation requires at least one denoising step")
    started = time.monotonic()
    inspect = checkpoint_inspector or inspect_depth_checkpoint
    compatibility = inspect(settings.checkpoint)
    if not compatibility.compatible or compatibility.checkpoint is None:
        raise ValueError("; ".join(compatibility.errors))
    source = load_depth_image(settings.depth_image)
    normalized = normalize_depth(
        source,
        settings.normalization,
        metadata={"source": source.info.document()},
    )
    resized = resize_depth(normalized.values, width, height)
    configured_regions = _depth_regions(settings, regions)
    override_depths: dict[str, np.ndarray] = {}
    for region in settings.regions:
        if region.override_image is None:
            continue
        override_source = load_depth_image(region.override_image)
        override_normalized = normalize_depth(override_source, settings.normalization)
        override_depths[region.region_id] = resize_depth(
            override_normalized.values,
            width,
            height,
        )
    if override_depths:
        if not allow_override:
            raise ValueError("override depth mode is disabled by the active feature flags")
        resized = compose_override_depth(
            resized,
            configured_regions,
            override_depths,
            feather_pixels=settings.feather_pixels,
        )
    fields = tuple(
        compose_effective_depth_field(
            settings,
            configured_regions,
            width=width,
            height=height,
            progress=step / max(steps - 1, 1),
            allow_override=allow_override,
        )
        for step in range(steps)
    )
    return DepthControlPreparation(
        checkpoint_path=settings.checkpoint,
        normalized=normalized,
        resized_values=resized,
        schedule=DepthScheduleController(fields),
        source_histogram=MappingProxyType(depth_histogram(source.values)),
        normalized_histogram=MappingProxyType(depth_histogram(resized)),
        checkpoint=MappingProxyType(compatibility.document()),
        preprocess_seconds=time.monotonic() - started,
    )
