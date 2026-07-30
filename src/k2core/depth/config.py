from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from k2core.depth.types import (
    DepthInvalidValuePolicy,
    DepthNormalizationMode,
    DepthRegionMode,
)


def _environment_flag(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class DepthFeatureFlags:
    control: bool = False
    regions: bool = False
    override: bool = False
    blender_bundle_import: bool = False

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "DepthFeatureFlags":
        values = os.environ if environment is None else environment
        return cls(
            control=_environment_flag(values, "K2_DEPTH_CONTROL_ENABLED"),
            regions=_environment_flag(values, "K2_DEPTH_REGIONS_ENABLED"),
            override=_environment_flag(values, "K2_DEPTH_OVERRIDE_ENABLED"),
            blender_bundle_import=_environment_flag(values, "K2_BLENDER_BUNDLE_IMPORT_ENABLED"),
        )

    def document(self) -> dict[str, bool]:
        return {
            "control": self.control,
            "regions": self.regions,
            "override": self.override,
            "blender_bundle_import": self.blender_bundle_import,
        }


@dataclass(frozen=True, slots=True)
class DepthNormalizationSettings:
    mode: DepthNormalizationMode = DepthNormalizationMode.PERCENTILE
    near_percentile: float = 1.0
    far_percentile: float = 99.0
    gamma: float = 1.0
    clamp: bool = True
    invert: bool = False
    invalid_value_policy: DepthInvalidValuePolicy = DepthInvalidValuePolicy.FAR
    camera_near: float | None = None
    camera_far: float | None = None
    checkpoint_minimum: float | None = None
    checkpoint_maximum: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", DepthNormalizationMode(self.mode))
        object.__setattr__(
            self,
            "invalid_value_policy",
            DepthInvalidValuePolicy(self.invalid_value_policy),
        )
        if not 0.0 <= self.near_percentile < self.far_percentile <= 100.0:
            raise ValueError("depth percentiles must satisfy 0 <= near < far <= 100")
        if not 0.01 <= self.gamma <= 10.0:
            raise ValueError("depth gamma must be between 0.01 and 10")
        if self.mode == DepthNormalizationMode.CAMERA_RANGE:
            if (
                self.camera_near is None
                or self.camera_far is None
                or self.camera_far <= self.camera_near
            ):
                raise ValueError("camera-range normalization requires near < far")
        if self.mode == DepthNormalizationMode.CHECKPOINT_REFERENCE:
            if (
                self.checkpoint_minimum is None
                or self.checkpoint_maximum is None
                or self.checkpoint_maximum <= self.checkpoint_minimum
            ):
                raise ValueError("checkpoint-reference normalization requires minimum < maximum")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        invert: bool | None = None,
    ) -> "DepthNormalizationSettings":
        values = payload if isinstance(payload, Mapping) else {}
        return cls(
            mode=DepthNormalizationMode(values.get("mode", "percentile")),
            near_percentile=float(
                values.get("near_percentile", values.get("percentile_near", 1.0))
            ),
            far_percentile=float(values.get("far_percentile", values.get("percentile_far", 99.0))),
            gamma=float(values.get("gamma", 1.0)),
            clamp=bool(values.get("clamp", True)),
            invert=bool(values.get("invert", False) if invert is None else invert),
            invalid_value_policy=DepthInvalidValuePolicy(values.get("invalid_value_policy", "far")),
            camera_near=(
                float(values["camera_near"]) if values.get("camera_near") is not None else None
            ),
            camera_far=(
                float(values["camera_far"]) if values.get("camera_far") is not None else None
            ),
            checkpoint_minimum=(
                float(values["checkpoint_minimum"])
                if values.get("checkpoint_minimum") is not None
                else None
            ),
            checkpoint_maximum=(
                float(values["checkpoint_maximum"])
                if values.get("checkpoint_maximum") is not None
                else None
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "near_percentile": self.near_percentile,
            "far_percentile": self.far_percentile,
            "gamma": self.gamma,
            "clamp": self.clamp,
            "invert": self.invert,
            "invalid_value_policy": self.invalid_value_policy.value,
            "camera_near": self.camera_near,
            "camera_far": self.camera_far,
            "checkpoint_minimum": self.checkpoint_minimum,
            "checkpoint_maximum": self.checkpoint_maximum,
        }


@dataclass(frozen=True, slots=True)
class DepthRegionSettings:
    region_id: str
    mode: DepthRegionMode = DepthRegionMode.INHERIT
    strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    override_image: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", DepthRegionMode(self.mode))
        if not self.region_id.strip():
            raise ValueError("depth region ID must not be empty")
        if not 0.0 <= self.strength <= 3.0:
            raise ValueError("regional depth strength must be between 0 and 3")
        if self.mode == DepthRegionMode.EMPHASIZE and self.strength < 1.0:
            raise ValueError("emphasized depth strength must be at least 1")
        if self.mode == DepthRegionMode.RELAX and self.strength > 1.0:
            raise ValueError("relaxed depth strength must not exceed 1")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError("regional depth schedule must satisfy 0 <= start <= end <= 1")
        if self.mode == DepthRegionMode.OVERRIDE and self.override_image is None:
            raise ValueError("override depth mode requires an override image")
        if self.mode != DepthRegionMode.OVERRIDE and self.override_image is not None:
            raise ValueError("only override depth mode may supply an override image")

    @property
    def multiplier(self) -> float:
        if self.mode == DepthRegionMode.IGNORE:
            return 0.0
        if self.mode == DepthRegionMode.INHERIT:
            return 1.0
        return self.strength


@dataclass(frozen=True, slots=True)
class DepthControlSettings:
    enabled: bool = False
    checkpoint: Path | None = None
    depth_image: Path | None = None
    global_strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    normalization: DepthNormalizationSettings = field(default_factory=DepthNormalizationSettings)
    feather_pixels: float = 32.0
    minimum_effective_strength: float = 0.0
    maximum_effective_strength: float = 3.0
    regions: tuple[DepthRegionSettings, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_effective_strength < self.maximum_effective_strength:
            raise ValueError("effective depth bounds must satisfy 0 <= minimum < maximum")
        if (
            not self.minimum_effective_strength
            <= self.global_strength
            <= (self.maximum_effective_strength)
        ):
            raise ValueError("global depth strength is outside the configured safe bounds")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError("global depth schedule must satisfy 0 <= start <= end <= 1")
        if not 0.0 <= self.feather_pixels <= 2048.0:
            raise ValueError("depth feathering must be between 0 and 2048 pixels")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("depth region settings must have unique region IDs")
        if self.enabled and self.checkpoint is None:
            raise ValueError("enabled depth control requires a checkpoint")
        if self.enabled and self.depth_image is None:
            raise ValueError("enabled depth control requires a depth image")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "DepthControlSettings":
        values = payload if isinstance(payload, Mapping) else {}
        normalization = DepthNormalizationSettings.from_payload(
            values.get("normalization"),
            invert=bool(values["invert"]) if "invert" in values else None,
        )
        return cls(
            enabled=bool(values.get("enabled", False)),
            checkpoint=(
                Path(str(values["checkpoint"])).expanduser() if values.get("checkpoint") else None
            ),
            depth_image=(
                Path(str(values["depth_image"])).expanduser() if values.get("depth_image") else None
            ),
            global_strength=float(values.get("global_strength", 1.0)),
            start_percent=float(values.get("start_percent", 0.0)),
            end_percent=float(values.get("end_percent", 1.0)),
            normalization=normalization,
            feather_pixels=float(values.get("feather_pixels", 32.0)),
            minimum_effective_strength=float(values.get("minimum_effective_strength", 0.0)),
            maximum_effective_strength=float(values.get("maximum_effective_strength", 3.0)),
            regions=tuple(
                DepthRegionSettings(
                    region_id=str(region["region_id"]),
                    mode=DepthRegionMode(region.get("mode", "inherit")),
                    strength=float(region.get("strength", 1.0)),
                    start_percent=float(region.get("start_percent", 0.0)),
                    end_percent=float(region.get("end_percent", 1.0)),
                    override_image=(
                        Path(str(region["override_image"])).expanduser()
                        if region.get("override_image")
                        else None
                    ),
                )
                for region in values.get("regions", ())
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "depth_image": str(self.depth_image) if self.depth_image else None,
            "global_strength": self.global_strength,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "invert": self.normalization.invert,
            "normalization": self.normalization.to_payload(),
            "feather_pixels": self.feather_pixels,
            "minimum_effective_strength": self.minimum_effective_strength,
            "maximum_effective_strength": self.maximum_effective_strength,
            "regions": [
                {
                    "region_id": region.region_id,
                    "mode": region.mode.value,
                    "strength": region.strength,
                    "start_percent": region.start_percent,
                    "end_percent": region.end_percent,
                    "override_image": (
                        str(region.override_image) if region.override_image else None
                    ),
                }
                for region in self.regions
            ],
        }
