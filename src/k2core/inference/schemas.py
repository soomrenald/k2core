"""Immutable, backend-neutral inference request and result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from k2core.model import ArtifactSet, RegisteredModel
from k2core.projector import DEFAULT_PROJECTOR_PRESET
from k2core.regional_prompting import (
    PromptEmphasis,
    prompt_emphases_from_payload,
    region_definitions_from_payload,
)
from k2core.regions import RegionDefinition
from k2core.sampling import DEFAULT_SAMPLER, DEFAULT_SCHEDULER


class InferenceMode(StrEnum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_EDIT = "image_edit"
    FACE_REFINEMENT = "face_refinement"


class DTypePolicy(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT8_E4M3FN = "float8_e4m3fn"


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    transformer_device: str = "auto"
    text_encoder_device: str = "auto"
    vae_device: str = "auto"
    compute_dtype: DTypePolicy = DTypePolicy.AUTO
    weight_dtype: DTypePolicy = DTypePolicy.AUTO
    cpu_offload: bool = False
    vae_tiling: bool = False


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Lifecycle settings consumed by either inference backend."""

    artifacts: ArtifactSet
    registered_model: RegisteredModel | None = None
    memory_policy: str = "safe_16gb"
    reserve_vram_gb: float = 4.0
    minimum_system_ram_gb: float = 14.0
    cpu_vae: bool = False
    oom_recovery: bool = True
    strict_loading: bool = True
    device_policy: DevicePolicy = field(default_factory=DevicePolicy)

    def __post_init__(self) -> None:
        if self.reserve_vram_gb < 0:
            raise ValueError("reserve_vram_gb must not be negative")
        if self.minimum_system_ram_gb < 0:
            raise ValueError("minimum_system_ram_gb must not be negative")


@dataclass(frozen=True, slots=True)
class LoadedPipeline:
    backend_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class LoraSpec:
    lora_id: str
    name: str
    path: Path
    strength: float = 1.0
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()
    routing_mode: str = "standard"
    trigger_phrase: str = ""

    def __post_init__(self) -> None:
        if not self.lora_id.strip():
            raise ValueError("LoRA ID must not be empty")
        if not -4.0 <= self.strength <= 4.0:
            raise ValueError("LoRA strength must be between -4 and 4")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LoraSpec":
        path = Path(str(payload["path"])).expanduser()
        return cls(
            lora_id=str(payload.get("id", path.stem)),
            name=str(payload.get("name", path.stem)),
            path=path,
            strength=float(payload.get("strength", 1.0)),
            global_scope=bool(payload.get("global", True)),
            region_ids=tuple(map(str, payload.get("region_ids", ()))),
            routing_mode=str(payload.get("routing_mode", "standard")),
            trigger_phrase=str(payload.get("trigger_phrase", "")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.lora_id,
            "name": self.name,
            "path": str(self.path),
            "strength": self.strength,
            "global": self.global_scope,
            "region_ids": list(self.region_ids),
            "routing_mode": self.routing_mode,
            "trigger_phrase": self.trigger_phrase,
        }


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    correlation_id: str
    phase: str
    step: int | None = None
    total_steps: int | None = None
    fraction: float | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError("progress fraction must be between zero and one")
        object.__setattr__(self, "detail", _immutable_mapping(self.detail))


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    correlation_id: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    cfg: float = 1.0
    negative_prompt: str = ""
    denoise: float = 1.0
    output_directory: Path = Path(".")
    filename_prefix: str = "baseline"
    regions: tuple[RegionDefinition, ...] = ()
    prompt_emphases: tuple[PromptEmphasis, ...] = ()
    loras: tuple[LoraSpec, ...] = ()
    regional_prompting: bool = True
    regional_prompt_strength: float = 1.0
    regional_outside_penalty: float = 1.0
    regional_feather_pixels: float = 128.0
    regional_subject_competition: bool = True
    regional_subject_fill: bool = True
    regional_late_step_scale: float = 0.35
    regional_lora_delta_adaptation: bool = False
    regional_lora_delta_adaptation_gain: float = 0.35
    projector_enabled: bool = False
    projector_preset: str = DEFAULT_PROJECTOR_PRESET
    projector_values: tuple[float, ...] = ()
    projector_multiplier: float = 1.0
    projector_identity_protection: float = 1.0
    post_upscale: bool = False
    upscale_scale: int = 2
    upscale_method: str = "lanczos"
    upscale_model_path: Path | None = None
    project_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            raise ValueError("correlation_id must not be empty")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("generation dimensions must be positive multiples of 16")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2147483647")
        if self.cfg != 1.0:
            raise ValueError("the current Krea2 Turbo path requires CFG 1.0")
        if not 0.0 <= self.regional_lora_delta_adaptation_gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        if not 0.0 <= self.projector_identity_protection <= 1.0:
            raise ValueError("projector identity protection must be between zero and one")
        object.__setattr__(self, "project_json", _immutable_mapping(self.project_json))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, correlation_id: str
    ) -> "GenerationRequest":
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        return cls(
            correlation_id=correlation_id,
            prompt=str(payload.get("prompt", "")),
            width=width,
            height=height,
            steps=int(payload.get("steps", 8)),
            seed=int(payload.get("seed", 0)),
            sampler=str(payload.get("sampler", DEFAULT_SAMPLER)),
            scheduler=str(payload.get("scheduler", DEFAULT_SCHEDULER)),
            output_directory=Path(payload["output_directory"]),
            filename_prefix=str(payload.get("filename_prefix", "baseline")),
            regions=region_definitions_from_payload(
                list(payload.get("regions", [])),
                canvas_width=width,
                canvas_height=height,
            ),
            prompt_emphases=prompt_emphases_from_payload(list(payload.get("prompt_emphases", []))),
            loras=tuple(LoraSpec.from_payload(item) for item in payload.get("loras", [])),
            regional_prompting=bool(payload.get("regional_prompting", True)),
            regional_prompt_strength=float(payload.get("regional_prompt_strength", 1.0)),
            regional_outside_penalty=float(payload.get("regional_outside_penalty", 1.0)),
            regional_feather_pixels=float(payload.get("regional_feather_pixels", 128.0)),
            regional_subject_competition=bool(payload.get("regional_subject_competition", True)),
            regional_subject_fill=bool(payload.get("regional_subject_fill", True)),
            regional_late_step_scale=float(payload.get("regional_late_step_scale", 0.35)),
            regional_lora_delta_adaptation=bool(
                payload.get("regional_lora_delta_adaptation", False)
            ),
            regional_lora_delta_adaptation_gain=float(
                payload.get("regional_lora_delta_adaptation_gain", 0.35)
            ),
            projector_enabled=bool(payload.get("projector_enabled", False)),
            projector_preset=str(payload.get("projector_preset", DEFAULT_PROJECTOR_PRESET)),
            projector_values=tuple(map(float, payload.get("projector_values", ()))),
            projector_multiplier=float(payload.get("projector_multiplier", 1.0)),
            projector_identity_protection=float(payload.get("projector_identity_protection", 1.0)),
            post_upscale=bool(payload.get("post_upscale", False)),
            upscale_scale=int(payload.get("upscale_scale", 2)),
            upscale_method=str(payload.get("upscale_method", "lanczos")),
            upscale_model_path=(
                Path(payload["upscale_model_path"]) if payload.get("upscale_model_path") else None
            ),
            project_json=(
                payload["project_json"] if isinstance(payload.get("project_json"), Mapping) else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class ImageEditRequest:
    correlation_id: str
    image_path: Path
    prompt: str
    regions: tuple[RegionDefinition, ...]
    loras: tuple[LoraSpec, ...]
    output_directory: Path | None = None
    reference_prompt: str = ""
    reference_regions: tuple[RegionDefinition, ...] = ()
    prompt_emphases: tuple[PromptEmphasis, ...] = ()
    seed: int = 0
    steps: int = 8
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    denoise: float = 0.15
    latent_feather_pixels: int = 64
    composite_feather_pixels: int = 48
    edit_entire_image: bool = False
    preserve_identity: bool = True
    reference_description_retention: float = 1.0
    regional_prompt_strength: float = 1.0
    regional_outside_penalty: float = 1.0
    regional_feather_pixels: float = 128.0
    regional_subject_competition: bool = True
    regional_subject_fill: bool = True
    regional_late_step_scale: float = 0.35
    regional_lora_delta_adaptation: bool = False
    regional_lora_delta_adaptation_gain: float = 0.35
    projector_enabled: bool = False
    projector_preset: str = DEFAULT_PROJECTOR_PRESET
    projector_values: tuple[float, ...] = ()
    projector_multiplier: float = 1.0
    projector_identity_protection: float = 1.0
    project_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            raise ValueError("correlation_id must not be empty")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError("image-edit denoise must be in (0, 1]")
        object.__setattr__(self, "project_json", _immutable_mapping(self.project_json))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, correlation_id: str) -> "ImageEditRequest":
        return cls(
            correlation_id=correlation_id,
            image_path=Path(payload["image_path"]),
            output_directory=(
                Path(payload["output_directory"]) if payload.get("output_directory") else None
            ),
            prompt=str(payload.get("prompt", "")),
            regions=region_definitions_from_payload(list(payload.get("regions", []))),
            reference_prompt=str(payload.get("reference_prompt", "")),
            reference_regions=region_definitions_from_payload(
                list(payload.get("reference_regions", []))
            ),
            prompt_emphases=prompt_emphases_from_payload(list(payload.get("prompt_emphases", []))),
            loras=tuple(LoraSpec.from_payload(item) for item in payload.get("loras", [])),
            seed=int(payload.get("seed", 0)),
            steps=int(payload.get("steps", 8)),
            sampler=str(payload.get("sampler", DEFAULT_SAMPLER)),
            scheduler=str(payload.get("scheduler", DEFAULT_SCHEDULER)),
            denoise=float(payload.get("denoise", 0.15)),
            latent_feather_pixels=int(payload.get("latent_feather_pixels", 64)),
            composite_feather_pixels=int(payload.get("composite_feather_pixels", 48)),
            edit_entire_image=bool(payload.get("edit_entire_image", False)),
            preserve_identity=bool(payload.get("preserve_identity", True)),
            reference_description_retention=float(
                payload.get("reference_description_retention", 1.0)
            ),
            regional_prompt_strength=float(payload.get("regional_prompt_strength", 1.0)),
            regional_outside_penalty=float(payload.get("regional_outside_penalty", 1.0)),
            regional_feather_pixels=float(payload.get("regional_feather_pixels", 128.0)),
            regional_subject_competition=bool(payload.get("regional_subject_competition", True)),
            regional_subject_fill=bool(payload.get("regional_subject_fill", True)),
            regional_late_step_scale=float(payload.get("regional_late_step_scale", 0.35)),
            regional_lora_delta_adaptation=bool(
                payload.get("regional_lora_delta_adaptation", False)
            ),
            regional_lora_delta_adaptation_gain=float(
                payload.get("regional_lora_delta_adaptation_gain", 0.35)
            ),
            projector_enabled=bool(payload.get("projector_enabled", False)),
            projector_preset=str(payload.get("projector_preset", DEFAULT_PROJECTOR_PRESET)),
            projector_values=tuple(map(float, payload.get("projector_values", ()))),
            projector_multiplier=float(payload.get("projector_multiplier", 1.0)),
            projector_identity_protection=float(payload.get("projector_identity_protection", 1.0)),
            project_json=(
                payload["project_json"] if isinstance(payload.get("project_json"), Mapping) else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class FaceRefinementRequest:
    correlation_id: str
    image_path: Path
    regions: tuple[RegionDefinition, ...]
    loras: tuple[LoraSpec, ...]
    output_directory: Path | None = None
    seed: int = 0
    steps: int = 8
    denoise: float = 0.15
    crop_size: int = 512
    padding: float = 2.0
    feather: float = 0.12
    blend: float = 0.5
    lora_scale: float = 0.5
    detector_threshold: float = 0.15
    detector_provider: str = "auto"
    selected_face_indices: tuple[int, ...] | None = None
    manual_face_paths: tuple[tuple[tuple[float, float], ...], ...] = ()
    project_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            raise ValueError("correlation_id must not be empty")
        object.__setattr__(self, "project_json", _immutable_mapping(self.project_json))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, correlation_id: str
    ) -> "FaceRefinementRequest":
        return cls(
            correlation_id=correlation_id,
            image_path=Path(payload["image_path"]),
            output_directory=(
                Path(payload["output_directory"]) if payload.get("output_directory") else None
            ),
            regions=region_definitions_from_payload(list(payload.get("regions", []))),
            loras=tuple(LoraSpec.from_payload(item) for item in payload.get("loras", [])),
            seed=int(payload.get("seed", 0)),
            steps=int(payload.get("steps", 8)),
            denoise=float(payload.get("denoise", 0.15)),
            crop_size=int(payload.get("crop_size", 512)),
            padding=float(payload.get("padding", 2.0)),
            feather=float(payload.get("feather", 0.12)),
            blend=float(payload.get("blend", 0.5)),
            lora_scale=float(payload.get("lora_scale", 0.5)),
            detector_threshold=float(payload.get("detector_threshold", 0.15)),
            detector_provider=str(payload.get("detector_provider", "auto")),
            selected_face_indices=(
                tuple(map(int, payload["selected_face_indices"]))
                if payload.get("selected_face_indices") is not None
                else None
            ),
            manual_face_paths=tuple(
                tuple((float(point[0]), float(point[1])) for point in path)
                for path in payload.get("manual_face_paths", ())
            ),
            project_json=(
                payload["project_json"] if isinstance(payload.get("project_json"), Mapping) else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    backend_id: str
    correlation_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))

    def to_payload(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class ImageEncodeRequest:
    correlation_id: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class LatentDecodeRequest:
    correlation_id: str
    latents: Any


@dataclass(frozen=True, slots=True)
class LatentResult:
    backend_id: str
    correlation_id: str
    latents: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageResult:
    backend_id: str
    correlation_id: str
    images: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


InferenceRequest = GenerationRequest | ImageEditRequest | FaceRefinementRequest


__all__ = [
    "DTypePolicy",
    "DevicePolicy",
    "FaceRefinementRequest",
    "GenerationRequest",
    "GenerationResult",
    "ImageEditRequest",
    "ImageEncodeRequest",
    "ImageResult",
    "InferenceMode",
    "InferenceRequest",
    "LatentDecodeRequest",
    "LatentResult",
    "LoadedPipeline",
    "LoraSpec",
    "PipelineConfig",
    "ProgressEvent",
]
