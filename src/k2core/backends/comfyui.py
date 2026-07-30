"""Behavior-preserving lifecycle adapter over the existing ComfyUI runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from k2core.backends import BackendCapabilities
from k2core.inference.backend import (
    CancellationToken,
    DiagnosticCallback,
    NullCancellationToken,
    ProgressCallback,
)
from k2core.inference.errors import UnsupportedFeatureError, convert_error
from k2core.inference.schemas import (
    FaceRefinementRequest,
    GenerationRequest,
    GenerationResult,
    ImageEditRequest,
    ImageEncodeRequest,
    ImageResult,
    InferenceRequest,
    LatentDecodeRequest,
    LatentResult,
    LoadedPipeline,
    PipelineConfig,
    ProgressEvent,
)


@dataclass(slots=True)
class ComfyUIBackend:
    """Expose ``ComfyBaselineRuntime`` without changing its generation semantics."""

    runtime: Any
    release_callback: Any = None
    backend_id: str = "comfyui"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset(
                {
                    "text_to_image",
                    "image_edit",
                    "face_refinement",
                    "ordinary_lora",
                    "post_upscale",
                    "projector",
                    "regional_prompting",
                    "regional_lora",
                }
            ),
            accelerator_vendors=frozenset({"cuda", "rocm"}),
            metadata={
                "native": False,
                "runtime_loaded": bool(getattr(self.runtime, "loaded", False)),
            },
        )

    def load(self, config: PipelineConfig) -> LoadedPipeline:
        try:
            payload = self.runtime.load(
                config.artifacts,
                memory_policy_key=config.memory_policy,
                reserve_vram_gb=config.reserve_vram_gb,
                minimum_system_ram_gb=config.minimum_system_ram_gb,
                cpu_vae=config.cpu_vae,
                oom_recovery=config.oom_recovery,
            )
        except Exception as error:
            structured = convert_error(
                error,
                backend_name=self.backend_id,
                phase="model_loading",
            )
            if structured is error:
                raise
            raise structured from error
        return LoadedPipeline(backend_id=self.backend_id, metadata=payload)

    def generate(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
        diagnostic: DiagnosticCallback | None = None,
    ) -> GenerationResult:
        token = cancellation or NullCancellationToken()
        token.raise_if_cancelled()

        def runtime_progress(step: int, total: int, detail: dict[str, Any]) -> None:
            token.raise_if_cancelled()
            if progress is not None:
                progress(
                    ProgressEvent(
                        correlation_id=request.correlation_id,
                        phase="diffusion",
                        step=step,
                        total_steps=total,
                        fraction=step / total if total else None,
                        detail=detail,
                    )
                )

        try:
            if isinstance(request, GenerationRequest):
                payload = self.runtime.generate(
                    prompt=request.prompt,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    sampler=request.sampler,
                    scheduler=request.scheduler,
                    seed=request.seed,
                    output_directory=request.output_directory,
                    filename_prefix=request.filename_prefix,
                    regions=request.regions,
                    emphases=request.prompt_emphases,
                    regional_prompting=request.regional_prompting,
                    regional_prompt_strength=request.regional_prompt_strength,
                    regional_outside_penalty=request.regional_outside_penalty,
                    regional_feather_pixels=request.regional_feather_pixels,
                    regional_subject_competition=request.regional_subject_competition,
                    regional_subject_fill=request.regional_subject_fill,
                    regional_late_step_scale=request.regional_late_step_scale,
                    regional_lora_delta_adaptation=(request.regional_lora_delta_adaptation),
                    regional_lora_delta_adaptation_gain=(
                        request.regional_lora_delta_adaptation_gain
                    ),
                    projector_enabled=request.projector_enabled,
                    projector_preset=request.projector_preset,
                    projector_values=request.projector_values,
                    projector_multiplier=request.projector_multiplier,
                    projector_identity_protection=(request.projector_identity_protection),
                    post_upscale=request.post_upscale,
                    upscale_scale=request.upscale_scale,
                    upscale_method=request.upscale_method,
                    upscale_model_path=request.upscale_model_path,
                    loras=[item.to_payload() for item in request.loras],
                    project_json=dict(request.project_json) or None,
                    progress=runtime_progress,
                    event=diagnostic,
                )
            elif isinstance(request, ImageEditRequest):
                payload = self.runtime.edit_image(
                    image_path=request.image_path,
                    output_directory=request.output_directory,
                    prompt=request.prompt,
                    regions=request.regions,
                    reference_prompt=request.reference_prompt,
                    reference_regions=request.reference_regions,
                    prompt_emphases=request.prompt_emphases,
                    loras=[item.to_payload() for item in request.loras],
                    seed=request.seed,
                    steps=request.steps,
                    sampler=request.sampler,
                    scheduler=request.scheduler,
                    denoise=request.denoise,
                    latent_feather_pixels=request.latent_feather_pixels,
                    composite_feather_pixels=request.composite_feather_pixels,
                    edit_entire_image=request.edit_entire_image,
                    preserve_identity=request.preserve_identity,
                    reference_description_retention=(request.reference_description_retention),
                    regional_prompt_strength=request.regional_prompt_strength,
                    regional_outside_penalty=request.regional_outside_penalty,
                    regional_feather_pixels=request.regional_feather_pixels,
                    regional_subject_competition=request.regional_subject_competition,
                    regional_subject_fill=request.regional_subject_fill,
                    regional_late_step_scale=request.regional_late_step_scale,
                    regional_lora_delta_adaptation=(request.regional_lora_delta_adaptation),
                    regional_lora_delta_adaptation_gain=(
                        request.regional_lora_delta_adaptation_gain
                    ),
                    projector_enabled=request.projector_enabled,
                    projector_preset=request.projector_preset,
                    projector_values=request.projector_values,
                    projector_multiplier=request.projector_multiplier,
                    projector_identity_protection=(request.projector_identity_protection),
                    project_json=dict(request.project_json) or None,
                    progress=runtime_progress,
                    event=diagnostic,
                )
            elif isinstance(request, FaceRefinementRequest):
                payload = self.runtime.refine_faces(
                    image_path=request.image_path,
                    output_directory=request.output_directory,
                    regions=request.regions,
                    loras=[item.to_payload() for item in request.loras],
                    seed=request.seed,
                    steps=request.steps,
                    denoise=request.denoise,
                    crop_size=request.crop_size,
                    padding=request.padding,
                    feather=request.feather,
                    blend=request.blend,
                    lora_scale=request.lora_scale,
                    detector_threshold=request.detector_threshold,
                    detector_provider=request.detector_provider,
                    selected_face_indices=request.selected_face_indices,
                    manual_face_paths=request.manual_face_paths,
                    project_json=dict(request.project_json) or None,
                    event=diagnostic,
                )
            else:
                raise TypeError(f"unsupported inference request: {type(request).__name__}")
            token.raise_if_cancelled()
        except Exception as error:
            structured = convert_error(
                error,
                backend_name=self.backend_id,
                phase="generation",
                correlation_id=request.correlation_id,
                gpu_work_started=True,
            )
            if structured is error:
                raise
            raise structured from error
        return GenerationResult(
            backend_id=self.backend_id,
            correlation_id=request.correlation_id,
            payload=payload,
        )

    def encode_image(self, request: ImageEncodeRequest) -> LatentResult:
        raise UnsupportedFeatureError(
            "Standalone image encoding is not exposed by the reference backend yet.",
            backend_name=self.backend_id,
            phase="image_encode",
            correlation_id=request.correlation_id,
        )

    def decode_latents(self, request: LatentDecodeRequest) -> ImageResult:
        raise UnsupportedFeatureError(
            "Standalone latent decoding is not exposed by the reference backend yet.",
            backend_name=self.backend_id,
            phase="latent_decode",
            correlation_id=request.correlation_id,
        )

    def unload(self) -> None:
        if self.release_callback is not None:
            self.release_callback()


__all__ = ["ComfyUIBackend"]
