"""Native K2 lifecycle backend and clean Krea2 generation path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from k2core.backends import BackendCapabilities
from k2core.backends.native_loading import NativeModelLoader, NativePipelineState
from k2core.backends.native_lora import apply_native_loras
from k2core.backends.native_qwen import build_qwen_text_encoder
from k2core.backends.native_sampling import (
    DenoisingCheckpoint,
    euler_flow_sample,
    prepare_noise,
    simple_sigmas,
)
from k2core.backends.native_text import prompt_token_count
from k2core.backends.native_transformer import build_krea2_transformer
from k2core.backends.native_vae import build_krea2_vae
from k2core.inference.backend import (
    CancellationToken,
    DiagnosticCallback,
    NullCancellationToken,
    ProgressCallback,
)
from k2core.inference.errors import (
    ConfigurationError,
    ModelCompatibilityError,
    UnsupportedFeatureError,
    convert_error,
)
from k2core.inference.schemas import (
    GenerationRequest,
    GenerationResult,
    ImageEncodeRequest,
    ImageResult,
    InferenceRequest,
    LatentDecodeRequest,
    LatentResult,
    LoadedPipeline,
    PipelineConfig,
    ProgressEvent,
)
from k2core.output import validate_filename_prefix
from k2core.regional_lora import (
    character_identity_triggers,
    compile_lora_delta_routes,
)
from k2core.regional_prompting import (
    BoundRegionalPromptPlan,
    RegionalPromptPlan,
    compile_regional_prompt_plan,
)
from k2core.spatial_attention import KreaSpatialAttentionOverride


@dataclass(slots=True)
class NativeK2Backend:
    backend_id: str = "native"
    loader: NativeModelLoader | None = None
    pipeline: NativePipelineState | None = None
    config: PipelineConfig | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset(
                {
                    "text_to_image",
                    "ordinary_lora",
                    "regional_prompting",
                    "regional_lora",
                }
            ),
            accelerator_vendors=frozenset({"cuda", "rocm"}),
            parameters=(
                {"name": "sampler", "values": ("euler",)},
                {"name": "scheduler", "values": ("simple",)},
                {"name": "cfg", "values": (1.0,)},
            ),
            metadata={
                "native": True,
                "release_ready": False,
                "developer_only": True,
                "clean_generation": True,
            },
        )

    def _unsupported(self, phase: str, *, correlation_id: str = ""):
        raise UnsupportedFeatureError(
            "The requested feature is not implemented by the native K2 backend.",
            backend_name=self.backend_id,
            phase=phase,
            remediation="Set K2LAB_BACKEND=comfyui.",
            retry_safe=True,
            correlation_id=correlation_id,
        )

    def load(self, config: PipelineConfig) -> LoadedPipeline:
        if config.registered_model is None:
            raise ConfigurationError(
                "Native loading requires a validated registered model.",
                backend_name=self.backend_id,
                phase="model_loading",
                remediation="Select a model from a K2Lab model registry.",
            )
        if not config.artifacts.complete:
            raise ModelCompatibilityError(
                "Native loading requires all three model artifacts.",
                backend_name=self.backend_id,
                phase="model_loading",
            )
        artifacts = {
            "transformer": config.artifacts.transformer,
            "text_encoder": config.artifacts.text_encoder,
            "vae": config.artifacts.vae,
        }
        registered = {
            "transformer": config.registered_model.transformer,
            "text_encoder": config.registered_model.text_encoder,
            "vae": config.registered_model.vae,
        }
        for kind, artifact in artifacts.items():
            if artifact.path.resolve() != registered[kind].path.resolve():
                raise ConfigurationError(
                    f"native {kind} artifact does not match the registered model",
                    backend_name=self.backend_id,
                    phase="model_loading",
                )
        if self.pipeline is not None:
            self.unload()
        self.loader = self.loader or NativeModelLoader()
        self.pipeline = self.loader.load(
            config.registered_model,
            device_policy=config.device_policy,
            strict=config.strict_loading,
        )
        self.config = config
        return LoadedPipeline(
            backend_id=self.backend_id,
            metadata={
                "model_name": self.pipeline.model_name,
                "strict_loading": config.strict_loading,
                "components": tuple(report.to_payload() for report in self.pipeline.reports()),
            },
        )

    def generate(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
        diagnostic: DiagnosticCallback | None = None,
    ) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            return self._unsupported(
                "generation",
                correlation_id=request.correlation_id,
            )
        token = cancellation or NullCancellationToken()
        token.raise_if_cancelled()
        gpu_work_started = False
        try:
            self._validate_clean_request(request)
            pipeline, config = self._require_loaded()
            registered_model = config.registered_model
            if registered_model is None or registered_model.tokenizer is None:
                raise ConfigurationError(
                    "Native clean generation requires a registered tokenizer.",
                    backend_name=self.backend_id,
                    phase="text_encoding",
                    correlation_id=request.correlation_id,
                    remediation="Rescan and validate the Krea2 model registry.",
                )

            emit = _progress_emitter(request.correlation_id, progress)
            regional_plan = _compile_native_regional_plan(request)
            conditioned_prompt = (
                regional_plan.prompt
                if regional_plan is not None and (regional_plan.regions or regional_plan.emphases)
                else request.prompt
            )
            if diagnostic is not None:
                diagnostic(
                    "Native clean generation started",
                    {
                        "correlation_id": request.correlation_id,
                        "width": request.width,
                        "height": request.height,
                        "steps": request.steps,
                        "seed": request.seed,
                        "sampler": request.sampler,
                        "scheduler": request.scheduler,
                    },
                )
            token.raise_if_cancelled()
            emit("text_encoding", fraction=0.0)
            text_encoder = build_qwen_text_encoder(
                pipeline.text_encoder,
                registered_model.tokenizer,
            )
            try:
                gpu_work_started = True
                encoding = text_encoder.encode(
                    conditioned_prompt,
                    device=config.device_policy.text_encoder_device,
                )
                bound_regional_plan = (
                    regional_plan.bind_tokens(
                        lambda prefix: prompt_token_count(
                            prefix,
                            text_encoder.tokenizer,
                        ),
                        conditioning_text_token_count=len(encoding.tokens.conditioned_ids),
                    )
                    if regional_plan is not None
                    and (regional_plan.regions or regional_plan.emphases)
                    else None
                )
            finally:
                text_encoder.unload()
            if diagnostic is not None and bound_regional_plan is not None:
                diagnostic(
                    "Unified spatial prompt prepared",
                    bound_regional_plan.summary(),
                )
            lora_routes = (
                compile_lora_delta_routes(
                    [item.to_payload() for item in request.loras],
                    width=request.width,
                    height=request.height,
                    text_token_count=len(encoding.tokens.conditioned_ids),
                    regional_plan=regional_plan,
                    bound_plan=bound_regional_plan,
                )
                if request.loras
                else ()
            )
            token.raise_if_cancelled()
            emit(
                "text_encoding",
                fraction=1.0,
                detail={"token_count": len(encoding.tokens.conditioned_ids)},
            )

            torch = _import_torch()
            execution_device = _execution_device(
                torch,
                config.device_policy.transformer_device,
            )
            latent = prepare_noise(
                _clean_latent_shape(request.width, request.height),
                request.seed,
                device="cpu",
                dtype=torch.float32,
            ).to(execution_device)
            sigmas = simple_sigmas(request.steps)
            attention_override = (
                KreaSpatialAttentionOverride(bound_regional_plan)
                if bound_regional_plan is not None
                and (bound_regional_plan.spans or bound_regional_plan.emphases)
                else None
            )
            transformer = build_krea2_transformer(
                pipeline.transformer,
                spatial_attention=attention_override,
            )

            def predict(current, sigma):
                token.raise_if_cancelled()
                return transformer.predict_velocity(
                    current,
                    encoding.conditioning,
                    sigma,
                    attention_mask=encoding.attention_mask,
                    device=config.device_policy.transformer_device,
                )

            def checkpoint(item: DenoisingCheckpoint) -> None:
                token.raise_if_cancelled()
                completed = item.step + 1
                if attention_override is not None:
                    attention_override.set_denoising_progress(
                        completed,
                        request.steps,
                    )
                emit(
                    "diffusion",
                    step=completed,
                    total_steps=request.steps,
                    fraction=completed / request.steps,
                    detail={
                        "sigma": item.sigma,
                        "sigma_next": item.sigma_next,
                    },
                )

            try:
                lora_reports = apply_native_loras(
                    transformer,
                    request.loras,
                    routes=lora_routes,
                )
                latent = euler_flow_sample(
                    predict,
                    latent,
                    sigmas,
                    checkpoint=checkpoint,
                )
            finally:
                if attention_override is not None:
                    attention_override.clear()
                transformer.unload()
            encoding = None
            token.raise_if_cancelled()

            emit("vae_decode", fraction=0.0)
            vae = build_krea2_vae(pipeline.vae)
            try:
                images = vae.decode(
                    latent,
                    device=config.device_policy.vae_device,
                )
            finally:
                vae.unload()
            token.raise_if_cancelled()
            emit("vae_decode", fraction=1.0)

            regional_summary = (
                _native_regional_summary(
                    regional_plan,
                    bound_regional_plan,
                    attention_override,
                )
                if regional_plan is not None
                and bound_regional_plan is not None
                and attention_override is not None
                else {
                    "backend": "disabled",
                    "region_count": 0,
                }
            )
            payload = _save_native_image(
                request,
                images,
                lora_reports,
                regional_summary=regional_summary,
            )
            if diagnostic is not None:
                diagnostic(
                    "Native clean generation complete",
                    {
                        "correlation_id": request.correlation_id,
                        "image_path": payload["image_path"],
                        "width": payload["width"],
                        "height": payload["height"],
                    },
                )
            return GenerationResult(
                backend_id=self.backend_id,
                correlation_id=request.correlation_id,
                payload=payload,
            )
        except Exception as error:
            structured = convert_error(
                error,
                backend_name=self.backend_id,
                phase="generation",
                correlation_id=request.correlation_id,
                gpu_work_started=gpu_work_started,
            )
            if structured is error:
                raise
            raise structured from error

    def encode_image(self, request: ImageEncodeRequest) -> LatentResult:
        return self._unsupported(
            "image_encode",
            correlation_id=request.correlation_id,
        )

    def decode_latents(self, request: LatentDecodeRequest) -> ImageResult:
        pipeline, config = self._require_loaded()
        vae = build_krea2_vae(pipeline.vae)
        try:
            images = vae.decode(
                request.latents,
                device=config.device_policy.vae_device,
            )
        finally:
            vae.unload()
        return ImageResult(
            backend_id=self.backend_id,
            correlation_id=request.correlation_id,
            images=images,
            metadata={"normalized_latents": True},
        )

    def unload(self) -> None:
        if self.pipeline is not None:
            (self.loader or NativeModelLoader()).unload(self.pipeline)
            self.pipeline = None
        self.config = None

    def _require_loaded(self) -> tuple[NativePipelineState, PipelineConfig]:
        if self.pipeline is None or self.config is None or not self.pipeline.loaded:
            raise ConfigurationError(
                "The native K2 pipeline must be loaded before generation.",
                backend_name=self.backend_id,
                phase="generation",
                remediation="Load a validated registered model before retrying.",
            )
        return self.pipeline, self.config

    def _validate_clean_request(self, request: GenerationRequest) -> None:
        if not request.prompt.strip():
            raise ValueError("native generation requires a non-empty prompt")
        if request.sampler != "euler":
            raise UnsupportedFeatureError(
                "Native clean generation currently supports only the Euler sampler.",
                backend_name=self.backend_id,
                phase="generation",
                correlation_id=request.correlation_id,
                remediation="Choose Euler or use the ComfyUI backend.",
            )
        if request.scheduler != "simple":
            raise UnsupportedFeatureError(
                "Native clean generation currently supports only the simple scheduler.",
                backend_name=self.backend_id,
                phase="generation",
                correlation_id=request.correlation_id,
                remediation="Choose the simple scheduler or use the ComfyUI backend.",
            )
        unsupported: list[str] = []
        if request.negative_prompt.strip():
            unsupported.append("negative prompts")
        if request.denoise != 1.0:
            unsupported.append("partial denoise")
        if request.projector_enabled:
            unsupported.append("projector controls")
        if request.post_upscale:
            unsupported.append("post-upscale")
        if unsupported:
            raise UnsupportedFeatureError(
                "Native clean generation does not yet support: " + ", ".join(unsupported) + ".",
                backend_name=self.backend_id,
                phase="generation",
                correlation_id=request.correlation_id,
                remediation="Use the ComfyUI backend for this request.",
            )


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise ConfigurationError(
            "Native clean generation requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="generation",
        ) from error
    return torch


def _execution_device(torch, requested: str):
    normalized = requested.strip().casefold()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _clean_latent_shape(width: int, height: int) -> tuple[int, int, int, int, int]:
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("native clean dimensions must be positive multiples of 16")
    return (1, 16, 1, height // 8, width // 8)


def _compile_native_regional_plan(
    request: GenerationRequest,
) -> RegionalPromptPlan | None:
    if not request.regional_prompting or not (request.regions or request.prompt_emphases):
        return None
    return compile_regional_prompt_plan(
        request.width,
        request.height,
        request.prompt,
        request.regions,
        strength=request.regional_prompt_strength,
        outside_penalty=request.regional_outside_penalty,
        falloff_pixels=request.regional_feather_pixels,
        subject_competition=request.regional_subject_competition,
        subject_fill=request.regional_subject_fill,
        late_step_scale=request.regional_late_step_scale,
        emphases=request.prompt_emphases,
        character_identity_triggers=character_identity_triggers(
            [item.to_payload() for item in request.loras]
        ),
    )


def _native_regional_summary(
    plan: RegionalPromptPlan,
    bound_plan: BoundRegionalPromptPlan,
    attention_override: KreaSpatialAttentionOverride,
) -> dict[str, Any]:
    plan_summary = plan.summary()
    bound_summary = bound_plan.summary()
    plan_regions = plan_summary["regions"]
    bound_regions = bound_summary["regions"]
    if len(plan_regions) != len(bound_regions):
        raise RuntimeError("native regional metadata plans do not align")
    return {
        **plan_summary,
        **bound_summary,
        "regions": [
            {**compiled, **bound}
            for compiled, bound in zip(
                plan_regions,
                bound_regions,
                strict=True,
            )
        ],
        "attention_calls": attention_override.matched_calls,
        **attention_override.summary(),
    }


def _progress_emitter(
    correlation_id: str,
    progress: ProgressCallback | None,
):
    def emit(
        phase: str,
        *,
        step: int | None = None,
        total_steps: int | None = None,
        fraction: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if progress is not None:
            progress(
                ProgressEvent(
                    correlation_id=correlation_id,
                    phase=phase,
                    step=step,
                    total_steps=total_steps,
                    fraction=fraction,
                    detail=detail or {},
                )
            )

    return emit


def _save_native_image(
    request: GenerationRequest,
    images: Any,
    lora_reports: Sequence[Any] = (),
    *,
    regional_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from PIL import Image, PngImagePlugin
    except ImportError as error:
        raise ConfigurationError(
            "Native image output requires Pillow.",
            technical_detail=str(error),
            backend_name="native",
            phase="image_output",
        ) from error

    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise RuntimeError(f"native VAE returned an unexpected image shape: {tuple(images.shape)}")
    array = (
        (
            images[0]
            .detach()
            .permute(1, 2, 0)
            .to(device="cpu", dtype=_import_torch().float32)
            .clamp(0, 1)
            .numpy()
            * 255.0
        )
        .round()
        .astype("uint8")
    )
    image = Image.fromarray(array, mode="RGB")
    output_directory = request.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = validate_filename_prefix(request.filename_prefix)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_directory / f"{prefix}_{stamp}_seed-{request.seed}.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("k2lab_mode", "krea2_turbo_native")
    metadata.add_text("backend", "native")
    metadata.add_text("correlation_id", request.correlation_id)
    metadata.add_text("prompt", request.prompt)
    metadata.add_text("seed", str(request.seed))
    metadata.add_text("steps", str(request.steps))
    metadata.add_text("sampler", request.sampler)
    metadata.add_text("scheduler", request.scheduler)
    metadata.add_text("cfg", str(request.cfg))
    metadata.add_text(
        "loras",
        json.dumps([report.to_payload() for report in lora_reports]),
    )
    metadata.add_text(
        "regional_prompting",
        json.dumps(
            regional_summary or {"backend": "disabled", "region_count": 0},
            separators=(",", ":"),
        ),
    )
    metadata.add_text("size", f"{image.width}x{image.height}")
    if request.project_json:
        metadata.add_text(
            "k2lab_project",
            json.dumps(dict(request.project_json), separators=(",", ":")),
        )
    image.save(output_path, pnginfo=metadata)
    return {
        "image_path": str(output_path),
        "width": image.width,
        "height": image.height,
        "base_width": request.width,
        "base_height": request.height,
        "steps": request.steps,
        "seed": request.seed,
        "filename_prefix": prefix,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "cfg": request.cfg,
        "backend": "native",
        "correlation_id": request.correlation_id,
        "loras": [report.to_payload() for report in lora_reports],
        "regional_prompting": regional_summary or {"backend": "disabled", "region_count": 0},
        "projector": {"enabled": False, "backend": "disabled"},
        "post_upscale": {"enabled": False, "backend": "disabled", "scale": 1},
    }


__all__ = ["NativeK2Backend"]
