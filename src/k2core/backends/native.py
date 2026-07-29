"""Native K2 lifecycle backend."""

from __future__ import annotations

from dataclasses import dataclass

from k2core.backends import BackendCapabilities
from k2core.backends.native_loading import NativeModelLoader, NativePipelineState
from k2core.inference.errors import (
    ConfigurationError,
    ModelCompatibilityError,
    UnsupportedFeatureError,
)
from k2core.inference.schemas import LoadedPipeline, PipelineConfig


@dataclass(slots=True)
class NativeK2Backend:
    backend_id: str = "native"
    loader: NativeModelLoader | None = None
    pipeline: NativePipelineState | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset(),
            accelerator_vendors=frozenset(),
            metadata={"native": True, "release_ready": False},
        )

    def _unsupported(self, phase: str):
        raise UnsupportedFeatureError(
            "The native K2 backend is not implemented.",
            backend_name=self.backend_id,
            phase=phase,
            remediation="Set K2LAB_BACKEND=comfyui.",
            retry_safe=True,
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
        return LoadedPipeline(
            backend_id=self.backend_id,
            metadata={
                "model_name": self.pipeline.model_name,
                "strict_loading": config.strict_loading,
                "components": tuple(
                    report.to_payload() for report in self.pipeline.reports()
                ),
            },
        )

    def generate(self, request, **kwargs):
        del request, kwargs
        return self._unsupported("generation")

    def encode_image(self, request):
        del request
        return self._unsupported("image_encode")

    def decode_latents(self, request):
        del request
        return self._unsupported("latent_decode")

    def unload(self) -> None:
        if self.pipeline is not None:
            (self.loader or NativeModelLoader()).unload(self.pipeline)
            self.pipeline = None


__all__ = ["NativeK2Backend"]
