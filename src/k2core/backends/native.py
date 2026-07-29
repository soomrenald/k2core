"""Developer-only native backend placeholder used during the gated migration."""

from __future__ import annotations

from dataclasses import dataclass

from k2core.backends import BackendCapabilities
from k2core.inference.errors import UnsupportedFeatureError


@dataclass(slots=True)
class NativeK2Backend:
    backend_id: str = "native"

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

    def load(self, config):
        del config
        return self._unsupported("model_loading")

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
        return None


__all__ = ["NativeK2Backend"]
