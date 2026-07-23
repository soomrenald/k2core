"""Protocol adapter over the shared in-process Comfy/Krea runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from k2core.backends import (
    BackendCapabilities,
    BackendResult,
    CancellationToken,
    ProgressCallback,
)
from k2core.regions import PixelBox, RegionDefinition


AssetResolver = Callable[[str], Path]


@dataclass(slots=True)
class ComfyKreaBackend:
    """Expose ``ComfyBaselineRuntime`` through the public K2 backend contracts."""

    runtime: Any
    asset_resolver: AssetResolver
    output_directory: Path
    release_callback: Callable[[], None] | None = None
    backend_id: str = "krea-comfyui"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset(
                {
                    "generate_image",
                    "edit_image",
                    "regional_image",
                    "regional_edit",
                    "face_refinement",
                }
            ),
            accelerator_vendors=frozenset({"cuda", "rocm"}),
            metadata={
                "prompt_backend": "krea-unified-spatial-attention-v6",
                "adapter_backend": "krea-regional-lora-delta-gating-v3",
                "runtime_loaded": bool(getattr(self.runtime, "loaded", False)),
            },
        )

    def validate_image_request(self, request: Mapping[str, Any]) -> tuple[str, ...]:
        errors = []
        operation = str(request.get("operation", ""))
        if operation not in {"generate_image", "edit_image"}:
            errors.append("operation must be generate_image or edit_image")
        if not str(request.get("prompt", "")).strip():
            errors.append("image request requires a prompt")
        if operation == "generate_image":
            for key in ("width", "height"):
                value = request.get(key)
                if not isinstance(value, int) or value <= 0 or value % 16:
                    errors.append(f"{key} must be a positive multiple of 16")
        if operation == "edit_image" and not request.get("source_asset_id"):
            errors.append("image edit requires source_asset_id")
        return tuple(errors)

    def validate_edit_request(self, request: Mapping[str, Any]) -> tuple[str, ...]:
        errors = []
        if not request.get("source_asset_id"):
            errors.append("frame edit requires source_asset_id")
        operation = str(request.get("operation", ""))
        if operation not in {
            "image_edit",
            "regional_edit",
            "face_refinement",
            "mannequin_guided",
        }:
            errors.append("unsupported frame edit operation")
        if operation == "face_refinement" and not request.get("user_confirmed_face_region"):
            errors.append("face refinement requires a user-confirmed region")
        return tuple(errors)

    def generate_image(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult:
        errors = self.validate_image_request(request)
        if errors:
            raise ValueError("; ".join(errors))
        cancellation.raise_if_cancelled()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if request["operation"] == "edit_image":
            result = self._edit(request, progress=progress, cancellation=cancellation)
        else:
            result = self.runtime.generate(
                prompt=str(request["prompt"]),
                width=int(request["width"]),
                height=int(request["height"]),
                steps=int(request.get("steps", 8)),
                sampler=str(request.get("sampler", "euler")),
                scheduler=str(request.get("scheduler", "simple")),
                seed=int(request.get("seed", 0)),
                output_directory=self.output_directory,
                filename_prefix=str(request.get("filename_prefix", "wan2lab-keyframe")),
                regions=_regions(request.get("regions", ())),
                regional_prompting=bool(request.get("regions")),
                regional_prompt_strength=float(request.get("regional_prompt_strength", 1.0)),
                regional_outside_penalty=float(request.get("regional_outside_penalty", 1.0)),
                regional_feather_pixels=float(request.get("regional_feather_pixels", 128.0)),
                loras=self._loras(request.get("adapter_routes", ())),
                progress=_progress_adapter(progress, cancellation),
                event=lambda message, payload: progress("runtime", None, {"message": message, **payload}),
            )
        return _result(result)

    def edit_frame(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult:
        errors = self.validate_edit_request(request)
        if errors:
            raise ValueError("; ".join(errors))
        result = self._edit(request, progress=progress, cancellation=cancellation)
        return _result(result)

    def refine_faces(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> BackendResult:
        errors = self.validate_edit_request(request)
        if errors:
            raise ValueError("; ".join(errors))
        if str(request.get("operation")) != "face_refinement":
            raise ValueError("refine_faces requires face_refinement operation")
        cancellation.raise_if_cancelled()
        source = self.asset_resolver(str(request["source_asset_id"]))
        region = request.get("region")
        manual_paths = ()
        regions = ()
        if isinstance(region, Mapping):
            x0, y0, x1, y1 = (float(region[key]) for key in ("x0", "y0", "x1", "y1"))
            manual_paths = (((x0, y0), (x1, y0), (x1, y1), (x0, y1)),)
            regions = (
                RegionDefinition(
                    region_id="confirmed-face",
                    name="Confirmed face",
                    box=PixelBox(x0, y0, x1, y1),
                    prompt=str(request.get("prompt", "")),
                    face_identity_prompt=str(request.get("prompt", "")),
                    spatial_role="subject",
                ),
            )
        settings = _settings(request)
        result = self.runtime.refine_faces(
            image_path=source,
            output_directory=self.output_directory,
            regions=regions,
            loras=self._loras(request.get("adapters", ())),
            seed=int(settings.get("seed", 0)),
            steps=int(settings.get("steps", 8)),
            denoise=float(settings.get("denoise", 0.15)),
            manual_face_paths=manual_paths,
            event=lambda message, payload: progress("face_refinement", None, {"message": message, **payload}),
        )
        cancellation.raise_if_cancelled()
        return _result(result)

    def release(self) -> None:
        if self.release_callback is not None:
            self.release_callback()

    def _edit(
        self,
        request: Mapping[str, Any],
        *,
        progress: ProgressCallback,
        cancellation: CancellationToken,
    ) -> Mapping[str, Any]:
        cancellation.raise_if_cancelled()
        settings = _settings(request)
        region_payload = request.get("regions", ())
        if not region_payload and isinstance(request.get("region"), Mapping):
            region = request["region"]
            region_payload = (
                {
                    "region_id": "edit-region",
                    "name": "Edit region",
                    "box": region,
                    "prompt": request.get("prompt", ""),
                },
            )
        result = self.runtime.edit_image(
            image_path=self.asset_resolver(str(request["source_asset_id"])),
            output_directory=self.output_directory,
            prompt=str(request.get("prompt", "")),
            regions=_regions(region_payload),
            loras=self._loras(request.get("adapter_routes", request.get("adapters", ()))),
            seed=int(request.get("seed", settings.get("seed", 0))),
            steps=int(request.get("steps", settings.get("steps", 8))),
            denoise=float(request.get("edit_strength", settings.get("denoise", 0.15))),
            edit_entire_image=not bool(region_payload),
            preserve_identity=bool(settings.get("preserve_identity", True)),
            progress=_progress_adapter(progress, cancellation),
            event=lambda message, payload: progress("runtime", None, {"message": message, **payload}),
        )
        cancellation.raise_if_cancelled()
        return result

    def _loras(self, items: object) -> list[dict[str, Any]]:
        if not isinstance(items, (list, tuple)):
            return []
        results = []
        for raw in items:
            if not isinstance(raw, Mapping):
                continue
            payload = dict(raw)
            asset_id = payload.get("path") or payload.get("asset_id") or payload.get("adapter_id")
            if asset_id:
                payload["path"] = str(self.asset_resolver(str(asset_id)))
            results.append(payload)
        return results


def _regions(items: object) -> tuple[RegionDefinition, ...]:
    if not isinstance(items, (list, tuple)):
        return ()
    results = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        box = raw.get("box")
        if isinstance(box, Mapping):
            coordinates = tuple(float(box[key]) for key in ("x0", "y0", "x1", "y1"))
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            coordinates = tuple(float(value) for value in box)
        else:
            raise ValueError("regional Krea request requires a four-coordinate box")
        results.append(
            RegionDefinition(
                region_id=str(raw.get("region_id", raw.get("id", "region"))),
                name=str(raw.get("name", "Region")),
                box=PixelBox(*coordinates),
                prompt=str(raw.get("prompt", "")),
                negative_prompt=str(raw.get("negative_prompt", "")),
                face_identity_prompt=str(raw.get("face_identity_prompt", "")),
                priority=int(raw.get("priority", 0)),
                spatial_role="subject",
            )
        )
    return tuple(results)


def _settings(request: Mapping[str, Any]) -> Mapping[str, Any]:
    value = request.get("settings", {})
    return value if isinstance(value, Mapping) else {}


def _progress_adapter(
    progress: ProgressCallback,
    cancellation: CancellationToken,
) -> Callable[[int, int, dict[str, Any]], None]:
    def callback(step: int, total: int, memory: dict[str, Any]) -> None:
        cancellation.raise_if_cancelled()
        progress("diffusion", step / total if total else None, {"step": step, "total": total, "memory": memory})

    return callback


def _result(payload: Mapping[str, Any]) -> BackendResult:
    path = payload.get("image_path")
    if not isinstance(path, str) or not path:
        raise RuntimeError("Krea runtime did not return an image path")
    return BackendResult(asset_paths=(Path(path),), metadata=dict(payload))


__all__ = ["ComfyKreaBackend"]
