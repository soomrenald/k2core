"""K2Lab-owned model registry and read-only legacy discovery."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from k2core.config import ModelDirectories
from k2core.model.artifacts import (
    ArtifactKind,
    ModelArtifact,
    discover_krea_transformers,
    discover_model_artifacts,
    read_safetensors_header,
)
from k2core.model.manifests import validate_tensor_header


REGISTRY_SCHEMA_VERSION = "k2lab-model-registry/1"
SUPPORTED_ARCHITECTURES = frozenset({"krea2"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ComponentReference:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        normalized_hash = self.sha256.strip().casefold()
        if not _SHA256_PATTERN.fullmatch(normalized_hash):
            raise ValueError(f"invalid SHA-256 for model component {self.path}")
        object.__setattr__(self, "sha256", normalized_hash)


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    name: str
    architecture: str
    transformer: ComponentReference
    text_encoder: ComponentReference
    vae: ComponentReference
    default_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("registered model name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "architecture", self.architecture.strip().casefold())
        object.__setattr__(self, "default_dtype", self.default_dtype.strip().casefold())

    def components(self) -> tuple[tuple[ArtifactKind, ComponentReference], ...]:
        return (
            (ArtifactKind.TRANSFORMER, self.transformer),
            (ArtifactKind.TEXT_ENCODER, self.text_encoder),
            (ArtifactKind.VAE, self.vae),
        )


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    models: tuple[RegisteredModel, ...]
    schema_version: str = REGISTRY_SCHEMA_VERSION
    source: str = "k2lab"

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model registry schema {self.schema_version!r}; "
                f"expected {REGISTRY_SCHEMA_VERSION!r}"
            )
        names: set[str] = set()
        for model in self.models:
            key = model.name.casefold()
            if key in names:
                raise ValueError(f"duplicate model name: {model.name}")
            names.add(key)

    def to_toml(self) -> str:
        lines = [
            f'schema_version = "{_toml_string(self.schema_version)}"',
            f'source = "{_toml_string(self.source)}"',
        ]
        for model in self.models:
            lines.extend(
                [
                    "",
                    "[[models]]",
                    f'name = "{_toml_string(model.name)}"',
                    f'architecture = "{_toml_string(model.architecture)}"',
                    f'default_dtype = "{_toml_string(model.default_dtype)}"',
                ]
            )
            for kind, component in model.components():
                lines.extend(
                    [
                        "",
                        f"[models.{kind.value}]",
                        f'path = "{_toml_string(str(component.path))}"',
                        f'sha256 = "{component.sha256}"',
                    ]
                )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ComponentValidation:
    kind: ArtifactKind
    configured_path: Path
    resolved_path: Path | None
    expected_sha256: str
    observed_sha256: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ModelValidation:
    name: str
    architecture: str
    components: tuple[ComponentValidation, ...]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors and all(component.valid for component in self.components)


@dataclass(frozen=True, slots=True)
class RegistryValidation:
    models: tuple[ModelValidation, ...]

    @property
    def valid(self) -> bool:
        return bool(self.models) and all(model.valid for model in self.models)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_registry(path: Path) -> ModelRegistry:
    configured_path = path.expanduser()
    with configured_path.open("rb") as handle:
        document = tomllib.load(handle)
    return model_registry_from_document(document, base=configured_path.parent)


def model_registry_from_document(
    document: dict[str, Any],
    *,
    base: Path | None = None,
) -> ModelRegistry:
    if not isinstance(document, dict):
        raise ValueError("model registry must be a TOML table")
    raw_models = document.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("model registry must contain one or more [[models]] tables")
    models = tuple(_registered_model(item, base=base) for item in raw_models)
    if not models:
        raise ValueError("model registry must contain one or more [[models]] tables")
    return ModelRegistry(
        models=models,
        schema_version=_required_string(document, "schema_version"),
        source=str(document.get("source", "k2lab")),
    )


def validate_model_registry(
    registry: ModelRegistry,
    *,
    verify_hashes: bool = True,
) -> RegistryValidation:
    validated_models: list[ModelValidation] = []
    hash_cache: dict[Path, str] = {}
    for model in registry.models:
        model_errors: list[str] = []
        if model.architecture not in SUPPORTED_ARCHITECTURES:
            model_errors.append(
                f"unsupported architecture {model.architecture!r}; expected 'krea2'"
            )
        components = tuple(
            _validate_component(
                kind,
                component,
                verify_hashes=verify_hashes,
                hash_cache=hash_cache,
            )
            for kind, component in model.components()
        )
        validated_models.append(
            ModelValidation(
                name=model.name,
                architecture=model.architecture,
                components=components,
                errors=tuple(model_errors),
            )
        )
    return RegistryValidation(models=tuple(validated_models))


def scan_legacy_comfyui_models(directories: ModelDirectories) -> ModelRegistry:
    """Create registry data from ComfyUI directories without moving or loading weights."""

    shared = discover_model_artifacts(directories)
    if shared.text_encoder is None:
        raise FileNotFoundError("could not discover a Qwen text encoder in the legacy paths")
    if shared.vae is None:
        raise FileNotFoundError("could not discover a Qwen image VAE in the legacy paths")
    if directories.diffusion_model_file is not None:
        transformers: Iterable[ModelArtifact] = (
            (shared.transformer,) if shared.transformer is not None else ()
        )
    else:
        transformers = discover_krea_transformers(directories.diffusion_models)
    transformers = tuple(transformers)
    if not transformers:
        raise FileNotFoundError("could not discover a Krea transformer in the legacy paths")

    hash_cache: dict[Path, str] = {}

    def reference(artifact: ModelArtifact) -> ComponentReference:
        resolved = artifact.path.expanduser().resolve(strict=True)
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        return ComponentReference(path=resolved, sha256=hash_cache[resolved])

    text_encoder = reference(shared.text_encoder)
    vae = reference(shared.vae)
    models = tuple(
        RegisteredModel(
            name=_registry_name(transformer.path),
            architecture="krea2",
            transformer=reference(transformer),
            text_encoder=text_encoder,
            vae=vae,
        )
        for transformer in transformers
    )
    return ModelRegistry(models=models, source="legacy_comfyui_scan")


def _validate_component(
    kind: ArtifactKind,
    component: ComponentReference,
    *,
    verify_hashes: bool,
    hash_cache: dict[Path, str],
) -> ComponentValidation:
    configured = component.path.expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    resolved: Path | None = None
    observed_hash: str | None = None
    if not configured.is_absolute():
        errors.append(f"{kind.value} path must be absolute: {component.path}")
    else:
        try:
            resolved = configured.resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"{kind.value} file does not exist: {configured}")
        except (OSError, RuntimeError) as error:
            errors.append(f"could not resolve {kind.value} path {configured}: {error}")
    if resolved is not None:
        if not resolved.is_file():
            errors.append(f"{kind.value} path is not a file: {resolved}")
        elif resolved.suffix.casefold() != ".safetensors":
            errors.append(f"{kind.value} file is not safetensors: {resolved}")
        else:
            try:
                header = read_safetensors_header(resolved)
                shape_errors, shape_warnings = validate_tensor_header(kind, header)
                errors.extend(shape_errors)
                warnings.extend(shape_warnings)
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"could not inspect {kind.value} header: {error}")
            if verify_hashes:
                try:
                    if resolved not in hash_cache:
                        hash_cache[resolved] = sha256_file(resolved)
                    observed_hash = hash_cache[resolved]
                except OSError as error:
                    errors.append(f"could not hash {kind.value} file {resolved}: {error}")
                if observed_hash is not None and observed_hash != component.sha256:
                    errors.append(
                        f"SHA-256 mismatch for {kind.value}: expected {component.sha256}, "
                        f"got {observed_hash}"
                    )
    return ComponentValidation(
        kind=kind,
        configured_path=component.path,
        resolved_path=resolved,
        expected_sha256=component.sha256,
        observed_sha256=observed_hash,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _registered_model(value: Any, *, base: Path | None) -> RegisteredModel:
    if not isinstance(value, dict):
        raise ValueError("each [[models]] entry must be a TOML table")
    return RegisteredModel(
        name=_required_string(value, "name"),
        architecture=_required_string(value, "architecture"),
        default_dtype=str(value.get("default_dtype", "bfloat16")),
        transformer=_component(value, ArtifactKind.TRANSFORMER, base=base),
        text_encoder=_component(value, ArtifactKind.TEXT_ENCODER, base=base),
        vae=_component(value, ArtifactKind.VAE, base=base),
    )


def _component(
    model: dict[str, Any],
    kind: ArtifactKind,
    *,
    base: Path | None,
) -> ComponentReference:
    value = model.get(kind.value)
    if not isinstance(value, dict):
        raise ValueError(f"model entry requires a [{kind.value}] component table")
    raw_path = Path(_required_string(value, "path")).expanduser()
    path = raw_path if raw_path.is_absolute() or base is None else base / raw_path
    return ComponentReference(path=path, sha256=_required_string(value, "sha256"))


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"required string field is missing: {key}")
    return item.strip()


def _registry_name(path: Path) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", path.stem.casefold()).strip("_")
    return name or "krea2"


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "ComponentReference",
    "ComponentValidation",
    "ModelRegistry",
    "ModelValidation",
    "REGISTRY_SCHEMA_VERSION",
    "RegisteredModel",
    "RegistryValidation",
    "SUPPORTED_ARCHITECTURES",
    "load_model_registry",
    "model_registry_from_document",
    "scan_legacy_comfyui_models",
    "sha256_file",
    "validate_model_registry",
]
