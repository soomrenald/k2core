from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


KREA2_DEPTH_PUBLIC_SHA256 = "fb80547ed79b47c1e3fea7bb9d36297e3917b2115fab6700ca1501350f9f483c"
KREA2_DEPTH_EXPECTED_BLOCKS = 28
KREA2_DEPTH_EXPECTED_RANK = 64
KREA2_DEPTH_EXPECTED_TARGETS = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.wo",
    "attn.gate",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)
_MAX_HEADER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DepthCheckpointInfo:
    path: Path
    sha256: str
    metadata: Mapping[str, str]
    tensor_count: int
    rank: int
    expanded_projection_key: str
    compatible_block_pairs: int
    source: str

    def document(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "tensor_count": self.tensor_count,
            "rank": self.rank,
            "expanded_projection_key": self.expanded_projection_key,
            "compatible_block_pairs": self.compatible_block_pairs,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class DepthCheckpointCompatibility:
    compatible: bool
    verified: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checkpoint: DepthCheckpointInfo | None

    def document(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "verified": self.verified,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checkpoint": self.checkpoint.document() if self.checkpoint else None,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise ValueError("safetensors header length is missing")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if not 2 <= header_length <= _MAX_HEADER_BYTES:
            raise ValueError("safetensors header length is invalid")
        if 8 + header_length > file_size:
            raise ValueError("safetensors header exceeds the file size")
        encoded_header = stream.read(header_length)
    try:
        header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("safetensors header is not valid UTF-8 JSON") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("safetensors metadata must be an object")
    metadata = {str(key): str(value) for key, value in raw_metadata.items()}
    shapes: dict[str, tuple[int, ...]] = {}
    for key, descriptor in header.items():
        if not isinstance(key, str) or not isinstance(descriptor, dict):
            raise ValueError("safetensors tensor descriptors are invalid")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[1] < offsets[0]
            or 8 + header_length + offsets[1] > file_size
        ):
            raise ValueError(f"safetensors descriptor for {key!r} is invalid")
        shapes[key] = tuple(shape)
    if not shapes:
        raise ValueError("safetensors checkpoint contains no tensors")
    return metadata, shapes


def _strip_prefixes(base: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in (
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                changed = True
    return base


def lora_pairs(keys: tuple[str, ...] | list[str]) -> tuple[tuple[str, str, str], ...]:
    available = set(keys)
    suffixes = (
        (".A", ".B"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_down", ".lora_up"),
        ("_lora.down.weight", "_lora.up.weight"),
    )
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for down_suffix, up_suffix in suffixes:
        for down_key in keys:
            if not down_key.endswith(down_suffix):
                continue
            base = down_key[: -len(down_suffix)]
            up_key = base + up_suffix
            identity = (down_key, up_key)
            if up_key in available and identity not in seen:
                pairs.append((base, down_key, up_key))
                seen.add(identity)
    return tuple(pairs)


def _block_target(base: str) -> tuple[int, str] | None:
    normalized = _strip_prefixes(base)
    if not normalized.startswith("blocks."):
        return None
    parts = normalized.split(".")
    if len(parts) < 4 or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def inspect_depth_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = KREA2_DEPTH_PUBLIC_SHA256,
) -> DepthCheckpointCompatibility:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.casefold() != ".safetensors":
        return DepthCheckpointCompatibility(
            compatible=False,
            verified=False,
            errors=("checkpoint must be a readable .safetensors file",),
            warnings=(),
            checkpoint=None,
        )
    try:
        metadata, shapes = _safetensors_header(resolved)
        sha256 = _sha256_file(resolved)
    except (OSError, ValueError) as error:
        return DepthCheckpointCompatibility(
            compatible=False,
            verified=False,
            errors=(f"invalid safetensors checkpoint: {error}",),
            warnings=(),
            checkpoint=None,
        )

    errors: list[str] = []
    warnings: list[str] = []
    projection_candidates = [
        key
        for key, shape in shapes.items()
        if len(shape) == 2
        and shape[1] == 128
        and (key.endswith("first.weight") or key.endswith("img_in.weight"))
    ]
    if len(projection_candidates) != 1:
        errors.append("exactly one 128-channel expanded Krea input projection is required")
    projection_key = projection_candidates[0] if len(projection_candidates) == 1 else ""

    ranks: set[int] = set()
    found_targets: set[tuple[int, str]] = set()
    for base, down_key, up_key in lora_pairs(list(shapes)):
        target = _block_target(base)
        if target is None:
            continue
        down_shape = shapes[down_key]
        up_shape = shapes[up_key]
        if len(down_shape) != 2 or len(up_shape) != 2:
            continue
        if up_shape[1] == down_shape[0]:
            ranks.add(down_shape[0])
            found_targets.add(target)
        elif down_shape[1] == up_shape[0]:
            ranks.add(down_shape[1])
            found_targets.add(target)
    if ranks != {KREA2_DEPTH_EXPECTED_RANK}:
        errors.append(f"all Krea depth LoRA pairs must use rank {KREA2_DEPTH_EXPECTED_RANK}")
    required_targets = {
        (block, target)
        for block in range(KREA2_DEPTH_EXPECTED_BLOCKS)
        for target in KREA2_DEPTH_EXPECTED_TARGETS
    }
    missing = required_targets - found_targets
    unexpected = {
        target
        for target in found_targets
        if target[0] not in range(KREA2_DEPTH_EXPECTED_BLOCKS)
        or target[1] not in KREA2_DEPTH_EXPECTED_TARGETS
    }
    if missing:
        errors.append(f"checkpoint is missing {len(missing)} required Krea block LoRA pairs")
    if unexpected:
        warnings.append(f"checkpoint contains {len(unexpected)} additional block LoRA pairs")

    verified = expected_sha256 is not None and sha256.casefold() == expected_sha256.casefold()
    if expected_sha256 is not None and not verified:
        errors.append("checkpoint SHA-256 does not match the configured trusted artifact")
    elif expected_sha256 is None:
        warnings.append("checkpoint structure is compatible but its source hash is unverified")
    source = "Patil/Krea-2-depth-controlnet" if sha256 == KREA2_DEPTH_PUBLIC_SHA256 else "custom"
    info = DepthCheckpointInfo(
        path=resolved,
        sha256=sha256,
        metadata=MappingProxyType(metadata),
        tensor_count=len(shapes),
        rank=next(iter(ranks), 0) if len(ranks) == 1 else 0,
        expanded_projection_key=projection_key,
        compatible_block_pairs=len(found_targets),
        source=source,
    )
    return DepthCheckpointCompatibility(
        compatible=not errors,
        verified=verified,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checkpoint=info,
    )
