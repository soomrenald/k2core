from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from k2core.depth.loader import load_depth_image
from k2core.depth.types import DepthImage


@dataclass(frozen=True, slots=True)
class BlenderDepthBundle:
    root: Path
    depth: DepthImage
    camera: Mapping[str, Any]
    objects: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "root": self.root.name,
            "depth": self.depth.info.document(),
            "camera": dict(self.camera),
            "object_count": len(self.objects),
            "template_version": self.manifest.get("template_version"),
            "blender_version": self.manifest.get("blender_version"),
        }


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Blender bundle metadata is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Blender bundle metadata must be an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_blender_depth_bundle(path: Path) -> BlenderDepthBundle:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Blender depth bundle must be a directory")
    manifest = _json_object(root / "export.json")
    camera = _json_object(root / "camera.json")
    objects_document = _json_object(root / "objects.json")
    if manifest.get("format") != "k2lab-blender-depth-bundle":
        raise ValueError("unsupported Blender depth bundle format")
    if int(manifest.get("version", 0)) != 1:
        raise ValueError("unsupported Blender depth bundle version")
    depth_name = str(manifest.get("depth_image", "depth_16bit.png"))
    depth_path = root / depth_name
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or depth_name not in checksums:
        raise ValueError("Blender bundle does not declare a depth-image checksum")
    if _sha256(depth_path) != str(checksums[depth_name]).casefold():
        raise ValueError("Blender bundle depth-image checksum does not match")
    depth = load_depth_image(depth_path)
    resolution = camera.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or [depth.info.width, depth.info.height] != [int(resolution[0]), int(resolution[1])]
    ):
        raise ValueError("Blender bundle camera and depth resolutions disagree")
    convention = camera.get("depth_convention")
    if convention != "near_white_far_black":
        raise ValueError("Blender bundle uses an unsupported depth convention")
    raw_objects = objects_document.get("objects")
    if not isinstance(raw_objects, list) or not all(isinstance(item, dict) for item in raw_objects):
        raise ValueError("Blender bundle objects.json is invalid")
    return BlenderDepthBundle(
        root=root,
        depth=depth,
        camera=MappingProxyType(camera),
        objects=tuple(MappingProxyType(item) for item in raw_objects),
        manifest=MappingProxyType(manifest),
    )
