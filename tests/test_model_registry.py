from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from k2core.config import ModelDirectories
from k2core.model import (
    ComponentReference,
    ModelRegistry,
    RegisteredModel,
    load_model_registry,
    scan_legacy_comfyui_models,
    sha256_file,
    validate_model_registry,
)


def _descriptor(shape: list[int]) -> dict[str, object]:
    return {"dtype": "BF16", "shape": shape, "data_offsets": [0, 1]}


def _transformer_header() -> dict[str, object]:
    tensors: dict[str, object] = {
        "first.weight": _descriptor([6144, 64]),
        "blocks.0.attn.wq.weight": _descriptor([6144, 6144]),
        "blocks.0.attn.wk.weight": _descriptor([1536, 6144]),
        "blocks.0.attn.wv.weight": _descriptor([1536, 6144]),
        "blocks.27.attn.wq.weight": _descriptor([6144, 6144]),
        "txtfusion.projector.weight": _descriptor([1, 12]),
        "txtfusion.layerwise_blocks.0.prenorm.scale": _descriptor([2560]),
        "last.linear.weight": _descriptor([64, 6144]),
        "quant.weight_scale": _descriptor([1]),
    }
    tensors.update(
        {f"blocks.{index}.marker": _descriptor([1]) for index in range(1, 27)}
    )
    return tensors


def _text_encoder_header() -> dict[str, object]:
    tensors: dict[str, object] = {
        "model.embed_tokens.weight": _descriptor([151936, 2560]),
        "model.layers.0.self_attn.q_proj.weight": _descriptor([4096, 2560]),
        "model.layers.35.self_attn.q_proj.weight": _descriptor([4096, 2560]),
        "model.norm.weight": _descriptor([2560]),
    }
    tensors.update(
        {f"model.layers.{index}.marker": _descriptor([1]) for index in range(1, 35)}
    )
    return tensors


def _vae_header() -> dict[str, object]:
    return {
        "encoder.conv1.weight": _descriptor([96, 3, 3, 3, 3]),
        "decoder.conv1.weight": _descriptor([384, 16, 3, 3, 3]),
        "conv1.weight": _descriptor([32, 32, 1, 1, 1]),
        "conv2.weight": _descriptor([16, 16, 1, 1, 1]),
    }


def _write_header(path: Path, header: dict[str, object]) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def _component(path: Path, digest: str | None = None) -> ComponentReference:
    return ComponentReference(path=path, sha256=digest or sha256_file(path))


def _registered_model(
    transformer: Path,
    text_encoder: Path,
    vae: Path,
    *,
    name: str = "krea2_turbo",
    architecture: str = "krea2",
) -> RegisteredModel:
    return RegisteredModel(
        name=name,
        architecture=architecture,
        transformer=_component(transformer),
        text_encoder=_component(text_encoder),
        vae=_component(vae),
    )


def _model_files(root: Path) -> tuple[Path, Path, Path]:
    transformer = root / "krea2_turbo_fp8_scaled.safetensors"
    text_encoder = root / "qwen3vl_4b_fp8_scaled.safetensors"
    vae = root / "qwen_image_vae.safetensors"
    _write_header(transformer, _transformer_header())
    _write_header(text_encoder, _text_encoder_header())
    _write_header(vae, _vae_header())
    return transformer, text_encoder, vae


class ModelRegistryTests(unittest.TestCase):
    def test_valid_registry_round_trip_and_header_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer, text_encoder, vae = _model_files(root)
            registry = ModelRegistry(
                models=(_registered_model(transformer, text_encoder, vae),)
            )
            registry_path = root / "models.toml"
            registry_path.write_text(registry.to_toml(), encoding="utf-8")

            loaded = load_model_registry(registry_path)
            result = validate_model_registry(loaded)

            self.assertTrue(result.valid)
            self.assertEqual(loaded, registry)
            self.assertTrue(all(item.observed_sha256 for item in result.models[0].components))

    def test_missing_file_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer, text_encoder, vae = _model_files(root)
            missing = root / "missing.safetensors"
            registry = ModelRegistry(
                models=(
                    RegisteredModel(
                        name="missing",
                        architecture="krea2",
                        transformer=ComponentReference(
                            path=missing,
                            sha256="0" * 64,
                        ),
                        text_encoder=_component(text_encoder),
                        vae=_component(vae),
                    ),
                )
            )

            result = validate_model_registry(registry)

            self.assertFalse(result.valid)
            self.assertIn("does not exist", result.models[0].components[0].errors[0])
            self.assertTrue(transformer.exists())

    def test_wrong_architecture_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = _model_files(root)
            registry = ModelRegistry(
                models=(_registered_model(*files, architecture="not-krea"),)
            )

            result = validate_model_registry(registry)

            self.assertFalse(result.valid)
            self.assertIn("unsupported architecture", result.models[0].errors[0])

    def test_wrong_tensor_shape_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer, text_encoder, vae = _model_files(root)
            invalid_header = _transformer_header()
            invalid_header["first.weight"] = _descriptor([1, 1])
            _write_header(transformer, invalid_header)
            registry = ModelRegistry(
                models=(_registered_model(transformer, text_encoder, vae),)
            )

            result = validate_model_registry(registry)

            self.assertFalse(result.valid)
            self.assertTrue(
                any(
                    "shape mismatch for first.weight" in error
                    for error in result.models[0].components[0].errors
                )
            )

    def test_symlink_resolves_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer, text_encoder, vae = _model_files(root)
            linked = root / "linked-transformer.safetensors"
            linked.symlink_to(transformer)
            registry = ModelRegistry(
                models=(
                    RegisteredModel(
                        name="linked",
                        architecture="krea2",
                        transformer=_component(linked),
                        text_encoder=_component(text_encoder),
                        vae=_component(vae),
                    ),
                )
            )

            result = validate_model_registry(registry)

            self.assertTrue(result.valid)
            self.assertEqual(result.models[0].components[0].configured_path, linked)
            self.assertEqual(
                result.models[0].components[0].resolved_path,
                transformer.resolve(),
            )

    def test_duplicate_model_names_are_rejected_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = _model_files(Path(directory))
            first = _registered_model(*files, name="Krea2")
            second = _registered_model(*files, name="krea2")

            with self.assertRaisesRegex(ValueError, "duplicate model name"):
                ModelRegistry(models=(first, second))

    def test_hash_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer, text_encoder, vae = _model_files(root)
            registry = ModelRegistry(
                models=(
                    RegisteredModel(
                        name="wrong-hash",
                        architecture="krea2",
                        transformer=_component(transformer, "0" * 64),
                        text_encoder=_component(text_encoder),
                        vae=_component(vae),
                    ),
                )
            )

            result = validate_model_registry(registry)

            self.assertFalse(result.valid)
            self.assertTrue(
                any(
                    "SHA-256 mismatch" in error
                    for error in result.models[0].components[0].errors
                )
            )

    def test_legacy_comfyui_scan_emits_valid_config_without_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusion = root / "models" / "diffusion_models"
            text = root / "models" / "text_encoders"
            vae_directory = root / "models" / "vae"
            diffusion.mkdir(parents=True)
            text.mkdir()
            vae_directory.mkdir()
            transformer, text_encoder, vae = _model_files(root)
            moved_transformer = diffusion / transformer.name
            moved_text = text / text_encoder.name
            moved_vae = vae_directory / vae.name
            transformer.rename(moved_transformer)
            text_encoder.rename(moved_text)
            vae.rename(moved_vae)
            before = {
                path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (moved_transformer, moved_text, moved_vae)
            }

            registry = scan_legacy_comfyui_models(
                ModelDirectories(diffusion, text, vae_directory)
            )
            validation = validate_model_registry(registry)

            self.assertTrue(validation.valid)
            self.assertEqual(registry.source, "legacy_comfyui_scan")
            self.assertIn(str(moved_transformer.resolve()), registry.to_toml())
            after = {
                path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (moved_transformer, moved_text, moved_vae)
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
