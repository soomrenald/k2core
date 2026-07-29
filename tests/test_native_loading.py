from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from k2core.backends import NativeK2Backend
from k2core.backends.native_loading import NativeModelLoader
from k2core.inference import (
    ConfigurationError,
    DTypePolicy,
    DevicePolicy,
    PipelineConfig,
    WeightMappingError,
)
from k2core.model import (
    ArtifactKind,
    ArtifactSet,
    ComponentReference,
    ModelArtifact,
    RegisteredModel,
    SafetensorsSummary,
    sha256_file,
)


def _descriptor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 1]}


def _transformer_header() -> dict[str, object]:
    tensors: dict[str, object] = {
        "first.weight": _descriptor([6144, 64]),
        "blocks.0.attn.wq.weight": _descriptor([6144, 6144], "F8_E4M3"),
        "blocks.0.attn.wk.weight": _descriptor([1536, 6144], "F8_E4M3"),
        "blocks.0.attn.wv.weight": _descriptor([1536, 6144], "F8_E4M3"),
        "blocks.27.attn.wq.weight": _descriptor([6144, 6144], "F8_E4M3"),
        "txtfusion.projector.weight": _descriptor([1, 12]),
        "txtfusion.layerwise_blocks.0.prenorm.scale": _descriptor([2560]),
        "last.linear.weight": _descriptor([64, 6144]),
        "blocks.0.attn.wq.weight_scale": _descriptor([1], "F32"),
    }
    tensors.update(
        {f"blocks.{index}.marker": _descriptor([1]) for index in range(1, 27)}
    )
    return tensors


def _text_encoder_header() -> dict[str, object]:
    tensors: dict[str, object] = {
        "model.embed_tokens.weight": _descriptor([151936, 2560]),
        "model.layers.0.self_attn.q_proj.weight": _descriptor([4096, 2560], "F8_E4M3"),
        "model.layers.35.self_attn.q_proj.weight": _descriptor([4096, 2560], "F8_E4M3"),
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
    document = dict(header)
    document["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(document, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


class FakeTensor:
    _ELEMENT_SIZES = {"torch.float8_e4m3fn": 1, "torch.float32": 4, "torch.bfloat16": 2}

    def __init__(self, shape: list[int], dtype: str) -> None:
        names = {
            "F8_E4M3": "torch.float8_e4m3fn",
            "F32": "torch.float32",
            "BF16": "torch.bfloat16",
            "U8": "torch.uint8",
        }
        self.shape = shape
        self.dtype = names[dtype]

    def is_floating_point(self) -> bool:
        return self.dtype != "torch.uint8"

    def to(self, *, dtype):
        converted = FakeTensor(self.shape, "BF16")
        converted.dtype = dtype
        return converted

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self._ELEMENT_SIZES.get(self.dtype, 1)


class FakeSafeOpen:
    def __init__(self, path: str, *, framework: str, device: str) -> None:
        del framework
        self.path = Path(path)
        self.device = device
        with self.path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(length))

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def keys(self):
        return tuple(key for key in self.header if key != "__metadata__")

    def metadata(self):
        return self.header.get("__metadata__", {})

    def get_tensor(self, key: str) -> FakeTensor:
        descriptor = self.header[key]
        return FakeTensor(descriptor["shape"], descriptor["dtype"])


class FakeCuda:
    def __init__(self) -> None:
        self.cache_releases = 0

    def is_available(self) -> bool:
        return False

    def empty_cache(self) -> None:
        self.cache_releases += 1


class FakeTorch:
    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"
    float32 = "torch.float32"
    float8_e4m3fn = "torch.float8_e4m3fn"
    cuda = FakeCuda()

    @staticmethod
    def device(name: str) -> str:
        if name not in {"cpu", "cuda"}:
            raise RuntimeError("invalid device")
        return name


def _fixture(root: Path, *, rogue_transformer_key: bool = False):
    headers = {
        ArtifactKind.TRANSFORMER: _transformer_header(),
        ArtifactKind.TEXT_ENCODER: _text_encoder_header(),
        ArtifactKind.VAE: _vae_header(),
    }
    if rogue_transformer_key:
        headers[ArtifactKind.TRANSFORMER]["rogue.weight"] = _descriptor([1])
    paths = {
        kind: root / f"{kind.value}.safetensors"
        for kind in ArtifactKind
    }
    for kind, path in paths.items():
        _write_header(path, headers[kind])
    references = {
        kind: ComponentReference(path=path, sha256=sha256_file(path))
        for kind, path in paths.items()
    }
    model = RegisteredModel(
        name="fixture",
        architecture="krea2",
        transformer=references[ArtifactKind.TRANSFORMER],
        text_encoder=references[ArtifactKind.TEXT_ENCODER],
        vae=references[ArtifactKind.VAE],
    )
    artifacts = ArtifactSet(
        *(
            ModelArtifact(
                kind=kind,
                path=paths[kind],
                size_bytes=paths[kind].stat().st_size,
                summary=SafetensorsSummary(0, (), (), "pt", False),
            )
            for kind in ArtifactKind
        )
    )
    return model, artifacts


def _supported_hashes(model: RegisteredModel):
    return {
        kind: frozenset({component.sha256})
        for kind, component in model.components()
    }


class NativeLoadingTests(unittest.TestCase):
    def test_strict_identity_mapping_counts_parameters_and_unloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = _fixture(Path(directory))
            loader = NativeModelLoader(supported_hashes=_supported_hashes(model))
            with (
                patch("k2core.backends.native_loading._import_torch", return_value=FakeTorch),
                patch(
                    "k2core.backends.native_loading._import_safe_open",
                    return_value=FakeSafeOpen,
                ),
            ):
                pipeline = loader.load(model)
                reports = pipeline.reports()
                self.assertTrue(pipeline.loaded)
                self.assertTrue(all(report.strict_match for report in reports))
                self.assertTrue(all(report.tensor_count == report.mapped_key_count for report in reports))
                self.assertGreater(reports[0].parameter_count, 0)
                self.assertEqual(reports[0].device, "cpu")
                loader.unload(pipeline)

            self.assertFalse(pipeline.loaded)
            self.assertTrue(all(not component.tensors for component in (
                pipeline.transformer,
                pipeline.text_encoder,
                pipeline.vae,
            )))

    def test_requested_dtype_conversion_is_applied_to_floating_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = _fixture(Path(directory))
            with (
                patch("k2core.backends.native_loading._import_torch", return_value=FakeTorch),
                patch(
                    "k2core.backends.native_loading._import_safe_open",
                    return_value=FakeSafeOpen,
                ),
            ):
                pipeline = NativeModelLoader(
                    supported_hashes=_supported_hashes(model)
                ).load(
                    model,
                    device_policy=DevicePolicy(
                        transformer_device="cpu",
                        text_encoder_device="cpu",
                        vae_device="cpu",
                        weight_dtype=DTypePolicy.BFLOAT16,
                    ),
                )

            self.assertEqual(
                pipeline.transformer.report.loaded_dtypes,
                (("torch.bfloat16", pipeline.transformer.report.tensor_count),),
            )

    def test_unmapped_key_fails_strict_loading_and_releases_prior_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = _fixture(Path(directory), rogue_transformer_key=True)
            with (
                patch("k2core.backends.native_loading._import_torch", return_value=FakeTorch),
                patch(
                    "k2core.backends.native_loading._import_safe_open",
                    return_value=FakeSafeOpen,
                ),
                self.assertRaises(WeightMappingError),
            ):
                NativeModelLoader(
                    supported_hashes=_supported_hashes(model)
                ).load(model)

    def test_unapproved_component_identity_fails_before_tensor_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = _fixture(Path(directory))
            with (
                patch(
                    "k2core.backends.native_loading._import_torch",
                    return_value=FakeTorch,
                ),
                patch("k2core.backends.native_loading._import_safe_open") as safe_open,
                self.assertRaisesRegex(
                    WeightMappingError,
                    "no approved strict state-dict mapping",
                ),
            ):
                NativeModelLoader().load(model)
            safe_open.assert_not_called()

    def test_backend_requires_registry_and_matches_artifact_paths(self) -> None:
        backend = NativeK2Backend()
        with self.assertRaises(ConfigurationError):
            backend.load(PipelineConfig(artifacts=ArtifactSet(None, None, None)))

        with tempfile.TemporaryDirectory() as directory:
            model, artifacts = _fixture(Path(directory))
            pipeline = unittest.mock.Mock()
            pipeline.model_name = model.name
            pipeline.reports.return_value = ()
            loader = unittest.mock.Mock()
            loader.load.return_value = pipeline
            backend.loader = loader
            manager = unittest.mock.Mock()
            manager.preflight.return_value = {"stage": "model_loading"}
            manager.staging_policy.return_value = DevicePolicy(
                transformer_device="cpu",
                text_encoder_device="cpu",
                vae_device="cpu",
            )
            manager.plan.to_payload.return_value = {"accelerator_backend": "fixture"}

            with patch(
                "k2core.backends.native.NativeDeviceManager",
                return_value=manager,
            ):
                result = backend.load(
                    PipelineConfig(
                        artifacts=artifacts,
                        registered_model=model,
                    )
                )

            self.assertEqual(result.backend_id, "native")
            self.assertEqual(result.metadata["device_plan"]["accelerator_backend"], "fixture")
            loader.load.assert_called_once_with(
                model,
                device_policy=manager.staging_policy.return_value,
                strict=True,
            )
            backend.unload()
            loader.unload.assert_called_once_with(pipeline)
            manager.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
