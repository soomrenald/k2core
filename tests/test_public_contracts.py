from __future__ import annotations

import unittest
from pathlib import Path

from k2core.assets import AssetRecord, AssetStore
from k2core.backends import (
    BackendCapabilities,
    FrameEditorBackend,
    ImageGeneratorBackend,
    VideoGeneratorBackend,
)
from k2core.runtime import AcceleratorDevice, AcceleratorVendor, ModelRuntime
from k2core.workers import WORKER_PROTOCOL_VERSION, CommandKind


class PublicContractTests(unittest.TestCase):
    def test_asset_record_requires_contained_hashed_asset(self) -> None:
        record = AssetRecord(
            asset_id="asset-1",
            relative_path="assets/image.png",
            sha256="a" * 64,
            media_type="image/png",
            byte_size=12,
            width=2,
            height=3,
        )
        self.assertEqual(record.relative_path, "assets/image.png")
        with self.assertRaisesRegex(ValueError, "project-relative"):
            AssetRecord("bad", "../escape", "a" * 64, "image/png", 1)

    def test_backend_protocols_are_runtime_checkable(self) -> None:
        class Backend:
            backend_id = "mock"

            def capabilities(self):
                return BackendCapabilities("mock", frozenset(), frozenset())

            def validate_image_request(self, request):
                return ()

            def generate_image(self, request, *, progress, cancellation):
                raise NotImplementedError

            def validate_edit_request(self, request):
                return ()

            def edit_frame(self, request, *, progress, cancellation):
                raise NotImplementedError

            def refine_faces(self, request, *, progress, cancellation):
                raise NotImplementedError

            def validate_segment_request(self, request):
                return ()

            def generate_segment(self, request, *, progress, cancellation):
                raise NotImplementedError

            def release(self):
                return None

        backend = Backend()
        self.assertIsInstance(backend, ImageGeneratorBackend)
        self.assertIsInstance(backend, FrameEditorBackend)
        self.assertIsInstance(backend, VideoGeneratorBackend)

    def test_runtime_and_asset_store_protocols_are_structural(self) -> None:
        self.assertTrue(getattr(ModelRuntime, "_is_runtime_protocol"))
        self.assertTrue(getattr(AssetStore, "_is_runtime_protocol"))
        device = AcceleratorDevice("0", "GPU", AcceleratorVendor.AMD, 1024)
        self.assertEqual(device.vendor.value, "amd")
        self.assertEqual(Path("assets/image.png").parts[0], "assets")

    def test_worker_protocol_is_explicit_and_keeps_existing_commands(self) -> None:
        self.assertEqual(WORKER_PROTOCOL_VERSION, "1")
        self.assertEqual(CommandKind.EDIT_IMAGE.value, "edit_image")
        self.assertEqual(CommandKind.REFINE_FACES.value, "refine_faces")


if __name__ == "__main__":
    unittest.main()

