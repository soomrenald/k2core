from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from k2core.backends import ComfyKreaBackend
from k2core.face_detail import DetectedFace
from k2core.regions import PixelBox


class Token:
    cancelled = False

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise InterruptedError


class Runtime:
    loaded = True

    def __init__(self, output: Path) -> None:
        self.output = output
        self.comfyui_root = output.parent
        self.face_detector_path = output.parent / "face-det.onnx"
        self.generate_request = None
        self.edit_request = None
        self.face_request = None

    def generate(self, **request):
        self.generate_request = request
        request["progress"](1, 1, {})
        return {"image_path": str(self.output), "width": 64, "height": 64}

    def edit_image(self, **request):
        self.edit_request = request
        request["progress"](1, 1, {})
        return {"image_path": str(self.output), "width": 64, "height": 64}

    def refine_faces(self, **request):
        self.face_request = request
        return {"image_path": str(self.output), "width": 64, "height": 64}


class ComfyKreaBackendTests(unittest.TestCase):
    def test_face_detection_returns_typed_candidates_without_refining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.new("RGB", (64, 64), "white").save(source)
            output.touch()
            runtime = Runtime(output)
            runtime.face_detector_path.touch()
            backend = ComfyKreaBackend(
                runtime,
                asset_resolver=lambda asset_id: root / asset_id,
                output_directory=root / "results",
            )
            detected = (DetectedFace(PixelBox(4, 5, 30, 40), 0.87),)
            with patch("k2core.backends.comfy_krea.OnnxNanoFaceDetector") as detector:
                detector.return_value.detect.return_value = detected
                detector.return_value.execution_provider = "CPUExecutionProvider"
                result = backend.detect_faces(
                    {
                        "source_asset_id": "source.png",
                        "threshold": 0.35,
                        "provider": "cpu",
                    },
                    progress=lambda *_item: None,
                    cancellation=Token(),
                )

        self.assertEqual(result.faces, detected)
        self.assertEqual(result.metadata["provider"], "CPUExecutionProvider")
        detector.assert_called_once_with(
            runtime.face_detector_path.resolve(),
            threshold=0.35,
            provider="cpu",
        )

    def test_generate_maps_regions_and_resolves_adapter_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.png"
            output.touch()
            runtime = Runtime(output)
            backend = ComfyKreaBackend(
                runtime,
                asset_resolver=lambda asset_id: root / asset_id,
                output_directory=root / "results",
            )
            progress = []
            result = backend.generate_image(
                {
                    "operation": "generate_image",
                    "prompt": "two people",
                    "width": 1024,
                    "height": 1024,
                    "seed": 4,
                    "regions": [
                        {
                            "region_id": "person-1",
                            "name": "Person",
                            "box": [10, 20, 300, 900],
                            "prompt": "red coat",
                        }
                    ],
                    "adapter_routes": [
                        {"adapter_id": "adapter-1", "asset_id": "lora-1", "strength": 0.8}
                    ],
                },
                progress=lambda *item: progress.append(item),
                cancellation=Token(),
            )
        self.assertEqual(result.asset_paths, (output,))
        self.assertEqual(runtime.generate_request["regions"][0].region_id, "person-1")
        self.assertTrue(runtime.generate_request["loras"][0]["path"].endswith("lora-1"))
        self.assertEqual(progress[0][0], "diffusion")

    def test_image_and_confirmed_face_edits_map_to_shared_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.png"
            source = root / "source"
            output.touch()
            source.touch()
            runtime = Runtime(output)
            backend = ComfyKreaBackend(
                runtime,
                asset_resolver=lambda asset_id: root / asset_id,
                output_directory=root / "results",
            )
            backend.edit_frame(
                {
                    "operation": "regional_edit",
                    "source_asset_id": "source",
                    "prompt": "repair hand",
                    "region": {"x0": 10, "y0": 10, "x1": 40, "y1": 50},
                },
                progress=lambda *_item: None,
                cancellation=Token(),
            )
            backend.refine_faces(
                {
                    "operation": "face_refinement",
                    "source_asset_id": "source",
                    "prompt": "stable identity",
                    "region": {"x0": 10, "y0": 10, "x1": 40, "y1": 50},
                    "user_confirmed_face_region": True,
                },
                progress=lambda *_item: None,
                cancellation=Token(),
            )
        self.assertEqual(runtime.edit_request["regions"][0].region_id, "edit-region")
        self.assertEqual(len(runtime.face_request["manual_face_paths"]), 1)


if __name__ == "__main__":
    unittest.main()
