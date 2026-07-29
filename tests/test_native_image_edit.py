from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, PngImagePlugin

from k2core.backends.native import _save_native_edit
from k2core.image_edit import ImageEditState
from k2core.inference import ImageEditRequest
from k2core.regions import PixelBox, RegionDefinition


class NativeImageEditTests(unittest.TestCase):
    def test_zero_strength_is_a_valid_no_op_contract(self) -> None:
        request = ImageEditRequest(
            correlation_id="zero-edit",
            image_path=Path("/tmp/source.png"),
            prompt="retain the source",
            regions=(),
            loras=(),
            denoise=0.0,
            edit_entire_image=True,
        )

        self.assertEqual(request.denoise, 0.0)
        self.assertEqual(ImageEditState(denoise=0.0).denoise, 0.0)

    def test_save_preserves_pixels_outside_regions_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            source = Image.new("RGB", (256, 256), "black")
            source_metadata = PngImagePlugin.PngInfo()
            source_metadata.add_text("artist", "fixture")
            source_metadata.add_text("k2lab_project", '{"source":true}')
            source.save(source_path, pnginfo=source_metadata)
            request = ImageEditRequest(
                correlation_id="regional-edit",
                image_path=source_path,
                output_directory=root,
                prompt="make the center white",
                regions=(
                    RegionDefinition(
                        "center",
                        "Center",
                        PixelBox(80, 80, 176, 176),
                        "white square",
                    ),
                ),
                loras=(),
                seed=7,
                denoise=0.5,
                latent_feather_pixels=8,
                composite_feather_pixels=4,
            )

            payload = _save_native_edit(
                request,
                source_path=source_path,
                source_image=source,
                source_metadata={
                    "artist": "fixture",
                    "k2lab_project": '{"source":true}',
                },
                candidate=Image.new("RGB", source.size, "white"),
                geometry=SimpleNamespace(aligned_width=256, aligned_height=256),
                conditioned_prompt="make the center white",
                target_regions=request.regions,
                lora_reports=(),
                regional_summary={"backend": "fixture", "region_count": 1},
            )

            with Image.open(payload["image_path"]) as output:
                self.assertEqual(output.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(output.getpixel((128, 128)), (255, 255, 255))
                self.assertEqual(output.info["artist"], "fixture")
                self.assertEqual(output.info["k2lab_project"], '{"source":true}')
                edit_summary = json.loads(output.info["image_edit"])
            self.assertEqual(edit_summary["composite_bounds"], [77, 77, 179, 179])
            self.assertEqual(payload["backend"], "native")

    def test_request_project_replaces_stale_source_project_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            source = Image.new("RGB", (256, 256), "black")
            source.save(source_path)
            request = ImageEditRequest(
                correlation_id="project-edit",
                image_path=source_path,
                output_directory=root,
                prompt="make the image white",
                regions=(),
                loras=(),
                edit_entire_image=True,
                project_json={"current": True},
            )

            payload = _save_native_edit(
                request,
                source_path=source_path,
                source_image=source,
                source_metadata={"k2lab_project": '{"stale":true}'},
                candidate=Image.new("RGB", source.size, "white"),
                geometry=SimpleNamespace(aligned_width=256, aligned_height=256),
                conditioned_prompt=request.prompt,
                target_regions=(),
                lora_reports=(),
                regional_summary={"backend": "disabled", "region_count": 0},
            )

            with Image.open(payload["image_path"]) as output:
                self.assertEqual(
                    json.loads(output.info["k2lab_project"]),
                    {"current": True},
                )


if __name__ == "__main__":
    unittest.main()
