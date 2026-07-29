from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from k2core.parity import ImageParityThresholds, compare_image_files


class ParityFixtureTests(unittest.TestCase):
    def test_exact_decoded_pixels_ignore_png_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
            Image.fromarray(pixels).save(reference)
            Image.fromarray(pixels).save(candidate)

            measured = compare_image_files(reference, candidate)

        self.assertTrue(measured.exact)
        self.assertEqual(
            measured.status(
                ImageParityThresholds(
                    minimum_cosine=1.0,
                    maximum_mean_absolute_error=0.0,
                    maximum_rmse=0.0,
                    maximum_absolute_error=0.0,
                    minimum_psnr_db=100.0,
                )
            ),
            "PASS",
        )

    def test_tolerated_difference_has_structured_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            base = np.full((8, 8, 3), 127, dtype=np.uint8)
            changed = base.copy()
            changed[0, 0] = 128
            Image.fromarray(base).save(reference)
            Image.fromarray(changed).save(candidate)

            measured = compare_image_files(reference, candidate)
            thresholds = ImageParityThresholds(
                minimum_cosine=0.999,
                maximum_mean_absolute_error=0.01,
                maximum_rmse=0.01,
                maximum_absolute_error=0.01,
                minimum_psnr_db=40.0,
            )

        self.assertFalse(measured.exact)
        self.assertTrue(measured.passes(thresholds))
        self.assertEqual(measured.status(thresholds), "PASS WITH APPROVED DIFFERENCE")

    def test_dimension_mismatch_fails_without_broadcasting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            Image.new("RGB", (8, 8), "white").save(reference)
            Image.new("RGB", (16, 8), "white").save(candidate)

            measured = compare_image_files(reference, candidate)

        self.assertFalse(measured.passes(ImageParityThresholds(0.0, 1.0, 1.0, 1.0, -1.0)))


if __name__ == "__main__":
    unittest.main()
