"""Image comparison primitives for versioned native-backend fixtures."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageParityThresholds:
    minimum_cosine: float
    maximum_mean_absolute_error: float
    maximum_rmse: float
    maximum_absolute_error: float
    minimum_psnr_db: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("minimum cosine must be between zero and one")
        for name in (
            "maximum_mean_absolute_error",
            "maximum_rmse",
            "maximum_absolute_error",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class ImageParityMeasurements:
    reference_size: tuple[int, int]
    candidate_size: tuple[int, int]
    reference_mode: str
    candidate_mode: str
    reference_pixel_sha256: str
    candidate_pixel_sha256: str
    cosine: float
    mean_absolute_error: float
    rmse: float
    maximum_absolute_error: float
    psnr_db: float
    exact_pixel_fraction: float

    @property
    def exact(self) -> bool:
        return (
            self.reference_size == self.candidate_size
            and self.reference_mode == self.candidate_mode
            and self.reference_pixel_sha256 == self.candidate_pixel_sha256
        )

    def passes(self, thresholds: ImageParityThresholds) -> bool:
        return (
            self.reference_size == self.candidate_size
            and self.reference_mode == self.candidate_mode
            and self.cosine >= thresholds.minimum_cosine
            and self.mean_absolute_error <= thresholds.maximum_mean_absolute_error
            and self.rmse <= thresholds.maximum_rmse
            and self.maximum_absolute_error <= thresholds.maximum_absolute_error
            and self.psnr_db >= thresholds.minimum_psnr_db
        )

    def status(self, thresholds: ImageParityThresholds) -> str:
        if self.exact:
            return "PASS"
        if self.passes(thresholds):
            return "PASS WITH APPROVED DIFFERENCE"
        return "FAIL"

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_size"] = list(self.reference_size)
        payload["candidate_size"] = list(self.candidate_size)
        return payload


def compare_image_files(
    reference_path: Path,
    candidate_path: Path,
) -> ImageParityMeasurements:
    """Compare decoded RGB pixels, excluding volatile container metadata."""

    with Image.open(reference_path) as reference_image:
        reference_mode = reference_image.mode
        reference_size = reference_image.size
        reference_pixels = np.asarray(reference_image.convert("RGB"))
    with Image.open(candidate_path) as candidate_image:
        candidate_mode = candidate_image.mode
        candidate_size = candidate_image.size
        candidate_pixels = np.asarray(candidate_image.convert("RGB"))

    if reference_pixels.shape != candidate_pixels.shape:
        return ImageParityMeasurements(
            reference_size=reference_size,
            candidate_size=candidate_size,
            reference_mode=reference_mode,
            candidate_mode=candidate_mode,
            reference_pixel_sha256=_pixel_sha256(reference_pixels),
            candidate_pixel_sha256=_pixel_sha256(candidate_pixels),
            cosine=0.0,
            mean_absolute_error=math.inf,
            rmse=math.inf,
            maximum_absolute_error=math.inf,
            psnr_db=-math.inf,
            exact_pixel_fraction=0.0,
        )

    reference = reference_pixels.astype(np.float64) / 255.0
    candidate = candidate_pixels.astype(np.float64) / 255.0
    difference = candidate - reference
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference.ravel()))
    candidate_norm = float(np.linalg.norm(candidate.ravel()))
    denominator = reference_norm * candidate_norm
    cosine = (
        float(np.dot(reference.ravel(), candidate.ravel()) / denominator)
        if denominator
        else float(np.array_equal(reference_pixels, candidate_pixels))
    )
    return ImageParityMeasurements(
        reference_size=reference_size,
        candidate_size=candidate_size,
        reference_mode=reference_mode,
        candidate_mode=candidate_mode,
        reference_pixel_sha256=_pixel_sha256(reference_pixels),
        candidate_pixel_sha256=_pixel_sha256(candidate_pixels),
        cosine=cosine,
        mean_absolute_error=float(absolute.mean()),
        rmse=rmse,
        maximum_absolute_error=float(absolute.max()),
        psnr_db=math.inf if rmse == 0.0 else float(20.0 * math.log10(1.0 / rmse)),
        exact_pixel_fraction=float(np.mean(reference_pixels == candidate_pixels)),
    )


def _pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
