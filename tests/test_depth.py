from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from k2core.depth import (
    KREA2_DEPTH_EXPECTED_TARGETS,
    DepthControlSettings,
    DepthFeatureFlags,
    DepthInvalidValuePolicy,
    DepthNormalizationMode,
    DepthNormalizationSettings,
    DepthRegion,
    DepthRegionMode,
    DepthRegionSettings,
    block_average_mask,
    compose_effective_depth_field,
    compose_override_depth,
    depth_histogram,
    depth_preview,
    feathered_box_mask,
    load_depth_image,
    load_blender_depth_bundle,
    normalize_depth,
    inspect_depth_checkpoint,
    resize_depth,
)
import k2core.depth.checkpoint as checkpoint_module
from k2core.regions import PixelBox


def _compatible_checkpoint_shapes(*, rank: int = 64) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {"first.weight": (3072, 128)}
    for block in range(28):
        for target in KREA2_DEPTH_EXPECTED_TARGETS:
            base = f"blocks.{block}.{target}"
            shapes[f"{base}.A"] = (rank, 3072)
            shapes[f"{base}.B"] = (3072, rank)
    return shapes


def test_feature_flags_are_disabled_by_default_and_parse_explicit_truth() -> None:
    assert DepthFeatureFlags.from_environment({}) == DepthFeatureFlags()
    flags = DepthFeatureFlags.from_environment(
        {
            "K2_DEPTH_CONTROL_ENABLED": "true",
            "K2_DEPTH_REGIONS_ENABLED": "1",
            "K2_DEPTH_OVERRIDE_ENABLED": "off",
            "K2_BLENDER_BUNDLE_IMPORT_ENABLED": "yes",
        }
    )
    assert flags.control
    assert flags.regions
    assert not flags.override
    assert flags.blender_bundle_import


def test_depth_checkpoint_layout_accepts_complete_public_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "depth.safetensors"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        checkpoint_module,
        "_safetensors_header",
        lambda _path: ({}, _compatible_checkpoint_shapes()),
    )
    monkeypatch.setattr(checkpoint_module, "_sha256_file", lambda _path: "0" * 64)

    report = inspect_depth_checkpoint(path, expected_sha256=None)

    assert report.compatible
    assert not report.verified
    assert report.checkpoint is not None
    assert report.checkpoint.rank == 64
    assert report.checkpoint.compatible_block_pairs == 28 * 8
    assert report.checkpoint.expanded_projection_key == "first.weight"


def test_depth_checkpoint_layout_rejects_wrong_rank_and_missing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "depth.safetensors"
    path.write_bytes(b"placeholder")
    shapes = _compatible_checkpoint_shapes(rank=32)
    del shapes["blocks.27.mlp.down.B"]
    monkeypatch.setattr(
        checkpoint_module,
        "_safetensors_header",
        lambda _path: ({}, shapes),
    )
    monkeypatch.setattr(checkpoint_module, "_sha256_file", lambda _path: "0" * 64)

    report = inspect_depth_checkpoint(path, expected_sha256=None)

    assert not report.compatible
    assert any("rank 64" in error for error in report.errors)
    assert any("missing 1" in error for error in report.errors)


def test_disabled_depth_payload_requires_no_paths_and_round_trips() -> None:
    settings = DepthControlSettings.from_payload({})
    assert not settings.enabled
    assert settings.checkpoint is None
    assert settings.depth_image is None
    assert settings.to_payload()["enabled"] is False


def test_enabled_depth_payload_requires_checkpoint_and_image() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        DepthControlSettings.from_payload({"enabled": True})
    with pytest.raises(ValueError, match="depth image"):
        DepthControlSettings.from_payload({"enabled": True, "checkpoint": "depth.safetensors"})


def test_depth_region_mode_strength_contracts() -> None:
    assert DepthRegionSettings("a", DepthRegionMode.IGNORE).multiplier == 0.0
    assert DepthRegionSettings("a", DepthRegionMode.INHERIT).multiplier == 1.0
    assert DepthRegionSettings("a", DepthRegionMode.EMPHASIZE, 1.5).multiplier == 1.5
    assert DepthRegionSettings("a", DepthRegionMode.RELAX, 0.25).multiplier == 0.25
    with pytest.raises(ValueError, match="at least 1"):
        DepthRegionSettings("a", DepthRegionMode.EMPHASIZE, 0.9)
    with pytest.raises(ValueError, match="must not exceed 1"):
        DepthRegionSettings("a", DepthRegionMode.RELAX, 1.1)
    with pytest.raises(ValueError, match="override image"):
        DepthRegionSettings("a", DepthRegionMode.OVERRIDE)


def test_loads_8_bit_png_and_builds_preview(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(source, mode="L").save(path)

    depth = load_depth_image(path)

    assert depth.info.format == "PNG"
    assert depth.info.bit_depth == 8
    assert depth.info.dtype == "uint8"
    assert depth.info.minimum == 0
    assert depth.info.maximum == 255
    assert np.array_equal(depth.values, source)
    assert np.array_equal(np.asarray(depth_preview(depth)), source)


def test_loads_verified_blender_depth_bundle(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth_16bit.png"
    Image.fromarray(np.array([[0, 1024], [32000, 65535]], dtype=np.uint16)).save(depth_path)
    checksum = hashlib.sha256(depth_path.read_bytes()).hexdigest()
    (tmp_path / "camera.json").write_text(
        json.dumps(
            {
                "resolution": [2, 2],
                "depth_convention": "near_white_far_black",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "objects.json").write_text(
        json.dumps({"objects": [{"name": "mannequin_001"}]}),
        encoding="utf-8",
    )
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "format": "k2lab-blender-depth-bundle",
                "version": 1,
                "depth_image": depth_path.name,
                "checksums": {depth_path.name: checksum},
            }
        ),
        encoding="utf-8",
    )

    bundle = load_blender_depth_bundle(tmp_path)

    assert bundle.depth.info.bit_depth == 16
    assert bundle.camera["depth_convention"] == "near_white_far_black"
    assert bundle.objects[0]["name"] == "mannequin_001"

    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "format": "k2lab-blender-depth-bundle",
                "version": 1,
                "depth_image": depth_path.name,
                "checksums": {depth_path.name: "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        load_blender_depth_bundle(tmp_path)


def test_loads_k2pose_blender_depth_bundle_format(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth_16bit.png"
    Image.fromarray(np.array([[0, 1024], [32000, 65535]], dtype=np.uint16)).save(depth_path)
    checksum = hashlib.sha256(depth_path.read_bytes()).hexdigest()
    (tmp_path / "camera.json").write_text(
        json.dumps(
            {
                "resolution": [2, 2],
                "depth_convention": "near_white_far_black",
                "origin": "top_left",
                "vertical_flip": False,
            }
        )
    )
    (tmp_path / "objects.json").write_text(json.dumps({"objects": []}))
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "format": "k2pose-blender-depth-bundle",
                "version": 1,
                "depth_image": depth_path.name,
                "checksums": {depth_path.name: checksum},
            }
        )
    )

    assert load_blender_depth_bundle(tmp_path).depth.info.bit_depth == 16


def test_loads_16_bit_png_without_truncation(tmp_path: Path) -> None:
    path = tmp_path / "depth16.png"
    source = np.array([[0, 257], [32768, 65535]], dtype=np.uint16)
    Image.fromarray(source).save(path)

    depth = load_depth_image(path)

    assert depth.info.format == "PNG"
    assert depth.info.bit_depth == 16
    assert depth.info.dtype == "uint16"
    assert np.array_equal(depth.values, source)


def test_loads_16_bit_tiff_without_truncation(tmp_path: Path) -> None:
    path = tmp_path / "depth16.tiff"
    source = np.array([[11, 1024], [4096, 64000]], dtype=np.uint16)
    Image.fromarray(source).save(path, format="TIFF")

    depth = load_depth_image(path)

    assert depth.info.format == "TIFF"
    assert depth.info.bit_depth == 16
    assert np.array_equal(depth.values, source)


def test_rejects_rgb_and_alpha_depth_images(tmp_path: Path) -> None:
    for mode in ("RGB", "RGBA"):
        path = tmp_path / f"color-{mode}.png"
        Image.new(mode, (4, 4)).save(path)
        with pytest.raises(ValueError, match="single-channel grayscale"):
            load_depth_image(path)


def test_minmax_normalization_and_inversion_are_explicit() -> None:
    source = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    normal = normalize_depth(
        source,
        DepthNormalizationSettings(mode=DepthNormalizationMode.MINMAX),
    )
    inverted = normalize_depth(
        source,
        DepthNormalizationSettings(
            mode=DepthNormalizationMode.MINMAX,
            invert=True,
        ),
    )
    assert np.allclose(normal.values, [[0.0, 1 / 3], [2 / 3, 1.0]])
    assert np.allclose(inverted.values, 1.0 - normal.values)


def test_none_normalization_respects_integer_bit_depth(tmp_path: Path) -> None:
    path = tmp_path / "depth16.png"
    source = np.array([[0, 16384], [32768, 65535]], dtype=np.uint16)
    Image.fromarray(source).save(path)
    depth = load_depth_image(path)

    normalized = normalize_depth(
        depth,
        DepthNormalizationSettings(mode=DepthNormalizationMode.NONE),
    )

    assert normalized.values[0, 0] == 0.0
    assert normalized.values[-1, -1] == 1.0
    assert normalized.values[1, 0] == pytest.approx(32768 / 65535)


def test_percentile_normalization_clamps_and_applies_gamma() -> None:
    source = np.arange(100, dtype=np.float32).reshape(10, 10)
    normalized = normalize_depth(
        source,
        DepthNormalizationSettings(
            mode=DepthNormalizationMode.PERCENTILE,
            near_percentile=10,
            far_percentile=90,
            gamma=2,
        ),
    )
    assert normalized.values.min() == 0.0
    assert normalized.values.max() == 1.0
    midpoint = (49 - 9.9) / (89.1 - 9.9)
    assert normalized.values[4, 9] == pytest.approx(midpoint**2)


def test_camera_and_checkpoint_ranges_require_explicit_bounds() -> None:
    with pytest.raises(ValueError, match="near < far"):
        DepthNormalizationSettings(mode=DepthNormalizationMode.CAMERA_RANGE)
    with pytest.raises(ValueError, match="minimum < maximum"):
        DepthNormalizationSettings(mode=DepthNormalizationMode.CHECKPOINT_REFERENCE)
    source = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    normalized = normalize_depth(
        source,
        DepthNormalizationSettings(
            mode=DepthNormalizationMode.CAMERA_RANGE,
            camera_near=1.0,
            camera_far=5.0,
        ),
    )
    assert np.allclose(normalized.values, [[0.0, 0.25], [0.5, 1.0]])


def test_invalid_depth_policy_is_deterministic() -> None:
    source = np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
    far = normalize_depth(
        source,
        DepthNormalizationSettings(
            mode=DepthNormalizationMode.MINMAX,
            invalid_value_policy=DepthInvalidValuePolicy.FAR,
        ),
    )
    near = normalize_depth(
        source,
        DepthNormalizationSettings(
            mode=DepthNormalizationMode.MINMAX,
            invalid_value_policy=DepthInvalidValuePolicy.NEAR,
        ),
    )
    assert np.array_equal(far.values[1], [0.0, 0.0])
    assert np.array_equal(near.values[1], [1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        normalize_depth(
            source,
            DepthNormalizationSettings(
                mode=DepthNormalizationMode.MINMAX,
                invalid_value_policy=DepthInvalidValuePolicy.ERROR,
            ),
        )


def test_constant_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="fully constant"):
        normalize_depth(
            np.ones((4, 4), dtype=np.uint16),
            DepthNormalizationSettings(mode=DepthNormalizationMode.MINMAX),
        )


def test_resize_depth_is_deterministic_and_preserves_range() -> None:
    source = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    first = resize_depth(source, 8, 4)
    second = resize_depth(source, 8, 4)
    assert first.shape == (4, 8)
    assert np.array_equal(first, second)
    assert 0.0 <= first.min() <= first.max() <= 1.0


def test_feathered_mask_has_no_rectangular_outside_step() -> None:
    mask = feathered_box_mask(12, 8, PixelBox(4, 2, 8, 6), 4)
    assert mask[3, 5] == 1.0
    assert 0.0 < mask[3, 2] < 1.0
    assert 0.0 < mask[3, 0] < mask[3, 2]
    hard = feathered_box_mask(12, 8, PixelBox(4, 2, 8, 6), 0)
    assert set(np.unique(hard)) == {0.0, 1.0}


def test_block_average_maps_pixels_to_image_tokens() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:4, :4] = 1.0
    tokens = block_average_mask(mask, 4)
    assert np.array_equal(tokens, [[1.0, 0.0], [0.0, 0.0]])


def test_regional_depth_priority_and_ignore_are_deterministic() -> None:
    low = DepthRegionSettings("low", DepthRegionMode.EMPHASIZE, 2.0)
    high = DepthRegionSettings("high", DepthRegionMode.IGNORE)
    settings = DepthControlSettings(
        global_strength=1.0,
        feather_pixels=0,
        regions=(low, high),
    )
    field = compose_effective_depth_field(
        settings,
        (
            DepthRegion(low, PixelBox(0, 0, 16, 16), priority=1),
            DepthRegion(high, PixelBox(0, 0, 16, 16), priority=10),
        ),
        width=32,
        height=16,
    )
    assert np.all(field.pixel_values[:, :16] == 0.0)
    assert np.all(field.pixel_values[:, 16:] == 1.0)
    assert np.array_equal(field.image_token_values, [[0.0, 1.0]])


def test_override_depth_blends_through_feathered_priority_masks() -> None:
    low = DepthRegionSettings(
        "low",
        DepthRegionMode.OVERRIDE,
        override_image=Path("low.png"),
    )
    high = DepthRegionSettings(
        "high",
        DepthRegionMode.OVERRIDE,
        override_image=Path("high.png"),
    )
    regions = (
        DepthRegion(low, PixelBox(0, 0, 8, 8), priority=0),
        DepthRegion(high, PixelBox(2, 2, 6, 6), priority=10),
    )
    result = compose_override_depth(
        np.full((8, 8), 0.5, dtype=np.float32),
        regions,
        {
            "low": np.zeros((8, 8), dtype=np.float32),
            "high": np.ones((8, 8), dtype=np.float32),
        },
        feather_pixels=0,
    )
    assert result[0, 0] == 0.0
    assert result[3, 3] == 1.0
    assert not result.flags.writeable


def test_regional_depth_weighted_overlap_and_clamping() -> None:
    emphasize = DepthRegionSettings("a", DepthRegionMode.EMPHASIZE, 3.0)
    settings = DepthControlSettings(
        global_strength=1.0,
        feather_pixels=4,
        regions=(emphasize,),
    )
    field = compose_effective_depth_field(
        settings,
        (DepthRegion(emphasize, PixelBox(4, 0, 12, 16), priority=1),),
        width=16,
        height=16,
    )
    assert field.pixel_values.max() == 3.0
    assert 1.0 < field.pixel_values[:, 2].mean() < 3.0


def test_global_and_regional_schedules_do_not_add_forwards_or_hard_edges() -> None:
    relax = DepthRegionSettings(
        "a",
        DepthRegionMode.RELAX,
        0.25,
        start_percent=0.25,
        end_percent=0.75,
    )
    settings = DepthControlSettings(
        global_strength=1.0,
        start_percent=0.1,
        end_percent=0.9,
        feather_pixels=2,
        regions=(relax,),
    )
    geometry = (DepthRegion(relax, PixelBox(0, 0, 16, 16), priority=1),)
    before = compose_effective_depth_field(settings, geometry, width=16, height=16, progress=0.0)
    early = compose_effective_depth_field(settings, geometry, width=16, height=16, progress=0.2)
    active = compose_effective_depth_field(settings, geometry, width=16, height=16, progress=0.5)
    assert np.all(before.pixel_values == 0.0)
    assert np.all(early.pixel_values == 1.0)
    assert np.all(active.pixel_values == 0.25)


def test_depth_histogram_records_all_pixels() -> None:
    histogram = depth_histogram(np.arange(16, dtype=np.uint16).reshape(4, 4), bins=4)
    assert sum(histogram["counts"]) == 16
    assert len(histogram["edges"]) == 5
