"""Deterministic visual diagnostics for compiled regional prompt plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k2core.model import sha256_file
from k2core.regional_prompting import (
    BoundRegionalPromptPlan,
    RegionalPromptPlan,
)


_COLORS = (
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (217, 119, 6),
    (147, 51, 234),
    (8, 145, 178),
)


@dataclass(frozen=True, slots=True)
class RegionalDiagnosticArtifacts:
    mask_preview: Path
    latent_mask_preview: Path
    token_assignment_preview: Path
    overlap_preview: Path
    summary: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        paths = {
            "mask_preview": self.mask_preview,
            "latent_mask_preview": self.latent_mask_preview,
            "token_assignment_preview": self.token_assignment_preview,
            "overlap_preview": self.overlap_preview,
        }
        return {
            **{
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "summary": self.summary,
        }


def render_regional_diagnostics(
    plan: RegionalPromptPlan,
    bound_plan: BoundRegionalPromptPlan,
    output_directory: Path,
    *,
    prefix: str = "regional",
) -> RegionalDiagnosticArtifacts:
    """Render masks and token ownership without changing inference state."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("regional diagnostics require Pillow") from error
    if len(plan.regions) != len(bound_plan.spans):
        raise ValueError("compiled and bound regional plans do not align")

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", prefix).strip(".-")
    if not safe_prefix:
        raise ValueError("regional diagnostic prefix must not be empty")

    grid_size = (plan.image_token_width, plan.image_token_height)
    mask_grid = Image.new("RGBA", grid_size, (255, 255, 255, 255))
    overlap_grid = Image.new("RGB", grid_size, (255, 255, 255))
    overlap_counts = [0] * plan.image_token_count
    for region_index, region in enumerate(plan.regions):
        color = _COLORS[region_index % len(_COLORS)]
        layer = Image.new("RGBA", grid_size, (0, 0, 0, 0))
        pixels = layer.load()
        for token_index, weight in enumerate(region.image_token_field):
            row, column = divmod(token_index, plan.image_token_width)
            pixels[column, row] = (*color, round(180 * weight))
        mask_grid = Image.alpha_composite(mask_grid, layer)
        for token_index, weight in enumerate(region.image_token_mask):
            if weight > 0.0:
                overlap_counts[token_index] += 1

    overlap_pixels = overlap_grid.load()
    owners = [0] * plan.image_token_count
    for region_index, region in enumerate(plan.regions, start=1):
        for token_index, weight in enumerate(region.image_token_mask):
            if weight > 0.0 and owners[token_index] == 0:
                owners[token_index] = region_index
    for token_index, count in enumerate(overlap_counts):
        row, column = divmod(token_index, plan.image_token_width)
        if count > 1:
            overlap_pixels[column, row] = (236, 72, 153)
        elif owners[token_index]:
            overlap_pixels[column, row] = _COLORS[(owners[token_index] - 1) % len(_COLORS)]

    mask_preview = output_directory / f"{safe_prefix}-mask-preview.png"
    latent_preview = output_directory / f"{safe_prefix}-latent-mask-preview.png"
    overlap_preview = output_directory / f"{safe_prefix}-overlap-preview.png"
    mask_grid.resize((plan.width, plan.height), Image.Resampling.NEAREST).convert("RGB").save(
        mask_preview
    )
    latent_scale = max(
        4,
        min(
            8,
            512 // max(plan.image_token_width, plan.image_token_height),
        ),
    )
    latent_image = mask_grid.convert("RGB").resize(
        (
            plan.image_token_width * latent_scale,
            plan.image_token_height * latent_scale,
        ),
        Image.Resampling.NEAREST,
    )
    latent_draw = ImageDraw.Draw(latent_image)
    for x in range(0, latent_image.width, latent_scale):
        latent_draw.line((x, 0, x, latent_image.height), fill=(64, 64, 64))
    for y in range(0, latent_image.height, latent_scale):
        latent_draw.line((0, y, latent_image.width, y), fill=(64, 64, 64))
    latent_image.save(latent_preview)
    overlap_grid.resize((plan.width, plan.height), Image.Resampling.NEAREST).save(overlap_preview)

    token_height = 80 + 44 * max(1, len(bound_plan.spans))
    token_image = Image.new("RGB", (1024, token_height), "white")
    draw = ImageDraw.Draw(token_image)
    draw.text(
        (20, 12),
        f"Text tokens: {bound_plan.text_token_count}; image tokens: {bound_plan.image_token_count}",
        fill="black",
    )
    bar_left, bar_right = 20, 1004
    draw.rectangle((bar_left, 38, bar_right, 58), outline=(80, 80, 80))
    for index, span in enumerate(bound_plan.spans):
        color = _COLORS[index % len(_COLORS)]
        x0 = bar_left + round((bar_right - bar_left) * span.start / bound_plan.text_token_count)
        x1 = bar_left + round((bar_right - bar_left) * span.end / bound_plan.text_token_count)
        draw.rectangle((x0, 39, max(x0 + 1, x1), 57), fill=color)
        draw.text(
            (20, 72 + 44 * index),
            f"{span.name} [{span.region_id}] tokens "
            f"{span.start}:{span.end}; role={span.spatial_role}",
            fill=color,
        )
    token_preview = output_directory / f"{safe_prefix}-token-assignment.png"
    token_image.save(token_preview)

    summary = {
        "image_token_grid": [
            plan.image_token_width,
            plan.image_token_height,
        ],
        "text_token_count": bound_plan.text_token_count,
        "region_count": len(plan.regions),
        "overlap_token_count": sum(count > 1 for count in overlap_counts),
        "regions": [
            {
                "id": region.region_id,
                "name": region.name,
                "text_token_span": [span.start, span.end],
                "hard_mask_token_count": sum(weight > 0.0 for weight in region.image_token_mask),
                "soft_field_sum": sum(region.image_token_field),
            }
            for region, span in zip(
                plan.regions,
                bound_plan.spans,
                strict=True,
            )
        ],
    }
    return RegionalDiagnosticArtifacts(
        mask_preview=mask_preview,
        latent_mask_preview=latent_preview,
        token_assignment_preview=token_preview,
        overlap_preview=overlap_preview,
        summary=summary,
    )


__all__ = [
    "RegionalDiagnosticArtifacts",
    "render_regional_diagnostics",
]
