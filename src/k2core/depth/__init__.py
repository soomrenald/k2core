"""Backend-neutral depth-control contracts and preprocessing."""

from k2core.depth.checkpoint import (
    KREA2_DEPTH_EXPECTED_BLOCKS,
    KREA2_DEPTH_EXPECTED_RANK,
    KREA2_DEPTH_EXPECTED_TARGETS,
    KREA2_DEPTH_PUBLIC_SHA256,
    DepthCheckpointCompatibility,
    DepthCheckpointInfo,
    inspect_depth_checkpoint,
    lora_pairs,
)
from k2core.depth.config import (
    DepthControlSettings,
    DepthFeatureFlags,
    DepthNormalizationSettings,
    DepthRegionSettings,
)
from k2core.depth.diagnostics import depth_histogram, depth_summary
from k2core.depth.loader import depth_preview, load_depth_image
from k2core.depth.masks import (
    block_average_mask,
    feathered_box_mask,
    resize_depth,
    resize_mask,
)
from k2core.depth.preprocess import normalize_depth
from k2core.depth.regional import (
    DepthRegion,
    EffectiveDepthField,
    compose_effective_depth_field,
)
from k2core.depth.types import (
    DepthImage,
    DepthImageInfo,
    DepthInvalidValuePolicy,
    DepthNormalizationMode,
    DepthPreprocessReport,
    DepthRegionMode,
    NormalizedDepth,
)

__all__ = [
    "KREA2_DEPTH_EXPECTED_BLOCKS",
    "KREA2_DEPTH_EXPECTED_RANK",
    "KREA2_DEPTH_EXPECTED_TARGETS",
    "KREA2_DEPTH_PUBLIC_SHA256",
    "DepthCheckpointCompatibility",
    "DepthCheckpointInfo",
    "DepthControlSettings",
    "DepthFeatureFlags",
    "DepthImage",
    "DepthImageInfo",
    "DepthInvalidValuePolicy",
    "DepthNormalizationMode",
    "DepthNormalizationSettings",
    "DepthPreprocessReport",
    "DepthRegion",
    "DepthRegionMode",
    "DepthRegionSettings",
    "EffectiveDepthField",
    "NormalizedDepth",
    "block_average_mask",
    "compose_effective_depth_field",
    "depth_histogram",
    "depth_preview",
    "depth_summary",
    "feathered_box_mask",
    "inspect_depth_checkpoint",
    "lora_pairs",
    "load_depth_image",
    "normalize_depth",
    "resize_depth",
    "resize_mask",
]
