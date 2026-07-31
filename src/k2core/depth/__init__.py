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
from k2core.depth.blender_bundle import (
    BlenderDepthBundle,
    load_blender_depth_bundle,
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
    compose_override_depth,
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
from k2core.depth.adapter import (
    DEPTH_ADAPTER_FORMAT,
    DEPTH_ATTACHMENT_KEY,
    DepthAdapterRuntimeReport,
    DepthControlLatent,
    KreaDepthAdapterError,
    KreaDepthInputProjection,
    attach_depth_control,
    clear_depth_control,
    depth_adapter_runtime_report,
    encode_depth_control,
    install_depth_adapter,
    process_depth_latent_for_model,
)
from k2core.depth.runtime import (
    DepthControlPreparation,
    DepthScheduleController,
    prepare_depth_control,
)

__all__ = [
    "KREA2_DEPTH_EXPECTED_BLOCKS",
    "KREA2_DEPTH_EXPECTED_RANK",
    "KREA2_DEPTH_EXPECTED_TARGETS",
    "KREA2_DEPTH_PUBLIC_SHA256",
    "DepthCheckpointCompatibility",
    "DepthCheckpointInfo",
    "BlenderDepthBundle",
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
    "DEPTH_ADAPTER_FORMAT",
    "DEPTH_ATTACHMENT_KEY",
    "DepthAdapterRuntimeReport",
    "DepthControlLatent",
    "DepthControlPreparation",
    "DepthScheduleController",
    "KreaDepthAdapterError",
    "KreaDepthInputProjection",
    "attach_depth_control",
    "block_average_mask",
    "compose_effective_depth_field",
    "compose_override_depth",
    "depth_histogram",
    "depth_adapter_runtime_report",
    "depth_preview",
    "depth_summary",
    "feathered_box_mask",
    "inspect_depth_checkpoint",
    "install_depth_adapter",
    "lora_pairs",
    "load_blender_depth_bundle",
    "load_depth_image",
    "normalize_depth",
    "prepare_depth_control",
    "process_depth_latent_for_model",
    "encode_depth_control",
    "clear_depth_control",
    "resize_depth",
    "resize_mask",
]
