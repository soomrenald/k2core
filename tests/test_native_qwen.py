from __future__ import annotations

import json
import unittest

from k2core.backends.native_qwen import (
    KREA2_QWEN_TAP_LAYERS,
    KREA2_TEXT_FEATURE_SIZE,
    _text_state_dict,
    _validate_quantization_markers,
    _validate_state_dict_shapes,
)
from k2core.inference import WeightMappingError


class FakeMarker:
    def __init__(self, payload: dict[str, object]) -> None:
        self.encoded = tuple(json.dumps(payload, separators=(",", ":")).encode())

    def tolist(self) -> list[int]:
        return list(self.encoded)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class NativeQwenTests(unittest.TestCase):
    def test_krea_taps_produce_reviewed_feature_width(self) -> None:
        self.assertEqual(
            KREA2_QWEN_TAP_LAYERS,
            (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        )
        self.assertEqual(KREA2_TEXT_FEATURE_SIZE, 30_720)

    def test_text_state_mapping_excludes_reviewed_vision_tower_and_markers(self) -> None:
        tensors = {
            "model.embed_tokens.weight": object(),
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.layers.0.self_attn.q_proj.weight_scale": object(),
            "model.layers.0.self_attn.q_proj.comfy_quant": object(),
            "model.visual.patch_embed.weight": object(),
        }

        state, ignored = _text_state_dict(tensors)

        self.assertEqual(
            set(state),
            {
                "embed_tokens.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight_scale",
            },
        )
        self.assertEqual(ignored, 1)

    def test_quantization_contract_requires_matching_scale_and_known_marker(self) -> None:
        marker = FakeMarker(
            {
                "format": "float8_e4m3fn",
                "full_precision_matrix_mult": False,
            }
        )
        tensors = {
            "model.layers.0.mlp.up_proj.comfy_quant": marker,
            "model.layers.0.mlp.up_proj.weight_scale": object(),
        }
        self.assertEqual(
            _validate_quantization_markers(tensors),
            frozenset({"layers.0.mlp.up_proj"}),
        )

        with self.assertRaises(WeightMappingError):
            _validate_quantization_markers(
                {"model.layers.0.mlp.up_proj.comfy_quant": marker}
            )

    def test_shape_validation_rejects_non_strict_mapping(self) -> None:
        _validate_state_dict_shapes(
            {"weight": FakeTensor((2, 3))},
            {"weight": FakeTensor((2, 3))},
        )
        with self.assertRaises(WeightMappingError):
            _validate_state_dict_shapes(
                {"weight": FakeTensor((2, 3))},
                {"weight": FakeTensor((3, 2))},
            )


if __name__ == "__main__":
    unittest.main()
