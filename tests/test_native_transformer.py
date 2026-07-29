from __future__ import annotations

import json
import unittest

from k2core.backends.native_transformer import (
    KREA2_CONDITIONING_DIM,
    KREA2_QUANTIZED_LINEAR_COUNT,
    _map_linear_module,
    _map_transformer_key,
    _transformer_state_dict,
    _validate_quantization_metadata,
)
from k2core.inference import WeightMappingError


class NativeTransformerMappingTests(unittest.TestCase):
    def test_reviewed_dimensions_and_linear_names(self) -> None:
        self.assertEqual(KREA2_CONDITIONING_DIM, 30_720)
        self.assertEqual(KREA2_QUANTIZED_LINEAR_COUNT, 256)
        self.assertEqual(
            _map_linear_module("blocks.27.attn.wo"),
            "transformer_blocks.27.attn.to_out.0",
        )
        self.assertEqual(
            _map_linear_module("txtfusion.refiner_blocks.1.mlp.down"),
            "text_fusion.refiner_blocks.1.ff.down",
        )

    def test_norm_modulation_and_standalone_parameters_map_explicitly(self) -> None:
        expected = {
            "blocks.3.mod.lin": "transformer_blocks.3.scale_shift_table",
            "blocks.3.prenorm.scale": "transformer_blocks.3.norm1.weight",
            "blocks.3.attn.qknorm.knorm.scale": (
                "transformer_blocks.3.attn.norm_k.weight"
            ),
            "txtfusion.layerwise_blocks.0.postnorm.scale": (
                "text_fusion.layerwise_blocks.0.norm2.weight"
            ),
            "txtmlp.0.scale": "txt_in.norm.weight",
            "last.modulation.lin": "final_layer.scale_shift_table",
            "first.bias": "img_in.bias",
        }
        self.assertEqual(
            {source: _map_transformer_key(source) for source in expected},
            expected,
        )

    def test_complete_mapping_rejects_unknown_and_duplicate_keys(self) -> None:
        state = _transformer_state_dict(
            {
                "first.weight": object(),
                "first.bias": object(),
                "last.norm.scale": object(),
            }
        )
        self.assertEqual(
            set(state),
            {"img_in.weight", "img_in.bias", "final_layer.norm.weight"},
        )
        with self.assertRaises(WeightMappingError):
            _transformer_state_dict({"unknown.weight": object()})

    def test_quantization_metadata_must_match_all_scale_tensors(self) -> None:
        layers = {
            f"blocks.{index // 8}.{(
                'attn.wq',
                'attn.wk',
                'attn.wv',
                'attn.wo',
                'attn.gate',
                'mlp.gate',
                'mlp.up',
                'mlp.down',
            )[index % 8]}": {"format": "float8_e4m3fn"}
            for index in range(28 * 8)
        }
        for group in ("layerwise_blocks", "refiner_blocks"):
            for block in range(2):
                for name in (
                    "attn.wq",
                    "attn.wk",
                    "attn.wv",
                    "attn.wo",
                    "attn.gate",
                    "mlp.gate",
                    "mlp.up",
                    "mlp.down",
                ):
                    layers[f"txtfusion.{group}.{block}.{name}"] = {
                        "format": "float8_e4m3fn"
                    }
        tensors = {f"{name}.weight_scale": object() for name in layers}
        mapped = _validate_quantization_metadata(
            {"_quantization_metadata": json.dumps({"layers": layers})},
            tensors,
        )
        self.assertEqual(len(mapped), 256)

        tensors.pop(next(iter(tensors)))
        with self.assertRaises(WeightMappingError):
            _validate_quantization_metadata(
                {"_quantization_metadata": json.dumps({"layers": layers})},
                tensors,
            )


if __name__ == "__main__":
    unittest.main()
