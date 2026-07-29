from __future__ import annotations

import unittest

from k2core.backends.native_lora import (
    _group_lora_tensors,
    _group_lokr_tensors,
    _map_lora_module,
)
from k2core.inference import WeightMappingError


class NativeLoraMappingTests(unittest.TestCase):
    def test_maps_current_krea_linear_names(self) -> None:
        expected = {
            "diffusion_model.blocks.0.attn.wq": ("transformer_blocks.0.attn.to_q"),
            "blocks.27.mlp.down": "transformer_blocks.27.ff.down",
            "diffusion_model.txtfusion.refiner_blocks.1.attn.wo": (
                "text_fusion.refiner_blocks.1.attn.to_out.0"
            ),
            "diffusion_model.first": "img_in",
        }
        self.assertEqual(
            {source: _map_lora_module(source) for source in expected},
            expected,
        )

    def test_groups_both_standard_pair_conventions_and_alpha(self) -> None:
        tensors = {
            "blocks.0.attn.wq.lora_A.weight": "a",
            "blocks.0.attn.wq.lora_B.weight": "b",
            "blocks.0.attn.wq.alpha": "alpha",
            "blocks.0.attn.wk.lora_down.weight": "down",
            "blocks.0.attn.wk.lora_up.weight": "up",
        }
        grouped = _group_lora_tensors(tensors)
        self.assertEqual(
            grouped["blocks.0.attn.wq"],
            {
                ".lora_A.weight": "a",
                ".lora_B.weight": "b",
                ".alpha": "alpha",
            },
        )
        self.assertEqual(
            grouped["blocks.0.attn.wk"],
            {
                ".lora_down.weight": "down",
                ".lora_up.weight": "up",
            },
        )

    def test_rejects_partial_unknown_and_bare_parameter_targets(self) -> None:
        with self.assertRaisesRegex(WeightMappingError, "complete pair"):
            _group_lora_tensors({"blocks.0.attn.wq.lora_A.weight": object()})
        with self.assertRaisesRegex(WeightMappingError, "unsupported"):
            _group_lora_tensors(
                {
                    "blocks.0.attn.wq.lora_A.weight": object(),
                    "blocks.0.attn.wq.lora_B.weight": object(),
                    "blocks.0.attn.wq.dora_scale": object(),
                }
            )
        with self.assertRaisesRegex(WeightMappingError, "bare"):
            _map_lora_module("diffusion_model.last.modulation.lin")

    def test_direct_lokr_requires_both_factors_and_rejects_decomposition(self) -> None:
        grouped = _group_lokr_tensors(
            {
                "blocks.0.attn.wq.lokr_w1": "w1",
                "blocks.0.attn.wq.lokr_w2": "w2",
                "blocks.0.attn.wq.alpha": "sentinel",
            }
        )
        self.assertEqual(
            grouped["blocks.0.attn.wq"],
            {
                ".lokr_w1": "w1",
                ".lokr_w2": "w2",
                ".alpha": "sentinel",
            },
        )
        with self.assertRaisesRegex(WeightMappingError, "incomplete"):
            _group_lokr_tensors({"blocks.0.attn.wq.lokr_w1": object()})
        with self.assertRaisesRegex(WeightMappingError, "decomposed"):
            _group_lokr_tensors(
                {
                    "blocks.0.attn.wq.lokr_w1": object(),
                    "blocks.0.attn.wq.lokr_w2": object(),
                    "blocks.0.attn.wq.lokr_w2_a": object(),
                }
            )


if __name__ == "__main__":
    unittest.main()
