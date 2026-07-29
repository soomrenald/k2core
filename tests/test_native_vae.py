from __future__ import annotations

import unittest

from k2core.backends.native_vae import (
    KREA2_VAE_LATENT_CHANNELS,
    KREA2_VAE_SCALE_FACTOR,
    _map_vae_key,
    _validate_state_dict_shapes,
    _vae_state_dict,
)
from k2core.inference import WeightMappingError


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class NativeVaeMappingTests(unittest.TestCase):
    def test_reviewed_latent_contract(self) -> None:
        self.assertEqual(KREA2_VAE_LATENT_CHANNELS, 16)
        self.assertEqual(KREA2_VAE_SCALE_FACTOR, 8)

    def test_maps_quant_middle_and_downsample_keys(self) -> None:
        expected = {
            "conv1.weight": "quant_conv.weight",
            "encoder.middle.0.residual.3.gamma": (
                "encoder.mid_block.resnets.0.norm2.gamma"
            ),
            "decoder.middle.1.to_qkv.weight": (
                "decoder.mid_block.attentions.0.to_qkv.weight"
            ),
            "encoder.downsamples.6.shortcut.bias": (
                "encoder.down_blocks.6.conv_shortcut.bias"
            ),
        }
        self.assertEqual(
            {source: _map_vae_key(source) for source in expected},
            expected,
        )

    def test_maps_grouped_decoder_resnets_and_upsamplers(self) -> None:
        expected = {
            "decoder.upsamples.4.residual.2.weight": (
                "decoder.up_blocks.1.resnets.0.conv1.weight"
            ),
            "decoder.upsamples.10.residual.6.bias": (
                "decoder.up_blocks.2.resnets.2.conv2.bias"
            ),
            "decoder.upsamples.11.resample.1.weight": (
                "decoder.up_blocks.2.upsamplers.0.resample.1.weight"
            ),
            "decoder.upsamples.7.time_conv.bias": (
                "decoder.up_blocks.1.upsamplers.0.time_conv.bias"
            ),
        }
        self.assertEqual(
            {source: _map_vae_key(source) for source in expected},
            expected,
        )

    def test_mapping_rejects_unknown_and_duplicate_keys(self) -> None:
        with self.assertRaises(WeightMappingError):
            _map_vae_key("encoder.unreviewed.weight")
        state = _vae_state_dict(
            {
                "conv1.weight": object(),
                "encoder.conv1.weight": object(),
            }
        )
        self.assertEqual(
            set(state),
            {"quant_conv.weight", "encoder.conv_in.weight"},
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
