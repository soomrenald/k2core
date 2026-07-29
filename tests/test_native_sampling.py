from __future__ import annotations

import unittest
from unittest.mock import patch

from k2core.backends import (
    euler_flow_image_sample,
    euler_flow_sample,
    partial_denoise_sigmas,
    prepare_noise,
    simple_sigmas,
    tokenize_prompt,
)


class FakeTokenizer:
    def __init__(self, encoded: list[int]) -> None:
        self.encoded = encoded
        self.text = ""

    def encode(self, text: str, *, add_special_tokens: bool):
        self.text = text
        assert not add_special_tokens
        return self.encoded


class NativeSamplingTests(unittest.TestCase):
    def test_simple_eight_step_schedule_matches_reference_indices(self) -> None:
        sigmas = simple_sigmas(8)

        self.assertEqual(len(sigmas), 9)
        self.assertEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:])))
        self.assertAlmostEqual(sigmas[1], 0.9567237271, places=9)

    def test_euler_flow_integration_uses_velocity_and_sigma_delta(self) -> None:
        checkpoints = []
        result = euler_flow_sample(
            lambda latent, sigma: latent * 0 + sigma,
            2.0,
            (1.0, 0.5, 0.0),
            checkpoint=checkpoints.append,
        )

        self.assertEqual(result, 1.25)
        self.assertEqual([item.step for item in checkpoints], [0, 1])
        self.assertEqual(checkpoints[0].model_output, 1.0)

    def test_partial_denoise_schedule_matches_reference_truncation(self) -> None:
        self.assertEqual(partial_denoise_sigmas(8, 1.0), simple_sigmas(8))
        self.assertEqual(partial_denoise_sigmas(8, 0.25), simple_sigmas(32)[-9:])
        self.assertEqual(partial_denoise_sigmas(8, 0.0), (0.0,))

    def test_image_flow_initializes_from_source_and_preserves_masked_lanes(self) -> None:
        checkpoints = []
        self.assertEqual(
            euler_flow_image_sample(
                lambda latent, sigma: latent * 0 + sigma,
                source_latent=2.0,
                noise=10.0,
                sigmas=(0.5, 0.25, 0.0),
                denoise_mask=0.0,
                checkpoint=checkpoints.append,
            ),
            2.0,
        )
        changed = euler_flow_image_sample(
            lambda latent, sigma: latent * 0 + sigma,
            source_latent=2.0,
            noise=10.0,
            sigmas=(0.5, 0.25, 0.0),
            denoise_mask=1.0,
        )
        self.assertNotEqual(changed, 2.0)
        self.assertEqual([item.step for item in checkpoints], [0, 1])

    def test_noise_uses_an_explicit_generator_without_global_manual_seed(self) -> None:
        class Generator:
            def manual_seed(self, seed):
                self.seed = seed
                return self

        class Torch:
            strided = "strided"

            def __init__(self):
                self.generator = Generator()

            def Generator(self, *, device):
                self.generator.device = device
                return self.generator

            def randn(self, shape, **kwargs):
                return shape, kwargs

        fake_torch = Torch()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            shape, options = prepare_noise(
                (1, 16, 1, 8, 8),
                17,
                device="cpu",
                dtype="float32",
            )
        self.assertEqual(shape, (1, 16, 1, 8, 8))
        self.assertEqual(fake_torch.generator.seed, 17)
        self.assertIs(options["generator"], fake_torch.generator)

    def test_prompt_template_retains_full_context_and_marks_conditioned_slice(self) -> None:
        tokenizer = FakeTokenizer([151644, 10, 151644, 872, 198, 20, 21])

        tokens = tokenize_prompt("red ceramic teapot", tokenizer)

        self.assertIn("red ceramic teapot", tokenizer.text)
        self.assertEqual(tokens.output_start, 5)
        self.assertEqual(tokens.conditioned_ids, (20, 21))

    def test_prompt_template_rejects_unrecognized_tokenizer_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "second"):
            tokenize_prompt("fixture", FakeTokenizer([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
