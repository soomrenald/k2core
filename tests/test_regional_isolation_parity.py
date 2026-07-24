from __future__ import annotations

import importlib.util
import unittest

from k2core.regional_lora import LoraDeltaRoute
from k2core.worker.runtime import LoraDeltaStatistics


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def _route(
    lora_id: str,
    image_mask: tuple[float, ...],
) -> LoraDeltaRoute:
    return LoraDeltaRoute(
        lora_id=lora_id,
        display_name=lora_id,
        strength=1.0,
        global_scope=False,
        region_ids=(lora_id,),
        region_names=(lora_id,),
        text_token_mask=(0.0, 0.0),
        image_token_mask=image_mask,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is tested in the ComfyUI worker environment")
class RegionalIsolationParityTests(unittest.TestCase):
    def test_modified_tokens_are_derived_from_gated_relative_lora_delta(self) -> None:
        route = _route("left", (1.0, 0.0, 0.0))
        statistics = LoraDeltaStatistics((route,))

        statistics.observe(
            route,
            torch.tensor([[0.0, 0.0, 0.20, 4.0, 4.0]]),
            route_kind="combined",
            reference_norms=torch.ones((1, 5)),
        )

        self.assertEqual(
            statistics.values["left"]["modified_image_flags"].tolist(),
            [[True, False, False]],
        )
        self.assertEqual(statistics.summary("left")["modified_image_tokens"], 1)

    def test_outside_image_queries_cannot_read_lora_modified_image_keys(self) -> None:
        route = _route("left", (1.0, 0.0, 0.0))
        statistics = LoraDeltaStatistics((route,))
        statistics.observe(
            route,
            torch.tensor([[0.0, 0.0, 0.20, 0.0, 0.0]]),
            route_kind="combined",
            reference_norms=torch.ones((1, 5)),
        )
        scores = torch.zeros((1, 1, 5, 5))

        statistics.apply_asymmetric_attention(
            scores,
            start=0,
            end=5,
            text_token_count=2,
        )

        modified_image = 2
        outside_image = 3
        self.assertEqual(
            float(scores[0, 0, outside_image, modified_image]),
            -5.0,
        )
        self.assertEqual(
            float(scores[0, 0, modified_image, outside_image]),
            0.0,
        )
        self.assertEqual(float(scores[0, 0, 0, modified_image]), 0.0)

    def test_cross_lora_attention_receives_the_original_separate_penalty(
        self,
    ) -> None:
        left = _route("left", (1.0, 0.0))
        right = _route("right", (0.0, 1.0))
        statistics = LoraDeltaStatistics((left, right))
        statistics.ATTENTION_ISOLATION_STRENGTH = 0.0
        reference = torch.ones((1, 4))
        statistics.observe(
            left,
            torch.tensor([[0.0, 0.0, 0.20, 0.0]]),
            route_kind="combined",
            reference_norms=reference,
        )
        statistics.observe(
            right,
            torch.tensor([[0.0, 0.0, 0.0, 0.20]]),
            route_kind="combined",
            reference_norms=reference,
        )
        scores = torch.zeros((1, 1, 4, 4))

        statistics.apply_asymmetric_attention(
            scores,
            start=0,
            end=4,
            text_token_count=2,
        )

        self.assertEqual(float(scores[0, 0, 2, 3]), -3.0)
        self.assertEqual(float(scores[0, 0, 3, 2]), -3.0)
