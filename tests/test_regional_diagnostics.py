from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from k2core.regional_diagnostics import render_regional_diagnostics
from k2core.regional_prompting import compile_regional_prompt_plan
from k2core.regions import PixelBox, RegionDefinition


class RegionalDiagnosticTests(unittest.TestCase):
    def test_renders_all_required_visual_diagnostics(self) -> None:
        plan = compile_regional_prompt_plan(
            64,
            32,
            "studio",
            (
                RegionDefinition(
                    "front",
                    "Front",
                    PixelBox(0, 0, 40, 32),
                    "red vase",
                    priority=2,
                ),
                RegionDefinition(
                    "behind",
                    "Behind",
                    PixelBox(24, 0, 64, 32),
                    "blue vase",
                    priority=1,
                ),
            ),
            falloff_pixels=16,
        )
        bound = plan.bind_tokens(
            lambda prefix: len(prefix.split()),
            conditioning_text_token_count=len(plan.prompt.split()),
        )
        with tempfile.TemporaryDirectory() as root:
            artifacts = render_regional_diagnostics(
                plan,
                bound,
                Path(root),
                prefix="two vases",
            )
            payload = artifacts.to_payload()

            for name in (
                "mask_preview",
                "latent_mask_preview",
                "token_assignment_preview",
                "overlap_preview",
            ):
                self.assertTrue(Path(payload[name]["path"]).is_file())
                self.assertEqual(len(payload[name]["sha256"]), 64)
            self.assertGreater(
                payload["summary"]["overlap_token_count"],
                0,
            )
            self.assertEqual(payload["summary"]["region_count"], 2)


if __name__ == "__main__":
    unittest.main()
