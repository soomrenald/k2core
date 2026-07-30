from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ThirdPartyNoticeTests(unittest.TestCase):
    def test_notice_records_dependencies_and_open_license_boundary(self) -> None:
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        for dependency in (
            "NumPy",
            "Pillow",
            "PyTorch",
            "Diffusers",
            "Transformers",
            "safetensors",
        ):
            self.assertIn(dependency, notice)
        self.assertIn("No ComfyUI source is copied or vendored", notice)
        self.assertIn("no declared first-party project license", notice)
