from __future__ import annotations

import unittest

from k2core.projector import (
    PROJECTOR_PRESETS,
    effective_projector_values,
    projector_token_delta_mask,
    validate_projector_values,
)


class ProjectorSettingsTests(unittest.TestCase):
    def test_reference_presets_have_twelve_values_and_multiplier_is_global(self) -> None:
        self.assertEqual(
            set(PROJECTOR_PRESETS),
            {"filter_bypass2", "filter_bypass3", "skc3vo", "z0jglf"},
        )
        for values in PROJECTOR_PRESETS.values():
            self.assertEqual(len(values), 12)
        effective = effective_projector_values(PROJECTOR_PRESETS["filter_bypass2"], 3.0)
        self.assertAlmostEqual(effective[8], -1.5351)
        self.assertAlmostEqual(effective[9], -2.6718)

    def test_projector_vector_rejects_wrong_column_count(self) -> None:
        with self.assertRaises(ValueError):
            validate_projector_values((0.0,) * 11)

    def test_identity_protection_scales_only_selected_text_tokens(self) -> None:
        self.assertEqual(
            projector_token_delta_mask(8, ((2, 5),), 1.0),
            (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        )
        self.assertEqual(
            projector_token_delta_mask(5, ((1, 3),), 0.5),
            (1.0, 0.5, 0.5, 1.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()

