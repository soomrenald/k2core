from __future__ import annotations

import unittest

from k2core.memory import (
    MEMORY_POLICIES,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
    oom_recovery_reserve_vram_gb,
)


class MemoryPolicyTests(unittest.TestCase):
    def test_safe_16gb_policy_keeps_four_gib_free(self) -> None:
        policy = memory_policy("safe_16gb")
        self.assertEqual(policy.reserve_vram_gb, 4.0)
        self.assertEqual(policy.minimum_system_ram_gb, 14.0)
        self.assertTrue(policy.oom_recovery)

    def test_policy_keys_are_unique(self) -> None:
        keys = [policy.key for policy in MEMORY_POLICIES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_saved_value_cannot_weaken_policy_floor(self) -> None:
        self.assertEqual(effective_reserve_vram_gb("safe_16gb", 2.0), 4.0)
        self.assertEqual(effective_reserve_vram_gb("emergency", 4.0), 5.5)
        self.assertEqual(effective_reserve_vram_gb("balanced", 3.5), 3.5)
        self.assertEqual(effective_minimum_system_ram_gb("safe_16gb", 12.0), 14.0)
        self.assertEqual(effective_minimum_system_ram_gb("emergency", 14.0), 16.0)
        self.assertEqual(effective_minimum_system_ram_gb("balanced", 13.0), 13.0)

    def test_custom_policy_allows_tuning_for_unlisted_gpu_sizes(self) -> None:
        self.assertEqual(effective_reserve_vram_gb("custom", 0.75), 0.75)
        self.assertEqual(effective_minimum_system_ram_gb("custom", 6.0), 6.0)

    def test_oom_recovery_reserve_scales_with_gpu_capacity(self) -> None:
        self.assertEqual(oom_recovery_reserve_vram_gb(1.0, 8.0), 1.5)
        self.assertEqual(oom_recovery_reserve_vram_gb(4.0, 16.0), 5.0)
        self.assertEqual(oom_recovery_reserve_vram_gb(3.0, 24.0), 4.5)
        self.assertEqual(oom_recovery_reserve_vram_gb(5.0, 8.0), 5.0)


if __name__ == "__main__":
    unittest.main()

