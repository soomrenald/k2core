from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from k2core.backends import NativeK2Backend
from k2core.backends.native_device import NativeDeviceManager
from k2core.inference import ConfigurationError, DTypePolicy, DevicePolicy, OutOfMemoryError


class FakeVersion:
    hip = "7.1"


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.empty_cache_calls = 0
        self.reset_calls = 0

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self) -> str:
        return "Fixture GPU"

    def mem_get_info(self) -> tuple[int, int]:
        return 6 * 1024**3, 8 * 1024**3

    def memory_allocated(self) -> int:
        return 2 * 1024**3

    def memory_reserved(self) -> int:
        return 3 * 1024**3

    def max_memory_allocated(self) -> int:
        return 4 * 1024**3

    def max_memory_reserved(self) -> int:
        return 5 * 1024**3

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    float8_e4m3fn = "float8_e4m3fn"
    version = FakeVersion()

    def __init__(self, *, available: bool = True) -> None:
        self.cuda = FakeCuda(available=available)

    @staticmethod
    def device(name: str) -> str:
        if name not in {"cpu", "cuda", "cuda:0"}:
            raise RuntimeError("invalid device")
        return name


class NativeDeviceManagerTests(unittest.TestCase):
    def test_auto_plan_detects_rocm_and_keeps_auto_weights_staged_on_cpu(self) -> None:
        torch = FakeTorch()
        manager = NativeDeviceManager(
            DevicePolicy(cpu_offload=True, vae_tiling=True),
            reserve_vram_gb=1.0,
            minimum_system_ram_gb=2.0,
            torch=torch,
            memory_reader=lambda: (12 * 1024**3, 16 * 1024**3),
        )

        self.assertEqual(manager.plan.accelerator_backend, "rocm")
        self.assertEqual(manager.plan.transformer_device, "cuda")
        self.assertEqual(manager.plan.text_encoder_device, "cuda")
        self.assertEqual(manager.plan.vae_device, "cuda")
        self.assertTrue(manager.plan.sequential_components)
        self.assertTrue(manager.plan.vae_tiling)
        self.assertEqual(manager.staging_policy().transformer_device, "cpu")
        snapshot = manager.preflight("fixture")
        self.assertEqual(snapshot["gpu_allocated_bytes"], 2 * 1024**3)
        self.assertEqual(snapshot["gpu_peak_reserved_bytes"], 5 * 1024**3)

    def test_explicit_cpu_is_required_when_no_accelerator_is_visible(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "explicitly select CPU"):
            NativeDeviceManager(
                DevicePolicy(),
                torch=FakeTorch(available=False),
                memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
            )

        manager = NativeDeviceManager(
            DevicePolicy(
                transformer_device="cpu",
                text_encoder_device="cpu",
                vae_device="cpu",
            ),
            torch=FakeTorch(available=False),
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )
        self.assertEqual(manager.plan.accelerator_backend, "cpu")
        self.assertFalse(manager.plan.accelerator_available)

    def test_cpu_vae_override_is_explicit_in_the_resolved_plan(self) -> None:
        manager = NativeDeviceManager(
            DevicePolicy(),
            cpu_vae=True,
            torch=FakeTorch(),
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )
        self.assertEqual(manager.plan.vae_device, "cpu")

    def test_preflight_enforces_system_ram_and_reserved_vram(self) -> None:
        with self.assertRaisesRegex(OutOfMemoryError, "system RAM"):
            NativeDeviceManager(
                DevicePolicy(),
                minimum_system_ram_gb=4.0,
                torch=FakeTorch(),
                memory_reader=lambda: (3 * 1024**3, 16 * 1024**3),
            ).preflight("load")

        with self.assertRaisesRegex(OutOfMemoryError, "accelerator memory"):
            NativeDeviceManager(
                DevicePolicy(),
                reserve_vram_gb=6.0,
                torch=FakeTorch(),
                memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
            ).preflight("load")

    def test_dtype_contract_distinguishes_fp8_weights_from_compute(self) -> None:
        manager = NativeDeviceManager(
            DevicePolicy(weight_dtype=DTypePolicy.FLOAT8_E4M3FN),
            torch=FakeTorch(),
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )
        self.assertEqual(manager.plan.weight_dtype, "float8_e4m3fn")
        with self.assertRaisesRegex(ConfigurationError, "not as a compute dtype"):
            NativeDeviceManager(
                DevicePolicy(compute_dtype=DTypePolicy.FLOAT8_E4M3FN),
                torch=FakeTorch(),
                memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
            )

    def test_cleanup_and_oom_classification_are_owned_by_manager(self) -> None:
        torch = FakeTorch()
        manager = NativeDeviceManager(
            DevicePolicy(),
            torch=torch,
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )

        manager.reset_peak_stats()
        cleanup = manager.cleanup_after_failure("transformer")

        self.assertEqual(torch.cuda.reset_calls, 1)
        self.assertEqual(torch.cuda.empty_cache_calls, 1)
        self.assertEqual(cleanup["failed_phase"], "transformer")
        self.assertTrue(manager.is_oom(RuntimeError("HIP out of memory")))
        self.assertFalse(manager.is_oom(RuntimeError("shape mismatch")))

    def test_request_has_at_most_one_explicit_oom_fallback(self) -> None:
        manager = NativeDeviceManager(
            DevicePolicy(),
            torch=FakeTorch(),
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )

        manager.begin_request(allow_oom_retry=True)

        self.assertTrue(manager.claim_oom_retry("vae_decode", "vae_tiling"))
        self.assertFalse(manager.claim_oom_retry("transformer", "cpu"))
        self.assertEqual(
            manager.recovery_summary(),
            {
                "retry_used": True,
                "retry_remaining": False,
                "events": (
                    {
                        "phase": "vae_decode",
                        "fallback": "vae_tiling",
                    },
                ),
            },
        )
        manager.begin_request(allow_oom_retry=False)
        self.assertFalse(manager.claim_oom_retry("vae_encode", "vae_tiling"))

    def test_backend_uses_one_tiled_vae_retry_after_oom(self) -> None:
        manager = NativeDeviceManager(
            DevicePolicy(),
            torch=FakeTorch(),
            memory_reader=lambda: (8 * 1024**3, 16 * 1024**3),
        )
        manager.begin_request(allow_oom_retry=True)
        first = Mock()
        second = Mock()
        attempts = 0

        def operation(vae):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("HIP out of memory")
            return vae

        diagnostics = []
        with patch(
            "k2core.backends.native.build_krea2_vae",
            side_effect=[first, second],
        ) as build:
            result = NativeK2Backend()._run_vae(
                SimpleNamespace(vae=object()),
                manager,
                phase="vae_decode",
                operation=operation,
                diagnostic=lambda message, payload: diagnostics.append((message, payload)),
            )

        self.assertIs(result, second)
        self.assertEqual(build.call_args_list[0].kwargs["tiling"], False)
        self.assertEqual(build.call_args_list[1].kwargs["tiling"], True)
        first.unload.assert_called_once_with()
        second.unload.assert_called_once_with()
        self.assertTrue(manager.recovery_summary()["retry_used"])
        self.assertIn("retrying once", diagnostics[0][0])


if __name__ == "__main__":
    unittest.main()
