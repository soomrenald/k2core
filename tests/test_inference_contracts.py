from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from k2core.backends import ComfyUIBackend, NativeK2Backend
from k2core.backends.native import (
    _clean_latent_shape,
    _compile_native_regional_plan,
)
from k2core.inference import (
    BackendInitializationError,
    ConfigurationError,
    GenerationRequest,
    ImageEditRequest,
    InferenceBackend,
    InvalidRequestError,
    LoraSpec,
    PipelineConfig,
    UnsupportedFeatureError,
    configured_backend_name,
    convert_error,
    select_backend,
)
from k2core.model import ArtifactSet
from k2core.regions import PixelBox, RegionDefinition


class Runtime:
    loaded = False

    def __init__(self) -> None:
        self.load_request = None
        self.generate_request = None
        self.edit_request = None

    def load(self, artifacts, **request):
        self.loaded = True
        self.load_request = (artifacts, request)
        return {"transformer": "ModelPatcher", "cpu_vae": request["cpu_vae"]}

    def generate(self, **request):
        self.generate_request = request
        request["progress"](1, 2, {"gpu_free_bytes": 12})
        return {"image_path": "/tmp/result.png", "seed": request["seed"]}

    def edit_image(self, **request):
        self.edit_request = request
        request["progress"](1, 1, {})
        return {"image_path": "/tmp/edit.png", "seed": request["seed"]}


class InferenceContractTests(unittest.TestCase):
    def test_backend_selection_defaults_to_comfyui_and_never_silently_falls_back(self) -> None:
        runtime = Runtime()
        comfyui = ComfyUIBackend(runtime)
        native = NativeK2Backend()

        with self.assertLogs("k2core.inference.selection", level="INFO") as captured:
            self.assertEqual(configured_backend_name({}).value, "comfyui")
        self.assertIn("selected inference backend=comfyui", captured.output[0])
        self.assertIs(select_backend(comfyui, native=native, environment={}), comfyui)
        self.assertIs(
            select_backend(
                comfyui,
                native=native,
                environment={"K2LAB_BACKEND": "native"},
            ),
            native,
        )
        with self.assertRaises(ConfigurationError):
            configured_backend_name({"K2LAB_BACKEND": "automatic"})
        with self.assertRaises(UnsupportedFeatureError):
            select_backend(
                comfyui,
                environment={"K2LAB_BACKEND": "native"},
            )

    def test_comfyui_backend_implements_lifecycle_and_preserves_result_payload(self) -> None:
        runtime = Runtime()
        backend = ComfyUIBackend(runtime)
        self.assertIsInstance(backend, InferenceBackend)
        loaded = backend.load(
            PipelineConfig(
                artifacts=ArtifactSet(None, None, None),
                memory_policy="custom",
                reserve_vram_gb=3.0,
                minimum_system_ram_gb=12.0,
                cpu_vae=True,
                oom_recovery=False,
            )
        )
        self.assertEqual(loaded.backend_id, "comfyui")
        self.assertEqual(runtime.load_request[1]["memory_policy_key"], "custom")
        self.assertTrue(runtime.load_request[1]["cpu_vae"])

        progress = []
        result = backend.generate(
            GenerationRequest(
                correlation_id="job-1",
                prompt="synthetic fixture",
                width=1024,
                height=1024,
                steps=8,
                seed=17,
                output_directory=Path("/tmp"),
            ),
            progress=progress.append,
        )
        self.assertEqual(
            result.to_payload(),
            {"image_path": "/tmp/result.png", "seed": 17},
        )
        self.assertNotIn("backend_id", result.to_payload())
        self.assertEqual(progress[0].correlation_id, "job-1")
        self.assertEqual(progress[0].fraction, 0.5)
        self.assertNotIn("cfg", runtime.generate_request)

    def test_generation_schema_validates_current_turbo_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            GenerationRequest(
                correlation_id="job",
                prompt="fixture",
                width=1000,
                height=1024,
                steps=8,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "CFG 1.0"):
            GenerationRequest(
                correlation_id="job",
                prompt="fixture",
                width=1024,
                height=1024,
                steps=8,
                seed=1,
                cfg=2.0,
            )

    def test_native_backend_advertises_regional_prompting_loras_and_editing(
        self,
    ) -> None:
        backend = NativeK2Backend()
        capabilities = backend.capabilities()
        self.assertEqual(
            capabilities.modes,
            frozenset(
                {
                    "text_to_image",
                    "image_edit",
                    "ordinary_lora",
                    "regional_prompting",
                    "regional_lora",
                }
            ),
        )
        self.assertTrue(capabilities.metadata["developer_only"])
        backend._validate_clean_request(
            GenerationRequest(
                correlation_id="native-regional-lora",
                prompt="fixture",
                width=512,
                height=512,
                steps=8,
                seed=1,
                output_directory=Path("/tmp"),
                regions=(
                    RegionDefinition(
                        "one-region",
                        "One",
                        PixelBox(0, 0, 256, 512),
                        "a vase",
                    ),
                ),
                loras=(
                    LoraSpec(
                        lora_id="one",
                        name="one",
                        path=Path("/tmp/one.safetensors"),
                        global_scope=False,
                        region_ids=("one-region",),
                    ),
                ),
            )
        )

    def test_native_regional_plan_uses_shared_k2core_semantics(self) -> None:
        request = GenerationRequest(
            correlation_id="native-regions",
            prompt="studio scene",
            width=512,
            height=256,
            steps=8,
            seed=1,
            regions=(
                RegionDefinition(
                    "left",
                    "Left vessel",
                    PixelBox(-10, 0, 260, 256),
                    "a red ceramic vase",
                    priority=2,
                    spatial_role="subject",
                ),
                RegionDefinition(
                    "empty",
                    "Empty",
                    PixelBox(256, 0, 512, 256),
                    "",
                ),
            ),
        )

        plan = _compile_native_regional_plan(request)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.regions), 1)
        self.assertEqual(plan.regions[0].box.x0, 0.0)
        self.assertEqual(
            (plan.image_token_width, plan.image_token_height),
            (32, 16),
        )
        self.assertIn("studio scene.", plan.prompt)
        self.assertIn("a red ceramic vase", plan.prompt)
        self.assertIsNone(_compile_native_regional_plan(replace(request, regional_prompting=False)))

    def test_native_clean_dimensions_map_to_reviewed_five_dimensional_latents(
        self,
    ) -> None:
        cases = {
            (256, 256): (1, 16, 1, 32, 32),
            (512, 768): (1, 16, 1, 96, 64),
            (1024, 1024): (1, 16, 1, 128, 128),
            (1536, 1024): (1, 16, 1, 128, 192),
        }
        self.assertEqual(
            {size: _clean_latent_shape(*size) for size in cases},
            cases,
        )
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            _clean_latent_shape(513, 512)

    def test_native_clean_generation_rejects_malformed_or_unavailable_requests(
        self,
    ) -> None:
        backend = NativeK2Backend()
        base = {
            "correlation_id": "native-invalid",
            "prompt": "fixture",
            "width": 512,
            "height": 512,
            "steps": 8,
            "seed": 1,
            "output_directory": Path("/tmp"),
        }
        with self.assertRaisesRegex(InvalidRequestError, "non-empty"):
            backend.generate(GenerationRequest(**{**base, "prompt": "  "}))
        with self.assertRaisesRegex(UnsupportedFeatureError, "Euler"):
            backend.generate(GenerationRequest(**base, sampler="heun"))
        with self.assertRaisesRegex(UnsupportedFeatureError, "simple"):
            backend.generate(GenerationRequest(**base, scheduler="normal"))
        with self.assertRaisesRegex(UnsupportedFeatureError, "negative prompts"):
            backend.generate(GenerationRequest(**base, negative_prompt="low quality"))
        with self.assertRaisesRegex(ConfigurationError, "must be loaded"):
            backend.generate(GenerationRequest(**base))

    def test_model_loading_runtime_errors_are_classified_as_initialization(self) -> None:
        structured = convert_error(
            RuntimeError("GPU accelerator is unavailable"),
            backend_name="comfyui",
            phase="model_loading",
        )
        self.assertIsInstance(structured, BackendInitializationError)
        self.assertTrue(structured.retry_safe)

    def test_image_edit_uses_same_backend_adapter(self) -> None:
        runtime = Runtime()
        backend = ComfyUIBackend(runtime)
        request = ImageEditRequest.from_payload(
            {
                "image_path": "/tmp/source.png",
                "output_directory": "/tmp",
                "prompt": "replace object",
                "regions": [],
                "loras": [],
                "seed": 23,
                "steps": 8,
                "denoise": 0.2,
            },
            correlation_id="edit-1",
        )
        result = backend.generate(request)
        self.assertEqual(result.to_payload()["image_path"], "/tmp/edit.png")
        self.assertEqual(runtime.edit_request["seed"], 23)


if __name__ == "__main__":
    unittest.main()
