from __future__ import annotations

import unittest

from k2core.backends.native_instrumentation import NativeInstrumentation
from k2core.inference import InstrumentationConfig
from k2core.regional_lora import LoraDeltaRoute
from k2core.regional_prompting import compile_regional_prompt_plan
from k2core.regions import PixelBox, RegionDefinition


class NativeInstrumentationTests(unittest.TestCase):
    def test_disabled_configuration_has_no_enabled_targets(self) -> None:
        instrumentation = NativeInstrumentation(InstrumentationConfig())

        self.assertFalse(instrumentation.enabled)
        self.assertEqual(
            instrumentation.target_kinds("diffusion_model.blocks.0.attn.wq"),
            (),
        )
        self.assertEqual(
            instrumentation.summary(),
            {
                "enabled": False,
                "toggles": {
                    "q_projection_deltas": False,
                    "k_projection_deltas": False,
                    "v_projection_deltas": False,
                    "hidden_state_deltas": False,
                    "attention_output_deltas": False,
                    "mlp_output_deltas": False,
                    "residual_deltas": False,
                    "token_modification_flags": False,
                    "attention_masks": False,
                },
                "records": [],
                "attention_masks": None,
            },
        )

    def test_target_toggles_and_attention_masks_are_explicit(self) -> None:
        configuration = InstrumentationConfig.from_payload(
            {
                "q_projection_deltas": True,
                "attention_output_deltas": True,
                "mlp_output_deltas": True,
                "residual_deltas": True,
                "attention_masks": True,
            }
        )
        instrumentation = NativeInstrumentation(configuration)
        self.assertEqual(
            instrumentation.target_kinds("diffusion_model.blocks.0.attn.wq"),
            ("q_projection_delta",),
        )
        self.assertEqual(
            instrumentation.target_kinds("diffusion_model.blocks.0.attn.wo"),
            ("attention_output_delta", "residual_input_delta"),
        )
        self.assertEqual(
            instrumentation.target_kinds("diffusion_model.blocks.0.mlp.down"),
            ("mlp_output_delta", "residual_input_delta"),
        )

        plan = compile_regional_prompt_plan(
            32,
            16,
            "studio",
            (
                RegionDefinition(
                    "right",
                    "Right",
                    PixelBox(16, 0, 32, 16),
                    "blue vase",
                ),
            ),
        )
        bound = plan.bind_tokens(
            len,
            conditioning_text_token_count=len(plan.prompt),
        )
        instrumentation.record_attention_masks(bound)

        masks = instrumentation.summary()["attention_masks"]
        self.assertEqual(masks["image_token_count"], 2)
        self.assertEqual(masks["regions"][0]["image_tokens_enabled"], 1)

    def test_delta_adaptation_is_bounded_and_independent_of_diagnostics(self) -> None:
        route = LoraDeltaRoute(
            lora_id="subject-style",
            display_name="Subject style",
            strength=1.0,
            global_scope=False,
            region_ids=("subject",),
            region_names=("Subject",),
            text_token_mask=(0.0, 1.0),
            image_token_mask=(1.0, 0.0),
        )
        instrumentation = NativeInstrumentation(
            InstrumentationConfig(),
            adaptation_routes=(route,),
        )

        class Scalar:
            def __init__(self, value: float) -> None:
                self.value = value

            def __truediv__(self, divisor: int):
                return Scalar(self.value / divisor)

            def sqrt(self):
                return Scalar(self.value**0.5)

            def item(self) -> float:
                return self.value

        state = instrumentation._adaptation_values[route.lora_id]
        state["step_text_energy"] = Scalar(4.0)
        state["step_text_count"] = 1
        state["step_image_energy"] = Scalar(16.0)
        state["step_image_count"] = 1

        self.assertEqual(
            instrumentation.regional_attention_scales(0.35),
            {"subject": 1.0},
        )
        self.assertEqual(instrumentation.summary()["records"], [])

        instrumentation.reset_step_measurements()
        state["step_text_energy"] = Scalar(400.0)
        state["step_text_count"] = 1
        state["step_image_energy"] = Scalar(1600.0)
        state["step_image_count"] = 1
        self.assertEqual(
            instrumentation.regional_attention_scales(1.0),
            {"subject": 1.5},
        )


if __name__ == "__main__":
    unittest.main()
