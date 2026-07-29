"""Opt-in native inference instrumentation with zero default tensor work."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from k2core.inference.schemas import InstrumentationConfig
from k2core.regional_lora import LoraDeltaRoute
from k2core.regional_prompting import BoundRegionalPromptPlan


@dataclass(slots=True)
class _DeltaRecord:
    calls: int = 0
    element_count: int = 0
    sum_squares: float = 0.0
    maximum_absolute: float = 0.0
    maximum_outside_route: float = 0.0
    modified_token_observations: int = 0
    token_observations: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "element_count": self.element_count,
            "rms": (sqrt(self.sum_squares / self.element_count) if self.element_count else 0.0),
            "maximum_absolute": self.maximum_absolute,
            "maximum_outside_route": self.maximum_outside_route,
            "modified_token_observations": self.modified_token_observations,
            "token_observations": self.token_observations,
        }


@dataclass(slots=True)
class NativeInstrumentation:
    config: InstrumentationConfig
    adaptation_routes: tuple[LoraDeltaRoute, ...] = ()
    _records: dict[tuple[str, str, str, str], _DeltaRecord] = field(default_factory=dict)
    _attention_masks: dict[str, Any] | None = None
    _adaptation_values: dict[str, dict[str, Any]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self._adaptation_values = {
            route.lora_id: {
                "route": route,
                "text_energy": None,
                "text_count": 0,
                "image_energy": None,
                "image_count": 0,
                "step_text_energy": None,
                "step_text_count": 0,
                "step_image_energy": None,
                "step_image_count": 0,
                "delta_reference": None,
            }
            for route in self.adaptation_routes
        }

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def target_kinds(self, source_module: str) -> tuple[str, ...]:
        lowered = source_module.casefold()
        kinds = []
        if self.config.q_projection_deltas and lowered.endswith((".wq", ".q_proj")):
            kinds.append("q_projection_delta")
        if self.config.k_projection_deltas and lowered.endswith((".wk", ".k_proj")):
            kinds.append("k_projection_delta")
        if self.config.v_projection_deltas and lowered.endswith((".wv", ".v_proj")):
            kinds.append("v_projection_delta")
        attention_output = lowered.endswith((".wo", ".out_proj"))
        mlp_output = ".mlp.down" in lowered or lowered.endswith(".down_proj")
        if self.config.attention_output_deltas and attention_output:
            kinds.append("attention_output_delta")
        if self.config.mlp_output_deltas and mlp_output:
            kinds.append("mlp_output_delta")
        if self.config.residual_deltas and (attention_output or mlp_output):
            kinds.append("residual_input_delta")
        if self.config.hidden_state_deltas:
            kinds.append("hidden_state_contribution")
        return tuple(kinds)

    def observe_delta(
        self,
        *,
        lora_id: str,
        source_module: str,
        route_kind: str,
        applied: Any,
        route_mask: Any | None,
        route: LoraDeltaRoute | None = None,
    ) -> None:
        kinds = self.target_kinds(source_module)
        adaptation = (
            route is not None and not route.global_scope and lora_id in self._adaptation_values
        )
        if not kinds and not self.config.token_modification_flags and not adaptation:
            return
        detached = applied.detach().float()
        absolute = detached.abs()
        sum_squares = float((detached * detached).sum().item())
        maximum = float(absolute.max().item()) if detached.numel() else 0.0
        outside = 0.0
        if route_mask is not None and detached.numel():
            outside = float((absolute * (route_mask == 0).to(absolute.dtype)).max().item())
        token_norms = None
        if self.config.token_modification_flags or adaptation:
            token_norms = detached.square().sum(dim=-1).sqrt()
        if adaptation:
            self._observe_adaptation(
                route,
                token_norms,
                route_kind=route_kind,
            )
        record_kinds = kinds or ("token_modification_flags",)
        if not kinds and not self.config.token_modification_flags:
            record_kinds = ()
        for kind in record_kinds:
            key = (lora_id, kind, source_module, route_kind)
            record = self._records.setdefault(key, _DeltaRecord())
            record.calls += 1
            record.element_count += int(detached.numel())
            record.sum_squares += sum_squares
            record.maximum_absolute = max(record.maximum_absolute, maximum)
            record.maximum_outside_route = max(
                record.maximum_outside_route,
                outside,
            )
            if token_norms is not None:
                record.modified_token_observations += int((token_norms > 0).sum().item())
                record.token_observations += int(token_norms.numel())

    @staticmethod
    def _add(previous: Any, value: Any) -> Any:
        return value if previous is None else previous + value

    def _observe_adaptation(
        self,
        route: LoraDeltaRoute,
        token_norms: Any,
        *,
        route_kind: str,
    ) -> None:
        state = self._adaptation_values[route.lora_id]
        batch = int(token_norms.shape[0])
        text_count = len(route.text_token_mask)
        enabled_text = sum(value > 0.0 for value in route.text_token_mask)
        if route_kind == "text_layerwise":
            text_norms = token_norms
            image_norms = None
            folded_batches = batch // text_count
            text_observations = folded_batches * enabled_text * int(token_norms.shape[1])
        elif route_kind == "text_projector":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text * int(token_norms.shape[2])
        elif route_kind == "text_refiner":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text
        else:
            text_norms = token_norms[:, :text_count]
            image_norms = token_norms[:, text_count:]
            text_observations = batch * enabled_text
        if enabled_text:
            text_energy = text_norms.square().sum()
            state["text_energy"] = self._add(state["text_energy"], text_energy)
            state["text_count"] += text_observations
            state["step_text_energy"] = self._add(
                state["step_text_energy"],
                text_energy,
            )
            state["step_text_count"] += text_observations
        enabled_image = sum(value > 0.0 for value in route.image_token_mask)
        if image_norms is not None and enabled_image:
            image_energy = image_norms.square().sum()
            state["image_energy"] = self._add(state["image_energy"], image_energy)
            state["image_count"] += batch * enabled_image
            state["step_image_energy"] = self._add(
                state["step_image_energy"],
                image_energy,
            )
            state["step_image_count"] += batch * enabled_image

    @staticmethod
    def _rms(energy: Any, count: int) -> float:
        if energy is None or count == 0:
            return 0.0
        return float((energy / count).sqrt().item())

    def regional_attention_scales(self, gain: float) -> dict[str, float]:
        if not 0.0 <= gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        region_values: dict[str, list[float]] = {}
        for state in self._adaptation_values.values():
            route = state["route"]
            components = [
                self._rms(state["step_text_energy"], state["step_text_count"]),
                self._rms(state["step_image_energy"], state["step_image_count"]),
            ]
            components = [value for value in components if value > 0.0]
            if not components:
                continue
            observed = sum(components) / len(components)
            reference = state["delta_reference"]
            if reference is None:
                reference = observed
            ratio = observed / max(float(reference), 1e-12)
            scale = min(1.5, max(0.5, 1.0 + gain * (ratio - 1.0)))
            state["delta_reference"] = 0.85 * float(reference) + 0.15 * observed
            for region_id in route.region_ids:
                region_values.setdefault(region_id, []).append(scale)
        return {region_id: sum(scales) / len(scales) for region_id, scales in region_values.items()}

    def reset_step_measurements(self) -> None:
        for state in self._adaptation_values.values():
            for prefix in ("text", "image"):
                state[f"step_{prefix}_energy"] = None
                state[f"step_{prefix}_count"] = 0

    def record_attention_masks(
        self,
        bound_plan: BoundRegionalPromptPlan | None,
    ) -> None:
        if not self.config.attention_masks or bound_plan is None:
            return
        self._attention_masks = {
            "text_token_count": bound_plan.text_token_count,
            "image_token_count": bound_plan.image_token_count,
            "regions": [
                {
                    "id": span.region_id,
                    "text_token_span": [span.start, span.end],
                    "text_tokens_enabled": span.end - span.start,
                    "image_tokens_enabled": sum(value > 0.0 for value in span.image_token_mask),
                    "image_mask_coverage": (
                        sum(span.image_token_mask) / len(span.image_token_mask)
                    ),
                }
                for span in bound_plan.spans
            ],
        }

    def summary(self) -> dict[str, Any]:
        records = []
        for (lora_id, kind, target, route_kind), record in sorted(self._records.items()):
            records.append(
                {
                    "lora_id": lora_id,
                    "kind": kind,
                    "target": target,
                    "route_kind": route_kind,
                    **record.payload(),
                }
            )
        return {
            "enabled": self.enabled,
            "toggles": {
                name: bool(getattr(self.config, name)) for name in self.config.__dataclass_fields__
            },
            "records": records,
            "attention_masks": self._attention_masks,
        }


def route_summary(route: LoraDeltaRoute | None) -> dict[str, Any] | None:
    return route.summary() if route is not None else None


__all__ = ["NativeInstrumentation"]
