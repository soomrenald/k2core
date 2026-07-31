from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import k2core.depth.adapter as adapter
from k2core.depth import (
    DepthScheduleController,
    EffectiveDepthField,
    KreaDepthInputProjection,
    clear_depth_control,
    encode_depth_control,
)


def _field(value: float) -> EffectiveDepthField:
    return EffectiveDepthField(
        pixel_values=np.full((16, 16), value, dtype=np.float32),
        image_token_values=np.full((1, 1), value, dtype=np.float32),
        region_multipliers={},
    )


def test_depth_schedule_advances_resets_and_rejects_divergence() -> None:
    schedule = DepthScheduleController((_field(1.0), _field(0.5), _field(0.0)))

    assert schedule.current_values().tolist() == [1.0]
    schedule.advance_after(0, 3)
    assert schedule.current_values().tolist() == [0.5]
    schedule.reset()
    assert schedule.current_values().tolist() == [1.0]
    with pytest.raises(RuntimeError, match="diverged"):
        schedule.advance_after(1, 3)
    with pytest.raises(RuntimeError, match="counts diverged"):
        schedule.advance_after(0, 2)


def test_depth_projection_preserves_native_path_and_weights_control_per_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    utilities = ModuleType("comfy.utils")
    management.cast_to_device = lambda value, *_args, **_kwargs: value
    utilities.repeat_to_batch_size = lambda value, _batch: value
    comfy.model_management = management
    comfy.utils = utilities
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utilities)

    original = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        original.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    expanded = torch.zeros((3, 8), dtype=torch.float32)
    expanded[:, 4:] = 2
    image_tokens = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    projection = KreaDepthInputProjection(
        expanded,
        image_features=4,
        original_first=original,
    )
    projection.control_tokens = torch.ones_like(image_tokens)
    projection.token_strength = torch.tensor([0.0, 2.0])

    output = projection(image_tokens)

    assert torch.equal(output[:, 0], original(image_tokens)[:, 0])
    assert torch.equal(
        output[:, 1],
        original(image_tokens)[:, 1] + torch.full((1, 3), 16.0),
    )


def test_depth_wrapper_restores_projection_and_schedule_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original = object()

    class Projection:
        control_features = 4
        control_tokens = None
        token_strength = None
        original_first = original

    projection = Projection()
    diffusion = SimpleNamespace(first=original, patch=2)
    monkeypatch.setattr(
        adapter,
        "_control_tokens",
        lambda *_args, **_kwargs: torch.ones((1, 4, 4)),
    )

    class Executor:
        class_obj = diffusion

        def __call__(self, *_args, **_kwargs):
            assert diffusion.first is projection
            assert projection.control_tokens is not None
            assert projection.token_strength is not None
            raise RuntimeError("synthetic denoiser failure")

    options = {
        adapter.DEPTH_LATENT_KEY: torch.zeros((1, 4, 4, 4)),
        adapter.DEPTH_TOKEN_STRENGTH_KEY: SimpleNamespace(
            current_values=lambda: np.ones(4, dtype=np.float32)
        ),
    }

    with pytest.raises(RuntimeError, match="synthetic denoiser failure"):
        adapter._depth_wrapper(projection, {})(
            Executor(),
            torch.zeros((1, 4, 4, 4)),
            transformer_options=options,
        )

    assert diffusion.first is original
    assert projection.control_tokens is None
    assert projection.token_strength is None


def test_depth_encoding_uses_inference_only_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"inference": False}

    class FakeTensor:
        def to(self, **_kwargs):
            return self

        def unsqueeze(self, _dimension):
            return self

    class InferenceMode:
        def __enter__(self):
            state["inference"] = True

        def __exit__(self, *_args):
            state["inference"] = False

    class FakeVae:
        def encode(self, _pixels):
            assert state["inference"]
            return "latent"

    fake_torch = SimpleNamespace(
        float32="float32",
        from_numpy=lambda _values: FakeTensor(),
        inference_mode=InferenceMode,
    )
    monkeypatch.setattr(adapter, "torch", fake_torch)
    monkeypatch.setattr(
        adapter,
        "process_depth_latent_for_model",
        lambda _model, latent: latent,
    )

    result = encode_depth_control(
        FakeVae(),
        object(),
        np.ones((16, 16), dtype=np.float32),
    )

    assert result.value == "latent"
    assert len(result.source_sha256) == 64
    assert state["inference"] is False


def test_clear_depth_control_removes_job_state_and_projection() -> None:
    original = object()
    projection = SimpleNamespace(
        control_tokens=object(),
        token_strength=object(),
        original_first=original,
    )
    diffusion = SimpleNamespace(first=projection)
    options = {
        adapter.DEPTH_LATENT_KEY: object(),
        adapter.DEPTH_TOKEN_STRENGTH_KEY: object(),
    }

    class FakePatcher:
        model_options = {"transformer_options": options}
        model = SimpleNamespace(diffusion_model=diffusion)

        @staticmethod
        def get_attachment(key):
            assert key == adapter.DEPTH_ATTACHMENT_KEY
            return {"projection": projection}

        @staticmethod
        def remove_injections(key):
            assert key == adapter.DEPTH_ATTACHMENT_KEY

    clear_depth_control(FakePatcher())

    assert options == {}
    assert diffusion.first is original
    assert projection.control_tokens is None
    assert projection.token_strength is None
