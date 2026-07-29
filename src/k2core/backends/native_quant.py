"""Small PyTorch helpers for reviewed scaled-FP8 native checkpoints."""

from __future__ import annotations

from typing import Any

from k2core.inference.schemas import DTypePolicy


def replace_submodule(model: Any, path: str, replacement: Any) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, name, replacement)


def scaled_fp8_linear_class(torch, nn, functional):
    class ScaledFP8Linear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            bias: bool,
            device: str,
            quantize_input: bool = False,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.quantize_input = quantize_input
            self.weight = nn.Parameter(
                torch.empty(
                    (out_features, in_features),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty((), device=device, dtype=torch.float32),
                requires_grad=False,
            )
            self.bias = (
                nn.Parameter(
                    torch.empty(out_features, device=device),
                    requires_grad=False,
                )
                if bias
                else None
            )

        def forward(self, inputs):
            if self.quantize_input:
                scale = torch.ones(
                    (),
                    device=inputs.device,
                    dtype=torch.float32,
                )
                quantized_inputs = (
                    inputs.clamp(
                        min=-torch.finfo(torch.float8_e4m3fn).max,
                        max=torch.finfo(torch.float8_e4m3fn).max,
                    )
                    .to(torch.float8_e4m3fn)
                )
                if inputs.device.type == "cuda" and hasattr(
                    functional,
                    "scaled_mm",
                ):
                    input_shape = inputs.shape
                    flattened = quantized_inputs.reshape(
                        -1,
                        input_shape[-1],
                    ).contiguous()
                    bias = (
                        self.bias.to(dtype=inputs.dtype)
                        if self.bias is not None
                        else None
                    )
                    output = functional.scaled_mm(
                        flattened,
                        self.weight.t(),
                        scale_a=scale,
                        scale_recipe_a=functional.ScalingType.TensorWise,
                        scale_b=self.weight_scale,
                        scale_recipe_b=functional.ScalingType.TensorWise,
                        swizzle_a=functional.SwizzleType.NO_SWIZZLE,
                        swizzle_b=functional.SwizzleType.NO_SWIZZLE,
                        bias=bias,
                        output_dtype=inputs.dtype,
                        use_fast_accum=False,
                    )
                    return output.reshape(*input_shape[:-1], self.out_features)
                inputs = (
                    quantized_inputs.to(dtype=inputs.dtype)
                    * scale.to(dtype=inputs.dtype)
                )
            weight = self.weight.to(dtype=inputs.dtype) * self.weight_scale.to(
                dtype=inputs.dtype
            )
            bias = (
                self.bias.to(dtype=inputs.dtype)
                if self.bias is not None
                else None
            )
            return functional.linear(inputs, weight, bias)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, scaled_fp8=True, "
                f"quantize_input={self.quantize_input}"
            )

    return ScaledFP8Linear


def apply_compute_dtype(model: Any, torch: Any, policy: DTypePolicy) -> None:
    """Convert BF16 full-precision lanes without altering FP8 weights or FP32 scales."""

    if policy in {DTypePolicy.AUTO, DTypePolicy.BFLOAT16}:
        return
    if policy != DTypePolicy.FLOAT16:
        raise ValueError(f"unsupported native executable compute dtype: {policy.value}")
    for parameter in model.parameters():
        if parameter.dtype == torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.float16)
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.dtype == torch.bfloat16:
                setattr(module, name, buffer.to(dtype=torch.float16))


__all__ = [
    "apply_compute_dtype",
    "replace_submodule",
    "scaled_fp8_linear_class",
]
