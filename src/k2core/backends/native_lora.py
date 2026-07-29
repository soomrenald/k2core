"""Strict non-destructive ordinary LoRA support for the native Krea2 graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from k2core.backends.native_quant import replace_submodule
from k2core.backends.native_transformer import (
    NativeKrea2Transformer,
    _map_linear_module,
)
from k2core.inference.errors import ConfigurationError, WeightMappingError
from k2core.inference.schemas import LoraSpec
from k2core.model import sha256_file


_PAIR_SUFFIXES = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
)
_AUXILIARY_SUFFIXES = (".alpha",)
_LOKR_SUFFIXES = (".lokr_w1", ".lokr_w2")


@dataclass(frozen=True, slots=True)
class NativeLoraTarget:
    source_module: str
    target_module: str
    down: Any
    up: Any
    rank: int
    alpha: float
    scale: float

    @property
    def storage_bytes(self) -> int:
        return int(self.down.numel() * self.down.element_size()) + int(
            self.up.numel() * self.up.element_size()
        )


@dataclass(frozen=True, slots=True)
class NativeLokrTarget:
    source_module: str
    target_module: str
    w1: Any
    w2: Any
    scale: float

    @property
    def storage_bytes(self) -> int:
        return int(self.w1.numel() * self.w1.element_size()) + int(
            self.w2.numel() * self.w2.element_size()
        )


@dataclass(frozen=True, slots=True)
class NativeLoraReport:
    lora_id: str
    display_name: str
    path: Path
    sha256: str
    strength: float
    tensor_count: int
    target_count: int
    target_modules: tuple[str, ...]
    adapter_types: tuple[str, ...]
    ranks: tuple[int, ...]
    adapter_bytes: int
    unmatched_keys: tuple[str, ...]
    status: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.lora_id,
            "display_name": self.display_name,
            "path": str(self.path),
            "sha256": self.sha256,
            "strength": self.strength,
            "tensor_count": self.tensor_count,
            "target_count": self.target_count,
            "target_modules": list(self.target_modules),
            "adapter_types": list(self.adapter_types),
            "ranks": list(self.ranks),
            "adapter_bytes": self.adapter_bytes,
            "unmatched_keys": list(self.unmatched_keys),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class _LoadedLora:
    specification: LoraSpec
    path: Path
    sha256: str
    tensor_count: int
    targets: tuple[NativeLoraTarget | NativeLokrTarget, ...]


def apply_native_loras(
    transformer: NativeKrea2Transformer,
    specifications: Sequence[LoraSpec],
) -> tuple[NativeLoraReport, ...]:
    """Validate every adapter, then install ordered forward deltas atomically."""

    if not specifications:
        return ()
    if not transformer.loaded:
        raise RuntimeError("native Krea2 transformer has been unloaded")

    torch, nn, functional = _import_runtime()
    loaded = tuple(_load_lora(specification) for specification in specifications)
    active = tuple(item for item in loaded if item.specification.strength != 0.0)

    target_adapters: dict[str, list[Any]] = {}
    for item in active:
        for target in item.targets:
            try:
                base = transformer.model.get_submodule(target.target_module)
            except AttributeError as error:
                raise WeightMappingError(
                    f"LoRA target does not exist: {target.source_module}",
                    technical_detail=target.target_module,
                    backend_name="native",
                    phase="lora",
                ) from error
            in_features = getattr(base, "in_features", None)
            out_features = getattr(base, "out_features", None)
            if isinstance(target, NativeLoraTarget):
                expected_down = (target.rank, in_features)
                expected_up = (out_features, target.rank)
                if (
                    tuple(target.down.shape) != expected_down
                    or tuple(target.up.shape) != expected_up
                ):
                    raise WeightMappingError(
                        f"LoRA target shapes are incompatible: {target.source_module}",
                        technical_detail=(
                            f"down={tuple(target.down.shape)} expected={expected_down}; "
                            f"up={tuple(target.up.shape)} expected={expected_up}"
                        ),
                        backend_name="native",
                        phase="lora",
                    )
                adapter = _delta_module(
                    torch,
                    nn,
                    functional,
                    target.down,
                    target.up,
                    target.scale,
                )
            else:
                valid_factors = target.w1.ndim == 2 and target.w2.ndim == 2
                expected_linear = (
                    (
                        int(target.w1.shape[0]) * int(target.w2.shape[0]),
                        int(target.w1.shape[1]) * int(target.w2.shape[1]),
                    )
                    if valid_factors
                    else None
                )
                if not valid_factors or expected_linear != (
                    out_features,
                    in_features,
                ):
                    raise WeightMappingError(
                        f"LoKr factors are incompatible: {target.source_module}",
                        technical_detail=(
                            f"w1={tuple(target.w1.shape)}; "
                            f"w2={tuple(target.w2.shape)}; "
                            f"expected linear=({out_features}, {in_features})"
                        ),
                        backend_name="native",
                        phase="lora",
                    )
                adapter = _lokr_delta_module(
                    torch,
                    nn,
                    functional,
                    target.w1,
                    target.w2,
                    target.scale,
                )
            target_adapters.setdefault(target.target_module, []).append(adapter)

    wrapper = _adapter_linear_class(nn)
    for target_module in sorted(target_adapters):
        base = transformer.model.get_submodule(target_module)
        replace_submodule(
            transformer.model,
            target_module,
            wrapper(base, target_adapters[target_module]),
        )

    return tuple(
        NativeLoraReport(
            lora_id=item.specification.lora_id,
            display_name=item.specification.name,
            path=item.path,
            sha256=item.sha256,
            strength=item.specification.strength,
            tensor_count=item.tensor_count,
            target_count=len(item.targets),
            target_modules=tuple(target.target_module for target in item.targets),
            adapter_types=tuple(
                sorted(
                    {
                        "lora" if isinstance(target, NativeLoraTarget) else "lokr"
                        for target in item.targets
                    }
                )
            ),
            ranks=tuple(
                sorted(
                    {target.rank for target in item.targets if isinstance(target, NativeLoraTarget)}
                )
            ),
            adapter_bytes=sum(target.storage_bytes for target in item.targets),
            unmatched_keys=(),
            status=(
                "disabled_zero_strength" if item.specification.strength == 0.0 else "applied_global"
            ),
        )
        for item in loaded
    )


def _load_lora(specification: LoraSpec) -> _LoadedLora:
    if not specification.global_scope or specification.region_ids:
        raise WeightMappingError(
            "Ordinary native LoRA loading requires global scope.",
            technical_detail=specification.name,
            backend_name="native",
            phase="lora",
            remediation="Use the ComfyUI backend until regional LoRA Gate 8.",
        )
    path = specification.path.expanduser().resolve(strict=True)
    if path.suffix.casefold() != ".safetensors":
        raise ValueError("native LoRA files must use the .safetensors format")
    if specification.strength == 0.0:
        return _LoadedLora(
            specification=specification,
            path=path,
            sha256=sha256_file(path),
            tensor_count=0,
            targets=(),
        )

    safe_open = _import_safe_open()
    with safe_open(str(path), framework="pt", device="cpu") as source:
        tensors = {key: source.get_tensor(key) for key in source.keys()}
    has_lora = any(
        key.endswith(suffix) for key in tensors for pair in _PAIR_SUFFIXES for suffix in pair
    )
    has_lokr = any(key.endswith(_LOKR_SUFFIXES) for key in tensors)
    if has_lora and has_lokr:
        raise WeightMappingError(
            "A native adapter file cannot mix LoRA and LoKr targets.",
            backend_name="native",
            phase="lora",
        )
    grouped = _group_lokr_tensors(tensors) if has_lokr else _group_lora_tensors(tensors)
    targets = []
    mapped_targets: set[str] = set()
    for source_module in sorted(grouped):
        values = grouped[source_module]
        target_module = _map_lora_module(source_module)
        if target_module in mapped_targets:
            raise WeightMappingError(
                f"multiple LoRA source names map to one target: {target_module}",
                technical_detail=source_module,
                backend_name="native",
                phase="lora",
            )
        mapped_targets.add(target_module)
        if has_lokr:
            w1, w2 = values[".lokr_w1"], values[".lokr_w2"]
            if w1.ndim != 2 or w2.ndim != 2:
                raise WeightMappingError(
                    f"native Krea2 supports linear LoKr factors only: {source_module}",
                    backend_name="native",
                    phase="lora",
                )
            targets.append(
                NativeLokrTarget(
                    source_module=source_module,
                    target_module=target_module,
                    w1=w1,
                    w2=w2,
                    scale=specification.strength,
                )
            )
        else:
            down, up = _select_pair(source_module, values)
            if down.ndim != 2 or up.ndim != 2:
                raise WeightMappingError(
                    f"native Krea2 supports linear LoRA matrices only: {source_module}",
                    backend_name="native",
                    phase="lora",
                )
            rank = int(down.shape[0])
            if rank <= 0 or int(up.shape[1]) != rank:
                raise WeightMappingError(
                    f"LoRA rank is incompatible: {source_module}",
                    technical_detail=f"down={tuple(down.shape)}; up={tuple(up.shape)}",
                    backend_name="native",
                    phase="lora",
                )
            alpha_tensor = values.get(".alpha")
            alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
            targets.append(
                NativeLoraTarget(
                    source_module=source_module,
                    target_module=target_module,
                    down=down,
                    up=up,
                    rank=rank,
                    alpha=alpha,
                    scale=specification.strength * alpha / rank,
                )
            )
    if not targets:
        raise WeightMappingError(
            f"LoRA contains no executable adapter targets: {specification.name}",
            backend_name="native",
            phase="lora",
        )
    return _LoadedLora(
        specification=specification,
        path=path,
        sha256=sha256_file(path),
        tensor_count=len(tensors),
        targets=tuple(targets),
    )


def _group_lora_tensors(
    tensors: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    unknown = []
    for key, tensor in tensors.items():
        matched = False
        for down_suffix, up_suffix in _PAIR_SUFFIXES:
            for suffix in (down_suffix, up_suffix):
                if key.endswith(suffix):
                    base = key[: -len(suffix)]
                    grouped.setdefault(base, {})[suffix] = tensor
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        for suffix in _AUXILIARY_SUFFIXES:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                grouped.setdefault(base, {})[suffix] = tensor
                matched = True
                break
        if not matched:
            unknown.append(key)
    if unknown:
        raise WeightMappingError(
            "LoRA contains unsupported or unmatched tensors.",
            technical_detail=", ".join(sorted(unknown)[:20]),
            backend_name="native",
            phase="lora",
            remediation="Use a complete standard linear LoRA without DoRA or LoKr tensors.",
        )
    for base, values in grouped.items():
        complete_pairs = [
            (down, up) for down, up in _PAIR_SUFFIXES if down in values and up in values
        ]
        if len(complete_pairs) != 1:
            raise WeightMappingError(
                f"LoRA target must contain exactly one complete pair: {base}",
                technical_detail=", ".join(sorted(values)),
                backend_name="native",
                phase="lora",
            )
    return grouped


def _group_lokr_tensors(
    tensors: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    unknown = []
    for key, tensor in tensors.items():
        matched = False
        for suffix in (*_LOKR_SUFFIXES, *_AUXILIARY_SUFFIXES):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                grouped.setdefault(base, {})[suffix] = tensor
                matched = True
                break
        if not matched:
            unknown.append(key)
    if unknown:
        raise WeightMappingError(
            "LoKr contains unsupported decomposed, Tucker, or DoRA tensors.",
            technical_detail=", ".join(sorted(unknown)[:20]),
            backend_name="native",
            phase="lora",
            remediation="Use the current direct w1/w2 linear LoKr format.",
        )
    for base, values in grouped.items():
        missing = sorted(set(_LOKR_SUFFIXES) - set(values))
        if missing:
            raise WeightMappingError(
                f"LoKr target is incomplete: {base}",
                technical_detail=", ".join(missing),
                backend_name="native",
                phase="lora",
            )
    return grouped


def _select_pair(source_module: str, values: Mapping[str, Any]) -> tuple[Any, Any]:
    for down_suffix, up_suffix in _PAIR_SUFFIXES:
        if down_suffix in values and up_suffix in values:
            return values[down_suffix], values[up_suffix]
    raise WeightMappingError(
        f"LoRA target is incomplete: {source_module}",
        backend_name="native",
        phase="lora",
    )


def _map_lora_module(source: str) -> str:
    normalized = source
    if normalized.startswith("diffusion_model."):
        normalized = normalized.removeprefix("diffusion_model.")
    if normalized == "last.modulation.lin":
        raise WeightMappingError(
            "LoRA targets a bare Krea2 modulation parameter with no forward module.",
            technical_detail=source,
            backend_name="native",
            phase="lora",
            remediation="Use a LoRA whose targets are executable linear modules.",
        )
    return _map_linear_module(normalized)


def _delta_module(torch, nn, functional, down, up, scale: float):
    class LoraDelta(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Parameter(down, requires_grad=False)
            self.up = nn.Parameter(up, requires_grad=False)
            self.scale = float(scale)

        def forward(self, inputs):
            dtype = self.down.dtype
            hidden = functional.linear(inputs.to(dtype=dtype), self.down)
            delta = functional.linear(hidden, self.up)
            return delta.to(dtype=inputs.dtype) * self.scale

    return LoraDelta()


def _lokr_delta_module(torch, nn, functional, w1, w2, scale: float):
    class LokrDelta(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Parameter(w1, requires_grad=False)
            self.w2 = nn.Parameter(w2, requires_grad=False)
            self.scale = float(scale)

        def forward(self, inputs):
            dtype = inputs.dtype
            first = self.w1.to(dtype=dtype)
            second = self.w2.to(dtype=dtype)
            input_groups = int(first.shape[1])
            grouped = inputs.reshape(*inputs.shape[:-1], input_groups, -1)
            hidden = functional.linear(grouped, second)
            crossed = hidden.transpose(-1, -2)
            output = functional.linear(crossed, first)
            output = output.transpose(-1, -2).reshape(
                *output.shape[:-2],
                -1,
            )
            return output * self.scale

    return LokrDelta()


def _adapter_linear_class(nn):
    class AdapterLinear(nn.Module):
        def __init__(self, base, adapters) -> None:
            super().__init__()
            self.base = base
            self.adapters = nn.ModuleList(adapters)
            self.in_features = base.in_features
            self.out_features = base.out_features

        def forward(self, inputs):
            output = self.base(inputs)
            for adapter in self.adapters:
                output = output + adapter(inputs).to(dtype=output.dtype)
            return output

    return AdapterLinear


def _import_runtime():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 LoRA execution requires PyTorch.",
            technical_detail=str(error),
            backend_name="native",
            phase="lora",
        ) from error
    return torch, nn, functional


def _import_safe_open():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 LoRA loading requires safetensors.",
            technical_detail=str(error),
            backend_name="native",
            phase="lora",
        ) from error
    return safe_open


__all__ = [
    "NativeLoraReport",
    "NativeLoraTarget",
    "NativeLokrTarget",
    "apply_native_loras",
]
