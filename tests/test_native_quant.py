from __future__ import annotations

import unittest

from k2core.backends.native_quant import apply_compute_dtype
from k2core.inference import DTypePolicy


class FakeTensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to(self, *, dtype: str) -> "FakeTensor":
        return FakeTensor(dtype)


class FakeParameter:
    def __init__(self, dtype: str) -> None:
        self.data = FakeTensor(dtype)

    @property
    def dtype(self) -> str:
        return self.data.dtype


class FakeModule:
    def __init__(self, **buffers: FakeTensor) -> None:
        self._buffers = dict(buffers)
        for name, value in buffers.items():
            setattr(self, name, value)

    def named_buffers(self, *, recurse: bool):
        self.assert_no_recurse(recurse)
        return tuple(self._buffers.items())

    @staticmethod
    def assert_no_recurse(recurse: bool) -> None:
        if recurse:
            raise AssertionError("buffer traversal must visit each module exactly once")


class FakeModel(FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.bf16_parameter = FakeParameter("bfloat16")
        self.fp8_parameter = FakeParameter("float8_e4m3fn")
        self.child = FakeModule(
            bf16_cache=FakeTensor("bfloat16"),
            fp32_scale=FakeTensor("float32"),
        )

    def parameters(self):
        return (self.bf16_parameter, self.fp8_parameter)

    def modules(self):
        return (self, self.child)


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"


class NativeComputeDTypeTests(unittest.TestCase):
    def test_fp16_converts_only_bf16_parameters_and_buffers(self) -> None:
        model = FakeModel()

        apply_compute_dtype(model, FakeTorch, DTypePolicy.FLOAT16)

        self.assertEqual(model.bf16_parameter.dtype, "float16")
        self.assertEqual(model.fp8_parameter.dtype, "float8_e4m3fn")
        self.assertEqual(model.child.bf16_cache.dtype, "float16")
        self.assertEqual(model.child.fp32_scale.dtype, "float32")

    def test_auto_and_bfloat16_preserve_checkpoint_dtypes(self) -> None:
        for policy in (DTypePolicy.AUTO, DTypePolicy.BFLOAT16):
            with self.subTest(policy=policy):
                model = FakeModel()
                apply_compute_dtype(model, FakeTorch, policy)
                self.assertEqual(model.bf16_parameter.dtype, "bfloat16")
                self.assertEqual(model.child.bf16_cache.dtype, "bfloat16")

    def test_unimplemented_compute_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported native executable"):
            apply_compute_dtype(FakeModel(), FakeTorch, DTypePolicy.FLOAT32)


if __name__ == "__main__":
    unittest.main()
