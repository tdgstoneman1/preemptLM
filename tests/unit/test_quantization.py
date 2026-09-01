from __future__ import annotations

import pytest

import mlx.core as mx

import attrs

from preempt.backends.mlx_metal.quantization import QuantSettings
from preempt.backends.mlx_metal.constants import MLX_QUANTIZED_TENSOR_PARTS
from preempt.backends.mlx_metal.utils import dtype_tag_from_arrays, make_encoding_tag


class TestQuantParams:
    def test_construction(self) -> None:
        q = QuantSettings(mode="affine", bits=4, group_size=64)
        assert q.mode == "affine"
        assert q.bits == 4
        assert q.group_size == 64

    def test_frozen(self) -> None:
        q = QuantSettings(mode="affine", bits=4, group_size=64)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            q.bits = 8  # type: ignore[misc]


class TestQuantizedTensorParts:
    def test_order(self) -> None:
        assert MLX_QUANTIZED_TENSOR_PARTS == ("weight", "scales", "biases")


class TestPayloadEncodingFor:
    def test_quantized(self) -> None:
        q = QuantSettings(mode="affine", bits=4, group_size=64)
        assert make_encoding_tag(q, "bf16") == "mlx-affine-q4-g64-bf16"

    def test_unquantized(self) -> None:
        assert make_encoding_tag(None, "bf16") == "mlx-unquantized-bf16"

    def test_f16(self) -> None:
        q = QuantSettings(mode="affine", bits=4, group_size=64)
        assert make_encoding_tag(q, "f16") == "mlx-affine-q4-g64-f16"


class TestScalarDtypeTag:
    def test_bf16(self) -> None:
        stacked = {"gate_proj.scales": mx.array([1.0], dtype=mx.bfloat16)}
        assert dtype_tag_from_arrays(stacked, quantized=True) == "bf16"

    def test_f16(self) -> None:
        stacked = {"gate_proj.scales": mx.array([1.0], dtype=mx.float16)}
        assert dtype_tag_from_arrays(stacked, quantized=True) == "f16"

    def test_disagreement_raises(self) -> None:
        stacked = {
            "gate_proj.scales": mx.array([1.0], dtype=mx.bfloat16),
            "up_proj.scales": mx.array([1.0], dtype=mx.float16),
        }
        with pytest.raises(ValueError, match="multiple"):
            dtype_tag_from_arrays(stacked, quantized=True)

    def test_unquantized_reads_weight_dtype(self) -> None:
        stacked = {"gate_proj.weight": mx.array([1.0], dtype=mx.float32)}
        assert dtype_tag_from_arrays(stacked, quantized=False) == "f32"

    def test_unsupported_dtype_raises(self) -> None:
        stacked = {"gate_proj.weight": mx.array([1.0], dtype=mx.float64)}
        with pytest.raises(ValueError, match="unsupported"):
            dtype_tag_from_arrays(stacked, quantized=False)

    def test_empty_raises(self) -> None:
        stacked: dict[str, mx.array] = {}
        with pytest.raises(ValueError):
            dtype_tag_from_arrays(stacked, quantized=True)
