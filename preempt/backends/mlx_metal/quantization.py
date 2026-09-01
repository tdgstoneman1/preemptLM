"""MLX quantization parameters and encoding tag utilities.

Provides data structures and formatting logic for extracting quantization
metadata from MLX checkpoints. This includes generating the `payload_encoding`
tags used to durably record exact scalar types (e.g., `bfloat16`) across
serialization boundaries.
"""

from __future__ import annotations

from typing import Literal
from collections.abc import Mapping

import attrs
from attrs import field

import mlx.core as mx

from .constants import (
    MLX_DTYPE_TAGS,
    MLX_ENCODING_QUANTIZED_TEMPLATE,
    MLX_ENCODING_UNQUANTIZED_TEMPLATE,
)
from .enums import MlxQuantMode


@attrs.define(frozen=True, kw_only=True)
class MlxQuantParams:
    """Quantization parameters for an MLX module.

    Specifies the structural metadata required to correctly decode and
    interpret quantized serialized weights.

    Parameters
    ----------
    mode : MlxQuantModes
        The quantization algorithm used. A string may be passed,
        e.g., 'affine'.
    bits : int
        The precision of each quantized weight, in bits
    group_size : int
        The number of individual weights sharing a single scale and bias
    """

    mode: MlxQuantMode = field()
    bits: int = field()
    group_size: int = field()

    @mode.validator  # type: ignore
    def validate_mode(self, attribute, value) -> MlxQuantMode:
        return MlxQuantMode(value)
