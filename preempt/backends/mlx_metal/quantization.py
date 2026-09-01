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
    SCALAR_DTYPE_TAGS,
    MLX_ENCODING_QUANTIZED_TEMPLATE,
    MLX_ENCODING_UNQUANTIZED_TEMPLATE,
)


@attrs.define(frozen=True, kw_only=True)
class MlxQuantParams:
    """Quantization parameters for an MLX module.

    Specifies the structural metadata required to correctly decode and
    interpret packed expert weights.

    Parameters
    ----------
    mode : str
        The quantization algorithm used (e.g., `"affine"`)
    bits : int
        The precision of each quantized weight, in bits
    group_size : int
        The number of individual weights sharing a single scale and bias
    """

    mode: str = field()  # TODO use literal instead of string
    bits: int = field()
    group_size: int = field()


# TODO rename scalar_tag to scalar_dtype for clarity
# def make_encoding_tag(quant: MlxQuantParams | None, scalar_tag: str) -> str:
#     """Generates the payload encoding tag for the expert bank.

#     Produces a formatted string identifying the quantization state and
#     exact scalar dtype of the tensor. This tag must be kept in sync with
#     the parsing logic in `preempt.core.encoding.parse_payload_encoding_tag()`
#     to ensure bit-exact decoding.

#     Parameters
#     ----------
#     quant : MlxQuantParams | None
#         The quantization parameters used for the experts, or `None` if
#         unquantized
#     scalar_tag : str
#         The short tag representing the scalar dtype (e.g. `"bf16"`) as
#         derived from `dtype_tag_from_arrays()`

#     Returns
#     -------
#     str
#         The formatted encoding tag (e.g., `"mlx-affine-q4-g64-bf16"` or
#         `"mlx-unquantized-bf16"`)
#     """
#     if quant is None:
#         # return f"mlx-unquantized-{scalar_tag}"
#         return MLX_ENCODING_UNQUANTIZED_TEMPLATE.substitute(scalar_tag=scalar_tag)

#     # return f"mlx-{quant.mode}-q{quant.bits}-g{quant.group_size}-{scalar_tag}"
#     return MLX_ENCODING_QUANTIZED_TEMPLATE.substitute(
#         mode=quant.mode,
#         bits=quant.bits,
#         group_size=quant.group_size,
#         scalar_tag=scalar_tag,
#     )
