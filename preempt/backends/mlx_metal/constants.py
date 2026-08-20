from typing import Any, Final

from string import Template

import mlx.core as mx

# Sorted in explicit blob order matching MLX `SwitchGLU`
SWIGLU_PROJECTION_NAMES: Final[tuple[str, str, str]] = (
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Structural components of a quantized MLX tensor in blob order
QUANTIZED_TENSOR_PARTS: Final[tuple[str, str, str]] = ("weight", "scales", "biases")

# The surrogate numpy dtype used to serialize `bfloat16` tensors
BIT_VIEWED_STORAGE_DTYPE: Final[str] = "uint16"

# Tags requiring BIT_VIEWED_STORAGE_DTYPE surrogate serialization
# Native numpy dtypes require no relabeling and omit entries here
BIT_VIEWED_SCALARS: Final[dict[str, mx.Dtype]] = {"bf16": mx.bfloat16}

# Durable string tags for exact decoding dtypes for serialization
# TODO add more dtypes, e.g. int8
# TODO make this an enum or something better than a tuple
SCALAR_DTYPE_TAGS: Final[tuple[tuple[Any, str], ...]] = (
    (mx.bfloat16, "bf16"),
    (mx.float16, "f16"),
    (mx.float32, "f32"),
)

MLX_ENCODING_QUANTIZED_TEMPLATE: Final[Template] = Template(
    "mlx-${mode}-q${bits}-g${group_size}-${scalar_tag}"
)
MLX_ENCODING_UNQUANTIZED_TEMPLATE: Final[Template] = Template(
    "mlx-unquantized-${scalar_tag}"
)
