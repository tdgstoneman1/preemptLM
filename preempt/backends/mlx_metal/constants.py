from typing import Any, Final

from string import Template

import mlx.core as mx

# Durable string tags for serialization
# TODO add all mlx dtypes
MLX_DTYPE_TAGS: Final[tuple[tuple[Any, str], ...]] = (
    (mx.bfloat16, "bf16"),
    (mx.float16, "f16"),
    (mx.float32, "f32"),
)

# Structural components of a quantized MLX tensor in blob order
MLX_QUANTIZED_TENSOR_PARTS: Final[tuple[str, str, str]] = ("weight", "scales", "biases")

MLX_QUANT_PARAMS: Final[tuple[str, str, str]] = ("group_size", "bits", "mode")


MLX_QUANTIZED_ENCODING_TEMPLATE: Final[Template] = Template(
    "mlx-${mode}-q${bits}-g${group_size}-${dtype}"
)
MLX_UNQUANTIZED_ENCODING_TEMPLATE: Final[Template] = Template(
    "mlx-unquantized-${dtype}"
)
# Sorted in explicit blob order matching MLX `SwitchGLU`
SWIGLU_PROJECTION_NAMES: Final[tuple[str, str, str]] = (
    "gate_proj",
    "up_proj",
    "down_proj",
)
