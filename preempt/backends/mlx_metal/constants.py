from typing import Any, Final

from string import Template

import mlx.core as mx

# Durable string dtype tags for serialization
MLX_DTYPE_TAGS: Final[tuple[tuple[Any, str], ...]] = (
    (mx.bool_, "bool_"),
    (mx.int16, "int8"),
    (mx.int16, "int16"),
    (mx.int32, "int32"),
    (mx.uint16, "uint8"),
    (mx.uint16, "uint16"),
    (mx.uint32, "uint32"),
    (mx.bfloat16, "bfloat16"),
    (mx.float16, "float16"),
    (mx.float32, "float32"),
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
# Matches MLX `SwitchGLU`, sorted in serialization order
SWITCHGLU_LINEAR_PROJ_NAMES: Final[tuple[str, str, str]] = (
    "gate_proj",
    "up_proj",
    "down_proj",
)
