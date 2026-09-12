import mlx.core as mx
import mlx.nn as nn


@mx.compile
def swiglu_activation(x_up: mx.array, x_gate: mx.array) -> mx.array:
    return nn.silu(x_gate) * x_up
