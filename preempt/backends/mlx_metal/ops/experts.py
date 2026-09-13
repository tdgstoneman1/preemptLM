from collections.abc import Generator, Mapping, Sequence

from attrs import asdict

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
)
from preempt.core.exceptions import EngineCompatibilityError
from preempt.datamodel.identity import ExpertKey
from preempt.engine.expert_io.loader import DiskBackedExpertLoader

from ..constants import SWITCHGLU_LINEAR_PROJ_NAMES, MLX_QUANT_PARAMS
from ..types import (
    ExpertLayerQuants,
    WeightsTensor,
    QuantizedWeightsTensor,
)
from ..quantization import QuantSettings


@mx.compile
def swiglu_activation(x_up: mx.array, x_gate: mx.array) -> mx.array:
    return nn.silu(x_gate) * x_up


# TODO remove
def expert_idx_to_key(
    expert_idx: int,
    *,
    model_fingerprint: str,
    block_idx: int,
) -> ExpertKey:
    """Helper for converting expert indices to `ExpertKey`."""
    return ExpertKey(
        model_fingerprint=model_fingerprint,
        block_idx=block_idx,
        expert_idx=expert_idx,
    )


def load_experts_from_bank(
    loader: DiskBackedExpertLoader,
    expert_idxs: mx.array,
    block_idx: int,
    num_experts: int,
    stream: mx.DeviceType | mx.Stream = mx.gpu,
) -> Generator[int, None, None]:
    idxs = [
        (block_idx * num_experts) + idx
        for idx in expert_idxs.flatten(stream=stream).tolist()  # type: ignore
    ]
    yield from loader.load(idxs)


def wrap_weight_map(
    weights: Mapping[str, mx.array],
    name: str,
    quants: ExpertLayerQuants | None,
) -> WeightsTensor | QuantizedWeightsTensor:
    """Helper for converting a weight map to `WeightsTensor` or
    `QuantizedWeightsTensor` if `quants` is provided
    """
    if quants is None:
        return WeightsTensor(
            weight=weights[f"{name}.weight"],
        )
    return QuantizedWeightsTensor(
        weight=weights[f"{name}.weight"],
        scales=weights[f"{name}.scales"],
        biases=weights.get(f"{name}.biases"),
        **asdict(quants[name]),
    )


def make_switchglu_weight_map(
    weights: Mapping[str, mx.array],
    quants: ExpertLayerQuants | None,
) -> dict[str, WeightsTensor | QuantizedWeightsTensor]:
    return {
        name: wrap_weight_map(weights, name, quants)
        for name in SWITCHGLU_LINEAR_PROJ_NAMES
    }


# TODO add support for bias
def get_expert_quants(
    switch_mlp: SwitchGLU,
    linear_projection_names: Sequence[str],
) -> ExpertLayerQuants | None:
    """Returns the quantization parameters for a switch layer's linear
    projection weights, or None if unquantized.

    Parameters
    ----------
    switch_mlp : SwitchGLU
        A fused multi-expert module containing the stacked weights for all
        routed experts in an MoE block
    linear_projection_names : Sequence[str]
        The layer's linear projection weight names

    Returns
    -------
    ExpertLayerQuants
        Per-projection quantization parameters keyed by name.

    Raises
    ------
    TypeError
        If a layer in `switch_mlp` is not an instance of `SwitchLinear` or
        `QuantizedSwitchLinear`
    EngineCompatibilityError
        If a layer in `switch_mlp` has a bias term (currently unsupported
        in the forward pass)
    """
    params: dict[str, QuantSettings] = {}
    for name in linear_projection_names:
        module = getattr(switch_mlp, name)

        if not isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
            raise TypeError(
                f"`{name}` is of unsupported type `{type(module).__name__}`. "
                "Only `SwitchLinear` and `QuantizedSwitchLinear` are currently "
                "supported."
            )
        if "bias" in module:
            raise EngineCompatibilityError(
                f"Layer {name!r} (in {type(module).__name__!r}) has a 'bias' "
                "term, which is currently unsupported."
            )
        if all(hasattr(module, attr) for attr in MLX_QUANT_PARAMS):
            params[name] = QuantSettings(
                group_size=int(module.group_size),  # type: ignore
                bits=int(module.bits),  # type: ignore
                mode=str(module.mode),  # type: ignore
            )

    return params or None
