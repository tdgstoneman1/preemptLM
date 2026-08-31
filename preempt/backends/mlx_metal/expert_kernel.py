"""Quantized MoE kernel logic for sequentially applying experts in the forward
pass.

Implements a streaming-compatible evaluation loop using `mx.quantized_matmul`,
allowing individual expert weights to be loaded, evaluated, and offloaded one
at a time to enforce strict memory bounds during inference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear, SwitchGLU

from preempt.engine.expert_batching import group_rows_by_expert

from preempt.core.exceptions import EngineIncompatibilityError

from .constants import MLX_QUANT_PARAMS


# TODO move dataclasses to separate module
# TODO add support for expert layer bias terms in forward pass
# TODO prepend attrs dataclass names with `Mlx` prefix
# TODO fix hallucinated 'row' terminology, confusing
@attrs.define(kw_only=True, frozen=True)
class ProjectionQuantParams:
    """Quantization parameters for a single projection stored as scalars.

    Parameters are captured during instantiation of instrumented module
    wrappers, such as like `InstrumentedQwen3NextMoE`. This avoids having
    to dynamically inspect the inner layer's weights which may be stripped
    or offloaded from memory during the forward pass.
    """

    group_size: int = field()
    bits: int = field()
    mode: str = field()


# TODO move to quantization/
@attrs.define(kw_only=True, frozen=True)
class SwitchQuantParams:
    """Quantization parameters for all projections within a single expert layer.

    Stored as a mapping keyed by projection name to maintain agnosticism across
    different MoE architectures and future conversion utilities.

    Attributes
    ----------
    params : Mapping[str, ProjectionQuantParams]
        Mapping of projection names to their respective quantization parameters
    """

    params: Mapping[str, ProjectionQuantParams] = field()


@attrs.define(kw_only=True, frozen=True, eq=False)
# TODO move to quantization/
class QuantizedProjection:
    """Quantized weight tensor for one of an expert's linear projections and
    its quantization parameters

    Attributes
    ----------
    weight : mx.array
        The expert's quantized projection weights
    scales : mx.array
        The expert's per-group scales
    biases : mx.array | None
        Per-group affine quantization biases, if applicable
    group_size : int
        Quantization group size
    bits : int
        Quantization bit width
    mode : str
        Quantization mode
    """

    weight: mx.array = field()
    scales: mx.array = field()
    biases: mx.array | None = field()
    group_size: int = field()
    bits: int = field()
    mode: str = field()


@attrs.define(kw_only=True, frozen=True, eq=False)
class UnquantizedProjection:
    weight: mx.array = field()


@attrs.define(kw_only=True, frozen=True, eq=False)
class ExpertProjections:
    """All quantized linear projection weights for one expert.

    Keyed by projection name to abstract away MoE architecture.
    Supports tensors in memory or read from disk.

    Attributes
    ----------
    projections : Mapping[str, QuantizedProjection]
        Mapping of projection names to weights arrays
    """

    projections: Mapping[str, QuantizedProjection | UnquantizedProjection] = field()


def describe_switch_quantization(  # TODO rename
    switch_mlp: SwitchGLU,
    projection_names: Sequence[str],
) -> SwitchQuantParams | None:
    """Reads quantization parameters for the linear projection layer weights
    in `switch_mlp`.

    Parameters
    ----------
    switch_mlp : SwitchGLU
        A fused multi-expert module containing the stacked weights for all
        expert layers in an MoE block
    projection_names : Sequence[str]
        Projection names to read (e.g. from `architecture.projection_names`)

    Returns
    -------
    SwitchQuantParams
        Per-projection quantization parameters keyed by name.

    Raises
    ------
    TypeError
        If a layer in `switch_mlp` is not an instance of `QuantizedSwitchLinear`
    EngineIncompatibilityError
        If a layer in `switch_mlp` has a bias (currently unsupported in the
        forward pass)
    """
    params: dict[str, ProjectionQuantParams] = {}

    for name in projection_names:
        module = getattr(switch_mlp, name)

        if not isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
            raise TypeError(
                f"`{name}` is of unsupported type `{type(module).__name__}`. Only "
                "`SwitchLinear` or `QuantizedSwitchLinear` currently supported."
            )
        # TODO add support for bias
        if "bias" in module:
            raise EngineIncompatibilityError(
                f"`{name}` layer (in `{type(module).__name__}`) has an additive `bias`. "
                "This is currently unsupported in the per-expert forward pass. "
            )

        if all(hasattr(module, attr) for attr in MLX_QUANT_PARAMS):
            params[name] = ProjectionQuantParams(
                group_size=int(module.group_size),  # type: ignore
                bits=int(module.bits),  # type: ignore
                mode=str(module.mode),  # type: ignore
            )
    if params:
        return SwitchQuantParams(params=params)


# TODO rename, 'rows' terminology is confusing
# TODO add support for unquantized weights!
# TODO docstring, 'input_dims' confusing
def project_rows(
    x_rows: mx.array, projection: QuantizedProjection | UnquantizedProjection
) -> mx.array:
    """Applies an expert's projection to its routed token assignments.

    Parameters
    ----------
    x_rows : mx.array
        Flattened token activations of shape `(n_assignments, input_dims)`
        routed to the expert
    projection : QuantizedProjection
        The projection's quantized weights and parameters for the projection

    Returns
    -------
    mx.array
        Projected output tensor, shape `(n_assignments, d_model)`, where
        `d_model` is the feature dimension of the projection
    """
    if isinstance(projection, UnquantizedProjection):
        return mx.matmul(x_rows, projection.weight.T)

    return mx.quantized_matmul(
        x_rows,
        projection.weight,
        projection.scales,
        projection.biases,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode=projection.mode,
    )


def _iter_expert_rows(  # TODO rename
    x_flat: mx.array,
    row_experts: Sequence[int],
    top_k: int,
) -> Iterator[tuple[int, mx.array, np.ndarray]]:
    """Yields flattened token activations grouped by their assigned expert.

    Guarantees output in ascending expert order, with each expert group in
    ascending assignment index order (which dictates the execution order for
    disk reads).

    Parameters
    ----------
    x_flat : mx.array
        Flattened input activations, shape `(batch * tokens, input_dims)`
    row_experts : Sequence[int]
        Sequence of expert indices corresponding to each routing assignment
    top_k : int
        Number of routing selections per token

    Yields
    ------
    tuple[int, mx.array, np.ndarray]
        A tuple containing the expert index, the subset of `x_flat` assigned
        to that expert, and an array of the original assignment indices
    """
    for expert_idx, rows in group_rows_by_expert(row_experts):
        row_array = np.asarray(rows, dtype=np.int32)

        yield expert_idx, x_flat[mx.array(row_array // top_k)], row_array


def _reassemble(
    outputs: Sequence[mx.array],
    permutation: Sequence[np.ndarray],
    leading_shape: Sequence[int],  # TODO rename
    top_k: int,
) -> mx.array:
    """Restores the original shape and sequence ordering of routed expert
    outputs.

    Concatenates per-expert outputs and inverses the permutation used during
    grouping.

    Parameters
    ----------
    outputs : Sequence[mx.array]
        Per-expert projected outputs
    permutation : Sequence[np.ndarray]
        The original activation assignment indices corresponding to each expert
        group
    leading_shape : Sequence[int]
        The leading `batch` and `tokens` dimensions of the original input shape
    top_k : int
        Number of MoE router-selected experts per token

    Returns
    -------
    mx.array
        Reassembled output tensor, shape `(*leading_shape, top_k, d_model)`,
        where `d_model` is the feature dimension of expert layer outputs
    """
    grouped = mx.concatenate(list(outputs), axis=0)
    inverse = np.argsort(np.concatenate(permutation)).astype(np.int32)

    return grouped[mx.array(inverse)].reshape(*leading_shape, top_k, -1)


def sequential_run_selected_experts(
    x: mx.array,
    row_experts: Sequence[int],  # TODO rename, these aren't rows!
    *,
    top_k: int,
    expert_forward_fn: Callable[
        [mx.array, Mapping[str, QuantizedProjection | UnquantizedProjection]], mx.array
    ],
    expert_load_weights_fn: Callable[[int], ExpertProjections],
) -> mx.array:
    """Sequentially runs the `top_k` selected experts for an MoE block.

    Replaces MLX fused grouped matrix multiplications with a sequential evaluation
    loop, enabling strict memory controls by loading and applying only one expert's
    weights at a time.

    Parameters
    ----------
    x : mx.array
        Array of hidden states, shape `(batch, tokens, input_dims)`, where
        `input_dims` is the model's hidden dimension size
    row_experts : Sequence[int]
        The router's selections, flattened in `(batch, tokens, top_k)` order
    top_k : int
        Router selections per token
    expert_forward_fn : Callable[[mx.array, Mapping[str, QuantizedProjection]], mx.array]
        Callback applying one expert's projections to its assigned activations
    expert_load_weights_fn : Callable[[int], ExpertProjections]
        Callback retrieving one expert's parameters, invoked immediately prior
        to computation to support streaming paradigms

    Returns
    -------
    mx.array
        Processed hidden states, shape `(batch, tokens, top_k, d_model)`,
        where `d_model` is the feature dimension of expert layer outputs

    Raises
    ------
    ValueError
        If the number of elements in `row_experts` does not equal the total
        number of tokens multiplied by `top_k`
    """
    n_tokens = math.prod(x.shape[:-1])
    n_rows = len(row_experts)

    if n_rows != n_tokens * top_k:
        raise ValueError(
            f"Number of router assignments ({n_rows}) does not match expected "
            f"count of {n_tokens * top_k} for {n_tokens} token(s) at `{top_k=}`."
        )

    x_flat = x.reshape(-1, x.shape[-1])
    outputs: list[mx.array] = []
    perm: list[np.ndarray] = []

    for expert_idx, x_rows, row_array in _iter_expert_rows(x_flat, row_experts, top_k):
        expert_proj = expert_load_weights_fn(expert_idx)
        y_rows = expert_forward_fn(x_rows, expert_proj.projections)
        # mx.async_eval(y_rows)

        outputs.append(y_rows)
        perm.append(row_array)

    return _reassemble(outputs, perm, x.shape[:-1], top_k)
