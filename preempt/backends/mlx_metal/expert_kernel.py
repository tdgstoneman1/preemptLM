from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

from functools import partial

import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import (
    _gather_sort,
    _scatter_unsort,
)

from .types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerWeights,
)

# TODO add support for expert layer bias terms in forward pass
# TODO fix hallucinated 'row' terminology, confusing


# TODO docstring, 'input_dims' confusing
def _linear_proj_matmul(
    x: mx.array, projection: WeightsTensor | QuantizedWeightsTensor
) -> mx.array:
    """Applies an expert's linear projection to its routed token assignments.

    Parameters
    ----------
    x : mx.array
        Flattened token activations of shape `(n_assignments, input_dims)`
        routed to the expert
    projection : WeightsTensor | QuantizedWeightsTensor
        The projection's quantized weights and parameters for the projection

    Returns
    -------
    mx.array
        Projected output tensor, shape `(n_assignments, d_model)`, where
        `d_model` is the feature dimension of the projection
    """
    if isinstance(projection, QuantizedWeightsTensor):
        return mx.quantized_matmul(
            x,
            projection.weight,
            projection.scales,
            projection.biases,
            transpose=True,
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
        )
    return mx.matmul(x, projection.weight.T)


def _group_token_assignments_by_expert(
    row_experts: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    toks_by_expert: dict[int, list[int]] = {}

    for i, expert in enumerate(row_experts):
        if expert < 0:
            raise ValueError()

        toks_by_expert.setdefault(expert, []).append(i)

    return tuple((expert, tuple(rows)) for expert, rows in toks_by_expert.items())


def _iter_expert_routed_tokens(
    x_flat: mx.array,
    row_experts: Sequence[int],  # TODO rename
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
    for expert_idx, rows in _group_token_assignments_by_expert(row_experts):
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
    grouped = mx.concatenate([*outputs], axis=0)
    inverse = np.argsort(np.concatenate(permutation)).astype(np.int32)

    return grouped[mx.array(inverse)].reshape(*leading_shape, top_k, -1)


def sequential_expert_matmul(
    x: mx.array,
    expert_idxs: mx.array,
    *,
    expert_forward_fn: Callable[
        [mx.array, Mapping[str, QuantizedWeightsTensor | WeightsTensor]], mx.array
    ],
    load_expert_fn: Callable[[int], ExpertLayerWeights],
) -> mx.array:
    """Sequentially runs the `top_k` selected experts for an MoE block.

    Replaces MLX fused grouped matrix multiplications with a sequential evaluation
    loop, enabling strict memory controls by loading and applying only one expert's
    weights at a time.

    Parameters
    ----------
    x : mx.array
        Array of hidden states, shape `(batch, tokens, d_model)`
    expert_idxs : Sequence[int]
        The router's selections, flattened in `(batch, tokens, top_k)` order
    expert_forward_fn : Callable[[mx.array, Mapping[str, QuantizedWeightsTensor]], mx.array]
        Callback applying one expert's projections to its assigned activations
    load_expert_fn : Callable[[int], ExpertLayerWeights]
        Callable that loads one expert's parameters (from disk if not already in memory)

    Returns
    -------
    mx.array
        Processed hidden states of shape `(batch, tokens, top_k, d_model)`,
        where `d_model` is the feature dimension of expert layer outputs

    Raises
    ------
    ValueError
        If the number of elements in `row_experts` does not equal the total
        number of tokens multiplied by `top_k`
    """
    _, _, top_k = expert_idxs.shape

    flat_idxs = [int(expert) for expert in expert_idxs.flatten().tolist()]  # type: ignore
    num_token_assignments = len(flat_idxs)
    n_tokens = math.prod(x.shape[:-1])

    if num_token_assignments != n_tokens * top_k:
        raise ValueError(
            f"Number of token assignments ({num_token_assignments}) does not match "
            f"expected count of {n_tokens * top_k!r} for {n_tokens!r} token(s) at "
            f"{top_k=!r}."
        )

    x_flat = x.reshape(-1, x.shape[-1])
    outputs: list[mx.array] = []
    perm: list[np.ndarray] = []

    for expert_idx, x_rows, row_array in _iter_expert_routed_tokens(
        x_flat, flat_idxs, top_k
    ):
        expert_proj = load_expert_fn(expert_idx)
        y_rows = expert_forward_fn(x_rows, expert_proj)
        # mx.async_eval(y_rows)
        outputs.append(y_rows)
        perm.append(row_array)

    return _reassemble(outputs, perm, x.shape[:-1], top_k)


@partial(mx.compile)
def apply_swiglu_activation(x_up: mx.array, x_gate: mx.array) -> mx.array:
    return nn.silu(x_gate) * x_up


def swiglu_forward_fn(
    x: mx.array,
    projections: Mapping[str, QuantizedWeightsTensor | WeightsTensor],
) -> mx.array:
    x_up = _linear_proj_matmul(x, projections["up_proj"])
    x_gate = _linear_proj_matmul(x, projections["gate_proj"])

    return _linear_proj_matmul(
        apply_swiglu_activation(x_up, x_gate), projections["down_proj"]
    )


def stacked_proj_matmul(
    x_2d: mx.array,
    indices_2d: mx.array,
    projections: Sequence[QuantizedWeightsTensor | WeightsTensor],
    is_quantized: bool,
) -> mx.array:
    """Performs matrix multiplication on 2D inputs and a stack of linear projection
    weights for multiple expert layers.

    Parameters
    ----------
    x_2d : mx.array
        2-dimensional input activations of shape `(batch * tokens, d_model)`
    indices_2d : mx.array
        Expert indices, shape `(batch * tokens, K)` (K=top_k for up/gate, K=1 for down)
    projections : Sequence[QuantizedWeightsTensor | WeightsTensor]
        Expert projection weights and optional quantization parameters
    is_quantized: bool
        Whether the projections are quantized

    Returns
    -------
    mx.array
        Output activations of shape `(batch * tokens, K, d_out)`
    """
    xs = mx.expand_dims(x_2d, (-2, -3))
    idx = indices_2d

    if do_sort := idx.size >= 64:
        xs, idx, inv_order = _gather_sort(xs, idx)
    else:
        inv_order = None

    if is_quantized:
        quant_proj = projections  # type: ignore
        w_stacked = mx.stack([p.weight for p in quant_proj], axis=0)
        scales_stacked = mx.stack([p.scales for p in quant_proj], axis=0)  # type: ignore
        biases_stacked = (
            mx.stack([p.biases for p in quant_proj], axis=0)  # type: ignore
            if quant_proj[0].biases is not None  # type: ignore
            else None
        )
        sample = quant_proj[0]
        y = mx.gather_qmm(  # type: ignore
            xs,
            w_stacked,
            scales_stacked,
            biases_stacked,
            rhs_indices=idx,
            transpose=True,
            group_size=sample.group_size,  # type: ignore
            bits=sample.bits,  # type: ignore
            mode=sample.mode,  # type: ignore
            sorted_indices=do_sort,
        )
    else:
        unquant_proj = projections  # type: ignore
        w_stacked = mx.stack([p.weight for p in unquant_proj], axis=0)  # type: ignore
        y = mx.gather_mm(  # type: ignore
            xs,
            w_stacked.swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=do_sort,
        )
    if do_sort:
        y = _scatter_unsort(y, inv_order, indices_2d.shape)

    return y.squeeze(-2)  # shape = (B * S, K, d_out)


def fused_expert_matmul(
    x: mx.array,
    expert_idxs: mx.array,
    load_expert_fn: Callable[[int], ExpertLayerWeights],
    is_quantized: bool,
) -> mx.array:
    """Fuses weights for router-selected experts and applies them in parallel.

    Parameters
    ----------
    x : mx.array
        Hidden states, shape `(B, S, d_model)`
    expert_idxs : mx.array
        Router top-k selections, shape `(B, S, top_k)`
    load_expert_fn : Callable[[int], ExpertLayerWeights]
        Callback retrieving projections for each unique expert
    is_quantized: bool
        Whether the model is quantized

    Returns
    -------
    mx.array
        Output activations of shape `(B, S, top_k, d_model)`
    """
    B, S, top_k = expert_idxs.shape
    d_model = x.shape[-1]

    # * Get unique experts and map them
    flat_indices = np.asarray(expert_idxs).flatten()
    unique_experts, inverse_map = np.unique(flat_indices, return_inverse=True)

    local_indices_2d = mx.array(inverse_map.reshape(B * S, top_k), dtype=mx.uint32)
    x_2d = x.reshape(B * S, d_model)

    # * Load projection weights for each unique expert
    expert_projs = [load_expert_fn(int(e)) for e in unique_experts]

    up_projs = [ep["up_proj"] for ep in expert_projs]
    gate_projs = [ep["gate_proj"] for ep in expert_projs]
    down_projs = [ep["down_proj"] for ep in expert_projs]

    # * Up and gate projections, shape: (B * S, d_model) -> (B * S, top_k, d_hidden)
    x_up = stacked_proj_matmul(
        x_2d, local_indices_2d, up_projs, is_quantized=is_quantized
    )
    x_gate = stacked_proj_matmul(
        x_2d, local_indices_2d, gate_projs, is_quantized=is_quantized
    )

    # * SwiGLU activation, shape: (B * S, top_k, d_hidden)
    x_swiglu = apply_swiglu_activation(x_up, x_gate)
    d_hidden = x_swiglu.shape[-1]

    # * Flatten for down proj: (B * S * top_k, d_hidden) with indices (B * S * top_k, 1)
    swiglu_2d = x_swiglu.reshape(B * S * top_k, d_hidden)
    down_indices_2d = local_indices_2d.reshape(B * S * top_k, 1)

    # * Down projection, shape: (B*S*top_k, 1, d_model) -> reshape to (B, S, top_k, d_model)
    x_down = stacked_proj_matmul(
        swiglu_2d, down_indices_2d, down_projs, is_quantized=is_quantized
    )
    return x_down.reshape(B, S, top_k, d_model)
