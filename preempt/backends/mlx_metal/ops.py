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

# TODO overhaul fragile token assignment grouping
# TODO add support for expert layer bias terms in forward pass


def _linear_proj_matmul(
    x: mx.array, projection: WeightsTensor | QuantizedWeightsTensor
) -> mx.array:
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


def _group_tok_routings_by_expert(
    flat_idxs: Sequence[int],
) -> dict[int, list[int]]:
    toks_by_expert: dict[int, list[int]] = {}

    for i, expert in enumerate(flat_idxs):
        assert expert >= 0, f"Got negative routed expert index: {expert!r}."
        toks_by_expert.setdefault(expert, []).append(i)

    return toks_by_expert


def _iter_expert_routed_tokens(
    x_flat: mx.array,
    flat_idxs: Sequence[int],
    top_k: int,
) -> Iterator[tuple[int, mx.array, mx.array]]:
    for expert_idx, toks in _group_tok_routings_by_expert(flat_idxs).items():
        assigned = mx.asarray(toks, dtype=mx.int32)
        yield expert_idx, x_flat[assigned // top_k], assigned


def _reassemble(
    outputs: Sequence[mx.array],
    permutation: list[mx.array],
    leading_dims: Sequence[int],
    top_k: int,
) -> mx.array:
    grouped = mx.concatenate([*outputs], axis=0)
    inverse = mx.argsort(mx.concatenate(permutation)).astype(mx.int32)

    return grouped[inverse].reshape(*leading_dims, top_k, -1)


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
    expert_idxs : mx.array
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
        If the number of total token assignments does not equal the total number of
        tokens * `top_k`
    """
    _, _, top_k = expert_idxs.shape

    flat_idxs = [int(expert) for expert in expert_idxs.flatten().tolist()]  # type: ignore
    num_token_assignments = len(flat_idxs)
    n_tokens = math.prod(x.shape[:-1])

    if num_token_assignments != n_tokens * top_k:
        raise ValueError(
            f"Number of token assignments ({num_token_assignments}) does not match "
            f"expected count of {n_tokens * top_k!r} for {n_tokens!r} token(s) and "
            f"{top_k=!r}."
        )

    x_flat = x.reshape(-1, x.shape[-1])
    outputs: list[mx.array] = []
    perms: list[mx.array] = []

    for expert_idx, routed_toks, perm in _iter_expert_routed_tokens(
        x_flat, flat_idxs, top_k
    ):
        expert_proj = load_expert_fn(expert_idx)
        activations = expert_forward_fn(routed_toks, expert_proj)
        outputs.append(activations)
        perms.append(perm)

    return _reassemble(outputs, perms, x.shape[:-1], top_k)


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

    local_indices_2d = mx.asarray(inverse_map.reshape(B * S, top_k), dtype=mx.uint32)
    x_2d = x.reshape(B * S, d_model)

    # * Load projection weights for each unique expert
    expert_projs = [load_expert_fn(int(e)) for e in unique_experts]

    up_projs = [proj["up_proj"] for proj in expert_projs]
    gate_projs = [proj["gate_proj"] for proj in expert_projs]
    down_projs = [proj["down_proj"] for proj in expert_projs]

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
