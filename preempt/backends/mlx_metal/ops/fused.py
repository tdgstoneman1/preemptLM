from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import mlx.core as mx

from mlx_lm.models.switch_layers import (
    _gather_sort,
    _scatter_unsort,
)

from preempt.engine.expert_io.loader import DiskBackedExpertLoader

from ..types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerQuants,
)
from ..expert_cache import MlxExpertCache

from .experts import (
    load_experts_from_bank,
    make_switchglu_weight_map,
    swiglu_activation,
)


def _linear_projs_matmul(
    x_2d: mx.array,
    idxs_2d: mx.array,
    linear_projs: Sequence[WeightsTensor],
) -> mx.array:
    xs = mx.expand_dims(x_2d, (-2, -3))
    idx = idxs_2d
    inv_order = None

    if do_sort := idx.size >= 64:
        xs, idx, inv_order = _gather_sort(xs, idx)

    w_stacked = mx.stack([p.weight for p in linear_projs], axis=0)
    y = mx.gather_mm(  # type: ignore
        xs,
        w_stacked.swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    if do_sort:
        y = _scatter_unsort(y, inv_order, idxs_2d.shape)

    # output shape = (B * S, K, d_out)
    return y.squeeze(-2)


def _quantized_linear_projs_matmul(
    x_2d: mx.array,
    idxs_2d: mx.array,
    linear_projs: Sequence[QuantizedWeightsTensor],
) -> mx.array:
    xs = mx.expand_dims(x_2d, (-2, -3))
    idx = idxs_2d
    inv_order = None

    if do_sort := idx.size >= 64:
        xs, idx, inv_order = _gather_sort(xs, idx)

    w_stacked = mx.stack([p.weight for p in linear_projs], axis=0)
    scales_stacked = mx.stack([p.scales for p in linear_projs], axis=0)
    biases_stacked = (
        mx.stack([p.biases for p in linear_projs], axis=0)  # type: ignore
        if linear_projs[0].biases is not None
        else None
    )
    sample = linear_projs[0]

    y = mx.gather_qmm(
        xs,
        w_stacked,
        scales_stacked,
        biases_stacked,
        rhs_indices=idx,
        transpose=True,
        group_size=sample.group_size,
        bits=sample.bits,
        mode=sample.mode,
        sorted_indices=do_sort,
    )
    if do_sort:
        y = _scatter_unsort(y, inv_order, idxs_2d.shape)

    # output shape = (B * S, K, d_out)
    return y.squeeze(-2)


def fused_expert_matmul(
    x: mx.array,
    expert_idxs: mx.array,
    *,
    expert_loader: DiskBackedExpertLoader,
    expert_cache: MlxExpertCache,
    block_idx: int,
    num_experts: int,
    quants: ExpertLayerQuants | None,
    stream: mx.DeviceType | mx.Stream = mx.gpu,
) -> mx.array:
    """Performs matrix multiplication on fused expert weights.

    Parameters
    ----------
    x : mx.array
        Hidden states, shape `(B, S, d_model)`
    expert_idxs : mx.array
        Router top-k selections, shape `(B, S, K)`
    # TODO

    Returns
    -------
    mx.array
        Output activations of shape `(B, S, K, d_model)`
    """
    B, S, K = expert_idxs.shape
    d_model = x.shape[-1]

    # * Get unique experts and map them
    flat_indices = np.asarray(expert_idxs).flatten()
    unique_experts, inverse_map = np.unique(
        flat_indices,
        return_inverse=True,
    )
    local_idxs_2d = mx.asarray(
        inverse_map.reshape(B * S, K),
        dtype=mx.uint32,
    )
    x_2d = x.reshape(B * S, d_model)

    # * Load expert linear proj weights (out of order)
    expert_projs = {
        idx: expert_cache.get(idx)
        for idx in load_experts_from_bank(
            expert_loader,
            mx.asarray(unique_experts, copy=False),
            num_experts,
            block_idx,
            stream=stream,
        )
    }
    # * Reorder weights
    expert_projs = [
        make_switchglu_weight_map(expert_projs[idx], quants) for idx in unique_experts
    ]

    up_projs = [proj["up_proj"] for proj in expert_projs]
    gate_projs = [proj["gate_proj"] for proj in expert_projs]
    down_projs = [proj["down_proj"] for proj in expert_projs]

    # * Up and gate linear_projs, shape: (B * S, d_model) -> (B * S, K, d_hidden)
    matmul_fn = _quantized_linear_projs_matmul if quants else _linear_projs_matmul
    x_up = matmul_fn(
        x_2d,
        local_idxs_2d,
        up_projs,  # type: ignore
    )
    x_gate = matmul_fn(
        x_2d,
        local_idxs_2d,
        gate_projs,  # type: ignore
    )
    # * SwiGLU activation, shape: (B * S, K, d_hidden)
    x_swiglu = swiglu_activation(x_up, x_gate)
    d_hidden = x_swiglu.shape[-1]

    # * Flatten for down proj: (B * S * K, d_hidden) with indices (B * S * K, 1)
    swiglu_2d = x_swiglu.reshape(B * S * K, d_hidden)
    down_indices_2d = local_idxs_2d.reshape(B * S * K, 1)

    # * Down projection, shape: (B*S*K, 1, d_model) -> reshape to (B, S, K, d_model)
    x_down = matmul_fn(
        swiglu_2d,
        down_indices_2d,
        down_projs,  # type: ignore
    )
    return x_down.reshape(B, S, K, d_model)
