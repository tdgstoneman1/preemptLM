from __future__ import annotations

from collections.abc import Generator, Callable, Iterator, Mapping, Sequence

from functools import partial

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import (
    _gather_sort,
    _scatter_unsort,
)

from preempt.engine.expert_io.loader import DiskBackedExpertLoader
from preempt.datamodel.identity import ExpertKey
from preempt.engine.expert_io.cache_manager import ExpertCacheManager

from ..types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerWeights,
    ExpertLayerQuants,
)
from ..expert_cache import MlxExpertCache
from ..utils import make_switchglu_weight_map

from .compiled import swiglu_activation


def _stacked_proj_matmul(
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
    unique_experts, inverse_map = np.unique(
        flat_indices,
        return_inverse=True,
    )
    local_indices_2d = mx.asarray(
        inverse_map.reshape(B * S, top_k),
        dtype=mx.uint32,
    )
    x_2d = x.reshape(B * S, d_model)

    # * Load projection weights for each unique expert
    expert_projs = [load_expert_fn(int(e)) for e in unique_experts]

    up_projs = [proj["up_proj"] for proj in expert_projs]
    gate_projs = [proj["gate_proj"] for proj in expert_projs]
    down_projs = [proj["down_proj"] for proj in expert_projs]

    # * Up and gate projections, shape: (B * S, d_model) -> (B * S, top_k, d_hidden)
    x_up = _stacked_proj_matmul(
        x_2d,
        local_indices_2d,
        up_projs,
        is_quantized=is_quantized,
    )
    x_gate = _stacked_proj_matmul(
        x_2d,
        local_indices_2d,
        gate_projs,
        is_quantized=is_quantized,
    )
    # * SwiGLU activation, shape: (B * S, top_k, d_hidden)
    x_swiglu = swiglu_activation(x_up, x_gate)
    d_hidden = x_swiglu.shape[-1]

    # * Flatten for down proj: (B * S * top_k, d_hidden) with indices (B * S * top_k, 1)
    swiglu_2d = x_swiglu.reshape(B * S * top_k, d_hidden)
    down_indices_2d = local_indices_2d.reshape(B * S * top_k, 1)

    # * Down projection, shape: (B*S*top_k, 1, d_model) -> reshape to (B, S, top_k, d_model)
    x_down = _stacked_proj_matmul(
        swiglu_2d,
        down_indices_2d,
        down_projs,
        is_quantized=is_quantized,
    )
    return x_down.reshape(B, S, top_k, d_model)
