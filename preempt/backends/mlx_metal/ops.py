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

from .types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerWeights,
    ExpertLayerQuants,
)
from .expert_cache import MlxExpertCache
from .utils import make_switchglu_weight_map

# TODO overhaul fragile token assignment grouping
# TODO add support for expert layer bias terms in forward pass


@mx.compile
def swiglu_activation(x_up: mx.array, x_gate: mx.array) -> mx.array:
    return nn.silu(x_gate) * x_up


def swiglu_forward(
    x: mx.array,
    projections: Mapping[str, QuantizedWeightsTensor | WeightsTensor],
) -> mx.array:
    x_up = _linear_proj_matmul(x, projections["up_proj"])
    x_gate = _linear_proj_matmul(x, projections["gate_proj"])

    return _linear_proj_matmul(
        swiglu_activation(x_up, x_gate),
        projections["down_proj"],
    )


def _group_routed_toks_by_expert(
    flat_idxs: mx.array,
) -> dict[int, list[int]]:
    toks_by_expert: dict[int, list[int]] = {}

    for i, expert in enumerate(flat_idxs.tolist()):  # type: ignore
        toks_by_expert.setdefault(int(expert), []).append(i)

    return toks_by_expert


def _iter_expert_routed_toks(
    x: mx.array,
    expert_idxs: mx.array,
    top_k: int,
) -> Iterator[tuple[int, mx.array, mx.array]]:
    flat_idxs = expert_idxs.flatten()

    for expert_idx, toks in _group_routed_toks_by_expert(flat_idxs).items():
        permutation = mx.asarray(toks, dtype=mx.int32)
        routed_toks = x.reshape((-1, x.shape[-1]))[permutation // top_k]

        yield expert_idx, routed_toks, permutation


def _reassemble(
    outputs: Sequence[mx.array],
    permutation: list[mx.array],
    leading_dims: Sequence[int],
    top_k: int,
) -> mx.array:
    grouped = mx.concatenate([*outputs], axis=0)
    inverse = mx.argsort(mx.concatenate(permutation)).astype(mx.int32)

    return grouped[inverse].reshape(*leading_dims, top_k, -1)


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
    model_fingerprint: str,
    block_idx: int,
) -> Generator[ExpertKey, None, None]:
    """Converts expert indices to `ExpertKey`, reads them concurrently from disk, and yields
    experts' keys for as they are loaded.
    """
    keys = [
        expert_idx_to_key(
            idx,
            model_fingerprint=model_fingerprint,
            block_idx=block_idx,
        )
        for idx in expert_idxs.flatten().tolist()  # type: ignore
    ]
    yield from loader.load(keys)


def sequential_expert_matmul(
    x: mx.array,
    expert_idxs: mx.array,
    *,
    expert_loader: DiskBackedExpertLoader,
    expert_cache: MlxExpertCache,
    top_k: int,
    block_idx: int,
    model_fingerprint: str,
    quants: ExpertLayerQuants | None,
) -> mx.array:
    toks_by_expert = {
        expert_idx: (toks, perm)
        for expert_idx, toks, perm in _iter_expert_routed_toks(x, expert_idxs, top_k)
    }
    outputs: list[mx.array] = []
    perms: list[mx.array] = []

    for expert_key in load_experts_from_bank(
        expert_loader, expert_idxs, model_fingerprint, block_idx
    ):
        routed_toks, perm = toks_by_expert[expert_key.expert_idx]
        weights = expert_cache.get(expert_key)
        weights = make_switchglu_weight_map(weights, quants)

        outputs.append(swiglu_forward(routed_toks, weights))
        perms.append(perm)

    return _reassemble(outputs, perms, x.shape[:-1], top_k)


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
