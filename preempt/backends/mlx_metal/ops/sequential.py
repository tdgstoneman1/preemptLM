from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import mlx.core as mx

from preempt.engine.expert_io.loader import DiskBackedExpertLoader

from ..types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerQuants,
)
from ..expert_cache import MlxExpertCache

from .experts import (
    swiglu_activation,
    load_experts_from_bank,
    make_switchglu_weight_map,
)

# TODO overhaul fragile token assignment grouping
# TODO add support for expert layer bias terms in forward pass


def _linear_proj_matmul(
    x: mx.array,
    projection: WeightsTensor | QuantizedWeightsTensor,
    stream: mx.DeviceType | mx.Stream = mx.gpu,
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
            stream=stream,
        )
    return mx.matmul(x, projection.weight.T, stream=stream)


# TODO use array ops
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
    outputs: list[mx.array],
    permutation: list[mx.array],
    leading_dims: Sequence[int],
    top_k: int,
    stream: mx.DeviceType | mx.Stream = mx.gpu,
) -> mx.array:
    grouped = mx.concatenate(
        outputs,
        axis=0,
        stream=stream,
    )
    inverse = mx.concatenate(permutation, stream=stream)
    inverse = mx.argsort(inverse, stream=stream)

    return grouped[inverse].reshape(*leading_dims, top_k, -1)


def sequential_expert_matmul(
    x: mx.array,
    expert_idxs: mx.array,
    *,
    expert_loader: DiskBackedExpertLoader,
    expert_cache: MlxExpertCache,
    top_k: int,
    block_idx: int,
    num_experts: int,
    quants: ExpertLayerQuants | None,
    stream: mx.DeviceType | mx.Stream = mx.gpu,
) -> mx.array:
    toks_by_expert = {
        expert_idx: (toks, perm)
        for expert_idx, toks, perm in _iter_expert_routed_toks(x, expert_idxs, top_k)
    }
    outputs: list[mx.array] = []
    perms: list[mx.array] = []

    for glob_idx in load_experts_from_bank(
        expert_loader, expert_idxs, block_idx, num_experts, stream
    ):
        routed_toks, perm = toks_by_expert[glob_idx % num_experts]
        weights = expert_cache.get(glob_idx)
        weights = make_switchglu_weight_map(weights, quants)

        outputs.append(swiglu_forward(routed_toks, weights))
        perms.append(perm)

    return _reassemble(outputs, perms, x.shape[:-1], top_k, stream=stream)
