from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping, Sequence

import mlx.core as mx

from preempt.engine.expert_io.loader import DiskBackedExpertLoader

from preempt.datamodel.identity import ExpertKey

from ..types import (
    WeightsTensor,
    QuantizedWeightsTensor,
    ExpertLayerQuants,
)
from ..expert_cache import MlxExpertCache
from ..utils import make_switchglu_weight_map

from .compiled import swiglu_activation

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
