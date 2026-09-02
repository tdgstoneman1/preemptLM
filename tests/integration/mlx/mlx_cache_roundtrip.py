"""Host smoke test: do store bytes become the checkpoint's own tensors again?

Reads real experts out of a packed store, installs them through
`MlxExpertCache`, and compares every resulting tensor against the same
tensor freshly re-sliced from the source checkpoint — dtype, shape, and bytes.

The dtype half is the point. `scales`/`biases` reach the store as `uint16`
because numpy has no `bfloat16`, and the store's `-bf16` encoding tag is the
only record of what they really are. A residency that converted rather than
re-labelled those bits would still produce a tensor of the right shape, the
model would still run, and every scale in it would be wrong. Comparing against
the checkpoint's own `mx.bfloat16` tensors is what turns that silent failure
into a visible one.

Also exercised: `resident_bytes` accounting, `KeyError` on a non-resident key,
the encoding-mismatch guard, and the property the whole residency design rests
on — that a graph built against an expert's weights still evaluates correctly
after that expert has been evicted, because eviction drops a Python reference
and MLX refcounting does the rest.

Usage (from the repo root, on the macOS host)::

    uv run tests/integration/mlx/mlx_residency_roundtrip.py \\
        --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \\
        --expert-bank expert-bank/qwen3.6-35b-4bit
"""

from __future__ import annotations

from collections.abc import Mapping

import argparse
import asyncio
from pathlib import Path

import attrs
import numpy as np

import mlx.core as mx

from preempt.backends.mlx_metal.expert_bank.qwen3_x import Qwen3_xArchAdapter
from preempt.backends.mlx_metal.expert_bank.serialization import (
    ShardTensorCache,
    build_tensor_shard_index,
    group_expert_tensors_by_layer,
)
from preempt.backends.mlx_metal.expert_cache import MlxExpertCache
from preempt.backends.mlx_metal.utils import mlx_to_numpy

from preempt.utils.hf_utils import resolve_model_dir

from preempt.expert_bank.encoding import parse_encoding_tag

from preempt.expert_bank.manifest import ExpertBankManifest
from preempt.expert_bank.banks import PreadExpertBank

from preempt.core.enums import ReadPriority

# TODO DE-SLOP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that `MlxExpertCache` rebuilds the checkpoint's "
        "own expert tensors from a packed store."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="MLX model id or local checkpoint directory the expert bank was packed from.",
    )
    parser.add_argument(
        "--expert-bank",
        required=True,
        type=Path,
        help="Expert bank directory holding `experts.bin` and `manifest.json`.",
    )
    return parser.parse_args()


def sample_pairs(manifest: ExpertBankManifest) -> tuple[tuple[int, int], ...]:
    """Pick three deterministic (layer, expert) pairs spanning the expert bank."""
    block_idxs = manifest.model_moe_spec.moe_block_idxs
    num_routed_experts = manifest.model_moe_spec.num_routed_experts

    return (
        (block_idxs[0], 0),
        (block_idxs[len(block_idxs) // 2], num_routed_experts // 2),
        (block_idxs[-1], num_routed_experts - 1),
    )


def source_expert_tensors(
    cache: ShardTensorCache,
    layer_tensors: Mapping[str, str],
    expert_idx: int,
) -> dict[str, mx.array]:
    """Slice one expert straight out of the checkpoint shards.

    Returns the tensors as `mx.array`s rather than numpy, so their real dtype
    (`bfloat16` for `scales`/`biases`) survives the comparison — which is the
    fact this script exists to check.
    """
    return {
        suffix: tensor[expert_idx]
        for suffix, tensor in cache.load_layer(layer_tensors).items()
    }


def compare_tensor(
    name: str,
    resident: mx.array,
    source: mx.array,
) -> None:
    """Fail unless `resident` is the checkpoint's tensor, bit for bit.

    Raises
    ------
    RuntimeError
        On any dtype, shape, or byte difference.
    """
    if resident.dtype != source.dtype:
        raise RuntimeError(
            f"{name}: residency built {resident.dtype}, checkpoint holds "
            f"{source.dtype}. A `bfloat16` tensor decoded as `uint16` (or "
            "converted through a float) corrupts every value it scales."
        )
    if tuple(resident.shape) != tuple(source.shape):
        raise RuntimeError(
            f"{name}: residency built shape {tuple(resident.shape)}, "
            f"checkpoint holds {tuple(source.shape)}."
        )

    # `mlx_to_numpy` bit-views bfloat16 as uint16 rather than casting, so this is a
    # byte comparison on both sides regardless of the tensor's real dtype.
    resident_bytes = mlx_to_numpy(resident)
    source_bytes = mlx_to_numpy(source)

    if not np.array_equal(resident_bytes, source_bytes):
        differing = int(np.count_nonzero(resident_bytes != source_bytes))
        raise RuntimeError(
            f"{name}: {differing} of {source_bytes.size} elements differ from "
            "the checkpoint."
        )


async def verify_expert(
    *,
    expert_bank: PreadExpertBank,
    residency: MlxExpertCache,
    cache: ShardTensorCache,
    layer_tensors: Mapping[str, str],
    block_idx: int,
    expert_idx: int,
) -> int:
    """Read, add, and byte-verify one expert. Returns its blob size."""
    key = expert_bank.key_for(block_idx, expert_idx)
    payload = await expert_bank.read(key, ReadPriority.DEMAND)
    residency.add(payload)

    tensors = residency.get(key)
    source = source_expert_tensors(cache, layer_tensors, expert_idx)

    if set(tensors) != set(source):
        raise RuntimeError(
            f"Residency built {sorted(tensors)!r}; the checkpoint layer holds "
            f"{sorted(source)!r}."
        )

    for spec in payload.tensor_specs:
        resident = tensors[spec.name]
        if tuple(resident.shape) != spec.shape:
            raise RuntimeError(
                f"{spec.name}: residency built shape {tuple(resident.shape)}, "
                f"spec records {spec.shape}."
            )
        compare_tensor(spec.name, resident, source[spec.name])

    print(
        f"Expert verified: layer {block_idx}, expert {expert_idx}, "
        f"{len(payload.tensor_specs)} tensors, {len(payload.data)} bytes match "
        "the checkpoint.",
        flush=True,
    )
    for spec in payload.tensor_specs:
        print(
            f"    {spec.name}: {tensors[spec.name].dtype} "
            f"{tuple(tensors[spec.name].shape)} (stored as {spec.dtype})",
            flush=True,
        )

    return len(payload.data)


async def verify_pending_graph_survives_eviction(
    *,
    expert_bank: PreadExpertBank,
    residency: MlxExpertCache,
    block_idx: int,
    expert_idx: int,
) -> None:
    """Evict an expert with a graph still pending against its weights.

    This is the property that let phase 3 drop the slot pool: eviction is a
    reference drop, so MLX keeps the buffer alive until the pending node runs.
    If this ever fails, the residency has grown a barrier or a mutation it
    should not have.
    """
    key = expert_bank.key_for(block_idx, expert_idx)
    payload = await expert_bank.read(key, ReadPriority.DEMAND)
    residency.add(payload)

    scales = residency.get(key)["gate_proj.scales"]
    pending = mx.sum(scales.astype(mx.float32))  # lazy: no math has run yet

    # An independent copy of the same bytes, evaluated now, is what the pending
    # node must still agree with once its own weights are no longer referenced.
    snapshot = mx.array(mlx_to_numpy(scales)).view(scales.dtype)
    expected = float(mx.sum(snapshot.astype(mx.float32)))

    residency.evict(key)

    observed = float(pending)  # forces evaluation, after the eviction
    if observed != expected:
        raise RuntimeError(
            f"Pending graph evaluated to {observed} after eviction; expected "
            f"{expected}. Eviction must be a reference drop and nothing else."
        )

    print(
        f"Pending-graph property verified: sum={observed} evaluated correctly "
        "after the expert was evicted.",
        flush=True,
    )


def verify_guards(residency: MlxExpertCache, store: PreadExpertBank) -> None:
    """Check the failures that must be loud rather than silent."""
    absent = store.key_for(0, 0, variant="not-a-variant")

    try:
        residency.get(absent)
    except KeyError:
        pass
    else:
        raise RuntimeError("`tensors` returned something for a non-resident key.")

    try:
        residency.evict(absent)
    except KeyError:
        pass
    else:
        raise RuntimeError("`evict` accepted a non-resident key.")

    print(
        "Guards verified: `tensors` and `evict` raise `KeyError` for a "
        "non-resident key.",
        flush=True,
    )


async def verify_encoding_guard(
    *, store: PreadExpertBank, residency: MlxExpertCache, block_idx: int
) -> None:
    """A payload from a differently encoded store must be refused, not decoded."""
    key = store.key_for(block_idx, 0)
    payload = await store.read(key, ReadPriority.DEMAND)
    foreign = attrs.evolve(payload, encoding="mlx-affine-q4-g64-f16")

    try:
        residency.add(foreign)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Residency installed a payload whose encoding names a different "
            "scalar type; that path silently corrupts every scale."
        )

    print(
        "Encoding guard verified: a mismatched `encoding` raises "
        "instead of decoding.",
        flush=True,
    )


async def run(args: argparse.Namespace) -> None:
    model_dir = resolve_model_dir(args.model)
    print(f"Model directory: {model_dir}", flush=True)
    print(f"Store directory: {args.expert_bank}", flush=True)

    arch = Qwen3_xArchAdapter()
    shard_index = build_tensor_shard_index(model_dir, arch)
    by_layer = group_expert_tensors_by_layer(shard_index, arch)
    cache = ShardTensorCache(shard_index=shard_index)

    with PreadExpertBank(args.expert_bank) as bank:
        manifest = bank.manifest
        encoding = parse_encoding_tag(manifest.encoding)
        print(f"Encoding: {manifest.encoding} -> {encoding!r}", flush=True)

        residency = MlxExpertCache(encoding=encoding)  # type: ignore
        installed_bytes = 0

        for block_idx, expert_idx in sample_pairs(manifest):
            if block_idx not in by_layer:
                raise RuntimeError(
                    f"Store names MoE layer {block_idx}, absent from the "
                    f"checkpoint's expert layers {sorted(by_layer)!r}."
                )

            installed_bytes += await verify_expert(
                expert_bank=bank,
                residency=residency,
                cache=cache,
                layer_tensors=by_layer[block_idx],
                block_idx=block_idx,
                expert_idx=expert_idx,
            )

            if residency.size() != installed_bytes:
                raise RuntimeError(
                    f"`size` is {residency.size()}; "
                    f"{installed_bytes} bytes have been installed."
                )
            if not bank.key_for(block_idx, expert_idx) not in residency:
                raise RuntimeError(
                    f"Expert (layer {block_idx}, expert {expert_idx}) is not "
                    "resident after `add`."
                )

        print(
            f"Accounting verified: {residency.size()} resident bytes "
            f"across 3 experts ({manifest.expert_num_bytes()} bytes each).",
            flush=True,
        )

        for block_idx, expert_idx in sample_pairs(manifest):
            residency.evict(bank.key_for(block_idx, expert_idx))

        if residency.size() != 0:
            raise RuntimeError(
                f"`size` is {residency.size()} after " "evicting every expert."
            )
        print("Eviction verified: residency is empty and holds 0 bytes.", flush=True)

        verify_guards(residency, bank)
        await verify_encoding_guard(
            store=bank, residency=residency, block_idx=sample_pairs(manifest)[0][0]
        )
        await verify_pending_graph_survives_eviction(
            expert_bank=bank,
            residency=residency,
            block_idx=sample_pairs(manifest)[0][0],
            expert_idx=0,
        )

    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    print("Residency round-trip smoke test passed.", flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
