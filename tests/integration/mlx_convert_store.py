"""Host smoke test: convert a few MoE layers into a packed store and byte-verify it.

Runs on macOS with MLX and real weights. Converts the first `--layers` MoE
layers of a checkpoint, reopens the result through `PackedExpertStore`, and
proves the stored bytes are the checkpoint's bytes for a sample of experts.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from preempt.backends.mlx_metal.convert import (
    QuantParams,
    ShardTensorCache,
    assemble_expert_blob,
    build_tensor_shard_index,
    convert_mlx_model_to_store,
    expert_arrays,
    group_expert_tensors_by_layer,
    payload_encoding_for,
    resolve_expert_quantization,
    resolve_model_dir,
)
from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ReadPriority
from preempt.storage.manifest import ExpertStoreManifest
from preempt.storage.packed_store import PackedExpertStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a few MoE layers to a packed expert store and verify it."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="MLX model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the store into (a `store` subdirectory).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of MoE layers to convert.",
    )
    parser.add_argument(
        "--expect-num-experts",
        type=int,
        default=256,
        help="Routed experts per layer the checkpoint must report.",
    )
    parser.add_argument(
        "--expect-top-k",
        type=int,
        default=8,
        help="Router top-k the checkpoint must report.",
    )
    parser.add_argument(
        "--expect-encoding",
        default="mlx-affine-q4-g64-bf16",
        help="Payload encoding tag the converted store must carry.",
    )
    return parser.parse_args()


def sample_pairs(manifest: ExpertStoreManifest) -> tuple[tuple[int, int], ...]:
    """Pick three deterministic (layer, expert) pairs spanning the store."""
    layer_idxs = manifest.topology.moe_layer_idxs
    num_experts = manifest.topology.num_experts

    return (
        (layer_idxs[0], 0),
        (layer_idxs[len(layer_idxs) // 2], num_experts // 2),
        (layer_idxs[-1], num_experts - 1),
    )


def verify_identity(manifest: ExpertStoreManifest, args: argparse.Namespace) -> None:
    """Check the manifest fields the compatibility gate is built on."""
    if manifest.model_id != args.model:
        raise RuntimeError(
            f"model_id is {manifest.model_id!r}; expected the id the engine "
            f"loads the model under, {args.model!r}."
        )
    if manifest.payload_encoding != args.expect_encoding:
        raise RuntimeError(
            f"payload_encoding is {manifest.payload_encoding!r}; "
            f"expected {args.expect_encoding!r}."
        )

    print(
        f"Identity verified: model_id={manifest.model_id!r}, "
        f"payload_encoding={manifest.payload_encoding!r}"
    )


def verify_topology(manifest: ExpertStoreManifest, args: argparse.Namespace) -> None:
    topology = manifest.topology

    if topology.num_experts != args.expect_num_experts:
        raise RuntimeError(
            f"num_experts is {topology.num_experts}; expected {args.expect_num_experts}."
        )
    if topology.top_k != args.expect_top_k:
        raise RuntimeError(f"top_k is {topology.top_k}; expected {args.expect_top_k}.")
    if len(topology.moe_layer_idxs) != args.layers:
        raise RuntimeError(
            f"Converted {len(topology.moe_layer_idxs)} layer(s); expected {args.layers}."
        )

    expected_blobs = args.layers * topology.num_experts
    if len(manifest.blobs) != expected_blobs:
        raise RuntimeError(
            f"Store holds {len(manifest.blobs)} blob(s); expected {expected_blobs}."
        )

    print(
        f"Topology verified: layers={topology.moe_layer_idxs}, "
        f"num_experts={topology.num_experts}, top_k={topology.top_k}, "
        f"blobs={len(manifest.blobs)}"
    )


def verify_compatibility_gate(
    store: PackedExpertStore, args: argparse.Namespace
) -> None:
    """Run the gate the engine will run: the store must accept the loaded model."""
    store.ensure_compatible(
        model_id=args.model,
        num_experts=args.expect_num_experts,
        top_k=args.expect_top_k,
        moe_layer_idxs=tuple(range(args.layers)),
    )

    print(
        f"ensure_compatible passed: model_id={args.model!r}, "
        f"num_experts={args.expect_num_experts}, top_k={args.expect_top_k}, "
        f"moe_layer_idxs={tuple(range(args.layers))}"
    )


def verify_guards(model_dir: Path) -> None:
    """Check that the converter's silent-corruption guards actually fire.

    The real checkpoint is uniform, so these cases are exercised against
    fabricated inputs — they are the paths that would otherwise stamp a store
    with quantization parameters its blobs do not use, or merge two expert
    stacks into one.
    """
    mixed_quant = {
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "model.layers.0.mlp.switch_mlp.gate_proj": {"bits": 8, "group_size": 64},
        }
    }
    try:
        resolve_expert_quantization(mixed_quant, (0, 1))
    except ValueError:
        pass
    else:
        raise RuntimeError("Mixed expert quantization did not raise.")

    uniform_override = {
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            **{
                f"model.layers.{layer}.mlp.switch_mlp.{projection}": {
                    "bits": 8,
                    "group_size": 32,
                }
                for layer in (0, 1)
                for projection in ("gate_proj", "up_proj", "down_proj")
            },
        }
    }
    quant = resolve_expert_quantization(uniform_override, (0, 1))
    if quant != QuantParams(mode="affine", bits=8, group_size=32):
        raise RuntimeError(f"Per-module override was not honored; got {quant!r}.")
    if payload_encoding_for(quant, "bf16") != "mlx-affine-q8-g32-bf16":
        raise RuntimeError("Encoding tag does not follow the resolved parameters.")

    two_stacks = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": model_dir,
        "vision_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": model_dir,
    }
    try:
        group_expert_tensors_by_layer(two_stacks)
    except ValueError:
        pass
    else:
        raise RuntimeError("Two expert stacks did not raise.")

    print(
        "Guards verified: mixed quantization raises, per-module override wins, "
        "two expert stacks raise."
    )


def verify_alignment(manifest: ExpertStoreManifest) -> None:
    misaligned = [
        blob for blob in manifest.blobs if blob.offset % manifest.alignment != 0
    ]
    if misaligned:
        raise RuntimeError(
            f"{len(misaligned)} blob offset(s) are not {manifest.alignment}-aligned."
        )

    print(f"Alignment verified: every blob offset is {manifest.alignment}-aligned.")


async def verify_bytes(
    *,
    store: PackedExpertStore,
    manifest: ExpertStoreManifest,
    model_dir: Path,
) -> None:
    shard_index = build_tensor_shard_index(model_dir)
    by_layer = group_expert_tensors_by_layer(shard_index)
    cache = ShardTensorCache(shard_index=shard_index)

    for layer_idx, expert_idx in sample_pairs(manifest):
        expected = assemble_expert_blob(
            expert_arrays(cache.load_layer(by_layer[layer_idx]), expert_idx),
            manifest.tensor_specs,
        )
        payload = await store.read(
            ExpertKey(
                model_fingerprint=manifest.model_fingerprint,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
            ),
            ReadPriority.DEMAND,
        )

        if payload.data != expected:
            raise RuntimeError(
                f"Blob bytes differ from the checkpoint for "
                f"(layer {layer_idx}, expert {expert_idx})."
            )
        if payload.encoding != manifest.payload_encoding:
            raise RuntimeError(
                f"Payload encoding is {payload.encoding!r}; "
                f"expected {manifest.payload_encoding!r}."
            )

        print(
            f"Bytes verified: layer {layer_idx}, expert {expert_idx}, "
            f"{len(payload.data)} bytes match the checkpoint."
        )


def main() -> None:
    args = parse_args()

    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    model_dir = resolve_model_dir(args.model)
    store_dir = args.output / "store"

    print(f"Model directory: {model_dir}")
    print(f"Store directory: {store_dir}")

    verify_guards(model_dir)

    manifest = convert_mlx_model_to_store(
        model_dir,
        store_dir,
        model_id=args.model,
        max_layers=args.layers,
        overwrite=True,
    )

    expert_nbytes = manifest.expert_nbytes()
    print(
        f"Converted {len(manifest.blobs)} expert blob(s); {expert_nbytes} bytes each, "
        f"{expert_nbytes * len(manifest.blobs)} payload bytes total."
    )
    print(f"Encoding: {manifest.payload_encoding}")
    print(f"Tensor specs: {[(s.name, s.dtype, s.shape) for s in manifest.tensor_specs]}")

    with PackedExpertStore(store_dir) as store:
        reopened = store.manifest
        verify_identity(reopened, args)
        verify_topology(reopened, args)
        verify_compatibility_gate(store, args)
        verify_alignment(reopened)
        asyncio.run(
            verify_bytes(store=store, manifest=reopened, model_dir=model_dir)
        )

    print("Store conversion smoke test passed.")


if __name__ == "__main__":
    main()
