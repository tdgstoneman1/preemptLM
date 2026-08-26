"""Host smoke test: convert a few MoE layers into an expert bank and byte-verify it.

Runs on macOS with MLX and real weights. Converts the first `--layers` MoE
layers of a checkpoint, reopens the result through `ExpertBank`, and
proves the stored bytes are the checkpoint's bytes for a sample of experts.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from preempt.backends.mlx_metal.architectures.qwen3_next import Qwen3NextMoEArchitecture
from preempt.backends.mlx_metal.convert import (
    ShardTensorCache,
    build_tensor_shard_index,
    convert_mlx_model_to_expert_bank,
    expert_ndarrays,
    group_expert_tensors_by_layer,
)
from preempt.backends.mlx_metal.quantization import (
    MlxQuantParams,
    make_encoding_tag,
)
from preempt.core.identity import ExpertKey
from preempt.core.protocols.expert_bank import ReadPriority

from preempt.storage.blob import assemble_expert_blob
from preempt.storage.manifest import ExpertBankManifest
from preempt.storage.expert_io import ExpertBank

from preempt.utils.hf_utils import resolve_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a few MoE layers to an expert bank and verify it."
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
        help="Directory to write the expert bank into (a `store` subdirectory).",
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
        help="Payload encoding tag the converted expert bank must carry.",
    )
    return parser.parse_args()


def sample_pairs(manifest: ExpertBankManifest) -> tuple[tuple[int, int], ...]:
    """Pick three deterministic (layer, expert) pairs spanning the expert_bank."""
    block_idxs = manifest.model_moe_spec.moe_block_idxs
    num_routed_experts = manifest.model_moe_spec.num_routed_experts

    return (
        (block_idxs[0], 0),
        (block_idxs[len(block_idxs) // 2], num_routed_experts // 2),
        (block_idxs[-1], num_routed_experts - 1),
    )


def verify_identity(manifest: ExpertBankManifest, args: argparse.Namespace) -> None:
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


def verify_topology(manifest: ExpertBankManifest, args: argparse.Namespace) -> None:
    model_moe_spec = manifest.model_moe_spec

    if model_moe_spec.num_routed_experts != args.expect_num_experts:
        raise RuntimeError(
            f"num_routed_experts is {model_moe_spec.num_routed_experts}; expected {args.expect_num_experts}."
        )
    if model_moe_spec.top_k != args.expect_top_k:
        raise RuntimeError(
            f"top_k is {model_moe_spec.top_k}; expected {args.expect_top_k}."
        )
    if len(model_moe_spec.moe_block_idxs) != args.layers:
        raise RuntimeError(
            f"Converted {len(model_moe_spec.moe_block_idxs)} layer(s); expected {args.layers}."
        )

    expected_blobs = args.layers * model_moe_spec.num_routed_experts
    if len(manifest.blobs) != expected_blobs:
        raise RuntimeError(
            f"Expert bank holds {len(manifest.blobs)} blob(s); expected {expected_blobs}."
        )

    print(
        f"Topology verified: layers={model_moe_spec.moe_block_idxs}, "
        f"num_routed_experts={model_moe_spec.num_routed_experts}, top_k={model_moe_spec.top_k}, "
        f"blobs={len(manifest.blobs)}"
    )


def verify_compatibility_gate(
    expert_bank: ExpertBank, args: argparse.Namespace
) -> None:
    """Run the gate the engine will run: the expert bank must accept the loaded model."""
    expert_bank.check_model_compatibility(
        model_id=args.model,
        num_routed_experts=args.expect_num_experts,
        top_k=args.expect_top_k,
        moe_block_idxs=tuple(range(args.layers)),
    )

    print(
        f"check_model_compatibility passed: model_id={args.model!r}, "
        f"num_routed_experts={args.expect_num_experts}, top_k={args.expect_top_k}, "
        f"moe_block_idxs={tuple(range(args.layers))}"
    )


def verify_guards(model_dir: Path) -> None:
    """Check that the converter's silent-corruption guards actually fire.

    The real checkpoint is uniform, so these cases are exercised against
    fabricated inputs — they are the paths that would otherwise stamp an
    expert bank with quantization parameters its blobs do not use, or merge
    two expert stacks into one.
    """
    arch = Qwen3NextMoEArchitecture()

    mixed_quant = {
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "model.layers.0.mlp.switch_mlp.gate_proj": {"bits": 8, "group_size": 64},
        }
    }
    try:
        arch.resolve_quantization(mixed_quant, (0, 1))
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
    quant = arch.resolve_quantization(uniform_override, (0, 1))
    if quant != MlxQuantParams(mode="affine", bits=8, group_size=32):
        raise RuntimeError(f"Per-module override was not honored; got {quant!r}.")
    if make_encoding_tag(quant, "bf16") != "mlx-affine-q8-g32-bf16":
        raise RuntimeError("Encoding tag does not follow the resolved parameters.")

    two_stacks = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": model_dir,
        "vision_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": model_dir,
    }
    try:
        group_expert_tensors_by_layer(two_stacks, arch)
    except ValueError:
        pass
    else:
        raise RuntimeError("Two expert stacks did not raise.")

    print(
        "Guards verified: mixed quantization raises, per-module override wins, "
        "two expert stacks raise."
    )


def verify_alignment(manifest: ExpertBankManifest) -> None:
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
    expert_bank: ExpertBank,
    manifest: ExpertBankManifest,
    model_dir: Path,
) -> None:
    arch = Qwen3NextMoEArchitecture()
    shard_index = build_tensor_shard_index(model_dir, arch)
    by_layer = group_expert_tensors_by_layer(shard_index, arch)
    cache = ShardTensorCache(shard_index=shard_index)

    for block_idx, expert_idx in sample_pairs(manifest):
        expected = assemble_expert_blob(
            expert_ndarrays(cache.load_layer(by_layer[block_idx]), expert_idx),
            manifest.tensor_specs,
        )
        payload = await expert_bank.read(
            ExpertKey(
                model_fingerprint=manifest.model_fingerprint,
                block_idx=block_idx,
                expert_idx=expert_idx,
            ),
            ReadPriority.DEMAND,
        )

        if payload.data != expected:
            raise RuntimeError(
                f"Blob bytes differ from the checkpoint for "
                f"(layer {block_idx}, expert {expert_idx})."
            )
        if payload.encoding != manifest.payload_encoding:
            raise RuntimeError(
                f"Payload encoding is {payload.encoding!r}; "
                f"expected {manifest.payload_encoding!r}."
            )

        print(
            f"Bytes verified: layer {block_idx}, expert {expert_idx}, "
            f"{len(payload.data)} bytes match the checkpoint."
        )


def main() -> None:
    args = parse_args()

    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    arch = Qwen3NextMoEArchitecture()
    model_dir = resolve_model_dir(args.model)
    expert_bank_dir = args.output / "store"

    print(f"Model directory: {model_dir}")
    print(f"Expert bank directory: {expert_bank_dir}")

    verify_guards(model_dir)

    manifest = convert_mlx_model_to_expert_bank(
        model_dir,
        expert_bank_dir,
        architecture=arch,
        model_id=args.model,
        max_moe_blocks=args.layers,
        overwrite=True,
    )

    expert_num_bytes = manifest.expert_num_bytes()
    print(
        f"Converted {len(manifest.blobs)} expert blob(s); {expert_num_bytes} bytes each, "
        f"{expert_num_bytes * len(manifest.blobs)} payload bytes total."
    )
    print(f"Encoding: {manifest.payload_encoding}")
    print(
        f"Tensor specs: {[(s.name, s.dtype, s.shape) for s in manifest.tensor_specs]}"
    )

    with ExpertBank(expert_bank_dir) as expert_bank:
        reopened = expert_bank.manifest
        verify_identity(reopened, args)
        verify_topology(reopened, args)
        verify_compatibility_gate(expert_bank, args)
        verify_alignment(reopened)
        asyncio.run(
            verify_bytes(
                expert_bank=expert_bank, manifest=reopened, model_dir=model_dir
            )
        )

    print("Expert bank conversion smoke test passed.")


if __name__ == "__main__":
    main()
