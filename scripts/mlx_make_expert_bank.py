from typing import Sequence

import argparse

from rich import print

import textwrap

from pathlib import Path

import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.backends.mlx_metal.expert_bank_conversion import model_to_expert_bank

from preempt.utils.hf_utils import resolve_model_dir

# TODO add support for custom architecture registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Writes MoE expert layers to an expert bank on disk."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model/repo id or local path to a model checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to local directory where expert bank will be saved.",
    )
    parser.add_argument(
        "--max-moe-blocks",
        type=int,
        default=None,
        help="Limits conversion to the first 'max_moe_blocks' MoE blocks in the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Name of a MoE architecture adapter. Currently, this must be "
        "an architecture registered in `DefaultMoEArchRegistry`.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    from preempt.backends.mlx_metal.registry import DefaultMoEArchRegistry

    registry = DefaultMoEArchRegistry()
    architecture = registry.get(args.architecture)

    model_dir = resolve_model_dir(args.model)
    print(
        f"\nConverting model {model_dir.absolute().as_posix()} "
        f"with architecture {args.architecture!r}.",
        end="\n",
    )
    manifest = model_to_expert_bank(
        model_dir,
        expert_bank_dir=args.output,
        architecture=architecture,
        model_id=args.model,
        max_moe_blocks=args.max_moe_blocks,
        overwrite=args.overwrite,
    )
    print(f"Saved expert bank to {args.output.absolute().as_posix()}")

    num_blobs = len(manifest.blobs)
    num_blocks = len(manifest.model_moe_spec.moe_block_idxs)
    expert_num_bytes = manifest.expert_num_bytes()
    expert_num_mb = expert_num_bytes / 1024**2
    total_mb = expert_num_mb * len(manifest.blobs)

    print(textwrap.dedent(f"""
    Saved {num_blobs} expert layer blobs for {num_blocks} MoE blocks.

    Size per expert: {expert_num_mb:,} MB ({total_mb:,} MB total)
    Model id: {manifest.model_id.lstrip(".").lstrip("/")!r}
    Encoding: {manifest.payload_encoding!r}
    Hash: {manifest.model_fingerprint!r}
    """))


if __name__ == "__main__":
    main()
