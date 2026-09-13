"""Example usage on macOS:

    uv run scripts/mlx_make_expert_bank.py\
        --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit\
        --output expert-bank/qwen3.6-35b-4bit\
        --arch qwen3.6\
        --overwrite
"""

from typing import Sequence

import argparse

from rich import print
import textwrap

from pathlib import Path

import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.backends.mlx_metal.registry import (
    DefaultArchClassRegistry,
)
from preempt.backends.mlx_metal.expert_bank.serialization import model_to_expert_bank

from preempt.utils.hf_utils import resolve_model_dir

# TODO remove 'architecture' arg and resolve from config.json or mlx-lm class via registry


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
        "--architecture",
        "--arch",
        type=str,
        required=True,
        help="Name of a MoE architecture adapter. Currently, this must be "
        "an architecture registered in `DefaultArchClassRegistry`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files in the output directory.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    arch_adapter = DefaultArchClassRegistry.get_arch_adapter(
        args.architecture, instantiate=True
    )
    model_dir = resolve_model_dir(args.model)
    print(
        f"\nConverting model {model_dir.absolute().as_posix()} "
        f"with architecture {args.architecture!r}.",
        end="\n",
    )
    manifest = model_to_expert_bank(
        model_dir,
        expert_bank_path=args.output,
        arch_adapter=arch_adapter,
        model_id=args.model,
        overwrite=args.overwrite,
    )
    print(f"Saved expert bank to {args.output.absolute().as_posix()!r}")

    num_blobs = len(manifest.blob_index)
    num_blocks = len(manifest.model_moe_spec.moe_block_idxs)
    expert_num_bytes = manifest.expert_num_bytes()
    expert_num_mb = expert_num_bytes / 1024**2
    total_mb = expert_num_mb * num_blobs

    print(textwrap.dedent(f"""
    Saved {num_blobs} expert layer blobs for {num_blocks} MoE blocks.

    Size per expert: {expert_num_mb:,} MB ({total_mb:,} MB total)
    Model id: {manifest.model_id.lstrip(".").lstrip("/")!r}
    Encoding: {manifest.encoding!r}
    Hash: {manifest.model_fingerprint!r}
    """))


if __name__ == "__main__":
    main()
