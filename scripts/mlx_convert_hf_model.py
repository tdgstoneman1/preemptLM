from typing import Any

import argparse

from rich import print

from pathlib import Path

import json

import asyncio

import functools

import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.utils.hf_utils import (
    resolve_model_dir,
    validate_and_init_out_dir,
    copy_non_weight_files,
    make_shard_map,
)
from preempt.utils.mlx_utils import sanitize_fn_for, convert_and_save_shard


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch Hugging Face model to MLX."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to the source Hugging Face model directory.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        required=True,
        help="Path to the destination directory for the converted MLX model.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Maximum number of shards to process concurrently.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite exsting contents in the output directory.",
    )
    return parser.parse_args()


async def main() -> None:
    args = get_args()

    # Resolve and validate paths
    ckpt_path = resolve_model_dir(args.hf_path)

    index_path = ckpt_path / "model.safetensors.index.json"
    assert index_path.is_file()

    config_path = ckpt_path / "config.json"
    assert config_path.is_file()

    # Copy non-weight files
    dst_path = Path(args.mlx_path)
    validate_and_init_out_dir(dst_path, overwrite=args.overwrite)
    copy_non_weight_files(ckpt_path, dst_path, verbose=True)

    # Map shards to prevent splitting blocks across shards
    tensor_index = json.loads(index_path.read_text())
    shard_map = make_shard_map(
        ckpt_path=ckpt_path, weight_map=tensor_index["weight_map"]
    )
    num_total_shards = len(shard_map)

    # Get official MLX pipeline to make weights compatible, see `mlx_lm.utils`
    config: dict[str, Any] = json.loads(config_path.read_text())
    sanitize_fn = sanitize_fn_for(config)

    # Schedule and run async tasks
    semaphore = asyncio.Semaphore(args.num_parallel)

    async def process_shard_worker(
        shard_map: dict[str, str],
        ckpt_path: Path,
        dst_path: Path,
        shard_idx: int,
        num_total_shards: int,
    ) -> tuple[str, list[str]]:
        nonlocal semaphore
        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(
                    convert_and_save_shard,
                    shard_map={k: Path(v) for k, v in shard_map.items()},
                    sanitize_fn=sanitize_fn,
                    ckpt_path=ckpt_path,
                    dst_path=dst_path,
                    shard_idx=shard_idx,
                    num_total_shards=num_total_shards,
                ),
            )

    tasks = [
        process_shard_worker(shard, ckpt_path, dst_path, i, num_total_shards)
        for i, shard in enumerate(shard_map)
    ]
    results = await asyncio.gather(*tasks)

    # Collect all shard names and paths into one mapping
    new_weight_map = {}
    for new_shard_name, tensor_names in results:
        for name in tensor_names:
            new_weight_map[name] = new_shard_name

    print("All model weights converted to MLX and saved to safetensors.\n")

    # Make the final safetensor index
    final_index = {
        "metadata": tensor_index["metadata"],
        "weight_map": {k: new_weight_map[k] for k in sorted(new_weight_map)},
    }
    with (dst_path / "model.safetensors.index.json").open("w") as f:
        json.dump(final_index, f, indent=4)

    print("Successfully converted model to MLX.")


if __name__ == "__main__":
    asyncio.run(main())
