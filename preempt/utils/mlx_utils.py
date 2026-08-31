from typing import Any
from collections.abc import Callable

from pathlib import Path

import gc

import mlx.core as mx
from mlx_lm.utils import _get_classes

from safetensors import safe_open

from rich import print


def sanitize_fn_for(
    config: dict[str, Any],
) -> Callable[[dict[str, mx.array]], dict[str, mx.array]]:

    model_class, model_args_class = _get_classes(config)
    model = model_class(model_args_class.from_dict(config))  # lazy loaded
    if hasattr(model, "sanitize"):
        return model.sanitize
    else:
        return lambda x: x


def convert_and_save_shard(
    ckpt_path: Path,
    dst_path: Path,
    shard_map: dict[str, Path],
    sanitize_fn: Callable[[dict[str, mx.array]], dict[str, mx.array]],
    shard_idx: int,
    num_total_shards: int,
) -> tuple[str, list[str]]:

    log_prefix = f"Shard {shard_idx}/{num_total_shards}"
    print(f"{log_prefix}: Converting pt tensors to MLX...")

    torch_tensors = {}
    for tensor_name, shard_path in shard_map.items():
        with safe_open(ckpt_path / shard_path, framework="pt", backend="pread") as f:
            torch_tensors[tensor_name] = f.get_tensor(tensor_name)

    print(f"{log_prefix}: Loaded {len(shard_map)} module weights from safetensors.")

    mlx_arrays = {
        name: mx.from_dlpack(pt_tensor) for name, pt_tensor in torch_tensors.items()
    }
    del torch_tensors
    gc.collect()

    mlx_arrays = sanitize_fn(mlx_arrays)
    print(f"{log_prefix}: Sanitized pt weight tensors for MLX.")

    shard_name = (
        f"model-{shard_idx + 1:05d}-of-{num_total_shards:05d}.safetensors"
        # if num_total_shards == 1
        # else "model.safetensors"
    )
    out_path = dst_path / shard_name
    mx.save_safetensors(out_path, mlx_arrays, metadata={"format": "mlx"})
    tensor_names = list(mlx_arrays.keys())

    del mlx_arrays
    gc.collect()

    print(f"{log_prefix}: Saved MLX shard to {out_path.as_posix()!r}\n")
    return shard_name, tensor_names
