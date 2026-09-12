from typing import Any, cast
from collections.abc import Callable, Mapping, Sequence

from attrs import asdict

from pathlib import Path

import gc

from rich import print

import ml_dtypes

import numpy as np

import mlx.core as mx

from mlx_lm import load
from mlx_lm.utils import _get_classes

from safetensors import safe_open

from preempt.core.protocols import ITokenizer
from preempt.core.exceptions import EngineCompatibilityError

from .types import MlxLoadedModel
from .quantization import QuantSettings
from .constants import (
    MLX_DTYPE_TAGS,
    MLX_QUANTIZED_ENCODING_TEMPLATE,
    MLX_UNQUANTIZED_ENCODING_TEMPLATE,
)
from .quantization import QuantSettings


# TODO docstring
def sanitize_fn_for(
    config: dict[str, Any],
) -> Callable[[dict[str, mx.array]], dict[str, mx.array]]:
    model_class, model_args_class = _get_classes(config)
    model = model_class(model_args_class.from_dict(config))  # lazy loaded

    if hasattr(model, "sanitize"):
        return model.sanitize
    else:
        return lambda x: x


# TODO docstring
# TODO logging
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


def mlx_to_numpy(array: mx.array, copy: bool = False) -> np.ndarray:
    """Converts an MLX array to NumPy.

    :Note: MLX arrays of dtype `mx.bfloat16` are cast to `ml_dtypes.bfloat16`
    to account for NumPy's lack of native `bfloat16` support.

    Parameters
    ----------
    array : mx.array
        An MLX array
    copy : bool
        Whether to copy the array during conversion, by default False

    Returns
    -------
    np.ndarray
        The converted array
    """
    if array.dtype == mx.bfloat16:
        return np.frombuffer(
            memoryview(array),
            dtype=ml_dtypes.bfloat16,
        ).reshape(array.shape)

    if copy:
        return np.array(array, copy=True)

    return np.asarray(array)


def load_mlx_model(model_id: str, *, lazy: bool = False) -> MlxLoadedModel:
    """Loads an MLX model-tokenizer pair.

    Parameters
    ----------
    model_id : str
        Hugging Face repo id or a local path accepted by `mlx_lm.load()`
    lazy : bool
        If True, evaluation and loading of model weights is deferred, otherwise
        all weights are evaluated eagerly and immediately loaded.

        :Note: `lazy=True` is required for model instrumentation when streaming
        expert weights from disk.

    Returns
    -------
    MlxLoadedModel
        The loaded `mlx_lm` model and its tokenizer
    """
    model, tokenizer = load(model_id, lazy=lazy)  # type: ignore
    return MlxLoadedModel(model=model, tokenizer=cast(ITokenizer, tokenizer))


def dtype_tag_from_arrays(arrays: Mapping[str, mx.array], quantized: bool) -> str:
    """Returns a dtype tag for `arrays`.

    Parameters
    ----------
    arrays : Mapping[str, mx.array]
        A mapping of names to arrays
    quantized : bool
        Whether arrays are quantized (and should be scanned for quantization
        parameters)

    Returns
    -------
    str
        Scalar dtype tag, e.g. 'bfloat16'

    Raises
    ------
    ValueError
        If no applicable arrays found in `arrays`.
    ValueError
        If arrays have multiple different dtypes.
    ValueError
        If the detected dtype is not currently supported for tagging.
    """
    names = sorted(
        name
        for name in arrays
        if not quantized or name.endswith((".scales", ".biases"))
    )
    if not names:
        raise ValueError()  # TODO add message

    dtypes = {str(arrays[name].dtype) for name in names}
    if len(dtypes) != 1:
        raise ValueError(f"Arrays have multiple dtypes: {sorted(dtypes)!r}")

    dtype = arrays[names[0]].dtype
    for candidate, tag in MLX_DTYPE_TAGS:
        if dtype == candidate:
            return tag

    raise ValueError(f"Arrays have unsupported dtype: {dtype!r}")


def make_encoding_tag(quant: QuantSettings | None, dtype_tag: str) -> str:
    """Returns an encoding tag for use with expert banks.

    This is a formatted string containing quantization and dtype information
    used to decode serialized expert weight blobs.
    \nExamples: `'mlx-affine-q4-g64-bfloat16'`, `'mlx-unquantized-bfloat16'`

    Parameters
    ----------
    quant : QuantSettings | None
        Weight quantization parameters, or `None` if unquantized
    dtype_tag : str
        A tag representing the weights' dtype. e.g. 'bfloat16'

    Returns
    -------
    str
        A formatted encoding tag
    """
    if quant is None:
        return MLX_UNQUANTIZED_ENCODING_TEMPLATE.substitute(dtype=dtype_tag)

    return MLX_QUANTIZED_ENCODING_TEMPLATE.substitute(
        mode=quant.mode,
        bits=quant.bits,
        group_size=quant.group_size,
        dtype=dtype_tag,
    )


# TODO make architecture agnostic
def transformer_block_idx_from_path(module_path: str) -> int | None:
    """Parses `module_path` and returns the index of the module's parent
    transformer block, or `None` if the path doesn't follow the expected
    pattern.

    Example: `'language_model.model.layers.22.mlp'` → `22`
    """
    parts = module_path.split(".")

    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])

    return None
