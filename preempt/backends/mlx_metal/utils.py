from typing import Any, cast
from collections.abc import Callable, Mapping, Sequence

from pathlib import Path

import gc

from rich import print

import numpy as np

import mlx.core as mx

from mlx_lm.utils import _get_classes
from mlx_lm.models.switch_layers import (
    SwitchLinear,
    QuantizedSwitchLinear,
    SwitchGLU,
)
from mlx_lm import load

from safetensors import safe_open

from preempt.core.protocols import ITokenizer
from preempt.core.exceptions import EngineCompatibilityError

from .types import MlxLoadedModel, ExpertLayerQuants
from .quantization import QuantSettings
from .constants import (
    MLX_DTYPE_TAGS,
    MLX_QUANTIZED_ENCODING_TEMPLATE,
    MLX_UNQUANTIZED_ENCODING_TEMPLATE,
    MLX_QUANT_PARAMS,
)
from .quantization import QuantSettings


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


# TODO use ml_dtypes to preserve bfloat16
def mlx_to_numpy(array: mx.array, copy: bool = False) -> np.ndarray:
    """Converts an MLX array to NumPy with zero mutation.

    :Note: MLX arrays of dtype `bfloat16` reinterpreted as `uint16`
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
        return np.array(array.view(mx.uint16), copy=copy)

    return np.array(array, copy=copy)


def load_mlx_model(model_id: str, *, lazy: bool = False) -> MlxLoadedModel:
    """Loads an MLX model and tokenizer via `mlx_lm`.

    Parameters
    ----------
    model_id : str
        Hugging Face repo id or local path accepted by `mlx_lm.load(...)`
    lazy : bool
        If False, model weights are evaluated eagerly at load time. If True, `mlx_lm`
        skips its internal weight evaluation, and all weights remain unevaluated
        mmap-backed arrays. This is required for model instrumentation when streaming
        expert weights from disk as it defers weight materialization, allowing expert
        weights to be stripped so only the dense backbone materializes in memory.
        by default False.

    Returns
    -------
    MlxLoadedModel
        The loaded `mlx_lm` model and its tokenizer
    """
    model, tokenizer = load(model_id, lazy=lazy)  # type: ignore
    return MlxLoadedModel(model=model, tokenizer=cast(ITokenizer, tokenizer))


def dtype_tag_from_arrays(arrays: Mapping[str, mx.array], quantized: bool) -> str:
    """Returns a the scalar dtype of the arrays in `stacked` as a string tag.

    Parameters
    ----------
    arrays : Mapping[str, mx.array]
        A mapping of names to `mx.array`s
    quantized : bool
        Whether arrays are quantized and should be scanned for quantization
        parameters

    Returns
    -------
    str
        Scalar dtype tag, e.g. 'bf16', 'f16', or 'f32'

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
    """Generates the payload encoding tag for the expert bank.

    Produces a formatted string identifying the quantization state and
    exact scalar dtype of the tensor. This tag must be kept in sync with
    the parsing logic in `preempt.core.encoding.parse_payload_encoding_tag()`
    to ensure bit-exact decoding.

    Parameters
    ----------
    quant : QuantSettings | None
        The quantization parameters used for the experts, or `None` if
        unquantized
    dtype_tag : str
        The short tag representing the scalar dtype (e.g. `"bf16"`) as
        derived from `dtype_tag_from_arrays()`

    Returns
    -------
    str
        The formatted encoding tag (e.g., `"mlx-affine-q4-g64-bf16"` or
        `"mlx-unquantized-bf16"`)
    """
    if quant is None:
        return MLX_UNQUANTIZED_ENCODING_TEMPLATE.substitute(dtype=dtype_tag)

    return MLX_QUANTIZED_ENCODING_TEMPLATE.substitute(
        mode=quant.mode,
        bits=quant.bits,
        group_size=quant.group_size,
        dtype=dtype_tag,
    )


def get_expert_quants(
    switch_mlp: SwitchGLU,
    projection_names: Sequence[str],
) -> ExpertLayerQuants | None:
    """Reads quantization parameters for the linear projection layer weights
    in `switch_mlp`.

    Parameters
    ----------
    switch_mlp : SwitchGLU
        A fused multi-expert module containing the stacked weights for all
        expert layers in an MoE block
    projection_names : Sequence[str]
        Projection names to read (e.g. from `architecture.projection_names`)

    Returns
    -------
    ExpertLayerQuants
        Per-projection quantization parameters keyed by name.

    Raises
    ------
    TypeError
        If a layer in `switch_mlp` is not an instance of `QuantizedSwitchLinear`
    EngineCompatibilityError
        If a layer in `switch_mlp` has a bias (currently unsupported in the
        forward pass)
    """
    params: dict[str, QuantSettings] = {}
    for name in projection_names:
        module = getattr(switch_mlp, name)

        if not isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
            raise TypeError(
                f"`{name}` is of unsupported type `{type(module).__name__}`. Only "
                "`SwitchLinear` and `QuantizedSwitchLinear` are currently supported."
            )
        if "bias" in module:  # TODO add support for bias
            raise EngineCompatibilityError(
                f"Experts layer {name!r} (in {type(module).__name__!r}) has an additive 'bias' "
                "which is currently unsupported in forward pass. "
            )
        if all(hasattr(module, attr) for attr in MLX_QUANT_PARAMS):
            params[name] = QuantSettings(
                group_size=int(module.group_size),  # type: ignore
                bits=int(module.bits),  # type: ignore
                mode=str(module.mode),  # type: ignore
            )

    return params or None
