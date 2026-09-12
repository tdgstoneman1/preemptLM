from typing import Literal, Any, NoReturn
from types import MappingProxyType
from collections.abc import Generator

from pathlib import Path
import hashlib
import shutil
import re
import math

from huggingface_hub import snapshot_download

from safetensors import safe_open

from preempt.core.constants import DEFAULT_MAX_SAFETENSOR_SHARD_MB

DTYPE_SIZES: MappingProxyType[str, int] = MappingProxyType(
    {
        "F64": 8,
        "I64": 8,
        "U64": 8,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "U16": 2,
        "BOOL": 1,
        "I8": 1,
        "U8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
)

BLOCK_IDX_RE = re.compile(
    r"(?P<prefix>(?:[a-zA-Z_\.]+))\.(?:blocks|layers)\.(?P<block_idx>\d+)"
)  # TODO find bettter place for this, only applies to Qwen3.X models


def resolve_model_dir(model_path_or_id: str) -> Path:
    """Resolves `model_path_or_id` and downloads the model from Hugging
    Face if not found locally.

    Parameters
    ----------
    model_path_or_id : str
        Path to local model checkpoint directory or Hugging Face repo ID

    Returns
    -------
    Path
        The resolved absolute path to the local model checkpoint directory
    """
    path = Path(model_path_or_id)
    return path if path.exists() else Path(snapshot_download(model_path_or_id))


def get_safetensor_size(
    path: Path, tensor_name: str, units: Literal["bytes", "kb", "mb", "gb"] = "mb"
) -> int | float:
    with safe_open(path, framework="pt", backend="pread") as f:
        if tensor_name not in f:
            raise ValueError()

        tensor_slice = f.get_slice(tensor_name)
        shape = tensor_slice.get_shape()
        dtype = tensor_slice.get_dtype()

        num_elements = math.prod(shape)
        element_size = DTYPE_SIZES[dtype]
        bytes_size = num_elements * element_size

        if units == "bytes":
            return bytes_size

        if units == "kb":
            return bytes_size / 1024

        if units == "mb":
            return bytes_size / 1024**2

        if units == "gb":
            return bytes_size / 1024**3

        raise ValueError()


def validate_and_init_out_dir(path: Path, overwrite: bool) -> None:
    if path.is_file():
        raise FileExistsError()
    if path.is_dir():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError()
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def copy_non_weight_files(
    ckpt_path: Path, dst_dir: Path, overwrite: bool = False, verbose: bool = False
) -> None:
    """Copies all non-safetensors files from `ckpt_path` to the `dst_dir`."""
    if verbose:
        print(f"Copying non-weight files from {ckpt_path} to {dst_dir}")

    src_files = [
        p
        for p in ckpt_path.glob("**/*")
        if p.is_file() and ".safetensors" not in p.name
    ]
    for path in src_files:
        relative_path = path.relative_to(ckpt_path)
        dst_path = dst_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst_path)


# TODO find better place for this
def make_shard_map(
    ckpt_path: Path,
    weight_map: dict[str, str],
    max_shard_mb: int | float = DEFAULT_MAX_SAFETENSOR_SHARD_MB,
) -> list[dict[str, str]]:
    # Separate layers belonging to parent blocks from others, e.g.
    # `model.language_model.layers.40.mlp.experts.down_proj` vs.
    # `model.language_model.lm_head.weight`
    grouped, other = [], []
    for name in weight_map:
        if BLOCK_IDX_RE.match(name):
            grouped.append(name)
        else:
            other.append(name)

    def loop_weights() -> Generator[str, Any, NoReturn]:
        while True:
            # Prioritize layers grouped in blocks to avoid splits across files
            yield from (name for name in grouped + other)

    shards: list[dict[str, str]] = []
    current_shard: dict[str, str] = {}
    current_shard_size = 0
    break_current_shard: bool = True

    weight_generator = loop_weights()
    current_name = next(weight_generator)
    remaining = weight_map.copy()

    while True:
        if not remaining:
            break
        if current_name not in remaining:
            current_name = next(weight_generator)
            continue

        tensor_path = remaining.pop(current_name)
        tensor_size = get_safetensor_size(ckpt_path / tensor_path, current_name)

        if tensor_size + current_shard_size > max_shard_mb and break_current_shard:
            shards.append(current_shard)
            current_shard = {current_name: tensor_path}
            current_shard_size = tensor_size
        else:
            current_shard[current_name] = tensor_path
            current_shard_size += tensor_size

        if not (match := BLOCK_IDX_RE.match(current_name)):
            current_name = next(weight_generator)
            break_current_shard = True
            continue

        prefix = match.groupdict()["prefix"]
        block = match.groupdict()["block_idx"]

        for tensor in remaining:
            if not (match_ := BLOCK_IDX_RE.match(tensor)):
                break_current_shard = True
                continue
            if (
                match_.groupdict()["prefix"] == prefix
                and match_.groupdict()["block_idx"] == block
            ):
                current_name = tensor
                break_current_shard = False
                break

    if shards and current_shard != shards[-1]:
        shards.append(current_shard)

    return shards


def hash_model_ckpt(ckpt_path: Path) -> str:
    """Hashes an MLX model checkpoint.

    Returns a SHA-256 digest of the checkpoint's `config.json` and the names
    and byte sizes of its `*.safetensors` shards.

    Parameters
    ----------
    ckpt_path : Path
        Path to a Hugging Face-style MLX model checkpoint

    Returns
    -------
    str
        Unique fingerprint for the checkpoint
    """
    digest = hashlib.sha256((ckpt_path / "config.json").read_bytes())
    for shard in sorted(ckpt_path.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())

    return digest.hexdigest()
