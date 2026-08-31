from __future__ import annotations

from collections.abc import Mapping

import hashlib
import json
from pathlib import Path

import gc

import attrs
from attrs import field

import numpy as np

import mlx.core as mx

from preempt.storage.blob import assemble_expert_blob, derive_tensor_specs
from preempt.storage.manifest import ExpertBankManifest
from preempt.storage.expert_io import ExpertBankWriter

from .quantization import make_encoding_tag
from .architecture import MoEArchAdapter


def hash_model_ckpt(ckpt_path: Path) -> str:
    """Hashes an MLX model checkpoint.

    Returns a SHA-256 digest of the checkpoint's `config.json` and the names
    and byte sizes of its `*.safetensors` shards.

    Parameters
    ----------
    ckpt_path : Path
        An MLX model's checkpoint directory

    Returns
    -------
    str
        The checkpoint's unique fingerprint.
    """
    digest = hashlib.sha256((ckpt_path / "config.json").read_bytes())
    for shard in sorted(ckpt_path.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())

    return digest.hexdigest()


def build_tensor_shard_index(
    ckpt_path: Path, architecture: MoEArchAdapter
) -> dict[str, Path]:
    """Indexes the file locations of all routed expert tensors in a checkpoint.

    Consults the `model.safetensors.index.json` weight map (or falls back to
    scanning a monolithic `model.safetensors` file) to map each expert tensor's
    full, dotted path (e.g., `model.layers.0.block_sparse_moe.w1.weight`)
    to the physical shard file that contains it.

    Parameters
    ----------
    ckpt_path : Path
        Local path to a Hugging Face model's checkpoint directory
    architecture : MoEArchAdapter
        The architecture definition providing the regex to identify expert
        weight tensors

    Returns
    -------
    dict[str, Path]
        Mapping of tensor dotted paths to tensor shard file paths
    """
    index_path = ckpt_path / "model.safetensors.index.json"
    # TODO move path parts like 'model.safetensors' to .constants

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {
            name: ckpt_path / shard
            for name, shard in weight_map.items()
            if architecture.expert_tensor_regex.match(name)
        }

    single = ckpt_path / "model.safetensors"
    return {name: single for name in mx.load(single) if regex.match(name)}  # type: ignore


def group_expert_tensors_by_layer(
    shard_index: Mapping[str, Path],
    architecture: MoEArchAdapter,
) -> dict[int, dict[str, str]]:
    """Returns a dictionary of weight tensors grouped and keyed by layer index.
    Each group's nested dictionary maps a tensor's path relative to its parent
    (e.g., `w1.weight`) to its full path within the model.

    Parameters
    ----------
    shard_index : Mapping[str, Path]
        Mapping of a tensor's full, dotted path to its file location
    architecture : MoEArchAdapter
        Adapter providing regex patterns to parse tensor paths

    Returns
    -------
    dict[int, dict[str, str]]
        A nested dictionary of tensors grouped by layer index

    Raises
    ------
    ValueError
        If tensor weights from multiple model prefixes are found.
    """
    by_layer: dict[int, dict[str, str]] = {}
    prefixes: set[str] = set()

    for name in shard_index:
        match = architecture.expert_tensor_regex.match(name)
        if match is None:
            continue

        prefixes.add(match.group("prefix"))
        suffix = f"{match.group('projection')}.{match.group('part')}"
        by_layer.setdefault(int(match.group("layer")), {})[suffix] = name

    if len(prefixes) > 1:
        raise ValueError(
            "Routed expert tensors come from multiple stacks; this converter "
            f"packs one: prefixes {sorted(prefixes)!r}."
        )

    return by_layer


def mlx_to_numpy(tensor: mx.array) -> np.ndarray:
    """Converts an MLX array to NumPy with zero mutation.

    :Note: MLX arrays of dtype `bfloat16` reinterpreted as `uint16`
    to account for NumPy's lack of native `bfloat16` support.

    Parameters
    ----------
    tensor : mx.array
        An MLX array

    Returns
    -------
    np.ndarray
        The converted array
    """
    if tensor.dtype == mx.bfloat16:
        return np.array(tensor.view(mx.uint16))

    return np.array(tensor)


@attrs.define
class ShardTensorCache:
    """Bounded cache managing the deferred loading of stacked expert tensors."""

    shard_index: Mapping[str, Path] = field()
    _loaded_path: Path | None = field(default=None, init=False)
    _loaded: dict[str, mx.array] = field(factory=dict, init=False)

    def get(self, name: str) -> mx.array:
        path = self.shard_index[name]
        if path != self._loaded_path:
            self._loaded = mx.load(path)  # type: ignore
            self._loaded_path = path

        return self._loaded[name]

    def load_layer(self, layer_weights: Mapping[str, str]) -> dict[str, mx.array]:
        """Loads `layer_tensors` from tensorshards.

        Parameters
        ----------
        layer_tensors : Mapping[str, str]
            A mapping of dotted tensor paths relative to parent modules (e.g.,
            `w1.weight`) to their full paths in the checkpoint

        Returns
        -------
        dict[str, mx.array]
            Mapping of dotted paths in dot notation to stacked weight tensors
        """
        # Sort by shard order to reduce disk reads for layers split between shards
        # TODO verify sorting actually helps, otherwise may introduce latency
        ordered = sorted(
            layer_weights.items(), key=lambda item: self.shard_index[item[1]]
        )
        return {path: self.get(name) for path, name in ordered}


def expert_ndarrays(
    stacked: Mapping[str, mx.array], expert_idx: int
) -> dict[str, np.ndarray]:
    """Returns a dictionary mapping tensor paths to numpy arrays for the cache's
    `expert_idx`th expert layer.
    """
    return {path: mlx_to_numpy(tensor[expert_idx]) for path, tensor in stacked.items()}


def model_to_expert_bank(
    ckpt_path: Path,
    expert_bank_dir: Path,
    *,
    architecture: MoEArchAdapter,
    model_id: str | None = None,
    max_moe_blocks: int | None = None,
    overwrite: bool = False,
) -> ExpertBankManifest:
    """Writes the routed expert layers of an MLX model to an expert bank on disk.

    Parameters
    ----------
    ckpt_dir : Path
        Local path to a Hugging Face model's checkpoint directory
    expert_bank_dir : Path
        Local directory where `experts.bin` and `manifest.json` will be written
    architecture : MoEArchAdapter
        Adapter defining the checkpoint's MoE structural patterns
    model_id : str | None
        A unique identifier used for compatibility checks when loading the saved expert bank.
        Defaults to the name of the model checkpoint, but should be explicitly provided for
        specific cached Hugging Face snapshots. By default `None`
    max_moe_blocks : int | None
        Limits conversion to the first `max_moe_blocks` MoE blocks. If `None`, all layers are
        converted. By default `None`
    overwrite : bool
        If `True`, an existing expert bank in the output directory will be overwritten.
        By default False

    Returns
    -------
    ExpertBankManifest
        The metadata manifest for the expert bank

    Raises
    ------
    ValueError
        If no routed expert tensors could be resolved from checkpoint.
    ValueError
        If a layer's tensors deviate from the structural layout expected for the given
        architecture.
    """
    shard_index = build_tensor_shard_index(ckpt_path, architecture)
    by_layer = group_expert_tensors_by_layer(shard_index, architecture)
    if not by_layer:
        raise ValueError(
            f"No routed expert tensors found in `{ckpt_path.as_posix()!r}`."
        )

    block_idxs = sorted(by_layer)
    if max_moe_blocks is not None:
        block_idxs = block_idxs[:max_moe_blocks]

    config = json.loads((ckpt_path / "config.json").read_text())
    cache = ShardTensorCache(shard_index=shard_index)

    first_stacked = cache.load_layer(by_layer[block_idxs[0]])
    order = architecture.validate_layer_tensors(by_layer[block_idxs[0]])
    specs = derive_tensor_specs(expert_ndarrays(first_stacked, 0), order)
    num_routed_experts = int(next(iter(first_stacked.values())).shape[0])

    quant = architecture.resolve_quantization(config, block_idxs)
    encoding = make_encoding_tag(
        quant,
        architecture.dtype_tag(first_stacked, quantized=quant is not None),
    )
    del first_stacked
    gc.collect()

    model_moe_spec = architecture.extract_model_moe_spec(
        config, block_idxs, num_routed_experts
    )
    with ExpertBankWriter(
        expert_bank_dir,
        model_id=model_id if model_id is not None else ckpt_path.name,
        model_fingerprint=hash_model_ckpt(ckpt_path),
        payload_encoding=encoding,
        tensor_specs=specs,
        model_moe_spec=model_moe_spec,
        overwrite=overwrite,
    ) as writer:
        for block_idx in block_idxs:
            layer_order = architecture.validate_layer_tensors(by_layer[block_idx])
            if layer_order != order:
                raise ValueError(
                    f"Layer {block_idx} holds tensors {layer_order!r}; the expert bank's "
                    f"layout is {order!r}."
                )

            stacked = cache.load_layer(by_layer[block_idx])
            for expert_idx in range(num_routed_experts):
                writer.add_expert(
                    block_idx=block_idx,
                    expert_idx=expert_idx,
                    data=assemble_expert_blob(
                        expert_ndarrays(stacked, expert_idx), specs
                    ),
                )
            del stacked
            gc.collect()

        return writer.finalize()
