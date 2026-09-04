from __future__ import annotations

from typing import Optional
from collections.abc import Mapping

import attrs
from attrs import field

from pathlib import Path

import json

import gc

import numpy as np

import mlx.core as mx

from preempt.expert_bank.blob import assemble_expert_blob, derive_tensor_specs
from preempt.expert_bank.manifest import ExpertBankManifest
from preempt.expert_bank.writer import ExpertBankWriter

from preempt.utils.hf_utils import hash_model_ckpt

from ..utils import mlx_to_numpy, make_encoding_tag

from .base_adapter import BaseMoEArchAdapter


def _make_shard_index(
    ckpt_path: Path, arch_adapter: BaseMoEArchAdapter
) -> dict[str, Path]:
    """Filters routed expert layer weights from a model checkpoint's
    weight map (`model.safetensors.index.json` and returns dict mapping
    dotted weight paths to safetensors file paths, for example:

    `'language_model.model.layers.0.mlp.switch_mlp.down_proj.weight'`
    → `'model-00001-of-00005.safetensors'`

    Parameters
    ----------
    ckpt_path : Path
        Local path to a Hugging Face-style model checkpoint
    arch_adapter : BaseMoEArchAdapter
        Architecture adapter providing regex patterns to identify routed
        expert layer weights

    Returns
    -------
    dict[str, Path]
        Mapping of weight tensor dotted paths to safetensors file paths
    """
    if (index_path := ckpt_path / "model.safetensors.index.json").exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {
            path: ckpt_path / shard
            for path, shard in weight_map.items()
            if arch_adapter.expert_weight_path_regex.match(path)
        }

    single = ckpt_path / "model.safetensors"
    return {path: single for path in mx.load(single) if regex.match(path)}  # type: ignore


def _group_weight_paths_by_block(
    shard_index: Mapping[str, Path],
    arch_adapter: BaseMoEArchAdapter,
) -> dict[int, dict[str, str]]:
    """Groups routed expert weight tensors by parent transformer block
    index.

    Each block index maps to a nested mapping of linear projection
    component names (e.g. `'gate_proj.weight'`, `'down_proj.scales'`)
    to their full, dotted tensor paths in the checkpoint (e.g.
    `'language_model.model.layers.0.mlp.switch_mlp.down_proj.weight'`).

    Parameters
    ----------
    shard_index : Mapping[str, Path]
        Mapping of dotted weight paths to checkpoint shard file paths
    arch_adapter : BaseMoEArchAdapter
        Architecture adapter providing regex patterns to parse and match
        weight paths

    Returns
    -------
    dict[int, dict[str, str]]
        Mapping of parent transformer block indices to grouped expert
        weight paths

    Raises
    ------
    ValueError
        If expert weight paths match multiple distinct module prefixes.
    """
    grouped_w_paths: dict[int, dict[str, str]] = {}
    prefixes: set[str] = set()

    for weight_path in shard_index:
        match = arch_adapter.expert_weight_path_regex.match(weight_path)
        if match is None:
            continue

        prefixes.add(match.group("prefix"))
        suffix = f"{match.group('projection')}.{match.group('part')}"
        grouped_w_paths.setdefault(int(match.group("layer")), {})[suffix] = weight_path

    if len(prefixes) > 1:
        raise ValueError(
            "Expected all routed expert weight paths to share a single module"
            f"prefix, but found multiple: {sorted(prefixes)!r}"
        )

    return grouped_w_paths


def _expert_ndarrays(
    weight_map: Mapping[str, mx.array], expert_idx: int
) -> dict[str, np.ndarray]:
    return {
        path: mlx_to_numpy(tensor[expert_idx]) for path, tensor in weight_map.items()
    }


@attrs.define
class ShardTensorCache:
    shard_index: Mapping[str, Path] = field()
    _loaded_fp: Path | None = field(default=None, init=False)
    _loaded: dict[str, mx.array] = field(factory=dict, init=False)

    def get(self, tensor_path: str) -> mx.array:
        if (fp := self.shard_index[tensor_path]) != self._loaded_fp:
            self._loaded = mx.load(fp)  # type: ignore
            self._loaded_fp = fp

        return self._loaded[tensor_path]

    def load_layer(self, layer_weights: Mapping[str, str]) -> dict[str, mx.array]:
        # Sort by shard order to reduce disk reads for layers split between files
        ordered = sorted(
            layer_weights.items(), key=lambda item: self.shard_index[item[1]]
        )
        return {path: self.get(tensor_path) for path, tensor_path in ordered}


# TODO rm 'arch_adapter', resolve from config.json or registry
def model_to_expert_bank(
    ckpt_path: Path,
    expert_bank_dir: Path,
    arch_adapter: BaseMoEArchAdapter,
    *,
    model_id: Optional[str] = None,
    overwrite: bool = False,
) -> ExpertBankManifest:
    """Serializes a MoE model's routed expert weights and persists them to
    an expert bank on disk.

    Parameters
    ----------
    ckpt_dir : Path
        Local path to a Hugging Face-style model checkpoint
    expert_bank_dir : Path
        Path to local directory where `experts.bin` and `manifest.json`
        will be written
    arch_adapter : BaseMoEArchAdapter
        Adapter defining the checkpoint's MoE structural patterns
    model_id : Optional[str]
        A unique identifier used for compatibility checks when loading
        the saved expert bank. Defaults to the name of the model checkpoint,
        but should be explicitly provided for specific cached Hugging Face
        snapshots, by default `None`
    overwrite : bool
        If `True`, overwrites existing expert bank files in `expert_bank_dir`,
        by default False

    Returns
    -------
    ExpertBankManifest
        The manifest for the expert bank

    Raises
    ------
    ValueError
        If no routed expert tensors could be resolved from the checkpoint.
    ValueError
        If a layer deviates from the structural layout expected for the
        architecture defined by `arch_adapter`.
    """
    shard_index = _make_shard_index(ckpt_path, arch_adapter)
    grouped_w_paths = _group_weight_paths_by_block(shard_index, arch_adapter)

    # * Prechecks
    if not grouped_w_paths:
        raise ValueError(f"No expert weights found in {ckpt_path.as_posix()!r}")

    block_idxs = sorted(grouped_w_paths)
    validated_path_groups = [
        arch_adapter.validate_weight_paths(grouped_w_paths[b]) for b in block_idxs
    ]
    if not all(pg == validated_path_groups[0] for pg in validated_path_groups):
        raise ValueError()  # TODO error msg

    config = json.loads((ckpt_path / "config.json").read_text())
    cache = ShardTensorCache(shard_index=shard_index)

    # * Get ExpertBankWriter constructor args
    model_id_ = model_id or ckpt_path.name
    model_fingerprint = hash_model_ckpt(ckpt_path)
    model_moe_spec = arch_adapter.get_model_moe_spec(config, block_idxs)

    # * Derive encoding and tensor_specs from the first layer
    dummy_w_paths = grouped_w_paths[block_idxs[0]]
    dummy_layer = cache.load_layer(dummy_w_paths)

    quant = arch_adapter.resolve_quantization(config, block_idxs)
    dtype_tag = arch_adapter.dtype_tag_for(dummy_layer, quantized=quant is not None)
    encoding = make_encoding_tag(
        quant, dtype_tag
    )  # TODO resolve from config.json + weight map

    w_ndarrays = _expert_ndarrays(dummy_layer, 0)
    tensor_specs = derive_tensor_specs(w_ndarrays, validated_path_groups[0])

    del dummy_w_paths, dummy_layer
    gc.collect()

    with ExpertBankWriter(
        expert_bank_dir,
        model_id=model_id_,
        model_fingerprint=model_fingerprint,
        encoding=encoding,
        tensor_specs=tensor_specs,
        model_moe_spec=model_moe_spec,
        overwrite=overwrite,
    ) as writer:
        for b in block_idxs:
            weights = cache.load_layer(grouped_w_paths[b])

            for e in range(model_moe_spec.num_routed_experts):
                data = assemble_expert_blob(_expert_ndarrays(weights, e), tensor_specs)
                writer.add_expert(
                    block_idx=b,
                    expert_idx=e,
                    data=data,
                )

            del weights, data
            gc.collect()

        return writer.finalize()
