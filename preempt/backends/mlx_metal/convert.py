"""Convert an MLX-quantized MoE checkpoint into a packed expert store.

Offline, one-time-per-model. MLX-layout-aware by design: this module knows
mlx_lm's stacked `[num_experts, ...]` tensor naming; the container layout it
feeds belongs entirely to `preempt.storage.writer.PackedStoreWriter`.

Blob bytes are the checkpoint's bytes, unmodified: nothing here dequantizes,
casts, or reorders within a tensor. numpy has no `bfloat16`, so a `bfloat16`
tensor is bit-viewed as `uint16` on the way out: its bytes stay exact, but its
`TensorSpec.dtype` reads `uint16` and cannot be told apart from `float16` by
shape or size. The logical scalar type is therefore recorded in the store's
`payload_encoding` tag, which reads `mlx-<mode>-q<bits>-g<group>-<scalar>`
(e.g. `mlx-affine-q4-g64-bf16`) or `mlx-unquantized-<scalar>`.

What a reader can recover from a store alone: the per-tensor byte layout
(`tensor_specs`), the quantization mode/bits/group size, and the scalar type
of `scales`/`biases`. What it cannot: anything else about the source model —
that is what `model_id` plus `model_fingerprint` are for.

Usage (macOS host)::

    python -m preempt.backends.mlx_metal.convert \
        --model <hf-id-or-local-dir> --output out/expert-store
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import argparse
import hashlib
import json
import re
from pathlib import Path

import attrs
from attrs import field
import numpy as np

import mlx.core as mx

from preempt.storage.blob import assemble_expert_blob, derive_tensor_specs
from preempt.storage.manifest import ExpertStoreManifest, ExpertTopology
from preempt.storage.writer import PackedStoreWriter

# Blob order: for each projection, weight then scales then biases.
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
QUANT_PARTS = ("weight", "scales", "biases")
TENSOR_ORDER = tuple(
    f"{projection}.{part}" for projection in EXPERT_PROJECTIONS for part in QUANT_PARTS
)

# A multimodal checkpoint namespaces its text stack (Qwen3.6-35B-A3B ships as
# `language_model.model.layers.<i>....`), so leading segments are optional and
# captured: two stacks matching this shape must never be merged silently.
_EXPERT_MODULE_PATTERN = (
    r"(?P<prefix>(?:[A-Za-z0-9_]+\.)*)model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)"
)
_EXPERT_TENSOR_RE = re.compile(
    rf"^{_EXPERT_MODULE_PATTERN}\.(?P<part>weight|scales|biases)$"
)
_EXPERT_MODULE_RE = re.compile(rf"^{_EXPERT_MODULE_PATTERN}$")

# Short, stable tags for the scalar type `scales`/`biases` decode as; numpy
# cannot represent bfloat16, so the tag is the only durable record of it.
_SCALAR_DTYPE_TAGS: tuple[tuple[Any, str], ...] = (
    (mx.bfloat16, "bf16"),
    (mx.float16, "f16"),
    (mx.float32, "f32"),
)


@attrs.define(frozen=True, kw_only=True)
class QuantParams:
    """Quantization parameters governing one routed-expert module."""

    mode: str = field()
    bits: int = field()
    group_size: int = field()


def resolve_model_dir(model: str) -> Path:
    """Resolve a model id or local path to a checkpoint directory.

    Parameters
    ----------
    model : str
        Local directory, or a Hugging Face repo id to snapshot-download.

    Returns
    -------
    Path
        Directory holding `config.json` and the safetensors shards.
    """
    path = Path(model)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download  # mlx_lm transitive dependency

    return Path(snapshot_download(model))


def compute_mlx_fingerprint(model_dir: Path) -> str:
    """Fingerprint the checkpoint a store is packed from.

    Cheap and stable: sha256 of `config.json` bytes plus the sorted
    `(name, size)` list of safetensors shards. Catches shape/layout/quant
    changes; does not detect in-place bit flips (acceptable — deep hashing
    20 GB per run is not).

    Parameters
    ----------
    model_dir : Path
        Checkpoint directory.

    Returns
    -------
    str
        Hex sha256 digest identifying the checkpoint.
    """
    digest = hashlib.sha256((model_dir / "config.json").read_bytes())

    for shard in sorted(model_dir.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())

    return digest.hexdigest()


def build_tensor_shard_index(model_dir: Path) -> dict[str, Path]:
    """Map expert tensor name to the shard file holding it.

    Parameters
    ----------
    model_dir : Path
        Checkpoint directory.

    Returns
    -------
    dict[str, Path]
        Routed-expert tensor names mapped to their safetensors shard; read
        from the safetensors index, or from the single `model.safetensors`
        when the checkpoint is unsharded.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {
            name: model_dir / shard
            for name, shard in weight_map.items()
            if _EXPERT_TENSOR_RE.match(name)
        }

    single = model_dir / "model.safetensors"
    return {
        name: single for name in mx.load(str(single)) if _EXPERT_TENSOR_RE.match(name)
    }


def group_expert_tensors_by_layer(
    shard_index: Mapping[str, Path],
) -> dict[int, dict[str, str]]:
    """Group expert tensor names by MoE layer.

    Parameters
    ----------
    shard_index : Mapping[str, Path]
        Expert tensor names mapped to their shard, as built by
        `build_tensor_shard_index`.

    Returns
    -------
    dict[int, dict[str, str]]
        Layer index mapped to `{tensor suffix: full tensor name}`, where the
        suffix is expert-relative, e.g. `gate_proj.weight`.

    Raises
    ------
    ValueError
        If matched tensors come from more than one namespace prefix — two
        stacks (say a vision tower with its own `switch_mlp`) key identically
        on (layer, suffix) and would silently overwrite each other.
    """
    by_layer: dict[int, dict[str, str]] = {}
    prefixes: set[str] = set()

    for name in shard_index:
        match = _EXPERT_TENSOR_RE.match(name)
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


def resolve_tensor_order(layer_tensors: Mapping[str, str]) -> tuple[str, ...]:
    """Pick the blob tensor order present in this checkpoint.

    An unquantized checkpoint has no `scales`/`biases`, so the canonical
    `TENSOR_ORDER` is filtered rather than assumed.

    Parameters
    ----------
    layer_tensors : Mapping[str, str]
        One layer's `{tensor suffix: full tensor name}` mapping.

    Returns
    -------
    tuple[str, ...]
        Tensor suffixes in blob order.

    Raises
    ------
    ValueError
        If a projection's `weight` tensor is missing, or the layer holds a
        tensor `TENSOR_ORDER` does not name.
    """
    order = tuple(suffix for suffix in TENSOR_ORDER if suffix in layer_tensors)

    missing = [
        f"{projection}.weight"
        for projection in EXPERT_PROJECTIONS
        if f"{projection}.weight" not in layer_tensors
    ]
    if missing:
        raise ValueError(f"Checkpoint is missing expert tensors: {missing!r}")

    unexpected = sorted(set(layer_tensors) - set(order))
    if unexpected:
        raise ValueError(f"Unrecognized expert tensors: {unexpected!r}")

    return order


def text_config_of(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Returns the config section describing the text stack.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed `config.json` of the source checkpoint.

    Returns
    -------
    Mapping[str, Any]
        `config["text_config"]` for a multimodal checkpoint, otherwise `config`
        itself — the routed-expert topology lives in whichever holds it.
    """
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def quantization_section(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the checkpoint's quantization section, wherever it lives.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed `config.json` of the source checkpoint.

    Returns
    -------
    Mapping[str, Any] | None
        The `quantization` mapping from the config root, else from
        `text_config`; `None` for an unquantized checkpoint.
    """
    for section in (config, text_config_of(config)):
        quantization = section.get("quantization")
        if isinstance(quantization, dict):
            return quantization
    return None


def resolve_expert_quantization(
    config: Mapping[str, Any], layer_idxs: Sequence[int]
) -> QuantParams | None:
    """Resolve the quantization the routed experts of `layer_idxs` actually use.

    A dynamically quantized checkpoint (Unsloth "UD" and friends) overrides
    bits/group size per module, so the config root's defaults may describe
    nothing the experts use. Per-module `switch_mlp` entries therefore win, and
    the root default applies only to modules with no entry of their own.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed `config.json` of the source checkpoint.
    layer_idxs : Sequence[int]
        MoE layer indices being packed.

    Returns
    -------
    QuantParams | None
        The parameters every packed expert module shares; `None` if the
        checkpoint is unquantized.

    Raises
    ------
    ValueError
        If the packed expert modules do not all share one set of parameters —
        one store carries one `payload_encoding`, so a mixed selection must
        fail here rather than stamp blobs with parameters they do not use.
    """
    quantization = quantization_section(config)
    if quantization is None:
        return None

    mode = str(quantization.get("mode", "affine"))
    default: QuantParams | None = None
    if "bits" in quantization and "group_size" in quantization:
        default = QuantParams(
            mode=mode,
            bits=int(quantization["bits"]),
            group_size=int(quantization["group_size"]),
        )

    overrides: dict[tuple[int, str], QuantParams | None] = {}
    for key, value in quantization.items():
        match = _EXPERT_MODULE_RE.match(key)
        if match is None:
            continue
        identity = (int(match.group("layer")), match.group("projection"))
        overrides[identity] = (
            QuantParams(
                mode=mode,
                bits=int(value["bits"]),
                group_size=int(value["group_size"]),
            )
            if isinstance(value, dict)
            else None  # an explicitly unquantized expert module
        )

    resolved = {
        overrides.get((layer_idx, projection), default)
        for layer_idx in layer_idxs
        for projection in EXPERT_PROJECTIONS
    }

    if resolved == {None}:
        return None
    if len(resolved) != 1:
        raise ValueError(
            "Routed experts are not uniformly quantized across the packed "
            f"layers: {sorted(repr(params) for params in resolved)!r}"
        )

    return resolved.pop()


def scalar_dtype_tag(stacked: Mapping[str, mx.array], *, quantized: bool) -> str:
    """Name the scalar type the blob's non-packed tensors decode as.

    Parameters
    ----------
    stacked : Mapping[str, mx.array]
        One layer's stacked tensors, keyed by expert-relative suffix.
    quantized : bool
        Whether the checkpoint is quantized; if so the scalar type is that of
        `scales`/`biases`, otherwise that of the weights themselves.

    Returns
    -------
    str
        Short tag, one of `bf16`, `f16`, `f32`.

    Raises
    ------
    ValueError
        If the relevant tensors disagree on dtype, or use a dtype this
        converter has no tag for.
    """
    names = sorted(
        name
        for name in stacked
        if not quantized or name.endswith((".scales", ".biases"))
    )
    if not names:
        raise ValueError("Layer holds no tensor to read a scalar dtype from.")

    dtypes = {str(stacked[name].dtype) for name in names}
    if len(dtypes) != 1:
        raise ValueError(f"Expert tensors disagree on dtype: {sorted(dtypes)!r}")

    dtype = stacked[names[0]].dtype
    for candidate, tag in _SCALAR_DTYPE_TAGS:
        if dtype == candidate:
            return tag

    raise ValueError(f"Unsupported expert scalar dtype: {dtype}")


def payload_encoding_for(quant: QuantParams | None, scalar_tag: str) -> str:
    """Build the store's opaque encoding tag.

    Parameters
    ----------
    quant : QuantParams | None
        Parameters the packed experts are quantized with; `None` if they are
        not quantized.
    scalar_tag : str
        Scalar type tag from `scalar_dtype_tag`.

    Returns
    -------
    str
        Encoding tag, e.g. `mlx-affine-q4-g64-bf16` or `mlx-unquantized-bf16`.
    """
    if quant is None:
        return f"mlx-unquantized-{scalar_tag}"
    return f"mlx-{quant.mode}-q{quant.bits}-g{quant.group_size}-{scalar_tag}"


def to_numpy(tensor: mx.array) -> np.ndarray:
    """Convert an MLX tensor to numpy without altering a single bit.

    Parameters
    ----------
    tensor : mx.array
        Tensor sliced out of a checkpoint shard.

    Returns
    -------
    np.ndarray
        The same bytes; a `bfloat16` tensor comes back as `uint16` because
        numpy has no `bfloat16` and casting would change the payload.
    """
    if tensor.dtype == mx.bfloat16:
        return np.array(tensor.view(mx.uint16))
    return np.array(tensor)


@attrs.define
class ShardTensorCache:
    """Load stacked expert tensors, holding at most one shard at a time.

    Consecutive MoE layers almost always live in the same shard, so caching
    the last-loaded one keeps a full-model conversion to a single pass over
    the checkpoint without ever holding two shards resident.
    """

    shard_index: Mapping[str, Path] = field()
    _loaded_path: Path | None = field(default=None, init=False)
    _loaded: dict[str, mx.array] = field(factory=dict, init=False)

    def get(self, name: str) -> mx.array:
        """Return one stacked `[num_experts, ...]` tensor.

        Parameters
        ----------
        name : str
            Full checkpoint tensor name.

        Returns
        -------
        mx.array
            The stacked tensor, still lazy.
        """
        path = self.shard_index[name]
        if path != self._loaded_path:
            self._loaded = mx.load(str(path))
            self._loaded_path = path
        return self._loaded[name]

    def load_layer(self, layer_tensors: Mapping[str, str]) -> dict[str, mx.array]:
        """Return one layer's stacked tensors, keyed by expert-relative suffix.

        Parameters
        ----------
        layer_tensors : Mapping[str, str]
            One layer's `{tensor suffix: full tensor name}` mapping.

        Returns
        -------
        dict[str, mx.array]
            Stacked tensors for that layer.
        """
        # Shard-major request order: a layer split across two shards would
        # otherwise reload each of them once per alternation.
        ordered = sorted(
            layer_tensors.items(), key=lambda item: self.shard_index[item[1]]
        )
        return {suffix: self.get(name) for suffix, name in ordered}


def expert_arrays(
    stacked: Mapping[str, mx.array], expert_idx: int
) -> dict[str, np.ndarray]:
    """Slice one expert out of a layer's stacked tensors.

    Parameters
    ----------
    stacked : Mapping[str, mx.array]
        One layer's stacked `[num_experts, ...]` tensors.
    expert_idx : int
        Router-local expert index to extract.

    Returns
    -------
    dict[str, np.ndarray]
        That expert's tensors, keyed by expert-relative suffix.
    """
    return {suffix: to_numpy(tensor[expert_idx]) for suffix, tensor in stacked.items()}


def convert_mlx_model_to_store(
    model_dir: Path,
    store_dir: Path,
    *,
    model_id: str | None = None,
    max_layers: int | None = None,
    overwrite: bool = False,
) -> ExpertStoreManifest:
    """Pack an MLX checkpoint's routed experts into a packed expert store.

    Memory stays bounded: one shard and one layer's stacked tensors are in
    flight at a time, regardless of checkpoint size.

    Parameters
    ----------
    model_dir : Path
        Checkpoint directory holding `config.json` and safetensors shards.
    store_dir : Path
        Destination directory for `experts.bin` and `manifest.json`.
    model_id : str | None
        Identifier to record in the manifest — the id the engine will load the
        model under, since `PackedExpertStore.ensure_compatible` compares
        against it. Defaults to the checkpoint directory name, which for a
        Hugging Face cache is a snapshot sha and matches nothing.
    max_layers : int | None
        Convert only the first `max_layers` MoE layers, by default all of them.
    overwrite : bool
        Whether to replace an existing store in `store_dir`, by default False.

    Returns
    -------
    ExpertStoreManifest
        The manifest written into `store_dir`.

    Raises
    ------
    ValueError
        If the checkpoint holds no routed expert tensors, or a layer's tensors
        disagree with the layout derived from the first layer.
    """
    shard_index = build_tensor_shard_index(model_dir)
    by_layer = group_expert_tensors_by_layer(shard_index)
    if not by_layer:
        raise ValueError(f"No routed expert tensors found in `{model_dir}`.")

    layer_idxs = sorted(by_layer)
    if max_layers is not None:
        layer_idxs = layer_idxs[:max_layers]

    config = json.loads((model_dir / "config.json").read_text())
    cache = ShardTensorCache(shard_index=shard_index)

    first_stacked = cache.load_layer(by_layer[layer_idxs[0]])
    order = resolve_tensor_order(by_layer[layer_idxs[0]])
    specs = derive_tensor_specs(expert_arrays(first_stacked, 0), order)
    num_experts = int(next(iter(first_stacked.values())).shape[0])

    quant = resolve_expert_quantization(config, layer_idxs)
    encoding = payload_encoding_for(
        quant, scalar_dtype_tag(first_stacked, quantized=quant is not None)
    )
    del first_stacked

    topology = ExpertTopology(
        moe_layer_idxs=tuple(layer_idxs),
        num_experts=num_experts,
        top_k=int(text_config_of(config)["num_experts_per_tok"]),
    )

    with PackedStoreWriter(
        store_dir,
        model_id=model_id if model_id is not None else model_dir.name,
        model_fingerprint=compute_mlx_fingerprint(model_dir),
        payload_encoding=encoding,
        tensor_specs=specs,
        topology=topology,
        overwrite=overwrite,
    ) as writer:
        for layer_idx in layer_idxs:
            layer_order = resolve_tensor_order(by_layer[layer_idx])
            if layer_order != order:
                raise ValueError(
                    f"Layer {layer_idx} holds tensors {layer_order!r}; the store's "
                    f"layout is {order!r}."
                )

            stacked = cache.load_layer(by_layer[layer_idx])
            for expert_idx in range(num_experts):
                writer.add_expert(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    data=assemble_expert_blob(
                        expert_arrays(stacked, expert_idx), specs
                    ),
                )
            del stacked

        return writer.finalize()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the converter CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Argument vector, by default `sys.argv[1:]`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pack an MLX MoE checkpoint's routed experts into a store."
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
        help="Destination store directory.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Convert only the first N MoE layers; default is all of them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing store in the output directory.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the converter from the command line."""
    args = parse_args()

    model_dir = resolve_model_dir(args.model)
    print(f"Converting {model_dir} -> {args.output}")

    manifest = convert_mlx_model_to_store(
        model_dir,
        args.output,
        model_id=args.model,
        max_layers=args.max_layers,
        overwrite=args.overwrite,
    )

    expert_nbytes = manifest.expert_nbytes()
    print(
        f"Packed {len(manifest.blobs)} expert blob(s) across "
        f"{len(manifest.topology.moe_layer_idxs)} layer(s); "
        f"{expert_nbytes} bytes each, "
        f"{expert_nbytes * len(manifest.blobs)} payload bytes total."
    )
    print(f"Model id: {manifest.model_id}")
    print(f"Encoding: {manifest.payload_encoding}")
    print(f"Fingerprint: {manifest.model_fingerprint}")


if __name__ == "__main__":
    main()
