"""Qwen3/Qwen3-Next MoE architecture definitions and conventions.

Centralizes SwiGLU expert layer definitions, tensor naming conventions, and
quantization patterns specific to Qwen3/Qwen3-Next models.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import re

import mlx.core as mx

from preempt.backends.mlx_metal.architecture import MoEArchitecture
from preempt.backends.mlx_metal.constants import (
    SWIGLU_PROJECTION_NAMES,
    QUANTIZED_TENSOR_PARTS,
)
from preempt.backends.mlx_metal.quantization import (
    MlxQuantParams,
    scalar_dtype_tag,
)
from preempt.storage.manifest import ModelMoESpec


# TODO rename to convey this is an adapter
# TODO rename 'config' arg in methods to 'ckpt'
class Qwen3NextMoEArchitecture(MoEArchitecture):
    """Architecture adapter for Qwen3 and Qwen3-Next MoE checkpoints.

    Supports SwiGLU experts containing three linear projections (`'gate_proj'`,
    `'up_proj'`, `'down_proj'`) which may be quantized with MLX. Module paths
    follow the pattern `'model.layers.<i>.mlp.switch_mlp.<projection>'`
    """

    @property
    def projection_names(self) -> tuple[str, ...]:
        """The names SwiGLU's three projections in blob order."""
        return SWIGLU_PROJECTION_NAMES

    @property
    def quantized_tensor_parts(self) -> tuple[str, ...]:  # TODO redundant, remove
        """Structural components of a quantized MLX tensor."""
        return QUANTIZED_TENSOR_PARTS

    @property
    def expert_module_pattern(self) -> str:
        """Regex matching Qwen3-next expert layer paths in a checkpoint.

        Captures `prefix` (for multimodal namespace detection), `layer`,
        and `projection`.
        """
        return (
            r"(?P<prefix>(?:[A-Za-z0-9_]+\.)*)model\.layers\.(?P<layer>\d+)"
            r"\.mlp\.switch_mlp\.(?P<projection>gate_proj|up_proj|down_proj)"
        )

    @property
    def layer_class_name(self) -> str:  # TODO redundant, remove or make classvar
        """Class name for MoE block used in Qwen3/Qwen3-Next implementations in
        `mlx_lm` (`Qwen3NextSparseMoeBlock`)"""
        return "Qwen3NextSparseMoeBlock"

    def expert_tensor_regex(self) -> re.Pattern[str]:
        """Regex matching absolute paths to expert tensors in dot notation.

        Captures the constituent component type (`weight`, `scales`, or `biases`)
        under the `<part>` group.
        """
        return re.compile(
            rf"^{self.expert_module_pattern}\.(?P<part>weight|scales|biases)$"
        )

    def expert_module_regex(self) -> re.Pattern[str]:
        """Regex matching base expert module paths in dot notation.

        Excludes parameter component suffixes (e.g. `.weight`, `.scales`).
        """
        return re.compile(rf"^{self.expert_module_pattern}$")

    def tensor_order(self) -> tuple[str, ...]:  # TODO make this a property
        """Storage and evaluation order for expert tensors.

        Yields permutations of all projection names paired with each quantized
        tensor part.
        """
        return tuple(
            f"{proj}.{part}"
            for proj in self.projection_names
            for part in self.quantized_tensor_parts
        )

    # TODO rename layer_tensors arg
    def validate_layer_tensors(
        self, layer_tensors: Mapping[str, str]
    ) -> tuple[str, ...]:
        """Validates weights in `layer_tensors` against the expected tensor
        order and projection weight names for the adapter's MoE architecture.

        Supports both quantized and unquantized checkpoints.

        Parameters
        ----------
        layer_tensors : Mapping[str, str]
            Mapping of a layer's tensor suffixes to absolute tensor paths (in dot
            notation)

        Returns
        -------
        tuple[str, ...]
            The subset of tensor suffixes present in this layer in adapter's
            `tensor_order`.

        Raises
        ------
        ValueError
            If a required projection's `weight` tensor is missing
        ValueError
            If `layer_tensors` contains an unexpected tensor name
        """
        order = self.tensor_order()
        # present = tuple(name for name in order if name in layer_tensors)
        # missing = [
        #     f"{proj}.weight"
        #     for proj in self.projection_names
        #     if f"{proj}.weight" not in layer_tensors
        # ]
        present = tuple(filter(lambda x: x in layer_tensors, order))
        missing = list(
            filter(lambda x: f"{x}.weight" not in layer_tensors, self.projection_names)
        )
        if missing:
            raise ValueError(
                "The following weight tensors are missing from `layer_tensors`: "
                f"{missing!r}"
            )

        unexpected = set(layer_tensors) - set(order)
        if unexpected:
            raise ValueError(
                "`layer_tensors` contains the following unexpected weight tensors: "
                f"{unexpected!r}"
            )

        return present

    # TODO verify block_idxs = transformer blocks
    def resolve_quantization(
        self, config: Mapping[str, Any], block_idxs: Sequence[int]
    ) -> MlxQuantParams | None:
        """Determines the effective quantization parameters used by routed experts.

        Handles dynamically quantized checkpoints (e.g. Unsloth UD format) which
        may define per-module `bits` and `group_size` overrides that take
        precedence over global/default configuration.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed contents of the source checkpoint's `config.json`
        block_idxs : Sequence[int]
            Indices of the resolved expert layers' parent MoE blocks

        Returns
        -------
        MlxQuantParams | None
            The quantization parameters shared by experts in the checkpoint,
            or `None` if experts are unquantized.

        Raises
        ------
        ValueError
            If quantization parameters are not uniform across all resolved expert
            layers
        """
        quantization = self._quantization_section(config)
        if quantization is None:
            return None

        mode = str(
            quantization.get("mode", "affine")
        )  # TODO check if affine is a valid default

        # if "bits" in quantization and "group_size" in quantization:
        if all(param in quantization for param in ("bits", "group_size")):
            default = MlxQuantParams(
                mode=mode,
                bits=int(quantization["bits"]),
                group_size=int(quantization["group_size"]),
            )
        else:
            default = None

        module_re = self.expert_module_regex()
        overrides: dict[tuple[int, str], MlxQuantParams | None] = {}

        for key, value in quantization.items():
            match = module_re.match(key)
            if match is None:
                continue

            identity = (int(match.group("layer")), match.group("projection"))
            overrides[identity] = (
                MlxQuantParams(
                    mode=mode,
                    bits=int(value["bits"]),
                    group_size=int(value["group_size"]),
                )
                if isinstance(value, dict)
                else None  # explicitly unquantized
            )

        resolved = {
            overrides.get((block_idx, projection), default)
            for block_idx in block_idxs
            for projection in self.projection_names
        }

        if resolved == {None}:
            return None

        if len(resolved) != 1:
            raise ValueError(
                f"Checkpoint's experts are not uniformly quantized: "
                f"{sorted(repr(p) for p in resolved)!r}"
            )

        return resolved.pop()

    def scalar_dtype_tag(
        self, stacked: Mapping[str, mx.array], *, quantized: bool
    ) -> str:
        """Returns a scalar dtype tag for the weight tensors in `stacked`."""
        return scalar_dtype_tag(stacked, quantized=quantized)

    # TODO rename, MoE spec is composed, not extracted
    # TODO verify block_idxs = transformer blocks
    def extract_model_moe_spec(
        self,
        config: Mapping[str, Any],
        block_idxs: Sequence[int],  # TODO rename
        num_routed_experts: int,
    ) -> ModelMoESpec:
        """Extracts the model's MoE specification from checkpoint.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed contents of the source checkpoint's `config.json`.
        block_idxs : Sequence[int]
            Indices of the MoE blocks.
        num_routed_experts : int
            Total number of routed experts per layer (usually detected
            dynamically from checkpoint's tensor layout).

        Returns
        -------
        ModelMoESpec
        """
        text_config = config.get("text_config")
        tc = text_config if isinstance(text_config, dict) else config

        return ModelMoESpec(
            moe_block_idxs=tuple(block_idxs),
            num_routed_experts=num_routed_experts,
            top_k=int(tc["num_experts_per_tok"]),
        )

    @staticmethod
    def _quantization_section(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Locates and returns the quantization parameters in `config`."""
        text_config = config.get("text_config")

        for section in (
            config,
            text_config if isinstance(text_config, dict) else {},
        ):
            quantization = section.get("quantization")
            if isinstance(quantization, dict):
                return quantization
