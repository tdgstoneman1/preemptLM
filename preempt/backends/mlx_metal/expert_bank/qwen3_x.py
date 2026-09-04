"""Qwen3/Qwen3-Next MoE architecture definitions and conventions.

Centralizes SwiGLU expert layer definitions, tensor naming conventions, and
quantization patterns specific to Qwen3/Qwen3-Next models.
"""

from __future__ import annotations

from typing import ClassVar, Any
from collections.abc import Mapping, Sequence

import re

import mlx.core as mx

from preempt.expert_bank.manifest import ModelMoESpec

from .base_adapter import BaseMoEArchAdapter

from ..constants import (
    SWIGLU_PROJECTION_NAMES,
    MLX_QUANTIZED_TENSOR_PARTS,
)
from ..quantization import QuantSettings
from ..utils import dtype_tag_from_arrays


class Qwen3_xArchAdapter(BaseMoEArchAdapter):
    """Architecture adapter for Qwen3.X and Qwen3-Next MoE blocks expert bank
    serialization

    Supports SwiGLU experts containing three linear projections (`'gate_proj'`,
    `'up_proj'`, `'down_proj'`) which may be quantized with MLX.

    Module paths follow the pattern
    `'language_model.model.layers.<layer idx>.mlp.switch_mlp.<projection>'`
    """

    _model_arch: ClassVar[str] = "Qwen3_5MoeForConditionalGeneration"
    _moe_class_name: ClassVar[str] = "Qwen3NextSparseMoeBlock"
    _path_pattern: ClassVar[str] = (
        r"(?P<prefix>(?:[A-Za-z0-9_]+\.)*)model\.layers\.(?P<layer>\d+)"
        r"\.mlp\.switch_mlp\.(?P<projection>gate_proj|up_proj|down_proj)"
    )

    @property
    def model_architecture(self) -> str:
        return self._model_arch

    @property
    def moe_class_name(self) -> str:
        return self._moe_class_name

    @property
    def linear_projection_names(self) -> tuple[str, ...]:
        """The names SwiGLU's three projections in blob order."""
        return SWIGLU_PROJECTION_NAMES

    @property
    def expert_layer_path_regex(self) -> re.Pattern[str]:
        """Regex matching base expert module paths in dot notation.

        Excludes parameter component suffixes (e.g. `.weight`, `.scales`).
        """
        return re.compile(rf"^{self._path_pattern}$")

    @property
    def expert_weight_path_regex(self) -> re.Pattern[str]:
        """Regex matching absolute paths to expert tensors in dot notation.

        Captures the constituent component type (`weight`, `scales`, or `biases`)
        under the `<part>` group.
        """
        return re.compile(rf"^{self._path_pattern}\.(?P<part>weight|scales|biases)$")

    # TODO make quant params optional
    @property
    def weight_order(self) -> tuple[str, ...]:
        """Serialization and evaluation order for expert weights.

        Yields permutations of all linear projection names paired with each quantized
        tensor part.
        """
        return tuple(
            f"{proj}.{part}"
            for proj in self.linear_projection_names
            for part in MLX_QUANTIZED_TENSOR_PARTS
        )

    def validate_weight_paths(self, tensor_paths: Mapping[str, str]) -> tuple[str, ...]:
        """Validates an MoE block's weight tensor paths against expected path
        names for the architecture and returns them in order.

        Parameters
        ----------
        tensor_paths : Mapping[str, str]
            Weight tensor paths relative to the layer mapped to their
            full paths within the model (both in dot notation)

        Returns
        -------
        tuple[str, ...]
            Validated relative weight tensor paths

        Raises
        ------
        ValueError
            If expected paths are missing from `tensor_paths`.
        ValueError
            If `tensor_paths` contains unexpected paths.
        """
        present = tuple(filter(lambda x: x in tensor_paths, self.weight_order))
        missing = list(
            filter(
                lambda x: f"{x}.weight" not in tensor_paths,
                self.linear_projection_names,
            )
        )
        if missing:
            raise ValueError(
                "The following are missing from `tensor_paths`: " f"{missing!r}"
            )

        unexpected = set(tensor_paths) - set(self.weight_order)
        if unexpected:
            raise ValueError(
                f"`tensor_paths` contains the following unexpected items: {unexpected!r}"
            )

        return present

    # TODO verify block_idxs = transformer blocks
    def resolve_quantization(
        self, config: Mapping[str, Any], block_idxs: Sequence[int]
    ) -> QuantSettings | None:
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
        QuantSettings | None
            The quantization parameters shared by experts in the checkpoint,
            or `None` if experts are unquantized.

        Raises
        ------
        ValueError
            If quantization parameters are not uniform across resolved expert
            layers
        """
        if (quantization := self._quantization_section(config)) is None:
            return None

        mode = str(quantization.get("mode", "affine"))

        if all(param in quantization for param in ("bits", "group_size")):
            default = QuantSettings(
                mode=mode,  # type: ignore
                bits=int(quantization["bits"]),
                group_size=int(quantization["group_size"]),
            )
        else:
            default = None

        overrides: dict[tuple[int, str], QuantSettings | None] = {}

        for key, value in quantization.items():
            match = self.expert_layer_path_regex.match(key)
            if match is None:
                continue

            identity = (int(match.group("layer")), match.group("projection"))
            overrides[identity] = (
                QuantSettings(
                    mode=mode,  # type: ignore
                    bits=int(value["bits"]),
                    group_size=int(value["group_size"]),
                )
                if isinstance(value, dict)
                else None  # explicitly unquantized
            )

        resolved = {
            overrides.get((block_idx, projection), default)
            for block_idx in block_idxs
            for projection in self.linear_projection_names
        }
        if resolved == {None}:
            return

        if len(resolved) != 1:
            raise ValueError(
                f"Expert layer weights in checkpoint are not uniformly quantized: "
                f"{sorted(repr(p) for p in resolved)!r}"
            )

        return resolved.pop()

    def dtype_tag_for(self, weights: Mapping[str, mx.array], *, quantized: bool) -> str:
        """Returns a dtype tag for a layer's weights.

        Inspects `'weights'` for unquantized models, and `'scales'` and `'biases'`
        for quantized ones.

        Parameters
        ----------
        weights : Mapping[str, mx.array]
            Mapping of weight names to weight arrays
        quantized : bool
            True if the model is quantized, False otherwise

        Returns
        -------
        str
            A short dtype tag, e.g., `"bf16"`, `"f16"`, or `"f32"`
        """
        return dtype_tag_from_arrays(weights, quantized=quantized)

    def get_model_moe_spec(
        self,
        config: Mapping[str, Any],
        block_idxs: Sequence[int],
    ) -> ModelMoESpec:
        """Returns MoE configuration from a model checkpoint.

        Parses `config` to create a `ModelMoESpec` object, which includes the
        number of experts, top-k routing, and the indices of the transformer
        blocks in the model containing MoE blocks.

        Parameters
        ----------
        config : Mapping[str, Any]
            The model's parsed `config.json`
        block_idxs : Sequence[int]
            Indices of the transformer blocks containing MoE layers.

        Returns
        -------
        ModelMoESpec
            The model's MoE specification
        """
        text_config = config.get("text_config")
        config_ = text_config if isinstance(text_config, dict) else config

        return ModelMoESpec(
            moe_block_idxs=tuple(block_idxs),
            num_routed_experts=config_["num_experts"],
            top_k=int(config_["num_experts_per_tok"]),
        )

    @staticmethod
    def _quantization_section(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            text_config = {}

        for section in (config, text_config):
            quant = section.get("quantization")
            if isinstance(quant, dict):
                return quant
