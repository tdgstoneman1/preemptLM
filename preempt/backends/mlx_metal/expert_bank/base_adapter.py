from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

from abc import ABC, abstractmethod

import re

import mlx.core as mx
import mlx.nn as nn

from preempt.expert_bank.manifest import ModelMoESpec

from ..quantization import QuantSettings


class BaseMoEArchAdapter(ABC):
    """*Abstract; do not instantiate*

    Defines MoE architecture-specific parameters for expert bank serialization with
    MLX models. Subclasses encapsulate a model family's parameter name patterns/layouts
    and quantization configurations.
    """

    @property
    @abstractmethod
    def model_architecture(self) -> str: ...

    @property
    @abstractmethod
    def moe_class_name(self) -> str:
        """Name of the `nn.Module` class used for MoE blocks in mlx-lm's implemtation
        of the model, e.g. "Qwen3NextSparseMoeBlock" for Qwen3-Next and Qwen3.x models.

        Used for filtering a loaded model's MoE blocks.

        Returns
        -------
        str
            Class name matching `type(module).__name__`
        """
        ...

    @property
    @abstractmethod
    def linear_projection_names(self) -> tuple[str, ...]:
        """Ordered names of expert projection layers stored in the expert blob.

        Returns
        -------
        tuple[str, ...]
            Projection names in storage serialization order, such as
            ("gate_proj", "up_proj", "down_proj") for SwiGLU architectures.
        """
        ...

    @property
    @abstractmethod
    def expert_weight_path_regex(self) -> re.Pattern[str]:
        """Regex for matching expert tensor names

        The pattern must include named capture groups for `"prefix"`, `"layer"`,
        `"projection"`, and `"part"` to deconstruct tensor names from a checkpoint.

        Returns
        -------
        re.Pattern[str]
            A compiled regex pattern
        """
        ...

    @property
    @abstractmethod
    def expert_layer_path_regex(self) -> re.Pattern[str]:
        """Regex for matching expert module paths.

        The pattern must include named capture groups for `"prefix"`, `"layer"`,
        and `"projection"` to deconstruct module paths from a checkpoint. This
        pattern should not match the final tensor part (e.g., `".weight"`).

        Returns
        -------
        re.Pattern[str]
            A compiled regex pattern.
        """
        ...

    @property
    @abstractmethod
    def weight_order(self) -> tuple[str, ...]:
        """Defines how weight tensors should be ordered during serialization.

        Determines how tensor parts for an MoE block's linear projections are
        concatenated, for example: `("gate_proj.weight", "gate_proj.scales",
        "gate_proj.biases", "up_proj.weight", ...)`

        Returns
        -------
        tuple[str, ...]
            The ordered tensor suffixes.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def resolve_quantization(
        self, config: Mapping[str, Any], block_idxs: Sequence[int]
    ) -> QuantSettings | None:
        """Resolves quantization parameters from the model's `config.json`.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed `config.json` from a Hugging Face-style model checkpoint.
        block_idxs : Sequence[int]
            Indices of the transformer blocks containing the target MoE blocks
            being serialized

        Returns
        -------
        QuantSettings | None
            The resolved quantization parameters, or `None` if unquantized

        Raises
        ------
        ValueError
            If experts are not uniformly quantized across the specified layers
        """
        ...

    @abstractmethod
    def dtype_tag_for(self, weights: Mapping[str, mx.array], *, quantized: bool) -> str:
        """Returns a dtype tag for a layer's weights.

        Inspects `'weights'` for unquantized models and `'scales'` and `'biases'`
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
        ...

    @abstractmethod
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
        ...
