from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

from abc import ABC, abstractmethod

import re

import mlx.core as mx

from preempt.expert_bank.manifest import ModelMoESpec

from ..quantization import QuantSettings


class BaseMoEArchAdapter(ABC):
    """*Abstract; do not instantiate*

    Defines MoE architecture-specific parameters for expert bank serialization
    with MLX models. Subclasses encapsulate a model backend's parameter name
    patterns/layouts and quantization configurations.
    """

    @property
    @abstractmethod
    def model_architecture(self) -> str: ...

    @property
    @abstractmethod
    def moe_class_name(self) -> str:
        """Name of the `nn.Module` class used for MoE blocks in mlx-lm's
        implemtation of the model, e.g. "Qwen3NextSparseMoeBlock" for Qwen3-Next
        and Qwen3.x models.

        Used for identifying a loaded model's MoE blocks.

        Returns
        -------
        str
            Class name matching `type(module).__name__`
        """
        ...

    @property
    @abstractmethod
    def linear_projection_names(self) -> tuple[str, ...]:
        """Ordered names of the linear projection layers stored in an expert weight
        blob.

        Returns
        -------
        tuple[str, ...]
            Projection names in storage serialization order, such as
            ("gate_proj", "up_proj", "down_proj") for SwiGLU architectures.
        """
        ...

    @property
    @abstractmethod
    def expert_layer_path_regex(self) -> re.Pattern[str]:
        """Regex matching expert module paths (in dot notation).

        Excludes parameter component suffixes (e.g. `.weight`, `.scales`).
        """
        ...

    @property
    @abstractmethod
    def expert_weight_path_regex(self) -> re.Pattern[str]:
        """Regex matching absolute paths to expert weights (in dot notation).

        Captures the constituent component type (`weight`, `scales`, or `biases`)
        under the `<part>` group.
        """
        ...

    @property
    @abstractmethod
    def weight_order(self) -> tuple[str, ...]:
        """Defines how weight tensors should be ordered during serialization."""
        ...

    @abstractmethod
    def validate_weight_paths(self, weight_paths: Mapping[str, str]) -> tuple[str, ...]:
        """Validates an MoE block's expert weight paths against those expected
        for the architecture and returns them in serialization order.

        Parameters
        ----------
        weight_paths : Mapping[str, str]
            Mapping of weight paths relative to parent layers to absolute paths
            within the model (all in dot notation)

        Returns
        -------
        tuple[str, ...]
            Validated relative weight paths (relative to parent layer)

        Raises
        ------
        ValueError
            If expected paths are missing from `weight_paths`.
        ValueError
            If `weight_paths` contains unexpected items.
        """
        ...

    @abstractmethod
    def resolve_quantization(
        self, config: Mapping[str, Any], block_idxs: Sequence[int]
    ) -> QuantSettings | None:
        """Determines the effective quantization parameters used by routed experts.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed contents of Hugging Face-style checkpoint's `config.json`
        block_idxs : Sequence[int]
            Indices of the resolved expert layers' parent MoE blocks

        Returns
        -------
        QuantSettings | None
            The quantization parameters shared by expert layers in the checkpoint,
            or `None` if unquantized.

        Raises
        ------
        ValueError
            If quantization parameters are not uniform layers.
        """
        ...

    @abstractmethod
    def dtype_tag_for(self, weights: Mapping[str, mx.array], quantized: bool) -> str:
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
            A short dtype tag, e.g., 'bfloat16'
        """
        ...

    @abstractmethod
    def get_model_moe_spec(
        self,
        config: Mapping[str, Any],
        block_idxs: Sequence[int],
    ) -> ModelMoESpec:
        """Returns a model's MoE specification.

        Parses `config` to create a `ModelMoESpec` object, which includes the
        number of experts, top-k routing, and the indices of the transformer
        blocks in the model containing MoE blocks.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed contents of a Hugging Face-style checkpoint's `config.json`
        block_idxs : Sequence[int]
            Indices of the model's transformer blocks containing MoE layers.

        Returns
        -------
        ModelMoESpec
            The model's MoE specification
        """
        ...
