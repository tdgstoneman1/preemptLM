from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

from abc import ABC, abstractmethod

import re

import mlx.core as mx

from preempt.storage.manifest import ModelMoESpec

from .quantization import MlxQuantParams


class MoEArchitecture(ABC):
    """*Abstract; do not instantiate*

    Defines architecture-specific parameters for MLX MoE checkpoint conversion
    and runtime execution.

    Subclasses encapsulate model-specific tensor naming patterns, projection
    layouts, and quantization configurations for a specific MoE family.
    """

    @property
    @abstractmethod
    def projection_names(self) -> tuple[str, ...]:
        """Ordered names of expert projection layers stored in the expert blob.

        Returns
        -------
        tuple[str, ...]
            Projection names in storage serialization order, such as
            ("gate_proj", "up_proj", "down_proj") for SwiGLU architectures.
        """
        ...

    # TODO use literal or enum for output type
    @property
    @abstractmethod
    def quantized_tensor_parts(self) -> tuple[str, ...]:
        """Component suffixes comprising a quantized tensor in blob storage order.

        Returns
        -------
        tuple[str, ...]
            Quantized tensor component names in serialization order, such as
            ("weight", "scales", "biases").
        """
        ...

    @property
    @abstractmethod
    def expert_module_pattern(self) -> str:
        """Uncompiled regex pattern matching expert module paths in a checkpoint.

        The pattern must define named capture groups for `"prefix"`, `"layer"`,
        and `"projection"`.

        Returns
        -------
        str
            Raw regex pattern string used to construct compiled tensor and
            module matchers.
        """
        ...

    @property
    @abstractmethod
    def layer_class_name(self) -> str:
        """Class name of the MoE block module type in the loaded model.

        Used for layer-discovery filtering.

        Returns
        -------
        str
            Class name matching `type(module).__name__`, such as
            "Qwen3NextSparseMoeBlock".
        """
        ...

    @abstractmethod
    def expert_tensor_regex(self) -> re.Pattern[str]:
        """Regex for matching expert tensor names

        The pattern must include named capture groups for `"prefix"`, `"layer"`,
        `"projection"`, and `"part"` to deconstruct tensor names from a checkpoint.

        Returns
        -------
        re.Pattern[str]
            A compiled regex pattern
        """
        ...

    @abstractmethod
    def expert_module_regex(self) -> re.Pattern[str]:
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

    @abstractmethod
    def tensor_order(self) -> tuple[str, ...]:
        """Defines the canonical order of tensors within an expert blob.

        This sequence determines how tensor parts from all projections are
        concatenated. For example: `("gate_proj.weight", "gate_proj.scales",
        "gate_proj.biases", "up_proj.weight", ...)`.

        Returns
        -------
        tuple[str, ...]
            The ordered tensor suffixes.
        """
        ...

    @abstractmethod
    def validate_layer_tensors(
        self, layer_tensors: Mapping[str, str]
    ) -> tuple[str, ...]:
        """Validates and returns the tensor order for a specific layer.

        Parameters
        ----------
        layer_tensors : Mapping[str, str]
            Tensor suffixes mapped to full tensor names for one layer

        Returns
        -------
        tuple[str, ...]
            The ordered tensor suffixes present in the layer.

        Raises
        ------
        ValueError
            If required tensors are missing or unexpected tensors are found.
        """
        ...

    @abstractmethod
    def resolve_quantization(
        self, config: Mapping[str, Any], block_idxs: Sequence[int]
    ) -> MlxQuantParams | None:
        """Determines quantization parameters from `config`.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed `config.json` from model checkpoint.
        block_idxs : Sequence[int]
            Indices of the MoE transformer blocks being converted.

        Returns
        -------
        MlxQuantParams | None
            The shared quantization parameters, or `None` if unquantized.

        Raises
        ------
        ValueError
            If experts are not uniformly quantized across the given layers.
        """
        ...

    @abstractmethod
    def scalar_dtype_tag(
        self, stacked: Mapping[str, mx.array], *, quantized: bool
    ) -> str:
        """Returns a tag representing the scalar dtype.

        For quantized models, this inspects `scales`/`biases`. For unquantized
        models, it inspects the `weight` tensors. The tag provides a durable
        record for dtypes like `bfloat16` that are not native to all tools.

        Parameters
        ----------
        stacked : Mapping[str, mx.array]
            A layer's stacked tensors.
        quantized : bool
            True if the model is quantized, False otherwise

        Returns
        -------
        str
            A short dtype tag, e.g., `"bf16"`, `"f16"`, or `"f32"`.
        """
        ...

    # TODO verify that block_idxs refers to transformer block
    @abstractmethod
    def extract_model_moe_spec(
        self,
        config: Mapping[str, Any],  # TODO rename, too vague
        block_idxs: Sequence[int],
        num_routed_experts: int,
    ) -> ModelMoESpec:
        """Extracts MoE-specific configuration from checkpoint.

        Parses the model's `config.json` to create a `ModelMoESpec` object,
        which includes the number of experts, top-k routing, and the indices
        of the transformer blocks that contain MoE layers.

        Parameters
        ----------
        config : Mapping[str, Any]
            The model's parsed `config.json`
        block_idxs : Sequence[int]
            Indices of the transformer blocks containing MoE layers.
        num_routed_experts : int
            The number of experts per MoE layer, detected from tensor shapes.

        Returns
        -------
        ModelMoESpec
            The model's MoE config for its manifest
        """
        ...
