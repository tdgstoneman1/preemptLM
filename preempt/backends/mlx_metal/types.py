from typing import TypeVar, NamedTuple
from collections.abc import Callable, Mapping

import attrs
from attrs import field

import mlx.nn as nn
import mlx.core as mx

from preempt.core.protocols import ITokenizer
from preempt.engine.layer_resolution import LayerCandidate

from .quantization import QuantSettings

MlxModuleT = TypeVar("MlxModuleT", bound=nn.Module)

ModuleWrapperFactory = Callable[[MlxModuleT, LayerCandidate], nn.Module]


@attrs.define(kw_only=True, frozen=True, slots=True)
class MlxLoadedModel:
    """An `mlx_lm` model-tokenizer pair.

    Attributes
    ----------
    model : nn.Module
        The `mlx_lm` model
    tokenizer : ITokenizer
        Tokenizer for encoding text and decoding generated token
        ids. Must implement `ITokenizer` protocol.
    """

    model: nn.Module = field()
    tokenizer: ITokenizer = field()


# TODO remove weights data classes, adds unnecessary boilerplate
@attrs.define(kw_only=True, frozen=True, slots=True)
class WeightsTensor:
    weight: mx.array


@attrs.define(kw_only=True, frozen=True, slots=True)
class QuantizedWeightsTensor(QuantSettings):
    weight: mx.array
    scales: mx.array
    biases: mx.array | None


ExpertLayerWeights = Mapping[str, WeightsTensor | QuantizedWeightsTensor]

# TODO this is reduntant, quant settings already present in QuantizedWeightsTensor
ExpertLayerQuants = Mapping[str, QuantSettings]
