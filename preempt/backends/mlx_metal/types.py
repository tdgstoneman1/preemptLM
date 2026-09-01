from typing import TypeVar
from collections.abc import Callable, Mapping

import attrs
from attrs import field

import mlx.nn as nn

from preempt.engine.layer_resolution import LayerCandidate

from preempt.core.protocols import ITokenizer

from .adapters.moe_arch_adapter import MoEArchAdapter

MlxModuleT = TypeVar("MlxModuleT", bound=nn.Module)

MlxWrapperFactory = Callable[[MlxModuleT, LayerCandidate], nn.Module]

MlxArchAdapterFactory = Callable[[], MoEArchAdapter]


# ExpertProjections = Mapping[str, ]
@attrs.define(kw_only=True, frozen=True, eq=False)
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
