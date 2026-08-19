from typing import TypeVar
from collections.abc import Callable

import mlx.nn as nn

from preempt.engine.layer_resolution import LayerCandidate

from .architecture import MoEArchitecture

MlxModuleT = TypeVar("MlxModuleT", bound=nn.Module)

MlxWrapperFactory = Callable[[MlxModuleT, LayerCandidate], nn.Module]

MlxMoEFactory = Callable[[], MoEArchitecture]
