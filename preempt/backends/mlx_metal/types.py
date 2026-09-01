from typing import TypeVar
from collections.abc import Callable

import mlx.nn as nn

from preempt.engine.layer_resolution import LayerCandidate

from .adapters.moe_arch_adapter import MoEArchAdapter

MlxModuleT = TypeVar("MlxModuleT", bound=nn.Module)

MlxWrapperFactory = Callable[[MlxModuleT, LayerCandidate], nn.Module]

MlxArchAdapterFactory = Callable[[], MoEArchAdapter]
