from typing import TypeVar
from collections.abc import Callable

import mlx.nn as nn

from preempt.engine.layer_resolution import LayerCandidate

ModuleT = TypeVar("ModuleT", bound=nn.Module)

MlxWrapperFactory = Callable[[ModuleT, LayerCandidate], nn.Module]
