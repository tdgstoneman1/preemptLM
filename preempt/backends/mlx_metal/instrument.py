from __future__ import annotations

from collections.abc import Iterable

import mlx.nn as nn
from mlx.utils import tree_unflatten

from preempt.engine.layer_resolution import LayerCandidate
from .types import MlxWrapperFactory


def mlx_instrument_model(
    model: nn.Module,
    candidates: Iterable[LayerCandidate],
    wrapper_factory: MlxWrapperFactory,
) -> None:
    replacements = []

    for candidate in candidates:
        module = dict(model.named_modules())[candidate.layer_path]
        replacements.append(
            (
                candidate.layer_path,
                wrapper_factory(module, candidate),
            )
        )
    model.update_modules(tree_unflatten(replacements))
