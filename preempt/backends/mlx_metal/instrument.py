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
    """Replaces each candidate `nn.Module` in `model` in place with an instrumented
    outer wrapper.

    Parameters
    ----------
    model : nn.Module
        The loaded MLX model to instrument.
    candidates : Iterable[LayerCandidate]
        Resolved layers whose modules are swapped for wrappers.
    wrapper_factory : MlxWrapperFactory
        Builds the wrapper that replaces a given module.

    Raises
    ------
    RuntimeError
        If any candidate path does not hold its wrapper afterwards. MLX has no
        hooks, so instrumentation is a structural rewrite via `update_modules`.
        a path it silently declines to replace would leave the layer untraced
        and the trace short by a whole layer's worth of rows, with nothing else
        to signal it. # TODO rewrite this slop, makes no sense
    """

    replacements = []
    wrappers: dict[str, nn.Module] = {}

    for layer in candidates:
        module = dict(model.named_modules())[layer.layer_path]
        wrapper = wrapper_factory(module, layer)

        wrappers[layer.layer_path] = wrapper
        replacements.append((layer.layer_path, wrapper))

    model.update_modules(tree_unflatten(replacements))
    updated = dict(model.named_modules())

    not_instrumented = [
        path for path, wrapper in wrappers.items() if updated.get(path) is not wrapper
    ]
    if not_instrumented:
        raise RuntimeError(
            f"Failed to apply instrumentation to these {len(not_instrumented)} "
            f"layer(s): {not_instrumented!r}"
        )
