"""Instrumentation for MLX MoE layers. MLX doesn't have PyTorch-style forward hooks,
so instrumentation requires in-place replacement of target layers with instrumented
wrapper modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import mlx.nn as nn
from mlx.utils import tree_unflatten

from preempt.engine.layer_resolution import (
    TargetLayers,
    LayerCandidate,
    resolve_target_layers,
)

from ..types import ModuleWrapperFactory

# TODO move to dedicated constants module
_INNER_ATTR = "inner"
_SWITCH_MLP_ATTR = "switch_mlp"  # TODO make sure 'switch_mlp' isn't Qwen specific


def mlx_instrument_model(
    model: nn.Module,
    candidates: Iterable[LayerCandidate],
    wrapper_factory: ModuleWrapperFactory,
) -> None:
    """Replaces target layers of an MLX model in place with instrumented
    wrapper modules.

    Parameters
    ----------
    model : nn.Module
        MLX model to instrument
    candidates : Iterable[LayerCandidate]
        Target layers to replace, specifying module paths and block indices
    wrapper_factory : ModuleWrapperFactory
        Factory function that takes the original submodule and its candidate
        metadata, and returns an instrumented wrapper module to replace it
        with.

    Raises
    ------
    RuntimeError
        If any candidate layer is not successfully replaced with a wrapper in
        the model's module tree.
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
            f"Failed to apply instrumentation to the following {len(not_instrumented)} "
            f"layer(s):\n{not_instrumented!r}"
        )


def mlx_strip_instrumented_expert_weights(
    model: nn.Module,
    candidates: Iterable[LayerCandidate],
) -> None:
    """Strips expert weights from instrumented layers prior to evaluation, and
    replaces each instrumented layer's `switch_mlp` module with an empty module.

    When a model is loaded with `lazy=True`, this removes expert weights from the
    model's module tree before evaluation. A subsequent `mx.eval(model.parameters())`
    then materializes only the dense backbone (attention, embeddings, routers, and
    shared experts) in memory, and routed expert weights are loaded as needed from
    disk during generation.

    :Note: Must be called after `mlx_instrument_model(...)` and before evaluating model
    weights.

    Parameters
    ----------
    model : nn.Module
        Instrumented MLX model whose weights have not yet been evaluated.
    candidates : Iterable[LayerCandidate]
        Target layers containing instrumented wrappers whose expert weights will
        be stripped.

    Raises
    ------
    RuntimeError
        If a candidate layer is missing the expected inner wrapper and `switch_mlp`
        submodule
    RuntimeError
        If parameter groups remain in `switch_mlp` after stripping
    """
    modules = dict(model.named_modules())

    for candidate in candidates:
        wrapper = modules.get(candidate.layer_path)
        inner = getattr(wrapper, _INNER_ATTR, None)
        switch_mlp = getattr(inner, _SWITCH_MLP_ATTR, None)

        if not isinstance(inner, nn.Module) or not isinstance(switch_mlp, nn.Module):
            raise RuntimeError(
                f"Cannot strip expert weights from {candidate.layer_path!r} because "
                f"it lacks the expected {_INNER_ATTR}.{_SWITCH_MLP_ATTR} module."
            )

        # Replace switch_mlp with param-free module so subsequent mx.eval() calls
        # only materialize dense backbone weights.
        inner.update_modules(tree_unflatten([(_SWITCH_MLP_ATTR, nn.Module())]))

        residual = dict(getattr(inner, _SWITCH_MLP_ATTR).parameters())
        if residual:
            raise RuntimeError(
                f"Layer {candidate.layer_path!r} still contains {len(residual)} parameter "
                f"groups after stripping ({sorted(residual)!r})."
            )


# TODO verify this is architecture agnostic
def transformer_block_idx_from_path(module_path: str) -> int | None:
    """Parses `module_path` (in dot notation) and returns the index of the
    module's parent transformer block or `None` if the path doesn't follow
    the expected pattern.

    Example: `'language_model.model.layers.22.mlp'` → `22`
    """
    parts = module_path.split(".")

    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])

    return None


def iter_layer_candidates(model: nn.Module) -> Iterator[LayerCandidate]:
    """Yields a `LayerCandidate` for every named child module in `model`."""

    for path, module in model.named_modules():
        if not path:
            continue

        yield LayerCandidate(
            layer_path=path,
            layer_class=type(module).__name__,
            block_idx=transformer_block_idx_from_path(path),
        )


def resolve_mlx_target_layers(
    model: nn.Module,
    config: TargetLayers,
) -> dict[str, tuple[LayerCandidate, ...]]:
    """Resolves the target layers in `config` against named child modules in
    `model` keyed by target name.
    """
    return resolve_target_layers(
        candidates=iter_layer_candidates(model),
        config=config,
    )
