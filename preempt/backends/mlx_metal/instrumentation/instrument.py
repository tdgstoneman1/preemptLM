"""Instrumentation for MLX MoE blocks.

MLX doesn't have PyTorch-style forward hooks, so the workaround to access hidden
states is to wrap layers with outer instrumentation `nn.Module`s.
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
from ..utils import transformer_block_idx_from_path

_SWITCH_MLP_ATTR = "switch_mlp"  # TODO make dynamic and architecture-agnostic
_INNER_ATTR = "inner"


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
        The MLX model to instrument
    candidates : Iterable[LayerCandidate]
        Target layers to replace
    wrapper_factory : ModuleWrapperFactory
        Factory function that takes layers from `LayerCandidates` and returns them
        wrapped in an outer instrumentation module.

    Raises
    ------
    RuntimeError
        If instrumentation failed
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
    moe_blocks: Iterable[LayerCandidate],
) -> None:
    """Strips weights from instrumented expert layers and replaces `switch_mlp`s
    with an empty module prior to MLX evaluation.

    When an instrumented model is loaded lazily, calling this removes expert weights
    from the model's parameter tree. Subsequent `mx.eval(model.parameters())` calls only
    load the dense backbone weights in memory (attention, embeddings, routers, shared
    experts, etc.), and routed expert weights are then loaded as needed during inference
    from disk.

    :Note: Must be called *after* `mlx_instrument_model()` and *before* `mx.eval()`

    Parameters
    ----------
    model : nn.Module
        Instrumented MLX model whose weights have not yet been evaluated.
    moe_blocks : Iterable[LayerCandidate]
        Target MoE blocks whose expert layer weights are to be stripped from the model

    Raises
    ------
    RuntimeError
        If a target MoE block is missing the expected instrumentation and `switch_mlp`
        submodule.
    RuntimeError
        If experts weights still remain in the model's tree after stripping.
    """
    modules = dict(model.named_modules())

    for block in moe_blocks:
        wrapper = modules.get(block.layer_path)
        inner = getattr(wrapper, _INNER_ATTR, None)
        switch_mlp = getattr(inner, _SWITCH_MLP_ATTR, None)

        if not isinstance(inner, nn.Module) or not isinstance(switch_mlp, nn.Module):
            raise RuntimeError(
                f"Cannot strip expert weights from {block.layer_path!r} because "
                f"it lacks the expected '{_INNER_ATTR}.{_SWITCH_MLP_ATTR}' module "
                "for routed experts."
            )
        # Replace switch_mlp with a dummy placeholder module
        inner.update_modules(tree_unflatten([(_SWITCH_MLP_ATTR, nn.Module())]))
        if remaining := dict(getattr(inner, _SWITCH_MLP_ATTR).parameters()):
            raise RuntimeError(
                f"Layer {block.layer_path!r} still contains {len(remaining)} routed "
                f"expert parameter groups after stripping: {sorted(remaining)!r}."
            )


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
