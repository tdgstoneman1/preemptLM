from __future__ import annotations

from collections.abc import Iterator

import mlx.nn as nn

from preempt.config.target_layers import TargetLayerConfig

from preempt.engine.layer_resolution import LayerCandidate, resolve_target_layers


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
    config: TargetLayerConfig,
) -> dict[str, tuple[LayerCandidate, ...]]:
    """Resolves the target layers in `config` against named child modules in
    `model` keyed by target name.
    """
    return resolve_target_layers(
        candidates=iter_layer_candidates(model),
        config=config,
    )
