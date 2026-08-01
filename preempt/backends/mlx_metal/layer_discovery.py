from __future__ import annotations

from collections.abc import Iterator

import mlx.nn as nn

from preempt.config.target_layers import TargetLayerConfig

from preempt.engine.layer_resolution import LayerCandidate, resolve_target_layers


def transformer_block_idx_from_path(module_path: str) -> int | None:
    """Returns transformer block index from a module path with dot
    notation, such as `language_model.model.layers.22.mlp`."""

    parts = module_path.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])

    return None


def iter_layer_candidates(model: nn.Module) -> Iterator[LayerCandidate]:
    for path, module in model.named_modules():
        if not path:
            continue

        yield LayerCandidate(
            layer_path=path,
            layer_class=type(module).__name__,
            layer_idx=transformer_block_idx_from_path(path),
        )


def resolve_mlx_target_layers(
    model: nn.Module,
    config: TargetLayerConfig,
) -> dict[str, tuple[LayerCandidate, ...]]:

    return resolve_target_layers(
        candidates=iter_layer_candidates(model),
        config=config,
    )
