from __future__ import annotations

from typing import Optional
from collections.abc import Iterable

import attrs
from attrs import field

from fnmatch import fnmatchcase

from preempt.core.config.pipeline import PipelineConfig
from preempt.core.config.target_layers import (
    TargetLayers,
    TargetLayerSpec,
    TargetLayerSearchParams,
)
from preempt.datamodel.expert_bank.banks import BaseExpertBank


@attrs.define(frozen=True, kw_only=True)
class LayerCandidate:
    """Identifiers used to match candidate layers against `TargetLayerSearchParams`.

    Attributes
    ----------
    layer_path : str
        Layer/module path in dot notation, e.g. `model.layers.3.mlp`
    layer_class : str
        Python class name of the layer/module, e.g. "Qwen3NextSparseMoeBlock"
    block_idx : int | None
        Index of the layer's parent transformer block parsed from `layer_path`,
        or `None` if the path does not contain one
    """

    layer_path: str = field()
    layer_class: str = field()
    block_idx: int | None = field(default=None)


def layer_is_match(
    candidate: LayerCandidate,
    search_params: TargetLayerSearchParams,
) -> bool:
    """Whether `candidate` satisfies each of the conditions in `search_params`.
    `layer_class` and `block_idx` are matched exactly, while `layer_path_glob`
    matches as a case-sensitive glob. Search parameter values of `None` do not
    restrict matching.
    """
    return (
        (
            search_params.layer_class is None
            or candidate.layer_class == search_params.layer_class
        )
        and (
            search_params.layer_path_glob is None
            or fnmatchcase(candidate.layer_path, search_params.layer_path_glob)
        )
        and (
            search_params.block_idx is None
            or candidate.block_idx == search_params.block_idx
        )
    )


def match_target_layers(
    candidates: Iterable[LayerCandidate],
    search_params: TargetLayerSearchParams,
) -> tuple[LayerCandidate, ...]:
    """Returns the subset of candidate layers that match `search_params`.

    Parameters
    ----------
    candidates : Iterable[LayerCandidate]
        Candidate layers to be evaulated against `search_params`
    search_params : TargetLayerSearchParams
        The search parameters used to evaluate `candidates`

    Returns
    -------
    tuple[LayerCandidate, ...]

    Raises
    ------
    ValueError
        If no layers match `search_params`
    ValueError
        If `search_params.count` is set and differs from the number of
        layers that match `search_params`
    """
    matches = tuple(
        filter(lambda layer: layer_is_match(layer, search_params), candidates)
    )
    if not matches:
        raise ValueError(
            "No layers found matching the provided search parameters: "
            f"`{search_params!r}`"
        )
    if search_params.count is not None and len(matches) != search_params.count:
        raise ValueError(
            f"Expected {search_params.count} layers to match the provided "
            f"search parameters, but found {len(matches)}."
        )

    return matches


def resolve_target_layers(
    candidates: Iterable[LayerCandidate],
    config: TargetLayers,
) -> dict[str, tuple[LayerCandidate, ...]]:
    candidates = tuple(candidates)
    return {
        spec.name: match_target_layers(candidates, spec.search_params)
        for spec in config.specs
    }


def ensure_no_target_layer_overlap(
    resolved: dict[str, tuple[LayerCandidate, ...]],
) -> None:
    """Ensures that layers match no more than one target.

    Raises
    ------
    ValueError
        If the same layer appears under more than one target name in
        `resolved`
    """
    owners: dict[LayerCandidate, str] = {}

    for target_name, layers in resolved.items():
        for layer in layers:
            previous_name = owners.setdefault(layer, target_name)
            if previous_name != target_name:
                raise ValueError(
                    f"Layer {layer.layer_path!r} matches both "
                    f"'{previous_name!r}' and '{target_name!r}'."
                )


def _target_layers_for_architecture(
    target_layer_class: str, target_layer_count: int | None
) -> TargetLayers:
    search_params = TargetLayerSearchParams(
        layer_class=target_layer_class, count=target_layer_count
    )
    return TargetLayers(
        specs=(
            TargetLayerSpec(
                name="instrumented-moe-block",  # TODO figure this out
                search_params=search_params,
            ),
        ),
    )


def target_layers_for_model(
    config: PipelineConfig,
    expert_bank: Optional[BaseExpertBank],
    target_layer_class: str,
) -> TargetLayers | None:
    if config.trace_settings is not None:
        return config.trace_settings.traced_layers

    elif expert_bank is not None:
        return _target_layers_for_architecture(
            target_layer_class,
            len(expert_bank.manifest.model_moe_spec.moe_block_idxs),
        )
