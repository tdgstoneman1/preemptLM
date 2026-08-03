import pytest

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSearchParams
from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
    layer_is_match,
    match_target_layers,
    resolve_target_layers,
)


def make_candidates() -> tuple[LayerCandidate, ...]:
    return tuple(
        LayerCandidate(
            layer_path=f"model.layers.{i}.mlp",
            layer_class="SparseMoeBlock" if i % 2 == 0 else "DenseMlp",
            layer_idx=i,
        )
        for i in range(6)
    )


def test_layer_is_match_by_class() -> None:
    params = TargetLayerSearchParams(layer_class="SparseMoeBlock")
    candidates = make_candidates()
    assert layer_is_match(candidates[0], params)
    assert not layer_is_match(candidates[1], params)


def test_layer_is_match_conjunction_of_criteria() -> None:
    params = TargetLayerSearchParams(
        layer_class="SparseMoeBlock", layer_path_glob="model.layers.*.mlp", layer_idx=2
    )
    candidates = make_candidates()
    assert layer_is_match(candidates[2], params)
    assert not layer_is_match(candidates[0], params)  # class+glob match, idx doesn't


def test_match_target_layers_count_mismatch_raises() -> None:
    params = TargetLayerSearchParams(layer_class="SparseMoeBlock", count=2)
    with pytest.raises(ValueError, match="unexpected number"):
        match_target_layers(make_candidates(), params)


def test_match_target_layers_no_match_raises() -> None:
    params = TargetLayerSearchParams(layer_class="DoesNotExist")
    with pytest.raises(ValueError, match="Could not find"):
        match_target_layers(make_candidates(), params)


def test_resolve_target_layers_groups_by_spec_name() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "target_layers": [
                {"name": "moe", "search_params": {"layer_class": "SparseMoeBlock", "count": 3}},
                {"name": "dense", "search_params": {"layer_class": "DenseMlp", "count": 3}},
            ]
        }
    )
    resolved = resolve_target_layers(make_candidates(), config)
    assert set(resolved) == {"moe", "dense"}
    assert [c.layer_idx for c in resolved["moe"]] == [0, 2, 4]


def test_ensure_no_target_layer_overlap_raises_on_shared_layer() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "target_layers": [
                {"name": "a", "search_params": {"layer_class": "SparseMoeBlock"}},
                {"name": "b", "search_params": {"layer_idx": 0}},
            ]
        }
    )
    resolved = resolve_target_layers(make_candidates(), config)
    with pytest.raises(ValueError, match="matches both"):
        ensure_no_target_layer_overlap(resolved)
