import pytest
from pydantic import ValidationError

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSearchParams


def test_search_params_require_at_least_one_criterion() -> None:
    with pytest.raises(ValidationError, match="At least one"):
        TargetLayerSearchParams()


def test_search_params_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        TargetLayerSearchParams(layer_class="X", not_a_field=1)  # type: ignore[call-arg]


def test_config_rejects_duplicate_target_names() -> None:
    spec = {"name": "same", "search_params": {"block_idx": 0}}
    with pytest.raises(ValidationError, match="unique"):
        TargetLayerConfig.model_validate({"target_layers": [spec, spec]})


def test_config_round_trips_from_toml_shaped_dict() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "version": 1,
            "target_layers": [
                {"name": "r", "search_params": {"layer_class": "Blk", "count": 40}}
            ],
        }
    )
    assert config.target_layers[0].search_params.count == 40
