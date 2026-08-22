import pytest

from pathlib import Path

from pydantic import ValidationError

from preempt.config.pipeline import PipelineConfig

from preempt.utils.io_utils import read_and_validate_toml

MINIMAL = {
    "llm": {
        "model_id": "some/model",
        "backend": "mlx_metal",
        "architecture": "qwen3-next",
    }
}
TRACED = {
    **MINIMAL,
    "trace_settings": {
        "output_path": "out/events.parquet",
        "target_layers": [
            {"name": "r", "search_params": {"layer_class": "Blk", "count": 40}}
        ],
    },
}


def test_minimal_config_defaults() -> None:
    config = PipelineConfig.model_validate(MINIMAL)
    assert config.trace_settings is None
    assert config.stream_settings is None
    assert config.generation_settings.max_tokens >= 1


def test_tracing_targets_convert_to_target_layer_config() -> None:
    config = PipelineConfig.model_validate(TRACED)
    assert config.trace_settings is not None
    tl_config = config.trace_settings.to_target_layer_config()
    assert tl_config.target_layers[0].search_params.count == 40


def test_duplicate_tracing_target_names_rejected_at_parse() -> None:
    bad = {
        **MINIMAL,
        "trace_settings": {
            "output_path": "x.parquet",
            "target_layers": [
                {"name": "same", "search_params": {"block_idx": 0}},
                {"name": "same", "search_params": {"block_idx": 1}},
            ],
        },
    }
    with pytest.raises(ValidationError, match="unique"):
        PipelineConfig.model_validate(bad)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({**MINIMAL, "surprise": 1})


def test_streaming_requires_positive_budget() -> None:
    bad = {
        **MINIMAL,
        "stream_settings": {"expert_bank_path": "s", "memory_bytes_budget": 0},
    }
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(bad)


def test_round_trips_from_toml_file(tmp_path: Path) -> None:
    toml = tmp_path / "p.toml"
    toml.write_text(
        '[llm]\nmodel_id = "m"\nbackend = "mlx_metal"\narchitecture = "a"\n'
        "[generation_settings]\nmax_tokens = 4\n"
        '[trace_settings]\noutput_path = "o.parquet"\n'
        '[[trace_settings.target_layers]]\nname = "r"\n'
        '[trace_settings.target_layers.search_params]\nlayer_class = "Blk"\n'
    )
    config = read_and_validate_toml(toml, PipelineConfig)
    assert config.generation_settings.max_tokens == 4
    assert config.trace_settings is not None
