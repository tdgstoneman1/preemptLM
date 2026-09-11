import pytest

from collections.abc import Callable

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.config.pipeline import (
    LlmConfig,
    GenerationSettings,
    StreamSettings,
    TraceSettings,
    PipelineConfig,
)
from preempt.config.target_layers import (
    TargetLayerSearchParams,
    TargetLayerSpec,
)


@pytest.fixture(scope="session")
def root_path() -> Path:
    return Path(__file__).parents[1]


@pytest.fixture(scope="session")
def converted_models_dir(root_path: Path) -> Path:
    return root_path / "converted-models"


@pytest.fixture(scope="session")
def test_dir(root_path: Path) -> Path:
    return root_path / "tests"


@pytest.fixture(scope="session")
def config_dir(test_dir: Path) -> Path:
    return test_dir / "configs"


@pytest.fixture(scope="session")
def quantized_expert_bank_path(test_dir: Path) -> Path:
    return test_dir / "expert-bank" / "quantized"


@pytest.fixture(scope="session")
def unquantized_expert_bank_path(test_dir: Path) -> Path:
    return test_dir / "expert-bank" / "unquantized"


@pytest.fixture(scope="session")
def quantized_stream_settings(quantized_expert_bank_path: Path) -> StreamSettings:
    return StreamSettings(
        expert_bank_path=quantized_expert_bank_path,
        memory_budget_gb=4,
        bypass_page_cache=True,
    )


@pytest.fixture(scope="session")
def unquantized_stream_settings(unquantized_expert_bank_path: Path) -> StreamSettings:
    return StreamSettings(
        expert_bank_path=unquantized_expert_bank_path,
        memory_budget_gb=4,
        bypass_page_cache=True,
    )


@pytest.fixture(scope="session")
def quantized_llm_config() -> LlmConfig:
    return LlmConfig(
        model_id="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        backend="mlx_metal",
        architecture="qwen3.6",
    )


# TODO either copy weights to tests/converted-models or use ROOT/converted-models
@pytest.fixture(scope="session")
def unquantized_llm_config() -> LlmConfig:
    return LlmConfig(
        model_id="converted-models/qwen3.6-35b",
        backend="mlx_metal",
        architecture="qwen3.6",
    )


@pytest.fixture(scope="session")
def trace_path(test_dir: Path) -> Path:
    return test_dir / "traces" / "test-trace.parquet"


@pytest.fixture(scope="session")
def trace_settings(trace_path: Path) -> TraceSettings:
    search_params = TargetLayerSearchParams(
        layer_class="Qwen3NextSparseMoeBlock", count=40
    )
    target_layers = TargetLayerSpec(
        name="Qwen3.6-35B-router", search_params=search_params
    )
    return TraceSettings(
        output_path=trace_path,
        target_layers=tuple([target_layers]),
        overwrite_output=True,
    )


@pytest.fixture(scope="session")
def generation_settings() -> GenerationSettings:
    return GenerationSettings(
        max_tokens=128,
        prefill_chunk_size=256,
    )


@pytest.fixture(scope="session")
def prompt() -> str:
    return (
        "Explain the differences between Newtonian physics and "
        "Einstein's relativistic physics."
    )


@pytest.fixture(scope="session")
def make_pipe_config(
    generation_settings: GenerationSettings,
    quantized_llm_config: LlmConfig,
    unquantized_llm_config: LlmConfig,
    quantized_stream_settings: StreamSettings,
    unquantized_stream_settings: StreamSettings,
    trace_settings: TraceSettings,
) -> Callable[[bool], PipelineConfig]:

    def _pipeline_config(quantized: bool) -> PipelineConfig:
        llm = quantized_llm_config if quantized else unquantized_llm_config
        stream_settings = (
            quantized_stream_settings if quantized else unquantized_stream_settings
        )
        return PipelineConfig(
            llm=llm,
            generation_settings=generation_settings,
            stream_settings=stream_settings,
            trace_settings=trace_settings,
        )

    return _pipeline_config
