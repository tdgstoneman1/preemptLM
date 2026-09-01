from __future__ import annotations

from typing import Self, Any

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator, computed_field

from preempt.config.target_layers import TargetLayers, TargetLayerSpec


class LlmConfig(BaseModel):
    """Model identifiers for the LLM used in a pipeline"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None)
    backend: str = Field(min_length=1)
    architecture: str = Field(
        min_length=1
    )  # TODO validate against HF snapshot `config.json`, e.g. `Qwen3_5MoeForConditionalGeneration` (Qwen3.6)


class GenerationSettings(BaseModel):
    """Decode settings for a pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=16, ge=1)
    prefill_chunk_size: int = Field(default=512, ge=1)


class TraceSettings(BaseModel):
    """Settings for MoE router tracing"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path = Field()
    batch_size: int = Field(default=1024, ge=1)
    capture_gate_logits: bool = Field(default=False)
    run_id_prefix: str = Field(default="trace", min_length=1)
    overwrite_output: bool = Field(default=False)
    target_layers: tuple[TargetLayerSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_layers_as_layer_config(self) -> Self:
        """Calls `TargetLayers` validators early to avoid deferring
        potential failures.
        """
        self.to_target_layer_config()

        return self

    def to_target_layer_config(self) -> TargetLayers:
        """Projects traced target layers onto `TargetLayers`."""
        return TargetLayers(target_layers=self.target_layers)


class StreamSettings(BaseModel):
    """Settings for streaming weights from expert bank"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expert_bank_path: Path = Field()
    bypass_page_cache: bool = Field(default=True)
    memory_budget_gb: int | float = Field()

    @computed_field
    @property
    def memory_bytes_budget(self) -> int:  # TODO rename
        return int(self.memory_budget_gb * 1024**3)


class PipelineConfig(BaseModel):
    """Top-level config for inference pipeline (read from TOML)"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: LlmConfig
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    trace_settings: TraceSettings | None = Field(default=None)
    stream_settings: StreamSettings | None = Field(default=None)

    version: int = Field(default=1, ge=1)
