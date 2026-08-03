from __future__ import annotations

from typing import Self

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSpec


class ModelConfig(BaseModel):
    """Identity of the model a pipeline run executes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    revision: str | None = Field(default=None)


class GenerationConfig(BaseModel):
    """Decode settings for a pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=16, ge=1)


class TracingConfig(BaseModel):
    """Router-trace capture settings, including the layers to instrument."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path = Field()
    batch_size: int = Field(default=1024, ge=1)
    capture_gate_logits: bool = Field(default=False)
    run_id_prefix: str = Field(default="trace", min_length=1)
    overwrite_output: bool = Field(default=False)
    targets: tuple[TargetLayerSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_targets_as_layer_config(self) -> Self:
        """Invokes `TargetLayerConfig` validators early instead of deferring
        potential failures.
        """
        self.to_target_layer_config()

        return self

    def to_target_layer_config(self) -> TargetLayerConfig:
        """
        Projects the tracing targets onto the layer-resolution config.

        Returns
        -------
        TargetLayerConfig
            The same targets in the form `resolve_target_layers` consumes.
        """
        return TargetLayerConfig(target_layers=self.targets)


class StreamingConfig(BaseModel):
    """Expert-weight streaming settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store_path: Path = Field()
    resident_bytes_budget: int = Field(gt=0)


class PipelineConfig(BaseModel):
    """Top-level TOML config describing one inference pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1)
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tracing: TracingConfig | None = Field(default=None)
    streaming: StreamingConfig | None = Field(default=None)
