from __future__ import annotations

from typing import Self, Optional

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
    computed_field,
)

from preempt.utils.io_utils import read_and_validate_toml, resolve_dotted_relative_path

from .target_layers import TargetLayers, TargetLayerSpec


# TODO validate arch against HF snapshot `config.json`,
# e.g. `Qwen3_5MoeForConditionalGeneration` (Qwen3.6)
class LlmConfig(BaseModel):
    """Model identifiers for an LLM."""

    model_config = ConfigDict(extra="allow")

    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None)
    backend: str = Field(min_length=1)
    architecture: str = Field(min_length=1)


class GenerationSettings(BaseModel):
    """Decode settings for text generation."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=16, ge=1)
    prefill_chunk_size: int = Field(default=512, ge=1)


# TODO refactor
class TraceSettings(BaseModel):
    """MoE router tracing settings"""

    model_config = ConfigDict(extra="forbid")

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
    """Streaming settings for disk reads from expert bank"""

    model_config = ConfigDict(extra="forbid")

    expert_bank_path: Path = Field()
    bypass_page_cache: bool = Field(default=True)
    memory_budget_gb: int | float = Field()

    @computed_field
    @property
    def memory_bytes_budget(self) -> int:  # TODO rename
        return int(self.memory_budget_gb * 1024**3)


# TODO add from_toml() classmethod to resolve paths relative to config
class PipelineConfig(BaseModel):
    """Top-level generation pipeline config (read from TOML)"""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1)

    llm: LlmConfig = Field()
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    trace_settings: Optional[TraceSettings] = Field(default=None)
    stream_settings: Optional[StreamSettings] = Field(default=None)

    _fp: Optional[Path] = PrivateAttr(default=None)

    @classmethod
    def from_toml(cls, fp: str | Path, resolve_relative_paths: bool = True) -> Self:
        model = read_and_validate_toml(fp, cls)
        model._fp = Path(fp)
        if resolve_relative_paths:
            model.resolve_paths_relative_to_config()

        return model

    def resolve_paths_relative_to_config(self) -> None:
        assert self._fp is not None

        if self.stream_settings:
            self.stream_settings.expert_bank_path = resolve_dotted_relative_path(
                self.stream_settings.expert_bank_path, self._fp
            )
        if self.trace_settings:
            self.trace_settings.output_path = resolve_dotted_relative_path(
                self.trace_settings.output_path, self._fp
            )
