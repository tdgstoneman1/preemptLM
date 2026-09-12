from __future__ import annotations

from typing import Self
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
    computed_field,
)

from preempt.utils.dataclass_utils import (
    read_and_validate_toml,
    resolve_dotted_relative_path,
)
from .target_layers import TargetLayers, TargetLayerSpec


# TODO validate arch against `config.json` in HF ckpt,
# e.g. `Qwen3_5MoeForConditionalGeneration` (Qwen3.6)
class LlmConfig(BaseModel):
    """Model identifiers for an LLM."""

    model_config = ConfigDict(extra="allow")

    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None)
    backend: str = Field(min_length=1)
    architecture: str = Field(min_length=1)


# TODO add field for matmul mode (sequential or fused)
# TODO add sampling options
# TODO add kv cache size
class GenerationSettings(BaseModel):
    """Decode settings for text generation."""

    max_tokens: int = Field(default=16, ge=1)
    prefill_chunk_size: int = Field(default=512, ge=1)


class TraceSettings(BaseModel):
    """MoE router trace settings"""

    output_path: Path = Field()
    batch_size: int = Field(default=1024, ge=1)
    capture_gate_logits: bool = Field(default=False)
    run_id_prefix: str = Field(default="trace", min_length=1)
    overwrite_output: bool = Field(default=False)
    target_layers: tuple[TargetLayerSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_target_layers_as_layer_config(self) -> Self:
        self.to_target_layer_config()
        return self

    def to_target_layer_config(self) -> TargetLayers:  # TODO rename/refactor
        """Projects traced target layers onto `TargetLayers`."""
        return TargetLayers(target_layers=self.target_layers)


# TODO add pread or mmap
# TODO add max concurrency field for expert bank I/O
class StreamSettings(BaseModel):
    """I/O settings for expert bank reads."""

    expert_bank_path: Path = Field()
    bypass_page_cache: bool = Field(default=True)
    memory_budget_gb: int | float = Field()

    @computed_field
    @property
    def memory_bytes_budget(self) -> int:  # TODO rename
        return int(self.memory_budget_gb * 1024**3)


class PipelineConfig(BaseModel):
    """Top-level generation pipeline config (read from TOML)"""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1)

    llm: LlmConfig = Field()
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    trace_settings: TraceSettings | None = Field(default=None)
    stream_settings: StreamSettings | None = Field(default=None)

    resolve_relative_paths: bool = Field(default=True)

    _fp: Path | None = PrivateAttr(default=None)

    # TODO optionally resolve model relative path
    @classmethod
    def from_toml(
        cls,
        fp: str | Path,
        resolve_relative_paths: bool | None = None,
    ) -> Self:
        """Initializes and returns a new `PipelineConfig` from a TOML config

        Parameters
        ----------
        fp : str | Path
            Path to config file
        resolve_relative_paths : bool
            Whether to resolve expert bank and trace output paths relative to `fp`. Set to True for
            paths like `'../../traces/trace.parquet'` where the absolute path cannot otherwise be
            resolved, by default True

        Returns
        -------
        Self
            A new `PipelineConfig` instance
        """
        model = read_and_validate_toml(fp, cls)
        model._fp = Path(fp)

        if model.resolve_relative_paths or resolve_relative_paths:
            model.resolve_paths_relative_to_config()

        return model

    def resolve_paths_relative_to_config(self) -> None:
        """Resolves expert bank and trace output paths relative to the config's file path. For
        example, with config path `'configs/mlx/my-config.toml'`, `'../../expert-bank/my-bank'`
        would resolve to `'expert-bank/my-bank'`.

        Raises
        ------
        AttributeError
            If the config wasn't originally created from a file.
        """
        if self._fp is None:
            raise AttributeError(
                "Cannot resolve paths relative to config file because `PipelineConfig` was not "
                "created from a file."
            )

        if self.stream_settings:
            self.stream_settings.expert_bank_path = resolve_dotted_relative_path(
                self.stream_settings.expert_bank_path, self._fp
            )
        if self.trace_settings:
            self.trace_settings.output_path = resolve_dotted_relative_path(
                self.trace_settings.output_path, self._fp
            )
