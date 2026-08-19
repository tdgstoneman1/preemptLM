from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetLayerSearchParams(BaseModel):
    """Search parameters identifying a set of target layers.

    A layer matches if it satisfies all of the provided search parameters (unset
    parameters of `None` are ignored and do not constrain matching). At least
    one of `layer_class`, `layer_path_glob`, or `block_idx` must be given.

    If set, `count` asserts exactly how many layers must match.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_class: str | None = Field(default=None, min_length=1)
    layer_path_glob: str | None = Field(default=None, min_length=1)
    block_idx: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_search_params(self) -> Self:
        if all(
            v is None for v in [self.layer_class, self.layer_path_glob, self.block_idx]
        ):
            raise ValueError(
                "At least one of `layer_class`, `layer_path_glob`, or "
                "`block_idx` must be provided."
            )
        return self


class TargetLayerSpec(BaseModel):
    """Name and search parameters for a single target layer"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    search_params: TargetLayerSearchParams


class TargetLayerConfig(BaseModel):
    """A set of unique named target layers"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1)
    target_layers: tuple[TargetLayerSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_for_name_duplicates(self) -> Self:
        names = [target.name for target in self.target_layers]

        if len(names) != len(set(names)):
            raise ValueError(
                "Target layer names must be unique."
            )  # TODO more descriptive message

        return self
