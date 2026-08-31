from __future__ import annotations

from typing import TypeVar

import tomllib
from pathlib import Path

from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def read_and_validate_toml(fp: str | Path, base_model: type[BaseModelT]) -> BaseModelT:
    """Reads a TOML file, then validates and returns its contents as a `base_model` Pydantic model."""
    with Path(fp).open("rb") as f:
        raw_spec = tomllib.load(f)

    return base_model.model_validate(raw_spec)
