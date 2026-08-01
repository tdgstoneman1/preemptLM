from __future__ import annotations

from typing import TypeVar

import tomllib
from pathlib import Path

from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def read_and_validate_toml(fp: str | Path, base_model: type[BaseModelT]) -> BaseModelT:
    """Reads TOML file and returns its contents as a validated Pydantic `BaseModel`"""
    with Path(fp).open("rb") as file:
        raw_spec = tomllib.load(file)

    return base_model.model_validate(raw_spec)
