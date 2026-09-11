from __future__ import annotations

from typing import Any, TypeVar

import tomllib
from pathlib import Path

from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def parse_toml(fp: str | Path) -> dict[str, Any]:
    with Path(fp).open("rb") as f:
        contents = tomllib.load(f)

    return contents


def read_and_validate_toml(fp: str | Path, base_model: type[BaseModelT]) -> BaseModelT:
    """Reads a TOML file, then validates and returns its contents as a `base_model` Pydantic model."""

    return base_model.model_validate(parse_toml(fp))


def resolve_dotted_relative_path(path: Path, relative_to: Path) -> Path:
    """E.g. `path = '../../scripts'` and `relative_to = 'preempt/utils/io_utils.py'`"""
    for i, part in enumerate(path.parts):
        if part != "..":
            break

    return relative_to.parents[i] / Path(*path.parts[i:])
