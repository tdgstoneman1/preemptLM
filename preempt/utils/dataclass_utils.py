from typing import Any, TypeVar
from collections.abc import Sequence, Generator

from pathlib import Path

import attrs

from pydantic import BaseModel

import tomllib


def recurse_attrs_fields(fields: Sequence[Any]) -> Generator[Any, Any, None]:
    for f in fields:
        if attrs.has(f.type):
            yield from recurse_attrs_fields(attrs.fields(f.type))
        else:
            yield f


def recurse_attrs_instance_fields(
    obj: object,
) -> Generator[tuple[attrs.Attribute, Any], Any, None]:
    fields = attrs.fields(type(obj))
    for f in fields:
        if attrs.has(f.type):
            yield from recurse_attrs_instance_fields(getattr(obj, f.name))
        else:
            yield f, getattr(obj, f.name)


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
