from __future__ import annotations

from typing import Any, Mapping, Final
from types import MappingProxyType
from collections.abc import Callable

import attrs

import pyarrow as pa

from datetime import datetime, UTC

__all__ = [
    "ARROW_ARGS_KEY",
    "ArrowFieldArgs",
    "arrow_metadata",
    "arrow_args_for",
    "arrow_schema_field",
    "serialize_utc",
]

ARROW_ARGS_KEY: Final[str] = "arrow_args"


@attrs.define(frozen=True)
class ArrowFieldArgs:
    arrow_type: pa.DataType
    serializer: Callable[[Any], Any] | None
    nullable: bool


def arrow_metadata(
    arrow_type: pa.DataType,
    serializer: Callable[[Any], Any] | None = None,
    nullable: bool = False,
) -> Mapping[str, ArrowFieldArgs]:
    arrow_args = ArrowFieldArgs(
        arrow_type,
        serializer,
        nullable,
    )
    return MappingProxyType({ARROW_ARGS_KEY: arrow_args})


def arrow_args_for(
    attribute: attrs.Attribute, cls: type, arrow_args_key: str = ARROW_ARGS_KEY
) -> ArrowFieldArgs:
    try:
        return attribute.metadata[arrow_args_key]

    except KeyError as e:
        raise RuntimeError(
            f"Field `{cls.__name__}.{attribute.name}` is missing "
            f"{arrow_args_key!r} in its metadata."
        ) from e


def arrow_schema_field(attribute: attrs.Attribute, cls: type) -> pa.Field:
    arrow_args = arrow_args_for(attribute, cls=cls)
    return pa.field(
        attribute.name,
        arrow_args.arrow_type,
        nullable=arrow_args.nullable,
    )


def serialize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("`timestamp` must be timezone-aware")

    return value.astimezone(UTC)
