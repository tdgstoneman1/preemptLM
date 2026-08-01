from typing import Any
from collections.abc import Sequence, Generator

import attrs


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
