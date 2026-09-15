from typing import Any, Generic, TypeVar
from collections.abc import Iterator, Sequence, Generator
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


T = TypeVar("T", bound=(Any))


class FixedLenthList(Generic[T]):
    def __init__(self, size, fill_value=None) -> None:
        self._items = [fill_value] * size
        self._size: int = size

    def __getitem__(self, index: int) -> T | None:
        return self._items[index]

    def __setitem__(self, index: int, value: T) -> None:
        self._items[index] = value

    def __delitem__(self, index: int) -> None:
        self._items[index] = None

    def __len__(self) -> int:
        return self._size

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def __reversed__(self) -> Iterator[T | None]:
        return reversed(self._items)

    def __repr__(self) -> str:
        return repr(self._items)


def make_slotted_container_class(
    name: str,
    num_slots: int,
    slot_name_prefix: str | None,
    default_fill_value: Any = None,
) -> type:
    prefix = slot_name_prefix if slot_name_prefix is not None else "_slot"
    slot_names = tuple(f"{prefix}{i}" for i in range(num_slots))

    def __init__(self, fill_value=default_fill_value) -> None:
        for name in slot_names:
            setattr(self, name, fill_value)

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, slice):
            return [
                getattr(self, slot_names[i]) for i in range(*index.indices(num_slots))
            ]
        try:
            slot_name = slot_names[index]
            return getattr(self, slot_name)

        except IndexError:
            raise IndexError(f"{index} index out of range") from None

    def __setitem__(self, index: int, value: Any) -> None:
        try:
            slot_name = slot_names[index]
            setattr(self, slot_name, value)

        except IndexError:
            raise IndexError(f"{index} index out of range") from None

    def __delitem__(self, index: int) -> None:
        try:
            slot_name = slot_names[index]
            setattr(self, slot_name, None)

        except IndexError:
            raise IndexError(f"{index} index out of range") from None

    def __len__(self) -> int:
        return num_slots

    def __contains__(self, item: Any) -> bool:
        for name in slot_names:
            if getattr(self, name) == item:
                return True

        return False

    class_dict = {
        "__slots__": slot_names,
        "__init__": __init__,
        "__getitem__": __getitem__,
        "__setitem__": __setitem__,
        "__delitem__": __delitem__,
        "__len__": __len__,
        "__contains__": __contains__,
    }
    return type(name, (object,), class_dict)
