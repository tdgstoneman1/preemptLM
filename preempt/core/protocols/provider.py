from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from ..identity import ExpertKey

# TODO rename module


@runtime_checkable
class IExpertProvider(Protocol):  # TODO rename
    """Ensures router-selected experts are loaded in memory and ready for
    downstream computation.
    """

    def acquire(self, keys: Sequence[ExpertKey]) -> None: ...
