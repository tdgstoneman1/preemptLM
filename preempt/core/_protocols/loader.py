from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from ..identity import ExpertKey

# @runtime_checkable
# class IExpertLoader(Protocol):
#     """Ensures MoE router-selected experts are loaded into memory and makes them
#     available for downstream computation.
#     """

#     def load(self, keys: Sequence[ExpertKey]) -> None: ...
