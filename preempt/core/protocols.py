from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from collections.abc import Sequence, Mapping, Hashable

from preempt.datamodel.identity import ExpertKey
from preempt.datamodel.experts import SerializedExpert


@runtime_checkable
class IExpertCache(Protocol):
    """Interface for caching experts in memory."""

    def __contains__(self, item) -> bool: ...

    def add(self, expert: SerializedExpert) -> None:
        """Adds expert to the cache."""
        ...

    def evict(self, key: ExpertKey) -> None:
        """Drops the data mapped to `key` from the cache."""
        ...

    def size(self) -> int:
        """Total size of the cache in bytes."""
        ...

    def get(self, key: Hashable) -> Mapping[str, Any]:
        """Returns cache weight tensors for `key`"""
        ...


@runtime_checkable
class IExpertLoader(Protocol):
    """Ensures MoE router-selected experts are loaded into memory and makes them
    available for downstream computation.
    """

    def load(self, keys: Sequence[ExpertKey]) -> None: ...


@runtime_checkable
class ITokenizer(Protocol):
    """Minimal tokenization interface for encoding text to token ids
    and vice versa.
    """

    eos_token_ids: set[int] | None

    @property
    def think_start_id(self) -> int | None: ...

    @property
    def think_end_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@runtime_checkable
class IModelRunner(Protocol):
    """Runs synchronous forward pass and returns the greedy-decoded
    next token.
    """

    def step(self, tokens: Sequence[int]) -> int:
        """Runs forward pass over `tokens` and returns the greedy-
        decoded next token id.
        """
        ...
