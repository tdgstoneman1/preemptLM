from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from collections.abc import Sequence, Mapping

from preempt.datamodel.identity import ExpertKey

from preempt.expert_bank.blob import SerializedExpert


@runtime_checkable
class IExpertCache(Protocol):
    """Interface for caching experts in memory."""

    _entries: Any

    def __contains__(self, item) -> bool: ...

    def add(self, expert: SerializedExpert) -> None:
        """Adds expert to the cache."""
        ...

    def get(self, key: ExpertKey) -> Mapping[str, Any]:
        """Returns cached weights for `key`"""
        ...

    def evict(self, key: ExpertKey) -> None:
        """Drops the data mapped to `key` from the cache."""
        ...

    def size(self) -> int:
        """Cache's memory footprint in bytes."""
        ...


@runtime_checkable
class IExpertLoader(Protocol):
    """Ensures MoE router-selected experts are loaded into memory and makes them
    available for downstream computation.
    """

    def __del__(self) -> None: ...

    def load(self, keys: ExpertKey | Sequence[ExpertKey]) -> None: ...

    def close(self) -> None: ...


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
    """Runs model text generation loop."""

    model: Any
    kv_cache: Any

    def step(self, tokens: Sequence[int]) -> int:
        """Runs forward pass over `tokens` and returns the greedy-
        decoded next token id.
        """
        ...
