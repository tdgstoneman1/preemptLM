from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from .identity import ExpertKey
from .enums import ReadPriority

from preempt.core.protocols.expert_bank import ExpertPayload


# TODO add `tensors` method to be consistent with `MlxExpertCache`?
@runtime_checkable
class IExpertCache(Protocol):
    """Interface for caching experts in memory."""

    def install(
        self, key: ExpertKey, payload: ExpertPayload
    ) -> None:  # TODO to cache_expert
        """Decodes `payload` into weights and caches them under `key`."""
        ...

    def evict(self, key: ExpertKey) -> None:  # TODO rename to `drop`
        """Drops the tensors mapped to `key` from the cache."""
        ...

    def is_resident(self, key: ExpertKey) -> bool:  # TODO rename to `is_cached`
        """Checks if `key` is in the cache."""
        ...

    def size(self) -> int:
        """Total size of the cache in bytes."""
        ...


@runtime_checkable
class IExpertBank(Protocol):
    """Read-only source of expert blobs keyed by `ExpertKey`"""

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload: ...


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
