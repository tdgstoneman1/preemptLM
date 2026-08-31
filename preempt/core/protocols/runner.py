from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence


@runtime_checkable
class ITokenCodec(Protocol):  # TODO rename
    """Minimal tokenization interface for encoding text to token ids
    and vice versa.
    """

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@runtime_checkable
class IModelRunner(Protocol):
    """Runs synchronous forward pass and returns the greedy-decoded
    next token.
    """

    def prepare(self) -> None:  # TODO remove
        """Resets per-sequence state (prompt cache). Call before the
        first (prefill) generation step.
        """
        ...

    def step(self, tokens: Sequence[int]) -> int:
        """Runs forward pass over `tokens` and returns the greedy-
        decoded next token id.
        """
        ...
