from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence


@runtime_checkable
class TokenCodec(Protocol):
    """Minimal tokenizer surface the engine needs."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@runtime_checkable
class ModelRunner(Protocol):
    """One synchronous forward pass + greedy sample.

    Sync by design: MLX decode is sync; the engine wraps `step` in
    `asyncio.to_thread` so the event loop stays free for I/O.
    """

    def prepare(self) -> None:
        """Resets per-sequence state (e.g. the prompt cache). Call before the
        first `step` of each generation."""
        ...

    def step(self, tokens: Sequence[int]) -> int:
        """Runs forward pass over `tokens`; returns the greedy next token id."""
        ...
