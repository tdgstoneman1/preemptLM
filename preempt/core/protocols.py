from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from collections.abc import Sequence


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
