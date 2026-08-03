from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from preempt.core.identity import ExpertKey


@runtime_checkable
class ExpertProvider(Protocol):
    """Sync in-forward hook: block until the given experts are resident.

    Called from the runner thread by instrumented MoE blocks right after
    top-k selection — the correctness path. Implementations must be
    thread-safe with respect to the engine's event loop.
    """

    def acquire(self, keys: Sequence[ExpertKey]) -> None: ...
