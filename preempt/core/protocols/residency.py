from __future__ import annotations

from typing import Protocol, runtime_checkable

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertPayload


@runtime_checkable
class ExpertResidency(Protocol):
    """Backend-owned: where payload bytes become live device tensors."""

    def install(self, key: ExpertKey, payload: ExpertPayload) -> None: ...

    def evict(self, key: ExpertKey) -> None: ...

    def is_resident(self, key: ExpertKey) -> bool: ...

    def resident_bytes(self) -> int: ...
