from __future__ import annotations

from typing import Protocol, runtime_checkable

from preempt.core.identity import ExpertKey
from preempt.core.protocols.expert_bank import ExpertPayload

# TODO rename module


@runtime_checkable
class IExpertResidency(Protocol):
    """Backend-specific layer that decodes payload bytes into live device
    tensors and manages their lifecycle in memory.
    """

    def install(self, key: ExpertKey, payload: ExpertPayload) -> None:  # TODO rename
        """Decodes `payload` and holds its weight tensors in memory under `key`."""
        ...

    def evict(self, key: ExpertKey) -> None:  # TODO rename to `drop`?
        """Drops the tensors currently held in memory under `key`."""
        ...

    def is_resident(self, key: ExpertKey) -> bool:  # TODO rename to `is_loaded`?
        """Whether tensors under `key` are currently held in memory"""
        ...

    def resident_bytes(self) -> int:  # TODO rename to `memory_footprint`?
        """Total memory footprint in bytes of all currently loaded experts"""
        ...
