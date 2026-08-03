from __future__ import annotations

from typing import Protocol, runtime_checkable

from enum import IntEnum

import attrs
from attrs import field, validators

from preempt.core.identity import ExpertKey


class ReadPriority(IntEnum):
    """Lower value = more urgent. Ordering lives in the scheduler's queue;
    stores just read what they are told."""

    DEMAND = 0
    PREFETCH = 1


@attrs.define(kw_only=True, frozen=True)
class ExpertPayload:
    """Opaque expert weight bytes plus the encoding tag needed to decode them."""

    key: ExpertKey = field()
    data: bytes = field()
    encoding: str = field(validator=validators.min_len(1))


@runtime_checkable
class ExpertStore(Protocol):
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload: ...
