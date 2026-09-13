from concurrent.futures import Future

import attrs
from attrs import field

from preempt.core.enums import ReadPriority

from .expert_bank.blob import SerializedExpert


@attrs.define(kw_only=True, order=False, eq=False, slots=True)
class LoadRequest:
    priority: ReadPriority = field()
    expert_idx: int = field()
    completion_handle: Future = field()

    def __lt__(self, other: "LoadRequest") -> bool:
        if not isinstance(other, LoadRequest):
            return NotImplemented

        return self.priority < other.priority


@attrs.define(kw_only=True, slots=True)
class CacheRequest:
    priority: ReadPriority = field()
    expert: SerializedExpert = field()
    completion_handle: Future = field()
