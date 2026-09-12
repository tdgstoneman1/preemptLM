import attrs
from attrs import field

from concurrent.futures import Future

from preempt.core.enums import ReadPriority

from preempt.expert_bank.blob import SerializedExpert

from .identity import ExpertKey


@attrs.define(kw_only=True, order=False, eq=False)
class LoadRequest:
    priority: ReadPriority = field()
    key: ExpertKey = field()
    completion_handle: Future = field()

    def __lt__(self, other: "LoadRequest") -> bool:
        if not isinstance(other, LoadRequest):
            return NotImplemented

        return self.priority < other.priority


@attrs.define(kw_only=True)
class CacheRequest:
    priority: ReadPriority = field()
    expert: SerializedExpert = field()
    completion_handle: Future = field()
