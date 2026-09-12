from collections.abc import Sequence

import attrs
from attrs import field

from concurrent.futures import Future

from preempt.core.enums import ReadPriority

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
