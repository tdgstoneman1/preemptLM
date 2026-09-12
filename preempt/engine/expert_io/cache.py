from typing import Any
from collections.abc import Mapping

from abc import ABC, abstractmethod

from preempt.datamodel.identity import ExpertKey

from preempt.expert_bank.blob import SerializedExpert

from .cache_manager import ExpertCacheManager


class BaseExpertCache(ABC):
    manager: ExpertCacheManager
    _entries: Any  # TODO rename to _experts

    @abstractmethod
    def __contains__(self, item) -> bool: ...

    @abstractmethod
    def add(self, expert: SerializedExpert) -> None:
        """Caches a serialized expert."""
        ...

    @abstractmethod
    def get(self, key: ExpertKey) -> Mapping[str, Any]:
        """Returns cached expert weights mapped to `key`"""
        ...

    @abstractmethod
    def evict(self, key: ExpertKey) -> None:
        """Drops the expert weights mapped to `key` from the cache."""
        ...

    @abstractmethod
    def size(self) -> int:
        """The cache's memory footprint in bytes."""
        ...
