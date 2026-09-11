from __future__ import annotations

from typing import Optional
from collections.abc import Sequence

from concurrent.futures import ThreadPoolExecutor

import time

from preempt.expert_bank.banks import BaseExpertBank
from preempt.core.protocols import IExpertCache
from preempt.datamodel.identity import ExpertKey
from preempt.core.enums import ReadPriority

from .expert_cache import ExpertCacheManager
from .metrics import GenerationMetrics


class DummyExpertLoader:
    """Dummy `IExpertLoader` interface.

    Use as a placeholder when not dynamically loading expert weights
    from disk (`load(...)` is a no-op).
    """

    def __del__(self) -> None: ...

    def load(self, keys: Sequence[ExpertKey]) -> None: ...

    def close(self) -> None: ...


class DiskBackedExpertLoader:
    """`IExpertLoader` interface that loads serialized expert layers
    from disk on demand.

    :Note: Failed disk reads are fatal.
    """

    _expert_bank: BaseExpertBank
    _cache: IExpertCache
    _cache_manager: ExpertCacheManager
    _executor: ThreadPoolExecutor
    _metrics: GenerationMetrics | None

    def __init__(
        self,
        *,
        expert_bank: BaseExpertBank,
        executor: Optional[ThreadPoolExecutor] = None,
        max_concurrent_bank_reads: int = 32,
        cache: IExpertCache,
        cache_manager: ExpertCacheManager,
        metrics: GenerationMetrics | None = None,
    ) -> None:
        """
        Parameters
        ----------
        expert_bank : BaseExpertBank
            Source of serialized expert weights read from disk
        executor : Optional[ThreadPoolExecutor]
            Executor managing concurrent disk reads from `expert_bank`. If not
            given, one is initialized with `max_workers` set to
            `max_concurrent_bank_reads`, by default None
        max_concurrent_bank_reads: int
            Sets the maximum number of concurrent threads that can execute reads
            from `expert_bank`. Overridden when `executor` is provided, by
            default 32
        cache : IExpertCache
            Decodes serialized expert weights into live device tensors and
            manages their lifecycle in memory
        cache_manager : ExpertCacheManager
            Tracks and manages cached experts within the configured allowed
            memory budget.
        metrics : GenerationMetrics | None
            Optional counters to accumulate into during generation, by default
            None
        """
        self._expert_bank = expert_bank
        self._executor = (
            executor
            if executor is not None
            else ThreadPoolExecutor(max_workers=max_concurrent_bank_reads)
        )
        self._cache = cache
        self._cache_manager = cache_manager
        self._metrics = metrics

    def __del__(self) -> None:
        self.close()

    def load(self, keys: ExpertKey | Sequence[ExpertKey]) -> None:
        start_t = time.perf_counter()
        cache_hits = cache_misses = 0
        futures = []
        keys_ = [keys] if isinstance(keys, ExpertKey) else keys

        for key in set(keys_):
            count = keys_.count(key)
            if self._cache_manager.touch(key):
                cache_hits += count
            else:
                futures.append(self._executor.submit(self._expert_bank.read_sync, key))
                cache_misses += count

        self._update_metrics("cache_hits", cache_hits)
        self._update_metrics("cache_misses", cache_misses)

        if not futures:
            return

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            expert = future.result()
            self._update_metrics("prefetched_bytes", len(expert.data))

            for victim in self._cache_manager.admit(expert.key, len(expert.data)):
                self._cache.evict(victim)

            self._cache.add(expert)

        self._update_metrics("demand_stall_s", time.perf_counter() - start_t)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._expert_bank.close()
        if hasattr(self._cache, "_entries"):
            self._cache._entries.clear()

    def _update_metrics(
        self,
        metric_name: str,
        value: int | float,
    ) -> None:
        if self._metrics is None:
            return

        try:
            current = getattr(self._metrics, metric_name)
            setattr(self._metrics, metric_name, current + value)

        except AttributeError:
            raise AttributeError() from None  # TODO add error msg

        except TypeError:
            raise TypeError() from None  # TODO add error msg
