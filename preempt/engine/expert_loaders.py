from __future__ import annotations

from collections.abc import Sequence

import asyncio
from concurrent.futures import Future

import time

from preempt.core.protocols import IExpertCache, IExpertBank, ReadPriority
from preempt.core.identity import ExpertKey

from .expert_cache import ExpertCacheManager
from .metrics import GenerationMetrics


class DummyExpertLoader:
    """Dummy `IExpertLoader` interface as a placeholder when models are fully loaded
    in memory and don't read read from disk (`load(...)` is a no-op).
    """

    def load(self, keys: Sequence[ExpertKey]) -> None:
        return None


class DiskBackedExpertLoader:
    """`IExpertLoader` interface that loads serialized expert layers from disk
    on demand.

    :Note: Failed disk reads are fatal. No fallback strategy currently in place
    since falling back to in-memory experts or skipping rows would make inference
    inexact.
    """

    _expert_bank: IExpertBank
    _cache: IExpertCache
    _cache_manager: ExpertCacheManager
    _loop: asyncio.AbstractEventLoop
    _metrics: GenerationMetrics | None

    def __init__(
        self,
        *,
        expert_bank: IExpertBank,
        cache: IExpertCache,
        cache_manager: ExpertCacheManager,
        loop: asyncio.AbstractEventLoop,  # TODO rename to 'event_loop'
        metrics: GenerationMetrics | None = None,
    ) -> None:
        """
        Parameters
        ----------
        expert_bank : IExpertBank
            Source of expert payloads. Read when an MoE router selects an expert that has not
            been loaded into memory and must be read from disk.
        residency : IExpertCache
            Decodes expert payloads into live device tensors and manages their lifecycle in
            memory.
        cache : ExpertCacheManager
            Tracks and manages cached experts within allowed memory budget
        loop : asyncio.AbstractEventLoop
            Event loop on which expert bank reads are scheduled
        metrics : GenerationMetrics | None
            Optional counters to accumulate into during a generation run
        """
        self._expert_bank = expert_bank
        self._cache = cache
        self._cache_manager = cache_manager
        self._loop = loop
        self._metrics = metrics

    def load(self, keys: Sequence[ExpertKey]) -> None:
        """Blocks thread until all requested experts are loaded. Disk reads are concurrent."""
        cache_misses = []
        cache_hits = 0
        for key in set(keys):
            if self._cache_manager.touch(key):
                cache_hits += keys.count(key) if isinstance(keys, list) else 1
            else:
                cache_misses.append(key)

        if self._metrics is not None:
            self._metrics.cache_hits += cache_hits
            self._metrics.cache_misses += len(cache_misses)

        if not cache_misses:
            return

        start_time = time.perf_counter()
        futures: list[Future] = [
            asyncio.run_coroutine_threadsafe(
                self._expert_bank.read(key, ReadPriority.DEMAND), self._loop
            )
            for key in cache_misses
        ]
        loaded_payloads = [f.result() for f in futures]
        elapsed = time.perf_counter() - start_time

        for payload in loaded_payloads:
            for victim in self._cache_manager.admit(payload.key, len(payload.data)):
                self._cache.evict(victim)

            self._cache.install(payload.key, payload)

        if self._metrics is not None:
            self._metrics.demand_stall_s += elapsed
            self._metrics.prefetched_bytes += len(payload.data)
