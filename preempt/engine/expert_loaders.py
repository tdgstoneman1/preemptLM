from __future__ import annotations

from collections.abc import Sequence

import asyncio

import time

from preempt.core.protocols.cache import IExpertCache
from preempt.core.protocols.expert_bank import IExpertBank, ReadPriority

from preempt.core.identity import ExpertKey

from .expert_cache import ExpertCacheManager
from .metrics import GenerationMetrics


class DummyExpertLoader:
    """Dummy `IExpertLoader` interface as a placeholder when models are fully loaded
    in memory and don't expert reads from disk (`load(...)` is a no-op).
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
        """Blocks until every expert in `keys` is loaded into memory.

        Parameters
        ----------
        keys : Sequence[ExpertKey]
            MoE router-selected experts to load. Currently, experts are loaded one at a
            time as needed for computation. Repeated keys are harmless (cache hit after
            the first call).

        Raises
        ------
        KeyError
            If the expert bank holds no blob for a requested key
        IOError
            If the expert bank returns fewer bytes than it recorded
        """
        for key in keys:
            if self._cache_manager.touch(key):
                if self._metrics is not None:
                    self._metrics.cache_hits += 1
                continue

            if self._metrics is not None:
                self._metrics.cache_misses += 1

            started = time.perf_counter()
            payload = asyncio.run_coroutine_threadsafe(
                self._expert_bank.read(key, ReadPriority.DEMAND), self._loop
            ).result()

            for victim in self._cache_manager.admit(key, len(payload.data)):
                self._cache.evict(victim)

            self._cache.install(key, payload)

            if self._metrics is not None:
                self._metrics.demand_stall_s += time.perf_counter() - started
