from __future__ import annotations

from collections.abc import Sequence

import asyncio

import time

from preempt.core.protocols.residency import IExpertResidency
from preempt.core.protocols.expert_bank import IExpertBank, ReadPriority

from preempt.core.identity import ExpertKey

from .expert_cache import ExpertCache
from .metrics import GenerationMetrics


class DummyExpertLoader:
    """Dummy `IExpertLoader` interface as a placeholder for models that are fully loaded
    in memory and don't require reading experts from disk (`load(...)` is a no-op).
    """

    def load(self, keys: Sequence[ExpertKey]) -> None:
        return None


class DiskBackedExpertLoader:
    """Implements `IExpertLoader` protocol and loads router-selected experts from disk
    on demand.

    :Note: A failed disk read is fatal and propagates. Currently, no fallback strategy
    exists since falling back to in-memory experts or skipping rows would make inference
    inexact.

    Parameters
    ----------
    expert_bank : IExpertBank
        Source of expert payloads. Read when an MoE router selects an expert that has not
        been loaded into memory and must be read from disk.
    residency : IExpertResidency
        Decodes expert payloads into live device tensors and manages their lifecycle in
        memory.
    cache : ExpertCache
        Tracks and manages cached experts within allowed memory budget
    loop : asyncio.AbstractEventLoop
        Event loop on which expert bank reads are scheduled
    metrics : GenerationMetrics | None
        Optional counters to accumulate into during a generation run
    """

    def __init__(
        self,
        *,
        expert_bank: IExpertBank,
        residency: IExpertResidency,
        cache: ExpertCache,
        loop: asyncio.AbstractEventLoop,
        metrics: GenerationMetrics | None = None,
    ) -> None:
        self._expert_bank = expert_bank
        self._residency = residency
        self._cache = cache
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
            if self._cache.touch(key):
                if self._metrics is not None:
                    self._metrics.cache_hits += 1
                continue

            if self._metrics is not None:
                self._metrics.cache_misses += 1

            started = time.perf_counter()
            payload = asyncio.run_coroutine_threadsafe(
                self._expert_bank.read(key, ReadPriority.DEMAND), self._loop
            ).result()

            for victim in self._cache.admit(key, len(payload.data)):
                self._residency.evict(victim)

            self._residency.install(key, payload)

            if self._metrics is not None:
                self._metrics.demand_stall_s += time.perf_counter() - started
