from __future__ import annotations

from typing import Generator, Optional
from collections.abc import Sequence

from concurrent.futures import as_completed, Future
import asyncio

import time

from preempt.expert_bank.banks import BaseExpertBank

from preempt.core.protocols import IExpertCache
from preempt.core.enums import ReadPriority

from preempt.datamodel.identity import ExpertKey
from preempt.datamodel.requests import LoadRequest, CacheRequest

from .expert_cache import ExpertCacheManager
from .metrics import GenerationMetrics


class DummyExpertLoader:
    """Dummy `IExpertLoader` interface.

    Use as a placeholder when not dynamically loading expert weights
    from disk (`load(...)` is a no-op).
    """

    def __del__(self) -> None: ...

    @property
    def cache_manager(self) -> None: ...

    def load(
        self,
        keys: ExpertKey | Sequence[ExpertKey],
    ) -> Generator[ExpertKey, None, None]: ...

    def enqueue_prefetch(
        self,
        keys: ExpertKey | Sequence[ExpertKey],
        await_free_slot: bool = True,
    ) -> list[Future]: ...

    def close(self) -> None: ...


# TODO validate num_worker_threads based on memory budget
# num_worker_threads = budget / expert_size -> need to account for edge
# case where all worker threads' experts simultaneously waiting to be consumed,
# meaning none of them can safely be evicted from cache
class DiskBackedExpertLoader:
    """`IExpertLoader` interface that manages concurrent expert bank I/O and
    schedules read and prefetch requests based on priority.
    """

    _expert_bank: BaseExpertBank
    _cache: IExpertCache
    _cache_manager: ExpertCacheManager

    _event_loop: asyncio.AbstractEventLoop
    _workers: list[asyncio.Task]
    _task_queue: asyncio.PriorityQueue
    _inflight: dict[ExpertKey, Future]

    _cache_update_queue: asyncio.Queue
    _cache_update_worker: asyncio.Task

    _metrics: GenerationMetrics | None

    def __init__(
        self,
        *,
        expert_bank: BaseExpertBank,
        cache: IExpertCache,
        cache_manager: ExpertCacheManager,
        event_loop: asyncio.AbstractEventLoop,
        num_worker_threads: int = 64,
        max_queue_size: int = 0,
        metrics: Optional[GenerationMetrics] = None,
    ) -> None:
        """
        Parameters
        ----------
        expert_bank : BaseExpertBank
            Source of serialized expert weights read from disk.
        cache : IExpertCache
            Decodes serialized expert weights into live device tensors and manages their lifecycle
            in memory
        cache_manager : ExpertCacheManager
            Tracks and manages cached experts within the configured allowed memory budget.
        event_loop : asyncio.AbstractEventLoop
            Event loop on which expert bank reads are scheduled
        num_worker_threads : int
            Sets the number of concurrent background workers available process read requests, by
            default 64
        max_queue_size : int
            Sets the maximum number of tasks allowed in the queue, by default 0 (infinite)
        metrics : GenerationMetrics | None
            Optional counters to accumulate into during generation, by default None
        """
        self._expert_bank = expert_bank
        self._cache = cache
        self._cache_manager = cache_manager

        self._event_loop = event_loop
        self._workers = [
            asyncio.create_task(self._worker_loop()) for _ in range(num_worker_threads)
        ]
        self._task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._inflight = {}

        # Queue ensures sequential updates to cache,
        # prevents race conditions and is faster than thread locking
        self._cache_update_queue = asyncio.Queue()
        self._cache_update_worker = asyncio.create_task(self._cache_loop())

        self._metrics = metrics

    def __del__(self) -> None:
        self.close()

    @property
    def cache_manager(self) -> ExpertCacheManager:
        return self._cache_manager

    async def _cache_loop(self) -> None:
        while True:
            request: CacheRequest = await self._cache_update_queue.get()
            expert = request.expert

            try:
                for target_key in self._cache_manager.admit(
                    expert.key, len(expert.data), request.priority
                ):
                    self._cache.evict(target_key)
                self._cache.add(expert)

                request.completion_handle.set_result(None)
                self._cache_update_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _worker_loop(self) -> None:
        while True:
            request: LoadRequest = await self._task_queue.get()
            key = request.key

            try:
                # Final check if expert already cached
                if key in self._cache:
                    request.completion_handle.set_result(None)
                    self._task_queue.task_done()
                    continue

                # Check if request already in flight
                if key in self._inflight:
                    first_worker_future = self._inflight[key]

                    def inflight_callback(future: Future):
                        try:
                            if exc := future.exception():
                                request.completion_handle.set_exception(exc)
                            else:
                                request.completion_handle.set_result(None)

                        except Exception as e:
                            request.completion_handle.set_exception(e)

                    first_worker_future.add_done_callback(inflight_callback)
                    self._task_queue.task_done()

                    continue

                # Make current request visible in other threads
                self._inflight[key] = request.completion_handle

                try:
                    expert = await self._expert_bank.read(key, request.priority)
                    cache_request = CacheRequest(
                        priority=request.priority,
                        expert=expert,
                        completion_handle=request.completion_handle,
                    )
                    await self._cache_update_queue.put(cache_request)

                finally:
                    del self._inflight[key]

                self._task_queue.task_done()

            except asyncio.CancelledError:
                break

            except Exception as e:
                if request:
                    if not request.completion_handle.done():
                        request.completion_handle.set_exception(e)

                    self._task_queue.task_done()

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

    def load(
        self,
        keys: ExpertKey | Sequence[ExpertKey],
    ) -> Generator[ExpertKey, None, None]:
        start_t = time.perf_counter()
        keys_ = [keys] if isinstance(keys, ExpertKey) else keys

        cache_hits = 0
        cache_misses = 0
        futures: dict[Future, ExpertKey] = {}  # Collects new and in-flight requests

        for key in set(keys_):
            count = keys_.count(key)

            # Check if expert already cached
            if self._cache_manager.touch(key):
                cache_hits += count
                yield key
                continue

            # Check if request already in progress
            if key in self._inflight:
                cache_misses += count
                existing = self._inflight[key]
                futures[existing] = key
                continue

            cache_misses += count
            future = Future()
            request = LoadRequest(
                priority=ReadPriority.DEMAND,
                key=key,
                completion_handle=future,
            )
            futures[future] = key

            asyncio.run_coroutine_threadsafe(
                self._task_queue.put(request),
                self._event_loop,
            ).result()  # Block until request is enqueued

        self._update_metrics("cache_hits", cache_hits)
        self._update_metrics("cache_misses", cache_misses)

        if not futures:
            return

        for future in as_completed(futures.keys()):
            future.result()
            yield futures[future]

        self._update_metrics("demand_stall_s", time.perf_counter() - start_t)

    def _enqueue_blocking(
        self,
        request: LoadRequest,
        future: Future,
        futures: list[Future],
    ) -> list[Future[None]]:
        enqueue = asyncio.run_coroutine_threadsafe(
            self._task_queue.put(request),
            self._event_loop,
        )
        enqueue.result()

        return [*futures, future]

    def _enqueue_nonblocking(
        self,
        request: LoadRequest,
        future: Future,
        futures: list[Future],
    ) -> list[Future[None]]:
        try:
            self._event_loop.call_soon_threadsafe(
                self._task_queue.put_nowait,
                request,
            )
        except asyncio.QueueFull:
            return futures

        return [*futures, future]

    def enqueue_prefetch(
        self,
        keys: ExpertKey | Sequence[ExpertKey],
        await_free_slot: bool = True,
    ) -> list[Future[None]]:
        keys_ = [keys] if isinstance(keys, ExpertKey) else keys
        futures: list[Future] = []

        for key in set(keys_):
            if self._cache_manager.touch(key) or key in self._inflight:
                continue

            future = Future()
            request = LoadRequest(
                priority=ReadPriority.PREFETCH,
                key=key,
                completion_handle=future,
            )
            futures = (
                self._enqueue_blocking(request, future, futures)
                if await_free_slot
                else self._enqueue_nonblocking(request, future, futures)
            )

        return futures

    def close(self) -> None:
        for worker in self._workers:
            worker.cancel()

        if hasattr(self._cache, "_entries"):
            self._cache._entries.clear()

        self._expert_bank.close()
