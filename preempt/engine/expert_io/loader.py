from __future__ import annotations

from typing import Generator, Optional
from collections.abc import Sequence

from concurrent.futures import as_completed, Future
import asyncio
import time

import attrs

from preempt.core.enums import ReadPriority

from preempt.datamodel.expert_bank.banks import BaseExpertBank
from preempt.datamodel.expert_bank.blob import SerializedExpert
from preempt.datamodel.requests import LoadRequest, CacheRequest

from ..metrics import GenerationMetrics
from .cache_manager import ExpertCacheManager
from .cache import BaseExpertCache

# TODO validate num_worker_threads based on memory budget
# num_worker_threads = budget / expert_size -> need to account for edge
# case where all worker threads' experts simultaneously waiting to be consumed,
# meaning none of them can safely be evicted from cache


@attrs.define(slots=True)
class DiskBackedExpertLoader:
    """Interfaces with expert bank, manages concurrent expert I/O, and schedules
    read and prefetch requests based on priority.
    """

    _expert_bank: BaseExpertBank
    _cache: BaseExpertCache
    _cache_manager: ExpertCacheManager

    _event_loop: asyncio.AbstractEventLoop
    _workers: list[asyncio.Task]
    _task_queue: asyncio.PriorityQueue
    _inflight: dict[int, Future]

    _cache_update_queue: asyncio.Queue
    _cache_update_worker: asyncio.Task

    _metrics: GenerationMetrics | None

    def __init__(
        self,
        *,
        expert_bank: BaseExpertBank,
        cache: BaseExpertCache,
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

        # Queue ensures sequential cache updates and prevents
        # race conditions without thread locking overhead
        self._cache_update_queue = asyncio.Queue()
        self._cache_update_worker = asyncio.create_task(self._cache_loop())

        self._metrics = metrics

    def __del__(self) -> None:
        self.close()

    async def _cache_loop(self) -> None:
        while True:
            req: CacheRequest = await self._cache_update_queue.get()
            expert = req.expert

            try:
                for target_key in self._cache_manager.admit(
                    req.expert.idx, len(req.expert.data), req.priority
                ):
                    self._cache.evict(target_key)
                self._cache.add(expert)

                req.completion_handle.set_result(None)
                self._cache_update_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _worker_loop(self) -> None:
        while True:
            req: LoadRequest = await self._task_queue.get()
            expert_idx = req.expert_idx
            try:
                if expert_idx in self._cache:
                    req.completion_handle.set_result(None)
                    self._task_queue.task_done()
                    continue

                # Check if request already in flight
                if expert_idx in self._inflight:
                    first_worker_future = self._inflight[expert_idx]

                    def inflight_callback(future: Future, request_=req) -> None:
                        try:
                            if exc := future.exception():
                                request_.completion_handle.set_exception(exc)
                            else:
                                request_.completion_handle.set_result(None)

                        except Exception as e:
                            request_.completion_handle.set_exception(e)

                    first_worker_future.add_done_callback(inflight_callback)
                    self._task_queue.task_done()
                    continue

                # Make current request visible in other threads
                self._inflight[expert_idx] = req.completion_handle

                try:
                    expert = await self._expert_bank.read(expert_idx, req.priority)
                    cache_request = CacheRequest(
                        priority=req.priority,
                        expert=expert,
                        completion_handle=req.completion_handle,
                    )
                    await self._cache_update_queue.put(cache_request)

                finally:
                    del self._inflight[expert_idx]

                self._task_queue.task_done()

            except asyncio.CancelledError:
                break

            except Exception as e:
                if req:
                    if not req.completion_handle.done():
                        req.completion_handle.set_exception(e)

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
        expert_idxs: int | Sequence[int],
    ) -> Generator[int, None, None]:
        start_t = time.perf_counter()
        idxs = [expert_idxs] if isinstance(expert_idxs, int) else expert_idxs

        cache_hits = 0
        cache_misses = 0
        futures: dict[Future, int] = {}  # Collects new and in-flight requests

        for idx in set(idxs):
            count = idxs.count(idx)
            # Check if expert already cached
            if self._cache_manager.touch(idx):
                cache_hits += count
                yield idx
                continue

            # Check if request already in progress
            if idx in self._inflight:
                cache_misses += count
                existing = self._inflight[idx]
                futures[existing] = idx
                continue

            cache_misses += count
            future = Future()
            request = LoadRequest(
                priority=ReadPriority.DEMAND,
                expert_idx=idx,
                completion_handle=future,
            )
            futures[future] = idx

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

    def load_sync(
        self,
        expert_idxs: Sequence[int],
    ) -> tuple[SerializedExpert, ...]:
        """Synchronously loads multiple experts directly from the bank."""

        async def _load_all() -> tuple[SerializedExpert, ...]:
            coros = [
                self._expert_bank.read(idx, ReadPriority.DEMAND) for idx in expert_idxs
            ]
            return tuple(await asyncio.gather(*coros))

        future = asyncio.run_coroutine_threadsafe(_load_all(), self._event_loop)
        return future.result()

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
        expert_idxs: int | Sequence[int],
        await_free_slot: bool = True,
    ) -> list[Future[None]]:
        idxs = [expert_idxs] if isinstance(expert_idxs, int) else expert_idxs
        futures: list[Future] = []

        for idx in set(idxs):
            if self._cache_manager.touch(idx) or idx in self._inflight:
                continue

            future = Future()
            request = LoadRequest(
                priority=ReadPriority.PREFETCH,
                expert_idx=idx,
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
