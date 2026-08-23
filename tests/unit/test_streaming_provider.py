from __future__ import annotations

import pytest

from collections.abc import Sequence

import asyncio

from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.enums import CachePolicy

from preempt.core.protocols.loader import IExpertLoader
from preempt.core.protocols.cache import IExpertCache
from preempt.core.protocols.expert_bank import ExpertPayload, IExpertBank, ReadPriority

from preempt.engine.expert_cache import ExpertCacheManager
from preempt.engine.metrics import GenerationMetrics
from preempt.engine.expert_loaders import DiskBackedExpertLoader

MODEL_HASH = "fp-test"
PAYLOAD_BYTES = 64
SPECS = (
    TensorSpec(
        name="w", dtype="uint8", shape=(PAYLOAD_BYTES,), num_bytes=PAYLOAD_BYTES
    ),
)


def key(expert_idx: int, block_idx: int = 0) -> ExpertKey:
    return ExpertKey(
        model_fingerprint=MODEL_HASH, block_idx=block_idx, expert_idx=expert_idx
    )


class FakeExpertBank:
    """`IExpertBank` over an in-memory blob table, recording every read."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.reads: list[tuple[ExpertKey, ReadPriority]] = []
        self.error = error

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        self.reads.append((key, priority))
        if self.error is not None:
            raise self.error
        return ExpertPayload(
            key=key,
            data=bytes(PAYLOAD_BYTES),
            encoding="mlx-affine-q4-g64-bf16",
            tensor_specs=SPECS,
        )


class FakeResidency:
    """`IExpertCache` mirroring `MlxExpertCache`'s strict `evict`.

    Evicting a key that is not resident raises, exactly as the MLX residency
    does — a silent no-op there would let the cache's accounting and the
    residency's accounting drift apart unnoticed.
    """

    def __init__(self) -> None:
        self.payloads: dict[ExpertKey, ExpertPayload] = {}
        self.installs: list[ExpertKey] = []
        self.evictions: list[ExpertKey] = []

    def install(self, key: ExpertKey, payload: ExpertPayload) -> None:
        self.payloads[key] = payload
        self.installs.append(key)

    def evict(self, key: ExpertKey) -> None:
        del self.payloads[key]  # KeyError for a non-resident key, by design
        self.evictions.append(key)

    def is_resident(self, key: ExpertKey) -> bool:
        return key in self.payloads

    def resident_bytes(self) -> int:
        return sum(len(payload.data) for payload in self.payloads.values())


def build(
    *,
    budget_experts: int = 4,
    expert_bank: FakeExpertBank | None = None,
    policy: CachePolicy = CachePolicy.LFRU,
) -> tuple[FakeExpertBank, FakeResidency, ExpertCacheManager, GenerationMetrics]:
    return (
        expert_bank if expert_bank is not None else FakeExpertBank(),
        FakeResidency(),
        ExpertCacheManager(
            budget_bytes=budget_experts * PAYLOAD_BYTES, policy=policy, sample_size=8
        ),
        GenerationMetrics(),
    )


async def load(
    *,
    expert_bank: IExpertBank,
    residency: IExpertCache,
    cache: ExpertCacheManager,
    metrics: GenerationMetrics | None,
    batches: Sequence[Sequence[ExpertKey]],
) -> DiskBackedExpertLoader:
    """Drive `load` off the loop thread, as the runner thread really does."""
    provider = DiskBackedExpertLoader(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        loop=asyncio.get_running_loop(),
        metrics=metrics,
    )
    for batch in batches:
        await asyncio.to_thread(provider.load, batch)
    return provider


def test_provider_satisfies_the_expert_provider_protocol() -> None:
    expert_bank, residency, cache, metrics = build()
    loop = asyncio.new_event_loop()
    try:
        provider = DiskBackedExpertLoader(
            expert_bank=expert_bank,
            residency=residency,
            cache=cache,
            loop=loop,
            metrics=metrics,
        )
        assert isinstance(provider, IExpertLoader)
    finally:
        loop.close()


async def test_a_miss_issues_exactly_one_demand_read_and_installs_it() -> None:
    expert_bank, residency, cache, metrics = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0),)],
    )

    assert expert_bank.reads == [(key(0), ReadPriority.DEMAND)]
    assert residency.is_resident(key(0))
    assert key(0) in cache
    assert metrics.cache_misses == 1
    assert metrics.cache_hits == 0


async def test_a_hit_issues_no_read() -> None:
    expert_bank, residency, cache, metrics = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0),), (key(0),)],
    )

    assert len(expert_bank.reads) == 1
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 1
    assert residency.installs == [key(0)]


async def test_a_repeated_key_within_one_batch_is_read_once() -> None:
    expert_bank, residency, cache, metrics = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0), key(1), key(0))],
    )

    assert [read_key for read_key, _ in expert_bank.reads] == [key(0), key(1)]
    assert metrics.cache_hits == 1


async def test_exceeding_the_budget_evicts_and_the_victim_leaves_residency() -> None:
    expert_bank, residency, cache, metrics = build(budget_experts=2)

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0), key(1)), (key(2),)],
    )

    # Both residents have frequency 1, so LFRU's tiebreak takes the older.
    assert residency.evictions == [key(0)]
    assert not residency.is_resident(key(0))
    assert residency.is_resident(key(1))
    assert residency.is_resident(key(2))


async def test_residency_bytes_track_the_cache_and_never_exceed_the_budget() -> None:
    budget_experts = 3
    expert_bank, residency, cache, metrics = build(budget_experts=budget_experts)

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(idx % 7),) for idx in range(30)],
    )

    assert residency.resident_bytes() == cache.resident_bytes
    assert residency.resident_bytes() <= budget_experts * PAYLOAD_BYTES
    assert cache.evictions > 0


async def test_lru_and_lfru_evict_different_victims_on_the_same_pattern() -> None:
    batches = [
        (key(0),),
        (key(0),),
        (key(0),),
        (key(1),),
        (key(2),),
        (key(2),),
        (key(1),),
        (key(3),),
    ]

    victims: dict[CachePolicy, list[ExpertKey]] = {}
    for policy in (CachePolicy.LFRU, CachePolicy.LRU):
        expert_bank, residency, cache, metrics = build(budget_experts=3, policy=policy)
        await load(
            expert_bank=expert_bank,
            residency=residency,
            cache=cache,
            metrics=metrics,
            batches=batches,
        )
        victims[policy] = residency.evictions

    assert victims[CachePolicy.LFRU] == [key(2)]
    assert victims[CachePolicy.LRU] == [key(0)]


async def test_stall_time_accumulates_only_for_demand_reads() -> None:
    expert_bank, residency, cache, metrics = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0), key(1))],
    )
    after_misses = metrics.demand_stall_s
    assert after_misses > 0.0

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[(key(0), key(1))],
    )
    assert metrics.demand_stall_s == after_misses


async def test_a_missing_blob_is_fatal_and_propagates() -> None:
    expert_bank, residency, cache, metrics = build(
        expert_bank=FakeExpertBank(error=KeyError("nope"))
    )

    with pytest.raises(KeyError):
        await load(
            expert_bank=expert_bank,
            residency=residency,
            cache=cache,
            metrics=metrics,
            batches=[(key(0),)],
        )

    # Nothing was silently substituted or skipped.
    assert residency.payloads == {}
    assert key(0) not in cache


async def test_a_short_read_is_fatal_and_propagates() -> None:
    expert_bank, residency, cache, metrics = build(
        expert_bank=FakeExpertBank(error=IOError("short"))
    )

    with pytest.raises(IOError):
        await load(
            expert_bank=expert_bank,
            residency=residency,
            cache=cache,
            metrics=metrics,
            batches=[(key(0),)],
        )

    assert residency.payloads == {}


async def test_a_failure_mid_batch_does_not_continue_with_the_experts_it_had() -> None:
    expert_bank = FakeExpertBank()
    residency = FakeResidency()
    cache = ExpertCacheManager(budget_bytes=8 * PAYLOAD_BYTES, sample_size=8)
    metrics = GenerationMetrics()

    provider = DiskBackedExpertLoader(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        loop=asyncio.get_running_loop(),
        metrics=metrics,
    )
    await asyncio.to_thread(provider.load, (key(0),))
    expert_bank.error = KeyError("gone")

    with pytest.raises(KeyError):
        await asyncio.to_thread(provider.load, (key(0), key(1), key(2)))

    # The batch aborted at the first failed read rather than proceeding.
    assert [read_key for read_key, _ in expert_bank.reads] == [key(0), key(1)]
    assert set(residency.payloads) == {key(0)}


async def test_metrics_are_optional() -> None:
    expert_bank, residency, cache, _ = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=None,
        batches=[(key(0),)],
    )

    assert residency.is_resident(key(0))


async def test_an_empty_batch_reads_nothing() -> None:
    expert_bank, residency, cache, metrics = build()

    await load(
        expert_bank=expert_bank,
        residency=residency,
        cache=cache,
        metrics=metrics,
        batches=[()],
    )

    assert expert_bank.reads == []
