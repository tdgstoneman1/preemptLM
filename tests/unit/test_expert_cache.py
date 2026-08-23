from __future__ import annotations

import pytest

from preempt.core.identity import ExpertKey
from preempt.core.enums import CachePolicy

from preempt.engine.expert_cache import ExpertCacheManager

MODEL_HASH = "fp-test"
UNIT = 100


def key(expert_idx: int, block_idx: int = 0) -> ExpertKey:
    return ExpertKey(
        model_fingerprint=MODEL_HASH, block_idx=block_idx, expert_idx=expert_idx
    )


def test_touch_misses_before_admit_and_hits_after() -> None:
    cache = ExpertCacheManager(budget_bytes=10 * UNIT)

    assert cache.touch(key(0)) is False
    assert key(0) not in cache

    assert cache.admit(key(0), UNIT) == ()
    assert key(0) in cache
    assert cache.touch(key(0)) is True


def test_counters_track_hits_misses_and_bytes_read() -> None:
    cache = ExpertCacheManager(budget_bytes=10 * UNIT)

    cache.touch(key(0))
    cache.admit(key(0), UNIT)
    cache.touch(key(0))
    cache.touch(key(0))
    cache.touch(key(1))

    assert cache.misses == 2
    assert cache.hits == 2
    assert cache.bytes_read == UNIT
    assert cache.evictions == 0
    assert cache.resident_bytes == UNIT
    assert len(cache) == 1


def test_admit_within_budget_evicts_nothing() -> None:
    cache = ExpertCacheManager(budget_bytes=3 * UNIT)

    assert cache.admit(key(0), UNIT) == ()
    assert cache.admit(key(1), UNIT) == ()
    assert cache.admit(key(2), UNIT) == ()
    assert cache.resident_bytes == 3 * UNIT
    assert cache.evictions == 0


def test_admit_evicts_until_the_new_entry_fits() -> None:
    cache = ExpertCacheManager(budget_bytes=3 * UNIT)
    for expert_idx in range(3):
        cache.admit(key(expert_idx), UNIT)

    # A double-sized newcomer must displace two residents, not one.
    evicted = cache.admit(key(3), 2 * UNIT)

    assert len(evicted) == 2
    assert cache.evictions == 2
    assert cache.resident_bytes == 3 * UNIT
    for victim in evicted:
        assert victim not in cache
    assert key(3) in cache


def test_evicted_keys_are_returned_exactly_once_each() -> None:
    cache = ExpertCacheManager(budget_bytes=4 * UNIT)
    for expert_idx in range(4):
        cache.admit(key(expert_idx), UNIT)

    evicted = cache.admit(key(4), 3 * UNIT)

    assert len(set(evicted)) == len(evicted)
    assert cache.resident_bytes == 4 * UNIT


def test_expert_larger_than_budget_raises_naming_both_sizes() -> None:
    cache = ExpertCacheManager(budget_bytes=UNIT)

    with pytest.raises(ValueError) as excinfo:
        cache.admit(key(0), 2 * UNIT)

    message = str(excinfo.value)
    assert str(2 * UNIT) in message
    assert str(UNIT) in message


def test_readmitting_a_resident_key_neither_evicts_nor_double_counts() -> None:
    cache = ExpertCacheManager(budget_bytes=2 * UNIT)
    cache.admit(key(0), UNIT)

    assert cache.admit(key(0), UNIT) == ()
    assert cache.resident_bytes == UNIT
    assert cache.bytes_read == UNIT
    assert len(cache) == 1


def test_resident_bytes_never_exceeds_budget_under_a_mixed_workload() -> None:
    budget = 7 * UNIT
    cache = ExpertCacheManager(budget_bytes=budget, sample_size=3, seed=17)

    for step in range(400):
        candidate = key((step * 7) % 23)
        if not cache.touch(candidate):
            cache.admit(candidate, UNIT + (step % 3) * 10)
        assert cache.resident_bytes <= budget

    assert cache.evictions > 0


def _scripted_cache(policy: CachePolicy) -> ExpertCacheManager:
    """Build a 3-entry cache whose LFRU and LRU victims differ.

    After the script: `A` is the least recently used but the most frequently
    used; `C` is the least frequently used among the recent pair. So LRU must
    take `A` and LFRU must take `C`.
    """
    cache = ExpertCacheManager(budget_bytes=3 * UNIT, policy=policy, sample_size=8)
    cache.admit(key(0), UNIT)  # A
    for _ in range(3):
        cache.touch(key(0))
    cache.admit(key(1), UNIT)  # B
    cache.admit(key(2), UNIT)  # C
    cache.touch(key(2))
    cache.touch(key(1))
    return cache


def test_lfru_evicts_the_least_frequently_used_entry() -> None:
    cache = _scripted_cache(CachePolicy.LFRU)

    assert cache.admit(key(9), UNIT) == (key(2),)


def test_lru_evicts_the_least_recently_used_entry() -> None:
    cache = _scripted_cache(CachePolicy.LRU)

    assert cache.admit(key(9), UNIT) == (key(0),)


def test_lfru_breaks_frequency_ties_by_recency() -> None:
    cache = ExpertCacheManager(budget_bytes=2 * UNIT, sample_size=8)
    cache.admit(key(0), UNIT)
    cache.admit(key(1), UNIT)
    cache.touch(key(0))
    cache.touch(key(1))

    # Equal frequency: the older last-access loses.
    assert cache.admit(key(2), UNIT) == (key(0),)


def _eviction_sequence(*, seed: int) -> tuple[int, ...]:
    """Run a fixed workload and return the expert index evicted at each step."""
    cache = ExpertCacheManager(budget_bytes=8 * UNIT, sample_size=3, seed=seed)
    evicted: list[int] = []
    for expert_idx in range(24):
        candidate = key(expert_idx % 12)
        if not cache.touch(candidate):
            evicted.extend(victim.expert_idx for victim in cache.admit(candidate, UNIT))
    return tuple(evicted)


def test_sampled_eviction_is_reproducible_for_a_given_seed() -> None:
    assert _eviction_sequence(seed=0) == _eviction_sequence(seed=0)
    # A pinned sequence, not merely "something got evicted": this is the
    # regression guard on sampling determinism.
    assert _eviction_sequence(seed=0) == (0, 4, 5, 2, 1, 8, 9, 6, 1, 7, 11, 4, 6, 3)


def test_a_different_seed_samples_different_victims() -> None:
    assert _eviction_sequence(seed=0) != _eviction_sequence(seed=1)


def test_sampling_is_bounded_by_sample_size_not_cache_size() -> None:
    cache = ExpertCacheManager(budget_bytes=64 * UNIT, sample_size=2, seed=3)
    for expert_idx in range(64):
        cache.admit(key(expert_idx), UNIT)
    # Make one entry unambiguously the best victim by frequency, then check
    # that a 2-of-64 sample does not reliably find it: sampling is real.
    for expert_idx in range(1, 64):
        cache.touch(key(expert_idx))

    evicted = cache.admit(key(100), UNIT)

    assert len(evicted) == 1
    # A full scan would be obliged to take `key(0)`; a 2-of-64 sample is not.
    assert evicted[0] != key(0)


def test_contains_reflects_admission_and_eviction() -> None:
    cache = ExpertCacheManager(budget_bytes=UNIT, sample_size=8)
    cache.admit(key(0), UNIT)
    assert key(0) in cache

    evicted = cache.admit(key(1), UNIT)

    assert evicted == (key(0),)
    assert key(0) not in cache
    assert key(1) in cache


def test_keys_from_different_layers_are_distinct_entries() -> None:
    cache = ExpertCacheManager(budget_bytes=4 * UNIT)
    cache.admit(key(0, block_idx=0), UNIT)

    assert cache.touch(key(0, block_idx=1)) is False
    cache.admit(key(0, block_idx=1), UNIT)

    assert len(cache) == 2


def test_zero_or_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertCacheManager(budget_bytes=0)


def test_sample_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertCacheManager(budget_bytes=UNIT, sample_size=0)
