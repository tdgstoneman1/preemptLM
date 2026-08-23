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
    manager = ExpertCacheManager(budget_bytes=10 * UNIT)

    assert manager.touch(key(0)) is False
    assert key(0) not in manager

    assert manager.admit(key(0), UNIT) == ()
    assert key(0) in manager
    assert manager.touch(key(0)) is True


def test_counters_track_hits_misses_and_bytes_read() -> None:
    manager = ExpertCacheManager(budget_bytes=10 * UNIT)

    manager.touch(key(0))
    manager.admit(key(0), UNIT)
    manager.touch(key(0))
    manager.touch(key(0))
    manager.touch(key(1))

    assert manager.misses == 2
    assert manager.hits == 2
    assert manager.bytes_read == UNIT
    assert manager.evictions == 0
    assert manager.cache_size == UNIT
    assert len(manager) == 1


def test_admit_within_budget_evicts_nothing() -> None:
    manager = ExpertCacheManager(budget_bytes=3 * UNIT)

    assert manager.admit(key(0), UNIT) == ()
    assert manager.admit(key(1), UNIT) == ()
    assert manager.admit(key(2), UNIT) == ()
    assert manager.cache_size == 3 * UNIT
    assert manager.evictions == 0


def test_admit_evicts_until_the_new_entry_fits() -> None:
    manager = ExpertCacheManager(budget_bytes=3 * UNIT)
    for expert_idx in range(3):
        manager.admit(key(expert_idx), UNIT)

    # A double-sized newcomer must displace two residents, not one.
    evicted = manager.admit(key(3), 2 * UNIT)

    assert len(evicted) == 2
    assert manager.evictions == 2
    assert manager.cache_size == 3 * UNIT
    for victim in evicted:
        assert victim not in manager
    assert key(3) in manager


def test_evicted_keys_are_returned_exactly_once_each() -> None:
    manager = ExpertCacheManager(budget_bytes=4 * UNIT)
    for expert_idx in range(4):
        manager.admit(key(expert_idx), UNIT)

    evicted = manager.admit(key(4), 3 * UNIT)

    assert len(set(evicted)) == len(evicted)
    assert manager.cache_size == 4 * UNIT


def test_expert_larger_than_budget_raises_naming_both_sizes() -> None:
    manager = ExpertCacheManager(budget_bytes=UNIT)

    with pytest.raises(ValueError) as excinfo:
        manager.admit(key(0), 2 * UNIT)

    message = str(excinfo.value)
    assert str(2 * UNIT) in message
    assert str(UNIT) in message


def test_readmitting_a_resident_key_neither_evicts_nor_double_counts() -> None:
    manager = ExpertCacheManager(budget_bytes=2 * UNIT)
    manager.admit(key(0), UNIT)

    assert manager.admit(key(0), UNIT) == ()
    assert manager.cache_size == UNIT
    assert manager.bytes_read == UNIT
    assert len(manager) == 1


def test_resident_bytes_never_exceeds_budget_under_a_mixed_workload() -> None:
    budget = 7 * UNIT
    manager = ExpertCacheManager(budget_bytes=budget, sample_size=3, seed=17)

    for step in range(400):
        candidate = key((step * 7) % 23)
        if not manager.touch(candidate):
            manager.admit(candidate, UNIT + (step % 3) * 10)
        assert manager.cache_size <= budget

    assert manager.evictions > 0


def _scripted_cache(policy: CachePolicy) -> ExpertCacheManager:
    """Build a 3-entry manager whose LFRU and LRU victims differ.

    After the script: `A` is the least recently used but the most frequently
    used; `C` is the least frequently used among the recent pair. So LRU must
    take `A` and LFRU must take `C`.
    """
    manager = ExpertCacheManager(budget_bytes=3 * UNIT, policy=policy, sample_size=8)
    manager.admit(key(0), UNIT)  # A
    for _ in range(3):
        manager.touch(key(0))

    manager.admit(key(1), UNIT)  # B
    manager.admit(key(2), UNIT)  # C
    manager.touch(key(2))
    manager.touch(key(1))

    return manager


def test_lfru_evicts_the_least_frequently_used_entry() -> None:
    manager = _scripted_cache(CachePolicy.LFRU)

    assert manager.admit(key(9), UNIT) == (key(2),)


def test_lru_evicts_the_least_recently_used_entry() -> None:
    manager = _scripted_cache(CachePolicy.LRU)

    assert manager.admit(key(9), UNIT) == (key(0),)


def test_lfru_breaks_frequency_ties_by_recency() -> None:
    manager = ExpertCacheManager(budget_bytes=2 * UNIT, sample_size=8)
    manager.admit(key(0), UNIT)
    manager.admit(key(1), UNIT)
    manager.touch(key(0))
    manager.touch(key(1))

    # Equal frequency: the older last-access loses.
    assert manager.admit(key(2), UNIT) == (key(0),)


def _eviction_sequence(*, seed: int) -> tuple[int, ...]:
    """Run a fixed workload and return the expert index evicted at each step."""
    manager = ExpertCacheManager(budget_bytes=8 * UNIT, sample_size=3, seed=seed)
    evicted: list[int] = []
    for expert_idx in range(24):
        candidate = key(expert_idx % 12)
        if not manager.touch(candidate):
            evicted.extend(
                victim.expert_idx for victim in manager.admit(candidate, UNIT)
            )
    return tuple(evicted)


def test_sampled_eviction_is_reproducible_for_a_given_seed() -> None:
    assert _eviction_sequence(seed=0) == _eviction_sequence(seed=0)
    # A pinned sequence, not merely "something got evicted": this is the
    # regression guard on sampling determinism.
    assert _eviction_sequence(seed=0) == (0, 4, 5, 2, 1, 8, 9, 6, 1, 7, 11, 4, 6, 3)


def test_a_different_seed_samples_different_victims() -> None:
    assert _eviction_sequence(seed=0) != _eviction_sequence(seed=1)


def test_sampling_is_bounded_by_sample_size_not_cache_size() -> None:
    manager = ExpertCacheManager(budget_bytes=64 * UNIT, sample_size=2, seed=3)
    for expert_idx in range(64):
        manager.admit(key(expert_idx), UNIT)
    # Make one entry unambiguously the best victim by frequency, then check
    # that a 2-of-64 sample does not reliably find it: sampling is real.
    for expert_idx in range(1, 64):
        manager.touch(key(expert_idx))

    evicted = manager.admit(key(100), UNIT)

    assert len(evicted) == 1
    # A full scan would be obliged to take `key(0)`; a 2-of-64 sample is not.
    assert evicted[0] != key(0)


def test_contains_reflects_admission_and_eviction() -> None:
    manager = ExpertCacheManager(budget_bytes=UNIT, sample_size=8)
    manager.admit(key(0), UNIT)
    assert key(0) in manager

    evicted = manager.admit(key(1), UNIT)

    assert evicted == (key(0),)
    assert key(0) not in manager
    assert key(1) in manager


def test_keys_from_different_layers_are_distinct_entries() -> None:
    manager = ExpertCacheManager(budget_bytes=4 * UNIT)
    manager.admit(key(0, block_idx=0), UNIT)

    assert manager.touch(key(0, block_idx=1)) is False
    manager.admit(key(0, block_idx=1), UNIT)

    assert len(manager) == 2


def test_zero_or_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertCacheManager(budget_bytes=0)


def test_sample_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertCacheManager(budget_bytes=UNIT, sample_size=0)
