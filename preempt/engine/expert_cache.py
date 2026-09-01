from __future__ import annotations

from collections.abc import Sequence

import attrs
from attrs import field

import random

from preempt.core.identity import ExpertKey
from preempt.core.enums import CachePolicy

# TODO rename module


@attrs.define(kw_only=True)
class _Entry:
    """Metadata for one cached expert

    Attributes
    ----------
    num_bytes : int
        Payload size charged against the cache budget. Must match the byte count
        that the in-memory layer uses as drift will cause silent memory overrun.
    freq : int
        Demand access count since admission (LFRU frequency term).
    last : int
        Logical clock of most recent access (LFRU recency term).
    slot : int
        Position in the cache's key list for O(1) eviction via swap-remove.
    """

    num_bytes: int = field()
    freq: int = field()
    last: int = field()
    slot: int = field()


class ExpertCacheManager:
    """Manager that tracks experts in memory and manages eviction decisions.

    Uses random sampling to avoid full scans during eviction decisions, with LFRU
    (Least Frequently Recently Used) as the default policy.

    **Note:** This does not track which experts are actively used in a forward pass.
    When memory budget is less than the *largest* set of unique experts required by
    *any one layer* across *all tokens*, experts may be prematurely evicted immediately
    upon loading.

    With the MLX backend, this will not cause an error as MLX's refcounting will still
    keep expert weights in memory during computation. However, it can still hurt performance
    by causing excessive re-reads from disk.

    With other backends, insufficient memory budget may cause downstream `KeyError`s.
    To safely avoid this and possble performance penalties, initialize cache manager
    with a memory budget greater than the size of `experts_per_token * tokens_in_sequence`
    (where `experts_per_token` typically refers to top-k, and `tokens_in_sequence`
    the max sequence length).
    """

    _budget_bytes: int  # TODO rename to '_memory_budget'
    _policy: CachePolicy
    _sample_size: int  # TODO rename
    _rng: random.Random

    _entries: dict[ExpertKey, _Entry]
    _keys: list[ExpertKey]  # TODO rename to '_expert_keys'
    _resident_bytes: int  # TODO rename
    _clock: int

    _hits: int  # TODO rename to '_num_hits'
    _misses: int  # TODO rename to '_num_misses'
    _evictions: int  # TODO rename to '_num_evictions'
    _bytes_read: int

    def __init__(
        self,
        *,
        budget_bytes: int,
        policy: CachePolicy = CachePolicy.LFRU,
        sample_size: int = 5,
        seed: int = 67,
    ) -> None:
        """
        Parameters
        ----------
        budget_bytes : int
            Maximum memory footprint allowed across all cached expert layers
        policy : CachePolicy
            Eviction ranking policy (LFRU or LRU), by default CachePolicy.LFRU
        sample_size : int
            Number of experts sampled for an eviction decision. When the number of cached
            experts exceeds this, random sampling is used. Otherwise, all cached experts
            are considered. By default 5
        seed : int
            Random seed for sampling candidate layers to evict. Deterministic sampling
            enables exact eviction assertions in tests, by default 67

        Raises
        ------
        ValueError
            If `budget_bytes` < 1
        ValueError
            If `sample_size` < 1
        """
        if budget_bytes < 1:
            raise ValueError(f"`budget_bytes` must be positive, got `{budget_bytes=}`.")

        if sample_size < 1:
            raise ValueError(f"`sample_size` must be positive, got `{sample_size=}`.")

        self._budget_bytes = budget_bytes
        self._policy = policy
        self._sample_size = sample_size
        self._rng = random.Random(seed)

        self._entries = dict()
        self._keys = list()
        self._resident_bytes = 0
        self._clock = 0

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bytes_read = 0

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def policy(self) -> CachePolicy:
        return self._policy

    @property
    def cache_size(self) -> int:
        return self._resident_bytes

    @property
    def hits(self) -> int:
        """Number of demand accesses where an expert was already cached"""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of demand accesses where an expert had to be read from disk"""
        return self._misses

    @property
    def evictions(self) -> int:
        """Number of evictions required to make room for new entries"""
        return self._evictions

    @property
    def bytes_read(self) -> int:
        """Total number of bytes read from expert bank on disk"""
        return self._bytes_read

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def touch(self, key: ExpertKey) -> bool:  # TODO rename
        """Records a demand access for `key` and reports whether it hit.

        Parameters
        ----------
        key : ExpertKey
            Key for an MoE router-selected expert

        Returns
        -------
        bool
            `True` if the key was found in cache (frequency and recency updated),
            `False` otherwise (caller should read the expert and call `admit(...)`).
        """
        entry = self._entries.get(key)

        if entry is None:
            self._misses += 1
            return False

        self._hits += 1
        self._clock += 1

        entry.freq += 1
        entry.last = self._clock

        return True

    def admit(self, key: ExpertKey, num_bytes: int) -> tuple[ExpertKey, ...]:
        """Makes room for new expert under `key` and records it.

        Parameters
        ----------
        key : ExpertKey
            Key identifying an expert
        num_bytes : int
            The expert's payload size in bytes

        Returns
        -------
        tuple[ExpertKey, ...]
            Expert keys displaced to make room for the new expert, in the order
            they were evicted. Caller must remove exactly these from the cache
            and in the same order.

        Raises
        ------
        ValueError
            If a single expert's size exceeds cache's total memory budget
        """
        if num_bytes > self._budget_bytes:
            raise ValueError(
                f"The size of expert {key!r} ({num_bytes} bytes) exceeds the "
                f"cache's total memory budget ({self._budget_bytes} bytes)."
            )

        if key in self._entries:
            return tuple()

        self._bytes_read += num_bytes

        evicted: list[ExpertKey] = []
        while self._resident_bytes + num_bytes > self._budget_bytes:
            victim = self._select_victim()
            self._remove(victim)

            self._evictions += 1
            evicted.append(victim)

        self._clock += 1
        self._entries[key] = _Entry(
            num_bytes=num_bytes, freq=1, last=self._clock, slot=len(self._keys)
        )
        self._keys.append(key)
        self._resident_bytes += num_bytes

        return tuple(evicted)

    def _select_victim(self) -> ExpertKey:  # TODO rename
        candidates: Sequence[ExpertKey]

        if len(self._keys) <= self._sample_size:
            candidates = self._keys
        else:
            # Sampling is cheaper w/ replacement than w/o, impact negligible when
            # cache size >> sample_size (see `waste/src/ecache.c:378` for similar approach)
            candidates = [
                self._keys[self._rng.randrange(len(self._keys))]
                for _ in range(self._sample_size)
            ]

        return min(candidates, key=self._rank)

    def _rank(self, key: ExpertKey) -> tuple[int, int]:
        entry = self._entries[key]

        if self._policy is CachePolicy.LRU:
            return (0, entry.last)

        return (entry.freq, entry.last)

    def _remove(self, key: ExpertKey) -> None:
        entry = self._entries.pop(key)
        moved = self._keys.pop()

        if moved != key:
            self._keys[entry.slot] = moved
            self._entries[moved].slot = entry.slot

        self._resident_bytes -= entry.num_bytes
