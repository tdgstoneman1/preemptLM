from __future__ import annotations

import attrs
from attrs import field

import numpy as np

from preempt.datamodel.identity import ExpertKey

from preempt.core.enums import CacheEvictionPolicy, ReadPriority


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
    can_evict : bool
        Flag indicating whether the entry can safely be evicted. For `DEMAND` reads,
        this should be False until the entry is consumed by the caller in order to prevent
        premature eviction.

    """

    num_bytes: int = field()
    freq: int = field()
    last: int = field()
    slot: int = field()
    can_evict: bool = field()


# TODO add true LRU and LFRU as a baseline for experiments
class ExpertCacheManager:
    """Manager that tracks experts in memory and manages eviction decisions.

    Uses Redis-style approximate LRU and LFRU with random sampling to avoid full scans during
    eviction decisions, with LFRU as the default policy.

    **Note:** This does not track which experts are actively used in a forward pass.  When memory
    budget is less than the *largest* set of unique experts required by *any one layer* across *all
    tokens*, experts may be prematurely evicted immediately upon loading.

    With the MLX backend, this will not cause an error as MLX's refcounting will still keep expert
    weights in memory during computation. However, it can still hurt performance by causing
    excessive re-reads from disk.

    With other backends, insufficient memory budget may cause downstream `KeyError`s.  To safely
    avoid this and possible performance penalties, initialize the cache manager with a memory budget
    greater than `expert_size * experts_per_token * tokens_in_sequence` (where `experts_per_token`
    usually means top-k, and `tokens_in_sequence` is max sequence length).
    """

    _budget_bytes: int
    _policy: CacheEvictionPolicy
    _eviction_sample_size: int
    _rng: np.random.Generator

    _entries: dict[ExpertKey, _Entry]
    _keys: list[ExpertKey]
    _bytes_size: int
    _clock: int

    _num_hits: int
    _num_misses: int
    _num_evictions: int
    _num_bytes_read: int

    def __init__(
        self,
        *,
        budget_bytes: int,
        policy: CacheEvictionPolicy = CacheEvictionPolicy.LFRU,
        sample_size: int = 5,
        seed: int = 67,
    ) -> None:
        """
        Parameters
        ----------
        budget_bytes : int
            Maximum memory footprint allowed across all cached expert layers
        policy : CacheEvictionPolicy
            Eviction ranking policy (LFRU or LRU), by default CacheEvictionPolicy.LFRU
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
        self._eviction_sample_size = sample_size
        self._rng = np.random.default_rng(seed)

        self._entries = dict()
        self._keys = list()
        self._bytes_size = 0
        self._clock = 0

        self._num_hits = 0
        self._num_misses = 0
        self._num_evictions = 0
        self._num_bytes_read = 0

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def policy(self) -> CacheEvictionPolicy:
        """The cache's eviction policy (LRU or LFRU)"""
        return self._policy

    @property
    def budget_bytes(self) -> int:
        """The cache's maximum allowable memory footprint in bytes"""
        return self._budget_bytes

    @property
    def cache_size(self) -> int:
        """The cache's memory footprint in bytes"""
        return self._bytes_size

    @property
    def hits(self) -> int:
        """Number of cache hits where a requested expert was already cached"""
        return self._num_hits

    @property
    def misses(self) -> int:
        """Number of cache misses requiring experts to be read from disk"""
        return self._num_misses

    @property
    def evictions(self) -> int:
        """Number of evictions required to make room for new entries"""
        return self._num_evictions

    @property
    def bytes_read(self) -> int:
        """Total number of bytes read from expert bank"""
        return self._num_bytes_read

    def touch(self, key: ExpertKey) -> bool:  # TODO rename
        """Records a request for `key` and reports whether it hit.

        Parameters
        ----------
        key : ExpertKey
            Key identifying an MoE expert

        Returns
        -------
        bool
            True if the key was found in the cache (frequency and recency updated), False
            otherwise.
        """
        if (entry := self._entries.get(key)) is None:
            self._num_misses += 1
            return False

        self._num_hits += 1
        self._clock += 1

        entry.freq += 1
        entry.last = self._clock

        return True

    def admit(
        self,
        key: ExpertKey,
        num_bytes: int,
        priority: ReadPriority,
    ) -> tuple[ExpertKey, ...]:
        """Makes room for new expert under `key` and records it.

        Parameters
        ----------
        key : ExpertKey
            Key identifying a serialized expert
        num_bytes : int
            Size of the expert's weights in bytes
        priority: ReadPriority
            Used to flag whether the entry can safely be evicted from the cache. This is to prevent
            entries with `DEMAND` priority from being prematurely evicted by concurrent `admit()`
            calls.

        Returns
        -------
        tuple[ExpertKey, ...]
            Keys for experts evicted to make room for the new expert, in the order they were evicted
            in.

        Raises
        ------
        ValueError
            If a single serialized expert's size exceeds cache's total memory budget
        """
        if num_bytes > self._budget_bytes:  # TODO move this check to helper method
            raise ValueError(
                f"The size of expert {key!r} ({num_bytes/1024**3} GB) exceeds the "
                f"cache's total memory budget ({self._budget_bytes/1024**3} GB)."
            )
        if key in self._entries:
            return tuple()

        self._num_bytes_read += num_bytes
        evicted: list[ExpertKey] = []

        while self._bytes_size + num_bytes > self._budget_bytes:
            target = self._select_eviction_target()
            self.evict(target)

            evicted.append(target)
            self._num_evictions += 1

        self._clock += 1

        self._entries[key] = _Entry(
            num_bytes=num_bytes,
            freq=1,
            last=self._clock,
            slot=len(self._keys),
            can_evict=priority != ReadPriority.DEMAND,
        )
        self._keys.append(key)
        self._bytes_size += num_bytes

        return tuple(evicted)

    def mark_entry_safe_to_evict(self, key: ExpertKey) -> None:
        # ! Maybe raise KeyError loudly, this could hide bugs
        if entry := self._entries.get(key):
            entry.can_evict = True

    def _select_eviction_target(self) -> ExpertKey:
        num_keys = len(self._keys)
        targets: list[ExpertKey] = []

        if num_keys <= self._eviction_sample_size:
            targets = [k for k in self._keys if self._entries[k].can_evict]
        else:
            max_attempts = self._eviction_sample_size * 10
            idxs = self._rng.integers(0, num_keys, size=max_attempts)

            for idx in idxs:
                key = self._keys[idx]

                if self._entries[key].can_evict:
                    targets.append(key)

                    if len(targets) == self._eviction_sample_size:
                        break

            if not targets:  # Fallback if cache highly locked
                targets = [k for k in self._keys if self._entries[k].can_evict]

        if not targets:
            raise RuntimeError(
                "No evictable experts found in the cache. The total size of in-flight "
                "expert bank reads likely exceeds the cache's confiugured memory budget "
                f"({self.budget_bytes/1024**3} GB)."
            )
        return min(targets, key=self._rank)

    def _rank(self, key: ExpertKey) -> tuple[int, int]:
        entry = self._entries[key]

        if self._policy is CacheEvictionPolicy.LRU:
            return (0, entry.last)

        return (entry.freq, entry.last)

    def evict(self, key: ExpertKey) -> None:
        entry = self._entries.pop(key)
        moved = self._keys.pop()

        if moved != key:
            self._keys[entry.slot] = moved
            self._entries[moved].slot = entry.slot

        self._bytes_size -= entry.num_bytes
