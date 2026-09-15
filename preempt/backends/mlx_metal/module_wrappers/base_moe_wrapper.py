from typing import Any
from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from ..types import ExpertLayerQuants, ModuleWrapperFactory
from ..expert_cache import MlxExpertCache
from ..recorder import MlxTraceRecorder

from preempt.core.enums import CacheSlotState

from preempt.engine.expert_io.loader import DiskBackedExpertLoader

from preempt.utils.dataclass_utils import FixedLenthList


class BaseMoEWrapper(ABC, nn.Module):
    """Base class for MoE block wrappers to enable tracing router selections and expert I/O.

    Multiple expert caching strategies are possible:

    - Shared: The block accesses routed experts from a global cache shared between all MoE blocks
      (v1 implementation). LFRU decisions are made by a global `ExpertCacheManager` based aggregated
      stats collected for all blocks.

    - Internal:
        - Each MoE block holds and manages its own cached experts.

    - Slotted: Weights are cached in a fixed number of preallocated slots These can be internal to
      blocks or shared globally.
        For internal cache, `_weight_slots` pins strong references to loaded weights (bytes) to
        prevent unintended garbage collection.

    - Nonslotted: Weights are cached in a dict-like container. Currently only supported for global
      caching.
    """

    inner: nn.Module
    model_fingerprint: str | None

    recorder: MlxTraceRecorder | None
    _capture_gate_logits: bool

    layer_path: str
    block_idx: int
    slot_size: int
    num_slots: int

    expert_loader: DiskBackedExpertLoader | None
    expert_cache: MlxExpertCache | None

    _quants: ExpertLayerQuants | None

    # LFRU terms
    _clock: int
    _slot_freq: np.ndarray
    _slot_recency: np.ndarray
    _eid_lookup_table: np.ndarray
    _sid_lookup_table: np.ndarray
    _slot_refs: FixedLenthList[bytes]

    _weight_slots: Any

    # TODO add option to use internal or global cache
    # TODO maybe add stream arg to constructor, here
    def __init__(
        self,
        *,
        inner: nn.Module,
        model_fingerprint: str | None,
        layer_path: str,
        block_idx: int,
        num_experts: int,
        recorder: MlxTraceRecorder | None,
        capture_gate_logits: bool,
        expert_loader: DiskBackedExpertLoader | None,
        expert_cache: MlxExpertCache | None,
        num_slots: int | None,
        slot_size: int | None,
    ) -> None:
        if expert_loader is not None and any(
            val is None for val in (expert_cache, num_slots, slot_size)
        ):
            raise ValueError()  # TODO error msg

        self.inner = inner
        self.model_fingerprint = model_fingerprint

        self.recorder = recorder
        self._capture_gate_logits = capture_gate_logits

        self.layer_path = layer_path
        self.block_idx = block_idx

        self.expert_loader = expert_loader
        self.expert_cache = expert_cache

        self.num_slots = num_slots or 0
        self.slot_size = slot_size or 0

        # Slot lookup tables
        self._eid_lookup_table = np.full(  # Maps eids to slots
            num_experts,
            CacheSlotState.EMPTY,
            np.int32,
        )
        self._sid_lookup_table = np.full(  # Maps slots to eids
            self.num_slots,
            CacheSlotState.EMPTY,
            np.int32,
        )
        # Strong refs to prevent premature gc
        self._slot_refs = FixedLenthList(self.num_slots)

        # LFRU terms
        self._clock: int = 0
        self._slot_freq = np.full(self.num_slots, 0, np.int64)
        self._slot_recency = np.full(self.num_slots, 0, np.int64)

        self._weight_slots = None

    @property
    def is_quantized(self) -> bool:
        return self._quants is not None

    @property
    def should_capture_traces(self) -> bool:
        return self.recorder is not None

    @property
    def should_stream_experts(self) -> bool:
        return self.expert_loader is not None

    @staticmethod
    @abstractmethod
    def make_wrapper_factory(
        *,
        model_fingerprint: str | None = None,
        recorder: MlxTraceRecorder | None = None,
        capture_gate_logits: bool = False,
        expert_loader: DiskBackedExpertLoader | None = None,
        expert_cache: MlxExpertCache | None = None,
        num_slots: int = 0,
        slot_size: int = 0,
        # stream: mx.Stream | mx.Device,
        **kwargs,
    ) -> ModuleWrapperFactory: ...

    def _touch(self, slots: np.ndarray) -> None:
        self._clock += 1
        np.add.at(self._slot_freq, slots, 1)
        self._slot_recency[slots] = self._clock

    def _choose_slots(self, n: int, protected: np.ndarray) -> np.ndarray:
        """Handles LFRU decisions internally, alternative external manager"""

        free = np.nonzero(self._sid_lookup_table < 0)[0]
        if free.size >= n:
            return free[:n]

        protected = self._eid_lookup_table[protected]
        protected = protected[protected >= 0]

        lfru_score = self._slot_freq * (self._clock + 1) + self._slot_recency
        lfru_score = np.delete(lfru_score, protected)
        # lfru_score[protected] = np.iinfo(np.int64).max
        evicted = np.argsort(lfru_score, kind="stable")[: n - free.size]

        return np.concatenate([free, evicted])
