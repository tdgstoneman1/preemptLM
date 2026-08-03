from __future__ import annotations

import attrs
from attrs import field, validators


@attrs.define(kw_only=True, frozen=True)
class StepMetrics:
    """Timing for one forward pass (prefill or decode)."""

    step_idx: int = field(validator=validators.ge(0))
    n_tokens: int = field(validator=validators.ge(1))
    duration_s: float = field(validator=validators.ge(0.0))


@attrs.define(kw_only=True)
class GenerationMetrics:
    """Per-generation counters.

    The streaming counters (`cache_*`, `demand_stall_s`, `*_bytes`) stay zero
    until the scheduler lands in phase 3; they exist now so the shape of
    `GenerationResult` is stable across phases.
    """

    steps: list[StepMetrics] = field(factory=list)
    records_written: int = field(default=0)
    cache_hits: int = field(default=0)
    cache_misses: int = field(default=0)
    demand_stall_s: float = field(default=0.0)
    prefetched_bytes: int = field(default=0)
    wasted_prefetch_bytes: int = field(default=0)

    @property
    def tokens_forwarded(self) -> int:
        return sum(step.n_tokens for step in self.steps)

    @property
    def total_duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)
