from __future__ import annotations

import attrs
from attrs import field, validators


@attrs.define(kw_only=True, frozen=True)
class StepMetrics:
    """Timing and token count for one call to `IModelRunner.step()`

    Attributes
    ----------
    step_idx : int
        Index of this step within a multi-step generation run
    n_tokens : int
        Number of tokens processed this step (multiple tokens during
        prefill or a single token thereafter)
    duration_s : float
        Wall time for the forward pass in seconds
    """

    step_idx: int = field(validator=validators.ge(0))
    n_tokens: int = field(validator=validators.ge(1))  # TODO rename to `num_tokens`
    duration_s: float = field(validator=validators.ge(0.0))
    generated_token_id: int = field(validator=validators.ge(0))


@attrs.define(kw_only=True)
class GenerationMetrics:
    """Aggregate counters and timing for a single generation run.

    Collects per-step metrics and cache/prefetch statistics.

    :Note: On generation runs where experts are not streamed,
    `cache_hits`, `cache_misses`, `demand_stall_s`, `prefetched_bytes`,
    and `wasted_prefetch_bytes` are still initialized but stay at 0.

    Attributes
    ----------
    steps : list[StepMetrics]
        Timing and token count for each generation step
    records_written : int
        Number of trace records written to the event sink
    cache_hits : int
        Number of times an expert was already in memory when requested
    cache_misses : int
        Number of times an expert was not in memory when requested and
        had to be loaded from disk
    demand_stall_s : float
        Total seconds blocked waiting for demand disk reads
    prefetched_bytes : int
        Total number of bytes loaded by prefetch (predicted experts)
    wasted_prefetch_bytes : int
        Total number of prefetched bytes loaded but never requested by
        the router
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
