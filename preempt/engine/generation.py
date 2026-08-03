from __future__ import annotations

from collections.abc import Callable, Sequence

import asyncio
import time

from preempt.core.protocols.runner import ModelRunner
from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceStepContext
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


async def generate_greedy(
    *,
    runner: ModelRunner,
    prompt_ids: Sequence[int],
    max_tokens: int,
    recorder: BaseEventRecorder | None = None,
    sink: BaseEventSink | None = None,
    sequence_id: int = 0,
    on_step: Callable[[StepMetrics], None] | None = None,
) -> tuple[list[int], GenerationMetrics]:
    """Greedily decode `max_tokens` ids: one batched prefill, then decode steps.

    The forward runs via `asyncio.to_thread` so the event loop stays free for
    I/O — the structural disk-parallel-compute overlap, in place before any
    real I/O exists. When `recorder` and `sink` are given, each step is
    bracketed by `start_step`/`flush`; both must be given together.

    Returns
    -------
    tuple[list[int], GenerationMetrics]
        Generated token ids and per-step metrics.
    """
    if max_tokens < 1:
        raise ValueError("`max_tokens` must be at least 1.")

    if not prompt_ids:
        raise ValueError("`prompt_ids` must be non-empty.")

    if (recorder is None) != (sink is None):
        raise ValueError("`recorder` and `sink` must be provided both or neither.")

    runner.prepare()

    generated: list[int] = []
    metrics = GenerationMetrics()
    tokens = list(prompt_ids)
    token_idx = 0

    for step_idx in range(max_tokens):
        started = time.perf_counter()

        if recorder is not None:
            recorder.start_step(
                TraceStepContext(
                    sequence_id=sequence_id,
                    token_idx=token_idx,
                    # A prefill spans many tokens; a single `token_id` is only
                    # meaningful for a one-token forward.
                    token_id=tokens[0] if len(tokens) == 1 else None,
                )
            )

        next_token = await asyncio.to_thread(runner.step, tokens)

        if recorder is not None and sink is not None:
            metrics.records_written += await recorder.flush(sink)

        step_metrics = StepMetrics(
            step_idx=step_idx,
            n_tokens=len(tokens),
            duration_s=time.perf_counter() - started,
        )
        metrics.steps.append(step_metrics)

        if on_step is not None:
            on_step(step_metrics)

        generated.append(next_token)
        token_idx += len(tokens)
        tokens = [next_token]

    return generated, metrics
