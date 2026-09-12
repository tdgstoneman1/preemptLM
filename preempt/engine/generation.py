from __future__ import annotations

from typing import Optional
from collections.abc import Callable, Sequence

import asyncio
import time

from preempt.core.protocols import IModelRunner

from preempt.datamodel.tracing.context import TraceStepContext

from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseTraceRecorder
from preempt.engine.sinks import BaseTraceSink

# TODO track total elapsed time during generation
# TODO track total bytes read


async def generate_greedy(
    *,
    runner: IModelRunner,
    prompt_ids: Sequence[int],
    eos_token_ids: set[int] | int | None,
    max_tokens: int,
    prefill_chunk_size: int = 512,
    recorder: Optional[BaseTraceRecorder] = None,
    sink: Optional[BaseTraceSink] = None,
    sequence_id: int = 0,
    on_step: Optional[Callable[[StepMetrics], None]] = None,
) -> tuple[list[int], GenerationMetrics]:
    """Greedy token generation using chunked prefill and single-token decoding.

    Each forward pass runs in a thread pool via `asyncio.to_thread` to keep the
    event loop free for concurrent I/O (such as reading expert weights from disk).

    Parameters
    ----------
    runner : IModelRunner
        Model runner implementing the `IModelRunner` interface
    prompt_ids : Sequence[int]
        Input token ids, must be non-empty.
    eos_token_ids: set[int] | int | None
        The tokenizer's EOS token ids
    max_tokens : int
        Max total tokens to generate including the first prefill output, must be
        >= 1.
    prefill_chunk_size : int
        Max tokens per prefill chunk. Controls activation memory and enables chunk-
        local expert reuse (no effect on correctness), by default default 512
    recorder : Optional[BaseTraceRecorder]
        Optional recorder for tracing. If provided, `sink` must also be given, by
        default None
    sink : Optional[BaseTraceSink]
        Optional sink for writing traced events. If provided, `recorder` must also
        be given, by default None
    sequence_id : int
        Identifier for the generation sequence passed to the recorder, by default 0
    on_step : Optional[Callable[[StepMetrics], None]]
        Optional callback invoked after each forward pass with per-step metrics,
        by default None

    Returns
    -------
    tuple[list[int], GenerationMetrics]
        Generated token ids with aggregated generation metrics

    Raises
    ------
    ValueError
        If `prompt_ids` is empty
    ValueError
        If `max_tokens` < 1
    ValueError
        If `prefill_chunk_size` < 1
    ValueError
        If `recorder` or `sink` is provided without the other
    """
    if len(prompt_ids) == 0:
        raise ValueError("`prompt_ids` must be non-empty.")

    if max_tokens < 1:
        raise ValueError("`max_tokens` must be at least 1.")

    if prefill_chunk_size < 1:
        raise ValueError("`prefill_chunk_size` must be at least 1.")

    if (recorder is None) != (sink is None):
        raise ValueError(
            "`recorder` and `sink` must both be provided or both be `None`, "
            f"but got `{type(recorder)=}` and `{type(sink)=}`."
        )
    _eos_token_ids = (
        {eos_token_ids}
        if isinstance(eos_token_ids, int) or eos_token_ids is None
        else eos_token_ids
    )
    metrics = GenerationMetrics()
    prompt = list(prompt_ids)
    token_idx = 0
    step_idx = 0

    async def forward(tokens: list[int]) -> int:
        nonlocal token_idx, step_idx
        started = time.perf_counter()

        if recorder is not None:
            recorder.start_trace(
                TraceStepContext(
                    sequence_id=sequence_id,
                    token_idx=token_idx,
                    token_id=tokens[0] if len(tokens) == 1 else None,
                )
            )
        next_token = await asyncio.to_thread(runner.step, tokens)

        if recorder is not None and sink is not None:
            metrics.records_written += await recorder.flush(sink)

        step_metrics = StepMetrics(
            step_idx=step_idx,
            num_tokens=len(tokens),
            duration_s=time.perf_counter() - started,
            generated_token_id=next_token,
        )
        metrics.steps.append(step_metrics)

        if on_step is not None:
            on_step(step_metrics)

        token_idx += len(tokens)
        step_idx += 1

        return next_token

    generated: list[int] = []

    # Prefill
    # TODO write dedicated prefill func
    first_token = 0
    for start in range(0, len(prompt), prefill_chunk_size):
        first_token = await forward(prompt[start : start + prefill_chunk_size])

    generated.append(first_token)
    tokens = [first_token]

    # Decode
    # TODO write dedicated greedy decode func
    for _ in range(max_tokens - 1):
        next_token = await forward(tokens)
        if next_token in _eos_token_ids:
            break

        generated.append(next_token)
        tokens = [next_token]

    return generated, metrics
