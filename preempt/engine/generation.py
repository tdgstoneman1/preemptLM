from __future__ import annotations

from collections.abc import Callable, Sequence

import asyncio

import time

from preempt.core.protocols.runner import IModelRunner
from preempt.core.sinks import BaseEventSink

from preempt.datamodel.tracing.context import TraceStepContext

from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


async def generate_greedy(
    *,
    runner: IModelRunner,
    prompt_ids: Sequence[int],
    max_tokens: int,
    prefill_chunk_size: int = 512,
    recorder: BaseEventRecorder | None = None,
    sink: BaseEventSink | None = None,
    sequence_id: int = 0,
    on_step: Callable[[StepMetrics], None] | None = None,
) -> tuple[list[int], GenerationMetrics]:
    """Greedily generates tokens from prompt using chunked prefill and single-token
    decoding.

    The prompt is processed in chunks of at most `prefill_chunk_size` tokens. Each
    chunk runs one forward pass, and `runner`'s KV cache carries context across chunks
    such that the results are identical to a single large prefill. The output of the
    last prefill chunk becomes the first generated token.

    Decode steps then run one token at a time until a total of `max_tokens` have been
    generated.

    Each forward pass runs in a thread pool via `asyncio.to_thread`. This keeps the
    event loop free for concurrent I/O (such as reading expert weights from disk).

    Parameters
    ----------
    runner : IModelRunner
        Model runner implementing `prepare()` and `step()`
    prompt_ids : Sequence[int]
        Input token ids, must be non-empty.
    max_tokens : int
        Max total tokens to generate including the first prefill output, must be >= 1.
    prefill_chunk_size : int, default 512
        Max tokens per prefill chunk. Controls activation memory and enables chunk-
        local expert reuse (no effect on correctness).
    recorder : BaseEventRecorder | None, default None
        Optional recorder for tracing. If provided, `sink` must also be given.
    sink : BaseEventSink | None, default None
        Optional sink for writing traced events. If provided, `recorder` must also be
        given.
    sequence_id : int, default 0
        Identifier for the generation sequence passed to the recorder
    on_step : Callable[[StepMetrics], None] | None, default None
        Optional callback invoked after each forward pass with per-step metrics

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
        If `recorder` and `sink` are mismatched (one provided without the other)
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

    runner.prepare()

    metrics = GenerationMetrics()
    prompt = list(prompt_ids)
    token_idx = 0
    step_idx = 0

    async def forward(tokens: list[int]) -> int:
        nonlocal token_idx, step_idx
        started = time.perf_counter()

        if recorder is not None:
            recorder.start_step(
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
            n_tokens=len(tokens),
            duration_s=time.perf_counter() - started,
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
        generated.append(next_token)
        tokens = [next_token]

    return generated, metrics
