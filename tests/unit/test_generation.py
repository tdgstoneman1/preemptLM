from typing import Any
from collections.abc import Mapping, Sequence

import pytest

import asyncio

from preempt.engine.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import StepMetrics
from preempt.engine.recorder import BaseEventRecorder


class ScriptedRunner:
    """Returns queued token ids; records every `step` call's inputs."""

    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[int]] = []

    def step(self, tokens: Sequence[int]) -> int:
        self.calls.append(list(tokens))
        return self.outputs.pop(0)


class ContextRunner:
    """Next token is a deterministic function of the full accumulated context.

    Models a KV cache: because the return depends only on every token fed so
    far and not on how those tokens were split across `step` calls, chunking a
    prompt cannot change the generated sequence — the property this fake exists
    to prove.
    """

    def __init__(self) -> None:
        self.context: list[int] = []
        self.calls: list[list[int]] = []

    def step(self, tokens: Sequence[int]) -> int:
        self.calls.append(list(tokens))
        self.context.extend(tokens)
        return (sum(self.context) + len(self.context)) % 50


class SpyRecorder(BaseEventRecorder):
    """Buffers one fake capture per step; counts flushes."""

    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )
        self.step_contexts: list[TraceStepContext] = []

    def start_trace(self, step_context: TraceStepContext) -> None:
        super().start_trace(step_context)
        self.step_contexts.append(step_context)

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseEventSink) -> int:
        self.stop_trace()
        return 3  # pretend 3 records per step


class NullSink(BaseEventSink):
    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...
    async def aclose(self) -> None: ...


async def test_prefill_then_single_token_decode_steps() -> None:
    runner = ScriptedRunner([11, 12, 13])
    tokens, metrics = await generate_greedy(
        runner=runner, prompt_ids=[1, 2, 3], eos_token_ids=None, max_tokens=3
    )
    assert tokens == [11, 12, 13]
    assert runner.calls == [[1, 2, 3], [11], [12]]
    assert metrics.tokens_forwarded == 5
    assert [s.n_tokens for s in metrics.steps] == [3, 1, 1]


async def test_prefill_chunks_reconstruct_prompt_in_order() -> None:
    runner = ScriptedRunner([10, 20, 30, 40])
    tokens, _ = await generate_greedy(
        runner=runner,
        prompt_ids=[1, 2, 3, 4, 5],
        eos_token_ids=None,
        max_tokens=2,
        prefill_chunk_size=2,
    )
    # Three prefill chunks then one decode step; the chunk slices, concatenated
    # in order, are exactly the prompt with nothing dropped or duplicated.
    prefill_calls = runner.calls[:3]
    assert prefill_calls == [[1, 2], [3, 4], [5]]
    assert [t for chunk in prefill_calls for t in chunk] == [1, 2, 3, 4, 5]
    # The first generated token is the final prefill chunk's `step` return.
    assert tokens[0] == 30
    assert runner.calls[3] == [30]
    assert tokens == [30, 40]


async def test_chunked_and_unchunked_prefill_match() -> None:
    prompt = [7, 3, 9, 1, 4, 8, 2]

    unchunked = ContextRunner()
    chunked = ContextRunner()

    tokens_unchunked, _ = await generate_greedy(
        runner=unchunked,
        prompt_ids=prompt,
        eos_token_ids=None,
        max_tokens=5,
        prefill_chunk_size=64,
    )
    tokens_chunked, _ = await generate_greedy(
        runner=chunked,
        prompt_ids=prompt,
        eos_token_ids=None,
        max_tokens=5,
        prefill_chunk_size=2,
    )

    assert unchunked.calls == [prompt, *[[t] for t in tokens_unchunked[:-1]]]
    assert tokens_chunked == tokens_unchunked


async def test_traced_run_stamps_step_contexts_and_counts_records() -> None:
    runner = ScriptedRunner([9, 8])
    recorder = SpyRecorder()
    tokens, metrics = await generate_greedy(
        runner=runner,
        prompt_ids=[5, 6],
        eos_token_ids=None,
        max_tokens=2,
        recorder=recorder,
        sink=NullSink(),
    )
    assert metrics.records_written == 6
    prefill, decode = recorder.step_contexts
    assert (prefill.token_idx, prefill.token_id) == (0, None)  # multi-token: id null
    assert (decode.token_idx, decode.token_id) == (2, 9)


async def test_traced_chunked_prefill_advances_token_idx_per_chunk() -> None:
    runner = ScriptedRunner([10, 20, 30])
    recorder = SpyRecorder()
    _, metrics = await generate_greedy(
        runner=runner,
        prompt_ids=[1, 2, 3, 4],
        eos_token_ids=None,
        max_tokens=2,
        prefill_chunk_size=2,
        recorder=recorder,
        sink=NullSink(),
    )
    # Two multi-token prefill chunks then one single-token decode: token_idx
    # advances by each chunk's length, token_id is null for the multi-token
    # chunks and set only for the single-token decode step.
    stamped = [(c.token_idx, c.token_id) for c in recorder.step_contexts]
    assert stamped == [(0, None), (2, None), (4, 20)]
    # SpyRecorder yields 3 records per forward; three forwards were bracketed.
    assert metrics.records_written == 9


async def test_recorder_without_sink_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        await generate_greedy(
            runner=ScriptedRunner([1]),
            prompt_ids=[1],
            eos_token_ids=None,
            max_tokens=1,
            recorder=SpyRecorder(),
        )


async def test_on_step_callback_sees_every_step() -> None:
    seen: list[StepMetrics] = []
    await generate_greedy(
        runner=ScriptedRunner([1, 2]),
        prompt_ids=[3],
        eos_token_ids=None,
        max_tokens=2,
        on_step=seen.append,
    )
    assert [s.step_idx for s in seen] == [0, 1]


async def test_runner_runs_off_event_loop() -> None:
    loop = asyncio.get_running_loop()

    class LoopAsserter(ScriptedRunner):
        def step(self, tokens: Sequence[int]) -> int:
            with pytest.raises(RuntimeError):
                asyncio.get_running_loop()  # no loop in the worker thread
            return super().step(tokens)

    await generate_greedy(
        runner=LoopAsserter([1]), prompt_ids=[1], eos_token_ids=None, max_tokens=1
    )
    assert loop is asyncio.get_running_loop()
