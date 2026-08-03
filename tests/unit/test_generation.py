import asyncio
from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import StepMetrics
from preempt.engine.recorder import BaseEventRecorder


class ScriptedRunner:
    """Returns queued token ids; records every `step` call's inputs."""

    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        self.prepared = False
        self.calls: list[list[int]] = []

    def prepare(self) -> None:
        self.prepared = True

    def step(self, tokens: Sequence[int]) -> int:
        assert self.prepared, "step() before prepare()"
        self.calls.append(list(tokens))
        return self.outputs.pop(0)


class SpyRecorder(BaseEventRecorder):
    """Buffers one fake capture per step; counts flushes."""

    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )
        self.step_contexts: list[TraceStepContext] = []

    def start_step(self, step_context: TraceStepContext) -> None:
        super().start_step(step_context)
        self.step_contexts.append(step_context)

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseEventSink) -> int:
        self.end_step()
        return 3  # pretend 3 records per step


class NullSink(BaseEventSink):
    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...
    async def aclose(self) -> None: ...


async def test_prefill_then_single_token_decode_steps() -> None:
    runner = ScriptedRunner([11, 12, 13])
    tokens, metrics = await generate_greedy(
        runner=runner, prompt_ids=[1, 2, 3], max_tokens=3
    )
    assert tokens == [11, 12, 13]
    assert runner.calls == [[1, 2, 3], [11], [12]]
    assert metrics.tokens_forwarded == 5
    assert [s.n_tokens for s in metrics.steps] == [3, 1, 1]


async def test_traced_run_stamps_step_contexts_and_counts_records() -> None:
    runner = ScriptedRunner([9, 8])
    recorder = SpyRecorder()
    tokens, metrics = await generate_greedy(
        runner=runner,
        prompt_ids=[5, 6],
        max_tokens=2,
        recorder=recorder,
        sink=NullSink(),
    )
    assert metrics.records_written == 6
    prefill, decode = recorder.step_contexts
    assert (prefill.token_idx, prefill.token_id) == (0, None)  # multi-token: id null
    assert (decode.token_idx, decode.token_id) == (2, 9)


async def test_recorder_without_sink_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        await generate_greedy(
            runner=ScriptedRunner([1]),
            prompt_ids=[1],
            max_tokens=1,
            recorder=SpyRecorder(),
        )


async def test_on_step_callback_sees_every_step() -> None:
    seen: list[StepMetrics] = []
    await generate_greedy(
        runner=ScriptedRunner([1, 2]),
        prompt_ids=[3],
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

    await generate_greedy(runner=LoopAsserter([1]), prompt_ids=[1], max_tokens=1)
    assert loop is asyncio.get_running_loop()
