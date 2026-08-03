from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.pipeline import InferencePipeline
from preempt.engine.recorder import BaseEventRecorder


class ScriptedRunner:
    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        self.prepare_calls = 0

    def prepare(self) -> None:
        self.prepare_calls += 1

    def step(self, tokens: Sequence[int]) -> int:
        return self.outputs.pop(0)


class StubCodec:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) for t in tokens)


class SpyRecorder(BaseEventRecorder):
    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseEventSink) -> int:
        self.end_step()
        return 1


class SpySink(BaseEventSink):
    def __init__(self) -> None:
        self.closed = False

    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...

    async def aclose(self) -> None:
        self.closed = True


async def test_generate_encodes_decodes_and_counts() -> None:
    runner = ScriptedRunner([65, 66])
    pipeline = InferencePipeline(runner=runner, tokenizer=StubCodec(), max_tokens=2)
    result = await pipeline.generate("hi")
    assert result.token_ids == [65, 66]
    assert result.text == "AB"
    assert result.metrics.tokens_forwarded == 3  # 2 prompt + 1 decode
    assert runner.prepare_calls == 1


async def test_untraced_pipeline_prepares_once_per_generate() -> None:
    # An untraced pipeline is reusable, and every generate must start from a
    # fresh cache -- one `prepare()` per call, never a shared one.
    runner = ScriptedRunner([65, 66, 67])
    pipeline = InferencePipeline(runner=runner, tokenizer=StubCodec(), max_tokens=1)

    await pipeline.generate("hi")
    assert runner.prepare_calls == 1

    await pipeline.generate("yo")
    assert runner.prepare_calls == 2


async def test_traced_generate_closes_sink_and_is_single_use() -> None:
    sink = SpySink()
    runner = ScriptedRunner([65, 66])
    pipeline = InferencePipeline(
        runner=runner,
        tokenizer=StubCodec(),
        max_tokens=1,
        recorder=SpyRecorder(),
        sink=sink,
    )
    result = await pipeline.generate("hi")
    assert result.metrics.records_written == 1
    assert sink.closed
    assert runner.prepare_calls == 1
    with pytest.raises(RuntimeError, match="single"):
        await pipeline.generate("again")
    assert runner.prepare_calls == 1


async def test_max_tokens_override_and_recorder_sink_pairing() -> None:
    pipeline = InferencePipeline(
        runner=ScriptedRunner([1, 2, 3]), tokenizer=StubCodec(), max_tokens=1
    )
    result = await pipeline.generate("xyz", max_tokens=3)
    assert len(result.token_ids) == 3

    with pytest.raises(ValueError, match="both"):
        InferencePipeline(
            runner=ScriptedRunner([1]),
            tokenizer=StubCodec(),
            max_tokens=1,
            recorder=SpyRecorder(),
        )
