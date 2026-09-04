from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from preempt.engine.sinks import BaseTraceSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.pipeline import GenerationPipeline
from preempt.engine.recorder import BaseTraceRecorder


class ScriptedRunner:
    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)

    def step(self, tokens: Sequence[int]) -> int:
        return self.outputs.pop(0)


class StubCodec:
    eos_token_ids: set[int] | None = {1}

    @property
    def think_start_id(self) -> int:
        return 101

    @property
    def think_end_id(self) -> int:
        return 102

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) for t in tokens)


class SpyRecorder(BaseTraceRecorder):
    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseTraceSink) -> int:
        self.stop_trace()
        return 1


class SpySink(BaseTraceSink):
    def __init__(self) -> None:
        self.closed = False

    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...

    async def aclose(self) -> None:
        self.closed = True


async def test_generate_encodes_decodes_and_counts() -> None:
    runner = ScriptedRunner([65, 66])
    pipeline = GenerationPipeline(runner=runner, tokenizer=StubCodec(), max_tokens=2)
    result = await pipeline.generate("hi")

    assert result.token_ids == [65, 66]
    assert result.text == "AB"
    assert result.metrics.tokens_forwarded == 3  # 2 prompt + 1 decode


async def test_traced_generate_closes_sink_and_is_single_use() -> None:
    sink = SpySink()
    runner = ScriptedRunner([65, 66])
    pipeline = GenerationPipeline(
        runner=runner,
        tokenizer=StubCodec(),
        max_tokens=1,
        recorder=SpyRecorder(),
        sink=sink,
    )
    result = await pipeline.generate("hi")
    assert result.metrics.records_written == 1
    assert sink.closed
    with pytest.raises(RuntimeError, match="single"):
        await pipeline.generate("again")


async def test_max_tokens_override_and_recorder_sink_pairing() -> None:
    pipeline = GenerationPipeline(
        runner=ScriptedRunner([1, 2, 3]), tokenizer=StubCodec(), max_tokens=1
    )
    result = await pipeline.generate("xyz", max_tokens=3)
    assert len(result.token_ids) == 3

    with pytest.raises(ValueError, match="both"):
        GenerationPipeline(
            runner=ScriptedRunner([1]),
            tokenizer=StubCodec(),
            max_tokens=1,
            recorder=SpyRecorder(),
        )
