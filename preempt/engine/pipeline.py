from __future__ import annotations

from typing import Optional
from collections.abc import Callable

import attrs
from attrs import field

from preempt.core.protocols.runner import IModelRunner, ITokenCodec
from preempt.core.sinks import BaseEventSink

from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


@attrs.define(kw_only=True)
class GenerationResult:
    """Token ids, decoded text, and metrics for a completed generation"""

    token_ids: list[int] = field()
    text: str = field()
    metrics: GenerationMetrics = field()


class GenerationPipeline:
    """Encodes a prompt, runs greedy generation, and decodes the result.

    Wraps a model runner, tokenizer, and optional tracing sink into a single
    `generate(...)` call.

    :Note: If a recorder and sink are provided, the pipeline
    can only be used once since the sink writes one output file and closes upon
    completion.
    """

    runner: IModelRunner
    tokenizer: ITokenCodec
    max_tokens: int
    prefill_chunk_size: int
    recorder: BaseEventRecorder | None
    sink: BaseEventSink | None
    on_step: Callable[[StepMetrics], None] | None

    _sink_consumed: bool

    def __init__(
        self,
        runner: IModelRunner,
        tokenizer: ITokenCodec,
        max_tokens: int,
        prefill_chunk_size: int = 512,
        recorder: Optional[BaseEventRecorder] = None,
        sink: Optional[BaseEventSink] = None,
        on_step: Optional[Callable[[StepMetrics], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        runner : IModelRunner
            Model runner implementing the `IModelRunner` interface
        tokenizer : ITokenCodec
            Tokenizer implementing the `ITokenCodec` interface
        max_tokens : int
            Max total tokens to generate including the first prefill output (can be
            overridden per call). Must be >= 1
        prefill_chunk_size : int, optional
            Max tokens per prefill chunk. Controls activation memory and enables
            chunk-local expert reuse (no effect on correctness), by default 512
        recorder : Optional[BaseEventRecorder], optional
            Optional recorder for tracing. If provided, `sink` must also be given,
            by default None
        sink : Optional[BaseEventSink], optional
            Optional sink for writing traced events. If provided, `recorder` must
            also be given, by default None
        on_step : Optional[Callable[[StepMetrics], None]], optional
            Optional callback invoked after each forward pass with per-step metrics,
            by default None

        Raises
        ------
        ValueError
            If either `recorder` or `sink` is provided without the other.
        """
        if (recorder is None) != (sink is None):
            raise ValueError(
                "`recorder` and `sink` must both be provided or both be `None`, "
                f"but got `{type(recorder)=}` and `{type(sink)=}`."
            )

        self.runner = runner
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.recorder = recorder
        self.sink = sink
        self.on_step = on_step

        self._sink_consumed = False

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        on_step: Optional[Callable[[StepMetrics], None]] = None,
    ) -> GenerationResult:
        """Runs greedy generation on text prompt and returns token ids, decoded text,
        and generation metrics.

        Parameters
        ----------
        prompt : str
            Input text to generate from
        max_tokens : int | None, default None
            Overrides the pipeline's default token budget for this call.

        Returns
        -------
        GenerationResult
            Generated token ids, their decoded text, and aggregated generation metrics.

        Raises
        ------
        RuntimeError
            If called again on a pipeline that was created with a recorder and sink
            (traced pipelines are single-use).
        """
        if self.sink is not None and self._sink_consumed:
            raise RuntimeError(
                "A traced pipeline is single-use, and this pipeline's sink has already"
                "written and closed its output file. Build a new pipeline to trace "
                "another generation run."
            )

        prompt_ids = self.tokenizer.encode(prompt)
        budget = max_tokens if isinstance(max_tokens, int) else self.max_tokens

        if self.sink is not None:
            self._sink_consumed = True

            async with self.sink as sink:
                token_ids, metrics = await generate_greedy(
                    runner=self.runner,
                    prompt_ids=prompt_ids,
                    eos_token_ids=self.tokenizer.eos_token_ids,
                    max_tokens=budget,
                    prefill_chunk_size=self.prefill_chunk_size,
                    recorder=self.recorder,
                    sink=sink,
                    on_step=on_step if on_step is not None else self.on_step,
                )
        else:
            token_ids, metrics = await generate_greedy(
                runner=self.runner,
                prompt_ids=prompt_ids,
                eos_token_ids=self.tokenizer.eos_token_ids,
                max_tokens=budget,
                prefill_chunk_size=self.prefill_chunk_size,
                on_step=on_step if on_step is not None else self.on_step,
            )

        return GenerationResult(
            token_ids=token_ids,
            text=self.tokenizer.decode(token_ids),
            metrics=metrics,
        )
