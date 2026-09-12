from __future__ import annotations

from typing import Optional
from collections.abc import Callable

from functools import partial

import attrs
from attrs import field

from preempt.core.protocols import IModelRunner, ITokenizer

from .sinks import BaseTraceSink
from .generation import generate_greedy
from .metrics import GenerationMetrics, StepMetrics
from .recorder import BaseTraceRecorder


@attrs.define(kw_only=True)
class GenerationResult:
    """Token ids, decoded text, and metrics for a completed generation"""

    token_ids: list[int] = field()
    text: str = field()
    metrics: GenerationMetrics = field()


class GenerationPipeline:
    """Configures and runs text generation loop with aggregated metrics.

    :Note: If a recorder and sink are provided, the pipeline
    can only be used once since the sink writes one output file and closes upon
    completion.
    """

    runner: IModelRunner
    tokenizer: ITokenizer
    max_tokens: int
    prefill_chunk_size: int
    recorder: BaseTraceRecorder | None
    sink: BaseTraceSink | None
    on_step: Callable[[StepMetrics], None] | None

    _sink_consumed: bool

    def __init__(
        self,
        runner: IModelRunner,
        tokenizer: ITokenizer,
        max_tokens: int,
        prefill_chunk_size: int = 512,
        recorder: Optional[BaseTraceRecorder] = None,
        sink: Optional[BaseTraceSink] = None,
        on_step: Optional[Callable[[StepMetrics], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        runner : IModelRunner
            Concrete implementation of the `IModelRunner` interface
        tokenizer : ITokenizer
            Tokenizer implementing the `ITokenizer` interface
        max_tokens : int
            Max total tokens to generate. can be overridden per call). Must be
            >= 1
        prefill_chunk_size : int
            Max tokens per prefill chunk. Controls activation memory and enables
            chunk-local expert reuse (no effect on correctness), by default 512
        recorder : Optional[BaseTraceRecorder]
            Optional recorder for tracing. If provided, `sink` must also be given,
            by default None
        sink : Optional[BaseTraceSink]
            Optional sink for persisting expert selection traces. If provided, `recorder`
            must also be given, by default None
        on_step : Optional[Callable[[StepMetrics], None]]
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
        max_tokens : Optional[int]
            Overrides the pipeline's default token budget for this call, by default None

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

        generate_fn = partial(
            generate_greedy,
            runner=self.runner,
            prompt_ids=prompt_ids,
            eos_token_ids=self.tokenizer.eos_token_ids,
            max_tokens=budget,
            prefill_chunk_size=self.prefill_chunk_size,
            on_step=on_step if on_step is not None else self.on_step,
        )
        if self.sink is None:
            token_ids, metrics = await generate_fn()
        else:
            async with self.sink as sink:
                token_ids, metrics = await generate_fn(
                    recorder=self.recorder, sink=sink
                )
            self._sink_consumed = True

        return GenerationResult(
            token_ids=token_ids,
            text=self.tokenizer.decode(token_ids),
            metrics=metrics,
        )
