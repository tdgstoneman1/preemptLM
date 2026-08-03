from __future__ import annotations

from collections.abc import Callable

import attrs
from attrs import field

from preempt.core.protocols.runner import ModelRunner, TokenCodec
from preempt.core.sinks import BaseEventSink
from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


@attrs.define(kw_only=True)
class GenerationResult:
    """One completed generation: its token ids, decoded text, and metrics."""

    token_ids: list[int] = field()
    text: str = field()
    metrics: GenerationMetrics = field()


class InferencePipeline:
    """Facade over one configured generation setup.

    Built at a composition root from injected concretes. When constructed
    with a `recorder`/`sink` pair the pipeline is single-use: the sink owns
    one output file and is closed when `generate` returns.
    """

    def __init__(
        self,
        *,
        runner: ModelRunner,
        tokenizer: TokenCodec,
        max_tokens: int,
        recorder: BaseEventRecorder | None = None,
        sink: BaseEventSink | None = None,
        on_step: Callable[[StepMetrics], None] | None = None,
    ) -> None:
        if (recorder is None) != (sink is None):
            raise ValueError("`recorder` and `sink` must be provided both or neither.")

        self._runner = runner
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._recorder = recorder
        self._sink = sink
        self._on_step = on_step
        self._sink_consumed = False

    async def generate(
        self, prompt: str, *, max_tokens: int | None = None
    ) -> GenerationResult:
        """Encode `prompt`, greedily decode, and decode the result back to text.

        Parameters
        ----------
        prompt : str
            Text prompt, encoded with the injected `TokenCodec`.
        max_tokens : int | None
            Per-call override of the pipeline's token budget, by default None.

        Returns
        -------
        GenerationResult
            Generated token ids, their decoded text, and generation metrics.

        Raises
        ------
        RuntimeError
            If a traced pipeline is reused after its sink has been closed.
        """
        if self._sink is not None and self._sink_consumed:
            raise RuntimeError(
                "A traced pipeline is single-use: its sink already wrote and "
                "closed one output file. Build a new pipeline to trace again."
            )

        prompt_ids = self._tokenizer.encode(prompt)
        budget = max_tokens if max_tokens is not None else self._max_tokens

        if self._sink is not None:
            self._sink_consumed = True

            async with self._sink as sink:
                token_ids, metrics = await generate_greedy(
                    runner=self._runner,
                    prompt_ids=prompt_ids,
                    max_tokens=budget,
                    recorder=self._recorder,
                    sink=sink,
                    on_step=self._on_step,
                )
        else:
            token_ids, metrics = await generate_greedy(
                runner=self._runner,
                prompt_ids=prompt_ids,
                max_tokens=budget,
                on_step=self._on_step,
            )

        return GenerationResult(
            token_ids=token_ids,
            text=self._tokenizer.decode(token_ids),
            metrics=metrics,
        )
