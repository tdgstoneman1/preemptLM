from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, NoReturn

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache

import numpy as np

from preempt.engine.metrics import StepMetrics


class MlxModelRunner:
    """Mlx implementation of `IModelRunner` protocol.

    Runs forward pass on a dedicated MLX stream to avoid interference with
    other MLX ops. Each generation step explicitly evaluates the sampled token
    and KV cache state to prevent computation graph buildup.

    :Note: Unlike `mlx_lm`, this uses synchronous `mx.eval` (vs. `mx.async_eval`).
    """

    model: nn.Module

    max_tokens: int | float
    prefill_chunk_size: int
    max_kv_size: int | None
    kv_cache: list[Any]
    stream: mx.ThreadLocalStream

    sampler: Callable[[mx.array], mx.array]
    on_step: Callable[[StepMetrics], None] | None

    def __init__(
        self,
        model: nn.Module,
        stream: mx.ThreadLocalStream,
        max_tokens: Optional[int] = None,
        prefill_chunk_size: int = 2048,
        max_kv_size: Optional[int] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        on_step: Optional[Callable[[StepMetrics], None]] = None,
    ) -> None:
        self.model = model
        self.stream = stream

        self.max_tokens = max_tokens if max_tokens is not None else np.inf
        self.prefill_chunk_size = prefill_chunk_size
        self.max_kv_size = max_kv_size
        self.kv_cache = make_prompt_cache(self.model, self.max_kv_size)

        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.on_step = on_step

    def step(self, tokens: Sequence[int]) -> int:
        with mx.stream(self.stream):
            input_ids = mx.asarray(list(tokens), dtype=mx.int32)
            logits = self.model(input_ids[None], cache=self.kv_cache)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            next_token = self.sampler(logprobs)

            mx.eval([c.state for c in self.kv_cache])

        mx.clear_cache()  # TODO optimize
        return int(next_token.item())  # type: ignore

    def _step(self, input_tokens: mx.array) -> NoReturn:  # ! CURRENTLY NOT WORKING
        raise NotImplementedError()

        with mx.stream(generation_stream):
            logits = self.model(input_tokens[None], cache=self.kv_cache)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)

            return self.sampler(logprobs)

    def prefill(self, prompt_tokens: mx.array) -> NoReturn:  # ! CURRENTLY NOT WORKING
        raise NotImplementedError()

        with mx.stream(generation_stream):
            total_prompt_tokens = len(prompt_tokens)
            prompt_processed_tokens = 0

            while total_prompt_tokens - prompt_processed_tokens > 1:
                remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
                n_to_process = min(self.prefill_chunk_size, remaining)

                self.model(prompt_tokens[:n_to_process][None], cache=self.kv_cache)

                mx.eval([c.state for c in self.kv_cache])

                prompt_processed_tokens += n_to_process
                prompt_tokens = prompt_tokens[n_to_process:]

                mx.clear_cache()

            y = self._step(input_tokens=prompt_tokens)

        return y

    def generate(self, tokens: list[int]) -> NoReturn:  # ! CURRENTLY NOT WORKING
        raise NotImplementedError()

        input_ids = mx.asarray(tokens, dtype=mx.int32)
        y = self.prefill(input_ids)
        mx.async_eval(y)  # schedule y computation

        n = 0
        while True:
            if n != self.max_tokens:
                next_y = self._step(y)
                mx.async_eval(next_y)  # schedule next_y computation
            if n == 0:
                mx.eval(y)  # block thread until y computed
            if n == self.max_tokens:
                break

            yield int(y.item())  # .item() does sync eval # type: ignore

            if n % 256 == 0:
                mx.clear_cache()

            y = next_y
            n += 1
