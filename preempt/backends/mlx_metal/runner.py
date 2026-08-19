from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache


class MlxModelRunner:
    """Mlx implementation of `IModelRunner` protocol.

    Runs forward passes on a dedicated MLX stream to avoid interference with
    other MLX ops. Each generation step explicitly evaluates the sampled token
    and KV cache state to prevent computation graph buildup, then clears MLX
    memory cache.

    **Note:** Unlike `mlx_lm`, this uses synchronous `mx.eval` (not `mx.async_eval`)
    and manually flattens cache state to handle optional `None` slots.
    """

    model: nn.Module
    _cache: list[Any] | None

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._cache = None

    def prepare(self) -> None:
        """Resets prompt cache. Call before the first `step` in a sequence."""
        self._cache = make_prompt_cache(self._model)

    # TODO tidy up docstring
    def step(self, tokens: Sequence[int]) -> int:
        """Runs one forward pass over `tokens` and greedily decodes and returns the
        next token id.

        Parameters
        ----------
        tokens : Sequence[int]
            Input sequence of token ids (entire prompt for prefill, 1 token
            per decode step thereafter)

        Returns
        -------
        int
            Output token id (argmax of the output logits)

        Raises
        ------
        RuntimeError
            If `prepare()` has not been called yet
        """
        if self._cache is None:
            raise RuntimeError(
                "`prepare()` must be called before running generation step."
            )

        with mx.stream(generation_stream):
            input_ids = mx.array([list(tokens)], dtype=mx.int32)
            logits = self._model(input_ids, cache=self._cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            mx.eval(next_token)

            # `state` is a list per cache entry and may hold `None` slots, flatten
            #  and only keep real arrays rather than pass tree directly to `mx.eval`
            state_arrays = [
                value
                for _, value in tree_flatten([entry.state for entry in self._cache])
                if isinstance(value, mx.array)
            ]
            if state_arrays:
                mx.eval(state_arrays)

        mx.clear_cache()
        return int(next_token.item())
