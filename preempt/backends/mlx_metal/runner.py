from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache


class MlxModelRunner:
    """`ModelRunner` for MLX: one forward pass + greedy argmax per `step`.

    Mirrors the memory discipline of `mlx_lm.generate.generate_step`: compute
    runs on the generation stream and the cache state is evaluated explicitly.
    That eval is load-bearing — this model's linear-attention layers return
    recurrent state as a *sibling* of the layer output, so evaluating only the
    sampled token leaves the state an unevaluated graph and every subsequent
    step re-derives the whole chain from step 0.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._cache: list[Any] | None = None

    def prepare(self) -> None:
        """Reset the prompt cache. Call before the first `step` of a sequence."""
        self._cache = make_prompt_cache(self._model)

    def step(self, tokens: Sequence[int]) -> int:
        """Run one forward pass over `tokens`; return the greedy next token id.

        Parameters
        ----------
        tokens : Sequence[int]
            Token ids to push through the model — the whole prompt on the
            prefill step, one token per decode step thereafter.

        Returns
        -------
        int
            The argmax token id of the final position's logits.

        Raises
        ------
        RuntimeError
            If `prepare()` has not been called.
        """
        if self._cache is None:
            raise RuntimeError("`prepare()` must be called before `step()`.")

        with mx.stream(generation_stream):
            input_ids = mx.array([list(tokens)], dtype=mx.int32)
            logits = self._model(input_ids, cache=self._cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            # `state` is a list per cache entry and may hold `None` slots, so
            # flatten and keep only real arrays instead of passing the tree
            # straight to `mx.eval`.
            state_arrays = [
                value
                for _, value in tree_flatten([entry.state for entry in self._cache])
                if isinstance(value, mx.array)
            ]
            if state_arrays:
                mx.eval(state_arrays)

        mx.clear_cache()
        return int(next_token.item())
