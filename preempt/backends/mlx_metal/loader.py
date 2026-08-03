from __future__ import annotations

from typing import cast

import attrs
from attrs import field

import mlx.nn as nn
from mlx_lm import load

from preempt.core.protocols.runner import TokenCodec


@attrs.define(kw_only=True, frozen=True, eq=False)
class MlxLoadedModel:
    """An `mlx_lm` model paired with its tokenizer.

    `eq=False` because `nn.Module` has no meaningful equality and comparing
    weights would be ruinous.
    """

    model: nn.Module = field()
    tokenizer: TokenCodec = field()


def load_mlx_model(model_id: str) -> MlxLoadedModel:
    """Load an MLX model + tokenizer via `mlx_lm`.

    Parameters
    ----------
    model_id : str
        Hugging Face repo id or local path accepted by `mlx_lm.load`.

    Returns
    -------
    MlxLoadedModel
        The loaded module and its tokenizer, typed as a `TokenCodec`.

    Notes
    -----
    `mlx_lm`'s `TokenizerWrapper` forwards `encode`/`decode` through an
    unannotated `__getattr__`; the cast to `TokenCodec` restores real
    signatures rather than silencing the diagnostic.
    """
    model, raw_tokenizer = load(model_id)  # type: ignore[misc]
    return MlxLoadedModel(model=model, tokenizer=cast(TokenCodec, raw_tokenizer))
