from __future__ import annotations

from typing import cast

import attrs
from attrs import field

import mlx.nn as nn

from mlx_lm import load

from preempt.core.protocols.runner import ITokenCodec


@attrs.define(kw_only=True, frozen=True, eq=False)
class MlxLoadedModel:
    """An `mlx_lm` model-tokenizer pair.

    Attributes
    ----------
    model : nn.Module
        The `mlx_lm` model
    tokenizer : ITokenCodec
        Tokenizer for encoding text and decoding generated token
        ids. Must implement `ITokenCodec` protocol.
    """

    model: nn.Module = field()
    tokenizer: ITokenCodec = field()


def load_mlx_model(model_id: str, *, lazy: bool = False) -> MlxLoadedModel:
    """Loads an MLX model and tokenizer via `mlx_lm`.

    Parameters
    ----------
    model_id : str
        Hugging Face repo id or local path accepted by `mlx_lm.load(...)`
    lazy : bool
        If False, model weights are evaluated eagerly at load time. If True, `mlx_lm`
        skips its internal weight evaluation, and all weights remain unevaluated
        mmap-backed arrays. This is required for model instrumentation when streaming
        expert weights from disk as it defers weight materialization, allowing expert
        weights to be stripped so only the dense backbone materializes in memory.

    Returns
    -------
    MlxLoadedModel
        The loaded `mlx_lm` model and its tokenizer
    """
    model, tokenizer = load(model_id, lazy=lazy)  # type: ignore
    return MlxLoadedModel(model=model, tokenizer=cast(ITokenCodec, tokenizer))
