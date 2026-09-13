from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import mlx.core as mx
from mlx.utils import tree_flatten

import numpy as np

from preempt.datamodel.expert_bank.encoding import (
    ExpertBankEncoding,
    parse_encoding_tag,
)
from preempt.datamodel.expert_bank.blob import SerializedExpert

from preempt.engine.expert_io.cache import BaseExpertCache
from preempt.engine.expert_io.cache_manager import ExpertCacheManager


@attrs.define(kw_only=True, frozen=True, eq=False, slots=True)
class CachedExpert:
    """A cached expert's weights and size in bytes.

    Attributes
    ----------
    weight_map : Mapping[str, mx.array]
        Mapping of weight names to MLX arrays
    num_bytes : int
        Total size of the expert's weights in bytes
    """

    weight_map: Mapping[str, mx.array] = field()
    num_bytes: int = field()


# TODO read weights concurrently
def decode_serialized_expert(
    expert: SerializedExpert, encoding: ExpertBankEncoding
) -> dict[str, mx.array]:
    """Decodes serialized expert into a dictionary of MLX arrays.

    Parameters
    ----------
    expert : SerializedExpert
        An serialized expert's weights in bytes, tensor specs, and encoding tag
    encoding : ExpertBankEncoding
        Expected expert encoding

    Returns
    -------
    dict[str, mx.array]
        Decoded MLX arrays mapped to tensor spec names, e.g. 'gate_proj.weight'

    Raises
    ------
    ValueError
        If expert encoding does not match expected encoding
    ValueError
        If tensor byte counts do not match shapes and dtypes
    ValueError
        If total decoded bytes do not match expert blob
    ValueError
        If a bit-viewed dtype cannot be resolved.
    """
    if (parsed := parse_encoding_tag(expert.encoding)) != encoding:
        raise ValueError(
            f"Encoding mismatch for {expert.idx}. Expected {encoding!r}, "
            f"but got {expert.encoding!r} ({parsed!r})."
        )

    weights: dict[str, mx.array] = {}
    offset = 0

    for spec in expert.tensor_specs:
        count = math.prod(spec.shape)
        weights[spec.name] = mx.asarray(
            np.frombuffer(expert.data, dtype=spec.dtype, count=count, offset=offset)
        ).reshape(spec.shape)

        offset += weights[spec.name].nbytes

    if offset != len(expert.data):
        raise ValueError(
            f"Blob for {expert.idx} is {len(expert.data)} bytes, but its "
            f"tensor spec accounts for {offset} bytes."
        )

    return weights


@attrs.define(kw_only=True, slots=True)
class MlxExpertCache(BaseExpertCache):
    """`IExpertCache` interface for MLX.

    Attributes
    ----------
    manager : ExpertCacheManager
        Manager that tracks experts in memory and handles eviction decisions.
    encoding : ExpertBankEncoding
        Encoding information for validating and decoding serialized experts
    """

    encoding: ExpertBankEncoding = field()
    manager: ExpertCacheManager

    _entries: dict[int, CachedExpert] = field(factory=dict, init=False)
    _bytes_size: int = field(default=0, init=False)

    def __contains__(self, item) -> bool:
        return item in self._entries

    def add(self, expert: SerializedExpert) -> None:
        """Decodes serialized expert weights as MLX arrays and maps them to `key`
        in the cache.

        If the cache already contains an entry for `key`, it is replaced and byte
        accounting is updated accordingly.

        Parameters
        ----------
        expert : SerializedExpert
            An expert layer's byte data, layout specs, and encoding info
        """
        num_bytes = len(expert.data)
        weight_map = decode_serialized_expert(
            expert, self.encoding
        )  # ! Use expert.encoding instead?
        mx.eval(tree_flatten(weight_map))

        if (previous := self._entries.get(expert.idx)) is not None:
            self._bytes_size -= previous.num_bytes

        self._entries[expert.idx] = CachedExpert(
            weight_map=weight_map, num_bytes=num_bytes
        )
        self._bytes_size += num_bytes

    def get(self, expert_idx: int) -> Mapping[str, mx.array]:
        """Returns cached expert mapped to `key`.

        Parameters
        ----------
        expert_idx : int
            Unique index denoting the cached expert's position within its model

        Returns
        -------
        Mapping[str, mx.array]
            Mapping of weight names to MLX arrays

        Raises
        ------
        KeyError
            If `key` does not exist in the cache.
        """
        entry = self._entries[expert_idx].weight_map
        self.manager.mark_entry_safe_to_evict(expert_idx)

        return entry

    def evict(self, expert_idx: int) -> None:
        """Drops an expert's weights from the cache.

        Parameters
        ----------
        expert_idx : int
            Unique index denoting the cached expert's position within its model

        Raises
        ------
        KeyError
            If `key` does not exist in the cache.
        """
        entry = self._entries.pop(expert_idx)
        self._bytes_size -= entry.num_bytes

    def size(self) -> int:
        """Returns the cache's memory footprint in bytes."""
        return self._bytes_size
