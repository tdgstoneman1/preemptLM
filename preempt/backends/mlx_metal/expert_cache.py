from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx
from mlx.utils import tree_flatten

from preempt.expert_bank.encoding import ExpertBankEncoding, parse_encoding_tag
from preempt.expert_bank.blob import SerializedExpert

from preempt.datamodel.identity import ExpertKey


@attrs.define(kw_only=True, frozen=True, eq=False)
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
        An expert layer's weight bytes, tensor specifications, and encoding tag
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
            f"Encoding mismatch for {expert.key!r}. Expected {encoding!r}, "
            f"but got {expert.encoding!r} ({parsed!r})."
        )

    tensors: dict[str, mx.array] = {}
    offset = 0

    for spec in expert.tensor_specs:
        count = math.prod(spec.shape)
        tensors[spec.name] = mx.asarray(
            np.frombuffer(expert.data, dtype=spec.dtype, count=count, offset=offset)
        ).reshape(spec.shape)
        offset += tensors[spec.name].nbytes

    if offset != len(expert.data):
        raise ValueError(
            f"Blob for {expert.key!r} is {len(expert.data)} bytes, but its "
            f"tensor spec accounts for {offset} bytes."
        )

    return tensors


@attrs.define(kw_only=True, eq=False)
class MlxExpertCache:
    """`IExpertCache` interface for MLX.

    Attributes
    ----------
    encoding : ExpertBankEncoding
        Encoding information for validating and decoding serialized experts
    """

    encoding: ExpertBankEncoding = field()
    _entries: dict[ExpertKey, CachedExpert] = field(factory=dict, init=False)
    _bytes_size: int = field(default=0, init=False)

    def add(self, expert: SerializedExpert) -> None:
        """Decodes serialized expert weights as MLX arrays and maps them to `key`
        in the cache.

        If the cache already contains an entry for `key`, it is replaced and byte
        accounting is updated accordingly.

        Parameters
        ----------
        key : ExpertKey
            Expert udentifier to key its weights to
        expert : SerializedExpert
            An expert layer's byte data, layout specs, and encoding info
        """
        num_bytes = len(expert.data)
        weight_map = decode_serialized_expert(
            expert, self.encoding
        )  # ! Use expert.encoding instead?
        mx.eval(tree_flatten(weight_map))

        if (previous := self._entries.get(expert.key)) is not None:
            self._bytes_size -= previous.num_bytes

        self._entries[expert.key] = CachedExpert(
            weight_map=weight_map, num_bytes=num_bytes
        )
        self._bytes_size += num_bytes

    def evict(self, key: ExpertKey) -> None:
        """Drops an expert's weights from the cache.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert to remove

        Raises
        ------
        KeyError
            If `key` does not exist in the cache.
        """
        entry = self._entries.pop(key)
        self._bytes_size -= entry.num_bytes

    def size(self) -> int:
        """Returns the cache's memory footprint in bytes."""
        return self._bytes_size

    def get(self, key: ExpertKey) -> Mapping[str, mx.array]:
        """Returns cached expert mapped to `key`.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the cached expert

        Returns
        -------
        Mapping[str, mx.array]
            Mapping of weight names to MLX arrays

        Raises
        ------
        KeyError
            If `key` does not exist in the cache.
        """
        return self._entries[key].weight_map

    def __contains__(self, item) -> bool:
        return item in self._entries
