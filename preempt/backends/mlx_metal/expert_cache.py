from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx

from preempt.expert_bank.encoding import ExpertBankEncoding, parse_encoding_tag

from preempt.datamodel.identity import ExpertKey
from preempt.datamodel.experts import SerializedExpert


@attrs.define(kw_only=True, frozen=True, eq=False)
class CachedExpert:
    """An cached expert's weight tensors and size in bytes.

    Attributes
    ----------
    tensors : Mapping[str, mx.array]
        Mapping of tensor names to MLX arrays.
    num_bytes : int
        Total size of the expert in bytes.
    """

    tensors: Mapping[str, mx.array] = field()
    num_bytes: int = field()


# TODO read weights concurrently
def decode_serialized_expert(
    expert: SerializedExpert, encoding: ExpertBankEncoding
) -> dict[str, mx.array]:
    """Decodes serialized expert into a dictionary of MLX arrays.

    Parameters
    ----------
    expert : SerializedExpert
        An expert layer's weights raw bytes, tensor specifications, and encoding tag
    encoding : ExpertBankEncoding
        Expected expert encoding

    Returns
    -------
    dict[str, mx.array]
        Decoded MLX arrays mapped to tensor spec names, e.g. "gate_proj.weight"

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
        # itemsize = np.dtype(spec.dtype).itemsize
        # assert spec.num_bytes == count * itemsize
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
        Metadata for validating and decoding serialized experts
    """

    _encoding: ExpertBankEncoding = field()
    _entries: dict[ExpertKey, CachedExpert] = field(factory=dict, init=False)
    _bytes_size: int = field(default=0, init=False)

    def add(self, expert: SerializedExpert) -> None:
        """Decodes serialized expert weights as MLX arrays and maps them to `key`
        in the cache.

        If `key` already exists in the cache, it is replaced and byte accounting
        is updated accordingly.

        Parameters
        ----------
        key : ExpertKey
            Identifier to map the expert weights to. Must match `expert.key`.
        expert : SerializedExpert
            Serialized expert containing byte data, layout specs, and
            encoding info.
        """
        num_bytes = len(expert.data)
        tensors = decode_serialized_expert(expert, self._encoding)
        mx.eval(tuple(tensors.values()))

        if (previous := self._entries.get(expert.key)) is not None:
            self._bytes_size -= previous.num_bytes

        self._entries[expert.key] = CachedExpert(tensors=tensors, num_bytes=num_bytes)
        self._bytes_size += num_bytes

    def evict(self, key: ExpertKey) -> None:
        """Drops an expert from the cache.

        Dropping the internal reference allows MLX reference counting to reclaim
        underlying array memory once active graph evaluations complete.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert to remove from memory

        Raises
        ------
        KeyError
            If `key` is not currently held in memory
        """
        entry = self._entries.pop(key)
        self._bytes_size -= entry.num_bytes

    def size(self) -> int:
        """Returns the total size of the cache in bytes."""
        return self._bytes_size

    def get(
        self, key: ExpertKey
    ) -> Mapping[str, mx.array]:  # TODO rename to weights_for
        """Returns cached expert weights mapped to `key`.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert whose weights to retrieve

        Returns
        -------
        Mapping[str, mx.array]
            Mapping of tensor names to MLX arrays

        Raises
        ------
        KeyError
            If `key` is not currently held in memory
        """
        return self._entries[key].tensors

    def __contains__(self, item) -> bool:
        return item in self._entries
