from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx

from preempt.expert_bank.encoding import ExpertBankEncoding, parse_encoding_tag

from preempt.datamodel.identity import ExpertKey, TensorSpec
from preempt.datamodel.experts import SerializedExpert

from .constants import BIT_VIEWED_STORAGE_DTYPE, BIT_VIEWED_SCALARS


@attrs.define(kw_only=True, frozen=True, eq=False)
class _CachedExpert:
    """In-memory MLX tensors and byte count for a serialized expert.

    Attributes
    ----------
    tensors : Mapping[str, mx.array]
        Mapping of tensor names to in-memory MLX arrays.
    num_bytes : int
        Total size of the expert in bytes.
    """

    tensors: Mapping[str, mx.array] = field()
    num_bytes: int = field()


def _view_dtype_for(  # TODO get rid of this
    spec: TensorSpec, encoding: ExpertBankEncoding
) -> mx.Dtype | None:
    """Determines target MLX dtype when reinterpreting bit-viewed tensor storage.

    Parameters
    ----------
    spec : TensorSpec
        Specification of the stored tensor
    encoding : ExpertBankEncoding
        Encoding metadata for the expert bank

    Returns
    -------
    mx.Dtype | None
        Target MLX dtype for array reinterpretation, or None if no bit-view
        reinterpretation is required.

    Raises
    ------
    ValueError
        If bit-viewed storage dtype does not map to a recognized target dtype in
        the encoding.
    """
    if spec.dtype != BIT_VIEWED_STORAGE_DTYPE:
        return None

    view = BIT_VIEWED_SCALARS.get(encoding.scalar)  # TODO use ml_dtypes bfloat16
    if view is None:
        # TODO rewrite slop message
        raise ValueError(
            f"Tensor {spec.name!r} is stored as {BIT_VIEWED_STORAGE_DTYPE!r}, "
            f"but the encoding names scalar {encoding.scalar!r}, which "
            "is representable in numpy and would not have been bit-viewed. "
            "Refusing to guess the real dtype."
        )

    return view


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
        itemsize = np.dtype(spec.dtype).itemsize

        if count * itemsize != spec.num_bytes:
            raise ValueError(
                f"Size mismatch for tensor spec {spec.name!r}: tensor shape "
                f"{spec.shape} requires {count * itemsize} bytes, but its tensor "
                f"spec records {spec.num_bytes} bytes."
            )

        # Tensor data contiguous within expert blob
        raw = np.frombuffer(
            expert.data, dtype=spec.dtype, count=count, offset=offset
        ).reshape(spec.shape)
        dtype = _view_dtype_for(spec, encoding)

        tensor = mx.array(raw)
        tensor = tensor.view(dtype) if dtype is not None else tensor

        tensors[spec.name] = tensor
        offset += spec.num_bytes

    if offset != len(expert.data):
        raise ValueError(
            f"Blob for {expert.key!r} is {len(expert.data)} bytes, but its "
            f"tensor specs account for {offset}."
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
    _entries: dict[ExpertKey, _CachedExpert] = field(factory=dict, init=False)
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

        self._entries[expert.key] = _CachedExpert(tensors=tensors, num_bytes=num_bytes)
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
