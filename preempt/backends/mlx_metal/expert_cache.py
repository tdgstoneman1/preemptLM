from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx

from preempt.expert_bank.encoding import PayloadEncoding, parse_payload_encoding_tag

from preempt.datamodel.identity import ExpertKey, TensorSpec
from preempt.datamodel.experts import ExpertPayload

from .constants import BIT_VIEWED_STORAGE_DTYPE, BIT_VIEWED_SCALARS


@attrs.define(kw_only=True, frozen=True, eq=False)
class _CachedExpert:
    """In-memory MLX tensors and payload byte count for a single expert.

    Attributes
    ----------
    tensors : Mapping[str, mx.array]
        Mapping of tensor names to in-memory MLX arrays.
    num_bytes : int
        Total size of the expert payload in bytes.
    """

    tensors: Mapping[str, mx.array] = field()
    num_bytes: int = field()


def _view_dtype_for(
    spec: TensorSpec, encoding: PayloadEncoding
) -> mx.Dtype | None:  # TODO rename
    """Determines target MLX dtype when reinterpreting bit-viewed tensor storage.

    Parameters
    ----------
    spec : TensorSpec
        Specification of the stored tensor
    encoding : PayloadEncoding
        Payload encoding metadata for the expert bank

    Returns
    -------
    mx.Dtype | None
        Target MLX dtype for array reinterpretation, or None if no bit-view
        reinterpretation is required.

    Raises
    ------
    ValueError
        If bit-viewed storage dtype does not map to a recognized target dtype in
        the payload encoding.
    """
    if spec.dtype != BIT_VIEWED_STORAGE_DTYPE:
        return None

    view = BIT_VIEWED_SCALARS.get(encoding.scalar)
    if view is None:
        # TODO rewrite slop message
        raise ValueError(
            f"Tensor {spec.name!r} is stored as {BIT_VIEWED_STORAGE_DTYPE!r}, "
            f"but the payload encoding names scalar {encoding.scalar!r}, which "
            "is representable in numpy and would not have been bit-viewed. "
            "Refusing to guess the real dtype."
        )

    return view


# TODO rename to decode_serialized_expert
def decode_expert_tensors(
    payload: ExpertPayload, encoding: PayloadEncoding
) -> dict[str, mx.array]:
    """Decodes serialized expert payload into a dictionary of MLX arrays.

    Parameters
    ----------
    payload : ExpertPayload
        Payload containing raw bytes, tensor specifications, and encoding tag
    encoding : PayloadEncoding
        Expected expert payload encoding

    Returns
    -------
    dict[str, mx.array]
        Decoded MLX arrays mapped to tensor spec names, e.g. "gate_proj.weight"

    Raises
    ------
    ValueError
        If payload encoding does not match expected encoding
    ValueError
        If tensor byte counts do not match shapes and dtypes
    ValueError
        If total decoded bytes do not match payload length
    ValueError
        If a bit-viewed dtype cannot be resolved.
    """
    payload_encoding = parse_payload_encoding_tag(payload.encoding)
    if payload_encoding != encoding:
        raise ValueError(
            f"Payload encoding mismatch for {payload.key!r}. Expected {encoding!r}, "
            f"but got {payload.encoding!r} ({payload_encoding!r})."
        )

    tensors: dict[str, mx.array] = {}
    offset = 0

    for spec in payload.tensor_specs:
        count = math.prod(spec.shape)
        itemsize = np.dtype(spec.dtype).itemsize

        if count * itemsize != spec.num_bytes:
            raise ValueError(
                f"Size mismatch for tensor spec {spec.name!r}: tensor shape "
                f"{spec.shape} requires {count * itemsize} bytes, but its tensor "
                f"spec records {spec.num_bytes} bytes."
            )

        # Tensor data contiguous within payload blob
        raw = np.frombuffer(
            payload.data, dtype=spec.dtype, count=count, offset=offset
        ).reshape(spec.shape)
        dtype = _view_dtype_for(spec, encoding)

        tensor = mx.array(raw)
        tensor = tensor.view(dtype) if dtype is not None else tensor

        tensors[spec.name] = tensor
        offset += spec.num_bytes

    if offset != len(payload.data):
        raise ValueError(
            f"Blob for {payload.key!r} is {len(payload.data)} bytes, but its "
            f"tensor specs account for {offset}."
        )

    return tensors


@attrs.define(kw_only=True, eq=False)
class MlxExpertCache:
    """`IExpertCache` interface for MLX.

    Attributes
    ----------
    encoding : PayloadEncoding
        Payload encoding metadata for validating and decoding serialized experts
    """

    _encoding: PayloadEncoding = field()
    _resident: dict[ExpertKey, _CachedExpert] = field(factory=dict, init=False)
    _resident_bytes: int = field(default=0, init=False)

    # TODO get rid of `key` and get it from `payload`
    def install(self, key: ExpertKey, payload: ExpertPayload) -> None:
        """Decodes serialized expert payload into MLX arrays and maps them to `key`
        in the cache.

        If `key`already exists in the cache, its corresponding weights are replaced
        and byte accounting is updated accordingly.

        Parameters
        ----------
        key : ExpertKey
            Identifier to map the expert weights to. Must match `payload.key`.
        payload : ExpertPayload
            Serialized expert payload containing byte data, layout specs, and
            encoding metadata.

        Raises
        ------
        ValueError
            If `payload.key` does not match `key`, or if payload decoding fails.
        """
        if payload.key != key:
            raise ValueError(
                f"Payload carries {payload.key!r} but was installed under {key!r}."
            )

        tensors = decode_expert_tensors(payload, self._encoding)
        mx.eval(tuple(tensors.values()))
        num_bytes = len(payload.data)

        previous = self._resident.get(key)
        if previous is not None:
            self._resident_bytes -= previous.num_bytes

        self._resident[key] = _CachedExpert(tensors=tensors, num_bytes=num_bytes)
        self._resident_bytes += num_bytes

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
        entry = self._resident.pop(key)
        self._resident_bytes -= entry.num_bytes

    def is_resident(self, key: ExpertKey) -> bool:
        return key in self._resident

    def size(self) -> int:
        """Returns the total size of the cache in bytes."""
        return self._resident_bytes

    def tensors(
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
        return self._resident[key].tensors
