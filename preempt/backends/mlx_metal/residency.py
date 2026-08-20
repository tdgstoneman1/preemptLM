from __future__ import annotations

from collections.abc import Mapping

import attrs
from attrs import field

import math

import numpy as np

import mlx.core as mx

from preempt.core.encoding import PayloadEncoding, parse_payload_encoding_tag
from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.protocols.expert_bank import ExpertPayload

from .constants import BIT_VIEWED_STORAGE_DTYPE, BIT_VIEWED_SCALARS

# TODO rename module


@attrs.define(kw_only=True, frozen=True, eq=False)
class _ResidentExpert:  # TODO rename
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
        raise ValueError(
            f"Tensor {spec.name!r} is stored as {BIT_VIEWED_STORAGE_DTYPE!r}, "
            f"but the payload encoding names scalar {encoding.scalar!r}, which "
            "is representable in numpy and would not have been bit-viewed. "
            "Refusing to guess the real dtype."
        )

    return view


# TODO double check that 'expert tensors' <-> weight tensors
def decode_expert_tensors(
    payload: ExpertPayload, encoding: PayloadEncoding
) -> dict[str, mx.array]:
    """Decodes serialized expert payload bytes from bank into a dictionary of
    MLX arrays.

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
                f"Size mismatch for tensor spec {spec.name!r}. Tensor shape "
                f"{spec.shape} of {spec.dtype} requires {count * itemsize} "
                f"bytes, but tensor spec records {spec.num_bytes} bytes."
            )

        # Tensor data is contiguous within payload blob
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
class MlxExpertResidency:  # TODO rename to `MlxExpertManager`
    """In-memory expert manager for MLX models. Decodes and holds expert weight
    tensors.

    Attributes
    ----------
    encoding : PayloadEncoding
        Payload encoding metadata for validating and decoding incoming expert
        payloads.
    """

    _encoding: PayloadEncoding = field()
    _resident: dict[ExpertKey, _ResidentExpert] = field(factory=dict, init=False)
    _resident_bytes: int = field(default=0, init=False)

    def install(self, key: ExpertKey, payload: ExpertPayload) -> None:
        """Decodes expert payload into in-memory MLX arrays and stores them under
        `key`.

        If `key` is already loaded, its tensors are replaced and byte accounting
        is updated accordingly.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert to store. Must match `payload.key`.
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
        num_bytes = len(payload.data)

        previous = self._resident.get(key)
        if previous is not None:
            self._resident_bytes -= previous.num_bytes

        self._resident[key] = _ResidentExpert(tensors=tensors, num_bytes=num_bytes)
        self._resident_bytes += num_bytes

    def evict(self, key: ExpertKey) -> None:
        """Removes an expert's tensors from memory.

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
        """Checks whether an expert's tensors are currently loaded in memory.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert to check

        Returns
        -------
        bool
            True if the expert is currently loaded in memory, False otherwise
        """
        return key in self._resident

    def resident_bytes(self) -> int:  # TODO rename for clarity
        """Returns the total byte size of all experts currently held in memory.

        Returns
        -------
        int
            Sum of payload byte sizes for all in-memory experts
        """
        return self._resident_bytes

    def tensors(self, key: ExpertKey) -> Mapping[str, mx.array]:
        """Retrieves in-memory MLX arrays for a loaded expert held under `key`.

        Parameters
        ----------
        key : ExpertKey
            Identifier for the expert whose tensors to retrieve

        Returns
        -------
        Mapping[str, mx.array]
            Mapping of tensor specification names to in-memory MLX arrays

        Raises
        ------
        KeyError
            If `key` is not currently held in memory
        """
        return self._resident[key].tensors
