from __future__ import annotations

from collections.abc import Mapping, Sequence

import attrs
from attrs import field, validators

import numpy as np

from preempt.datamodel.identity import ExpertKey, TensorSpec


@attrs.define(kw_only=True, frozen=True)
class SerializedExpert:
    """An expert layer's weights in bytes and the metadata required to decode them back
    into tensors.

    Attributes
    ----------
    key : ExpertKey
        Unique identifier for the expert layer
    data : bytes
        Concatenated bytes for the expert layer's weights (size = `sum(spec.num_bytes
        for spec in tensor_specs)`)
    encoding : str
        Quantization and dtype encoding tag, e.g. `'mlx-affine-q4-g64-bfloat16'`
    tensor_specs : tuple[TensorSpec, ...]
        Expert weight tensor specifications in the order they appear in `data`
    """

    key: ExpertKey = field()
    data: bytes = field()
    encoding: str = field(validator=validators.min_len(1))
    tensor_specs: tuple[TensorSpec, ...] = field()


def derive_tensor_specs(
    arrays: Mapping[str, np.ndarray], order: Sequence[str]
) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=name,
            dtype=str(arrays[name].dtype),
            shape=tuple(arrays[name].shape),
            num_bytes=arrays[name].nbytes,
        )
        for name in order
    )


def assemble_expert_blob(
    arrays: Mapping[str, np.ndarray], specs: Sequence[TensorSpec]
) -> bytes:
    parts: list[bytes] = []

    for spec in specs:
        array = arrays[spec.name]

        if str(array.dtype) != spec.dtype:
            raise ValueError(
                f"Tensor {spec.name!r}: dtype {array.dtype} != spec {spec.dtype}."
            )
        if tuple(array.shape) != spec.shape:
            raise ValueError(
                f"Tensor {spec.name!r}: shape {tuple(array.shape)} != spec {spec.shape}."
            )
        parts.append(np.ascontiguousarray(array).tobytes())

    return b"".join(parts)
