from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from preempt.datamodel.identity import TensorSpec


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
