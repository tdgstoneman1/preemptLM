from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from preempt.storage.manifest import TensorSpec


def derive_tensor_specs(
    arrays: Mapping[str, np.ndarray], order: Sequence[str]
) -> tuple[TensorSpec, ...]:
    """Build `TensorSpec`s for one expert's arrays, in the declared blob order.

    Parameters
    ----------
    arrays : Mapping[str, np.ndarray]
        One expert's tensors, keyed by name relative to that expert.
    order : Sequence[str]
        Tensor names in the order they will appear inside a blob.

    Returns
    -------
    tuple[TensorSpec, ...]
        One spec per name in `order`.

    Raises
    ------
    KeyError
        If `order` names a tensor absent from `arrays`.
    """
    return tuple(
        TensorSpec(
            name=name,
            dtype=str(arrays[name].dtype),
            shape=tuple(arrays[name].shape),
            nbytes=arrays[name].nbytes,
        )
        for name in order
    )


def assemble_expert_blob(
    arrays: Mapping[str, np.ndarray], specs: Sequence[TensorSpec]
) -> bytes:
    """Concatenate one expert's arrays into a blob, validating against `specs`.

    Parameters
    ----------
    arrays : Mapping[str, np.ndarray]
        One expert's tensors, keyed by name relative to that expert.
    specs : Sequence[TensorSpec]
        Per-expert tensor layout, in blob order.

    Returns
    -------
    bytes
        The fused blob, exactly `sum(spec.nbytes for spec in specs)` long.

    Raises
    ------
    KeyError
        If a spec names a tensor absent from `arrays`.
    ValueError
        If any array's dtype or shape drifts from its spec — a silent drift
        here would corrupt every read of the resulting store.
    """
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
