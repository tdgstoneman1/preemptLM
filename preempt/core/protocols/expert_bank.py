from __future__ import annotations

from typing import Protocol, runtime_checkable

import attrs
from attrs import field, validators

from ..identity import ExpertKey, TensorSpec
from ..enums import ReadPriority


# TODO move to datamodel/
@attrs.define(kw_only=True, frozen=True)
class ExpertPayload:
    """Raw weight bytes for one expert and the metadata required to decode them back
    into weight tensors.

    :Note: due to limitations with NumPy, `tensor_specs` stores `bfloat16` as `uint16`.
    For `bfloat16` tensors, this means `encoding` is the *only* correct record of the
    tensors' original dtype.

    Attributes
    ----------
    key : ExpertKey
        Identifier for the expert the weights belong to
    data : bytes
        Concatenated weight bytes for one expert (size equals `sum(spec.num_bytes
        for spec in tensor_specs)`)
    encoding : str
        Quantization format tag, e.g. `'mlx-affine-q4-g64-bf16'`.
    tensor_specs : tuple[TensorSpec, ...]
        Expert weight tensor specifications in the order they appear in `data`

    """

    key: ExpertKey = field()
    data: bytes = field()
    encoding: str = field(validator=validators.min_len(1))
    tensor_specs: tuple[TensorSpec, ...] = field()


@runtime_checkable
class IExpertBank(Protocol):
    """Read-only source of expert blobs keyed by `ExpertKey`"""

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload: ...
