from __future__ import annotations

import attrs
from attrs import field, validators

from pydantic import BaseModel, ConfigDict, Field


@attrs.define(kw_only=True, frozen=True, slots=True)
class ExpertKey:
    """Unique identifier for one routed expert's weights. Used as the cache and
    storage key throughout the preemptLM engine.

    Attributes
    ----------
    model_fingerprint : str
        Model fingerprint (SHA256 of config.json + safetensors shard manifest)
    block_idx : int
        Index of the MoE layer's parent transformer block
    expert_idx : int
        Routed expert index (within its MoE layer)
    variant : str
        Weight variant name, by default `'all'` for the full fused expert blob
    """

    model_fingerprint: str = field(validator=validators.min_len(1))
    block_idx: int = field(validator=validators.ge(0))
    expert_idx: int = field(validator=validators.ge(0))
    variant: str = field(default="all", validator=validators.min_len(1))


class TensorSpec(BaseModel):  # TODO use attrs
    """Specs for one of an expert's weight tensors, e.g. `gate_proj.weight`.

    Pure data description used by backends to reconstruct tensors from
    raw bytes.

    Attributes
    ----------
    name : str
        Tensor name (in dot notation) relative to an expert, e.g. `gate_proj.weight`
    dtype : str
        NumPy dtype name of the raw bytes, e.g. `uint32`. Note that `bfloat16`
        tensors appear as `uint16` here, which the expert bank's `encoding`
        scalar tag disambiguates.
    shape : tuple[int, ...]
        Tensor shape relative to an expert (no leading batch or expert dims)
    num_bytes : int
        Tensor size in bytes for one expert
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    num_bytes: int = Field(gt=0)
