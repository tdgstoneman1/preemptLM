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
    """Pure tensor data description used by backends to decode model parameters from
    bytes.

    Attributes
    ----------
    name : str
        A dotted path name relative to the parameter's parent layer, e.g.  `gate_proj.weight`
    dtype : str
        String representation of the numpy dtype
    shape : tuple[int, ...]
        Parameter shape relative to its parent layer (no leading batch or expert dims)
    num_bytes : int
        The parameter's size in bytes
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    num_bytes: int = Field(gt=0)
