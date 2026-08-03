from __future__ import annotations

import attrs
from attrs import field, validators


@attrs.define(kw_only=True, frozen=True)
class ExpertKey:
    """Fully qualified identity of one routed expert's weights.

    The cache, store, and residency key everywhere in the engine; bare
    integer expert ids never cross a module boundary. `variant` names the
    weight variant a key refers to — `"all"` is the full fused expert blob.
    """

    model_fingerprint: str = field(validator=validators.min_len(1))
    layer_idx: int = field(validator=validators.ge(0))
    expert_idx: int = field(validator=validators.ge(0))
    variant: str = field(default="all", validator=validators.min_len(1))
