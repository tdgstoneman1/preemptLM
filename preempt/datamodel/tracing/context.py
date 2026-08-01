import attrs
from attrs import field, validators

import pyarrow as pa

from ..arrow import arrow_metadata


@attrs.define(kw_only=True, frozen=True)
class TraceRunContext:
    run_id: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    instrumentation_version: str = field(
        metadata=arrow_metadata(pa.string()),
        default="0.1.0",
        validator=validators.min_len(1),
    )
    model_fingerprint: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    model_architecture: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    model_revision: str | None = field(
        metadata=arrow_metadata(pa.string(), nullable=True), default=None
    )


# TODO Expand later after adding batching/prompt-prefill tracing.
@attrs.define(kw_only=True, frozen=True)
class TraceStepContext:
    """Position metadata for single outer model invocation."""

    sequence_id: int = field(
        metadata=arrow_metadata(pa.int32()), validator=validators.ge(0)
    )
    token_idx: int = field(
        metadata=arrow_metadata(pa.int32()), validator=validators.ge(0)
    )
    token_id: int | None = field(
        metadata=arrow_metadata(pa.int64(), nullable=True),
        validator=validators.optional(validators.ge(0)),
        default=None,
    )
