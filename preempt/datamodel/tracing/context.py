from typing import Self

from datetime import datetime, UTC

import secrets

import attrs
from attrs import field, validators

import pyarrow as pa

from preempt.core.constants import RUN_ID_TIMESTAMP_FMT

from .arrow import arrow_metadata


def generate_run_id(prefix: str, *, timestamp_fn: datetime | None = None) -> str:
    """Generates a unique `run_id` formatted as `<prefix>-<timestamp>-<8 hex chars>`."""
    if not prefix:
        raise ValueError("`prefix` must be a non-empty string.")

    timestamp = timestamp_fn if timestamp_fn is not None else datetime.now(UTC)

    return f"{prefix}-{timestamp.strftime(RUN_ID_TIMESTAMP_FMT)}-{secrets.token_hex(4)}"


# TODO docstring
@attrs.define(kw_only=True, frozen=True)
class TraceRunContext:
    """Identifies one instrumentation run and the model it traced.

    Carries the run's unique identifier, HF-style model id, model architecture,
    revision number, and instrumentation version.
    """

    run_id: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    instrumentation_version: str = field(
        metadata=arrow_metadata(pa.string()),
        default="0.1.0",
        validator=validators.min_len(1),
    )
    model_id: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    model_architecture: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    model_revision: str | None = field(
        metadata=arrow_metadata(pa.string(), nullable=True), default=None
    )

    @classmethod
    def with_generated_run_id(
        cls,
        *,
        run_id_prefix: str,
        model_id: str,
        model_architecture: str,
        model_revision: str | None = None,
        instrumentation_version: str = "0.1.0",
    ) -> Self:
        """Instantiates `TraceRunContext` with a unique `run_id` generated from
        `run_id_prefix`.

        In general, this is the preferred way to create a new `TraceRunContext` as
        it guarantees `run_id` will be a unique value (rather than instantiating
        directly with `TraceRunContext(run_id=..., ...)`).
        """
        return cls(
            run_id=generate_run_id(run_id_prefix),
            model_id=model_id,
            model_architecture=model_architecture,
            model_revision=model_revision,
            instrumentation_version=instrumentation_version,
        )


# TODO Expand later after adding batching/prompt-prefill tracing.
@attrs.define(kw_only=True, frozen=True)
class TraceStepContext:
    """Position of one forward pass within a sequence."""

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
