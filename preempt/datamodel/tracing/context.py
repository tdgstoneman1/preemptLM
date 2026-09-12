from typing import Optional, Self
from datetime import datetime, UTC
import secrets

import attrs
from attrs import field, validators

import pyarrow as pa

from preempt.core.constants import TIMESTAMP_FMT, RUN_ID_TEMPLATE

from .arrow import arrow_metadata


def generate_run_id(prefix: str, *, timestamp: Optional[datetime] = None) -> str:
    """Generates a unique run id formatted as `<prefix>-<timestamp>-<8 hex chars>`."""
    if not prefix:
        raise ValueError("`prefix` must be a non-empty string.")

    timestamp = timestamp or datetime.now(UTC)
    return RUN_ID_TEMPLATE.substitute(
        prefix=prefix,
        timestamp=timestamp.strftime(TIMESTAMP_FMT),
        hex=secrets.token_hex(4),
    )


# TODO remove model_revision
# TODO import default instrumentation_version from dedicated module
@attrs.define(kw_only=True, frozen=True)
class TraceRunContext:
    """Identifiers and metadata for a traced generation run.

    Includes the run's unique id, instrumentation version, and the traced model
    id, architecture, and revision number.
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

        This is the preferred way to create a new `TraceRunContext` since it
        guarantees `run_id` will be unique.
        """
        return cls(
            run_id=generate_run_id(run_id_prefix),
            model_id=model_id,
            model_architecture=model_architecture,
            model_revision=model_revision,
            instrumentation_version=instrumentation_version,
        )


# TODO Add batching/prompt-prefill tracing.
@attrs.define(kw_only=True, frozen=True)
class TraceStepContext:
    """Identifiers for one forward pass within a generation run."""

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
