from typing import Final

from datetime import datetime, UTC

import attrs
from attrs import field, validators

import pyarrow as pa

from ..arrow import arrow_metadata, arrow_args_for, arrow_schema_field, serialize_utc
from .context import TraceRunContext, TraceStepContext

from preempt.utils.attrs_utils import recurse_attrs_instance_fields

EXPERT_ROUTING_SCHEMA_VERSION: Final[int] = 1
EXPERT_ROUTING_EVENT_TYPE: Final[str] = "expert_routing_event"


@attrs.define(kw_only=True, frozen=True)
class EventMetadata:
    event_idx: int = field(
        metadata=arrow_metadata(pa.int64()), validator=validators.ge(0)
    )
    timestamp: datetime = field(
        metadata=arrow_metadata(pa.timestamp("us", tz="UTC"), serializer=serialize_utc),
        factory=lambda: datetime.now(UTC),
    )

    def __attrs_post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("`timestamp` must be timezone-aware")

        if self.timestamp.utcoffset() is None:
            raise ValueError("`timestamp` must have a valid UTC offset")


@attrs.define(kw_only=True, frozen=True)
class LayerIdentifiers:
    layer_path: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    layer_class: str = field(
        metadata=arrow_metadata(pa.string()), validator=validators.min_len(1)
    )
    layer_idx: int = field(
        metadata=arrow_metadata(pa.int32()), validator=validators.ge(0)
    )


@attrs.define(kw_only=True)
class ExpertRoutingEvent:
    """MoE block expert routing for a single token.

    **Note:** `expert_weights` and `expert_ids` correspond to the *final* expert
    routing after applying any normalization to score logits.
    `gate_logits` corresponds to full-size logits array (used for training)
    and should be the same size as *total* number of experts, *not* top-k
    """

    schema_version: int = field(
        metadata=arrow_metadata(pa.int16()),
        default=EXPERT_ROUTING_SCHEMA_VERSION,
        init=False,
        validator=validators.ge(1),
    )
    run_context: TraceRunContext = field()
    step_context: TraceStepContext
    event_metadata: EventMetadata
    layer_identifiers: LayerIdentifiers

    expert_ids: tuple[int, ...] = field(
        metadata=arrow_metadata(
            pa.list_(pa.field("item", pa.int32(), nullable=False)),
            serializer=list,
        ),
        validator=validators.min_len(1),
        converter=tuple,
    )
    expert_weights: tuple[float, ...] = field(
        metadata=arrow_metadata(
            pa.list_(pa.field("item", pa.float32(), nullable=False)),
            serializer=list,
        ),
        validator=validators.min_len(1),
        converter=lambda arr: tuple(float(x) for x in arr),
    )
    # Gate logits optional since they only matter for training
    gate_logits: tuple[float, ...] | None = field(
        metadata=arrow_metadata(
            pa.list_(pa.field("item", pa.float32(), nullable=True)),
            serializer=list,
        ),
        validator=validators.optional(validators.min_len(1)),
        converter=lambda arr: tuple(float(x) for x in arr) if arr else None,
    )

    def __attrs_post_init__(self) -> None:
        if len(self.expert_ids) != len(self.expert_weights):
            raise ValueError(
                "`expert_ids` and `expert_weights` must contain the same number of values; "
                f"got `{len(self.expert_ids)=}` and `{len(self.expert_weights)=}`"
            )
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("`expert_ids` must only contain unique values.")

        if any(expert_id < 0 for expert_id in self.expert_ids):
            raise ValueError("`expert_ids` must be non-negative.")

        if any(weight < 0.0 for weight in self.expert_weights):
            raise ValueError("`expert_weights` must be non-negative.")

        if not any(weight > 0.0 for weight in self.expert_weights):
            raise ValueError("At least one expert weight must be positive.")

    def as_arrow_record(self) -> dict[str, pa.Field]:
        serialized: dict[str, pa.Field] = {}

        for f, value in recurse_attrs_instance_fields(attrs.fields(ExpertRoutingEvent)):
            arrow_args = arrow_args_for(f, cls=ExpertRoutingEvent)

            if arrow_args.serializer is not None:
                serialized[f.name] = arrow_args.serializer(value)

        return serialized

    @staticmethod
    def arrow_schema() -> pa.Schema:
        arrow_fields: list[pa.Field] = []

        for f in attrs.fields(ExpertRoutingEvent):
            if attrs.has(f.type):  # Unpack nested fields
                arrow_fields.extend(
                    arrow_schema_field(f_, f) for f_ in attrs.fields(f.type)
                )
            else:
                arrow_fields.append(arrow_schema_field(f, ExpertRoutingEvent))

        # TODO define base arrow metadata model
        return pa.schema(
            arrow_fields,
            metadata={
                b"preempt.event_type": str(EXPERT_ROUTING_EVENT_TYPE).encode(),
                b"preempt.schema_version": str(EXPERT_ROUTING_SCHEMA_VERSION).encode(),
            },
        )
