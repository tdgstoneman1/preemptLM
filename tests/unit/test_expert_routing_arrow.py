from datetime import UTC, datetime

import pyarrow as pa
import pytest

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_routing import (
    EXPERT_ROUTING_EVENT_TYPE,
    EXPERT_ROUTING_SCHEMA_VERSION,
    EventMetadata,
    ExpertRoutingEvent,
    LayerIdentifiers,
)


def make_event(*, gate_logits: tuple[float, ...] | None = None) -> ExpertRoutingEvent:
    return ExpertRoutingEvent(
        run_context=TraceRunContext(
            run_id="r-1", model_id="m", model_architecture="arch"
        ),
        step_context=TraceStepContext(sequence_id=0, token_idx=3, token_id=7),
        event_metadata=EventMetadata(event_idx=0, timestamp=datetime.now(UTC)),
        layer_identifiers=LayerIdentifiers(
            layer_path="model.layers.0.mlp", layer_class="Blk", layer_idx=0
        ),
        expert_ids=(4, 9),
        expert_weights=(0.7, 0.3),
        gate_logits=gate_logits,
    )


def test_schema_and_record_share_flat_field_names() -> None:
    schema = ExpertRoutingEvent.arrow_schema()
    record = make_event().as_arrow_record()
    assert set(record) == set(schema.names)


def test_schema_carries_event_type_and_version_metadata() -> None:
    metadata = ExpertRoutingEvent.arrow_schema().metadata
    assert metadata[b"preempt.event_type"] == EXPERT_ROUTING_EVENT_TYPE.encode()
    assert metadata[b"preempt.schema_version"] == str(EXPERT_ROUTING_SCHEMA_VERSION).encode()


def test_record_batch_round_trips_through_arrow() -> None:
    schema = ExpertRoutingEvent.arrow_schema()
    rows = [make_event().as_arrow_record(), make_event(gate_logits=(0.1, 0.9)).as_arrow_record()]
    batch = pa.RecordBatch.from_pylist(rows, schema)
    assert batch.num_rows == 2
    assert batch.to_pylist()[0]["expert_ids"] == [4, 9]
    assert batch.to_pylist()[0]["gate_logits"] is None


def test_mismatched_ids_and_weights_rejected() -> None:
    event = make_event()
    with pytest.raises(ValueError, match="same number"):
        ExpertRoutingEvent(
            run_context=event.run_context,
            step_context=event.step_context,
            event_metadata=event.event_metadata,
            layer_identifiers=event.layer_identifiers,
            expert_ids=(1, 2, 3),
            expert_weights=(0.5, 0.5),
            gate_logits=None,
        )
