from datetime import UTC, datetime

import pyarrow as pa
import pytest

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_selection import (
    EXPERT_SELECTION_TRACE_SCHEMA_VERSION,
    TraceMetadata,
    ExpertSelectionTrace,
    MoEBlockIdentifiers,
)


def make_event(*, gate_logits: tuple[float, ...] | None = None) -> ExpertSelectionTrace:
    return ExpertSelectionTrace(
        run_context=TraceRunContext(
            run_id="r-1", model_id="m", model_architecture="arch"
        ),
        step_context=TraceStepContext(sequence_id=0, token_idx=3, token_id=7),
        trace_metadata=TraceMetadata(event_idx=0, timestamp=datetime.now(UTC)),
        moe_block_identifiers=MoEBlockIdentifiers(
            path="model.layers.0.mlp", module_class="Blk", transformer_block_idx=0
        ),
        expert_ids=(4, 9),
        softmax_weights=(0.7, 0.3),
        gate_logits=gate_logits,
    )


def test_schema_and_record_share_flat_field_names() -> None:
    schema = ExpertSelectionTrace.arrow_schema()
    record = make_event().as_arrow_record()
    assert set(record) == set(schema.names)


def test_schema_carries_event_type_and_version_metadata() -> None:
    metadata = ExpertSelectionTrace.arrow_schema().metadata
    assert (
        metadata[b"preempt.schema_version"]
        == str(EXPERT_SELECTION_TRACE_SCHEMA_VERSION).encode()
    )


def test_record_batch_round_trips_through_arrow() -> None:
    schema = ExpertSelectionTrace.arrow_schema()
    rows = [
        make_event().as_arrow_record(),
        make_event(gate_logits=(0.1, 0.9)).as_arrow_record(),
    ]
    batch = pa.RecordBatch.from_pylist(rows, schema)
    assert batch.num_rows == 2
    assert batch.to_pylist()[0]["expert_ids"] == [4, 9]
    assert batch.to_pylist()[0]["gate_logits"] is None


def test_mismatched_ids_and_weights_rejected() -> None:
    event = make_event()
    with pytest.raises(ValueError, match="same number"):
        ExpertSelectionTrace(
            run_context=event.run_context,
            step_context=event.step_context,
            trace_metadata=event.trace_metadata,
            moe_block_identifiers=event.moe_block_identifiers,
            expert_ids=(1, 2, 3),
            softmax_weights=(0.5, 0.5),
            gate_logits=None,
        )
