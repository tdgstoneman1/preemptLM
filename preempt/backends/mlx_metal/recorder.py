from __future__ import annotations

from typing import Any
from collections.abc import Generator

from datetime import datetime, UTC

import attrs
from attrs import field

import mlx.core as mx

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_routing import (
    ExpertRoutingEvent,
    EventMetadata,
    LayerIdentifiers,
)

from preempt.core.sinks import BaseEventSink

from preempt.engine.recorder import BaseEventRecorder


@attrs.define(frozen=True)
class _PendingExpertRoutingEvent:
    expert_ids: mx.array = field()
    expert_weights: mx.array = field()
    gate_logits: mx.array | None = field()
    step_context: TraceStepContext = field()
    event_metadata: EventMetadata = field()
    layer_identifiers: LayerIdentifiers = field()


class MlxExpertRoutingRecorder(BaseEventRecorder):
    _buffer: list[_PendingExpertRoutingEvent]
    _emitted_event_idx: int

    def __init__(self, run_context: TraceRunContext) -> None:
        super().__init__(run_context)

        # `_event_idx` counts `capture()` calls, i.e. one per instrumented layer
        # per step. A single capture fans out to one event per (batch, token) at
        # flush time, so it cannot number the emitted events: a multi-token
        # prefill emits far more events than there were captures, and the next
        # step would restart inside the range already written. Track what has
        # actually been emitted separately.
        self._emitted_event_idx = 0

    def capture(
        self,
        *,
        layer_path: str,
        layer_class: str,
        layer_idx: int,
        expert_ids: mx.array,
        expert_weights: mx.array,
        gate_logits: mx.array | None,
    ) -> None:
        if self._step_context is None:
            raise RuntimeError(
                "`capture()` called without an active `TraceStepContext`, "
                "call `start_step()` before recording events."
            )
        event_metadata = EventMetadata(
            event_idx=self._event_idx, timestamp=datetime.now(UTC)
        )
        layer_identifiers = LayerIdentifiers(
            layer_path=layer_path, layer_class=layer_class, layer_idx=layer_idx
        )
        self._buffer.append(
            _PendingExpertRoutingEvent(
                expert_ids=expert_ids,
                expert_weights=expert_weights,
                gate_logits=gate_logits,
                step_context=self._step_context,
                event_metadata=event_metadata,
                layer_identifiers=layer_identifiers,
            )
        )
        self._event_idx += 1

    async def flush(self, sink: BaseEventSink) -> int:
        if self._step_context is None:
            raise RuntimeError("No capture step currently active.")
        try:
            return await self._to_sink(sink)

        finally:
            self.end_step()

    async def _to_sink(self, sink: BaseEventSink) -> int:
        if not self._buffer:
            return 0

        self._eval_arrays_in_buffer(self._buffer)
        n_flushed = 0

        for rec in self._materialize_buffered_events(self._emitted_event_idx):
            await sink.write(rec.as_arrow_record())
            n_flushed += 1

        self._emitted_event_idx += n_flushed

        return n_flushed

    def _materialize_buffered_events(
        self, current_event_idx: int
    ) -> Generator[ExpertRoutingEvent, Any, None]:

        for record in self._buffer:
            expert_ids = record.expert_ids.tolist()
            expert_weights = record.expert_weights.tolist()

            gate_logits = (
                record.gate_logits.tolist() if record.gate_logits is not None else None
            )

            for batch_idx, (batch_ids, batch_weights) in enumerate(
                zip(expert_ids, expert_weights, strict=True)
            ):
                for offset, (ids, weights) in enumerate(
                    zip(batch_ids, batch_weights, strict=True)
                ):
                    yield self._to_final_record(
                        record,
                        batch_idx,
                        offset,
                        current_event_idx,
                        ids,
                        weights,
                        (
                            gate_logits[batch_idx][offset]
                            if gate_logits is not None
                            else None
                        ),
                    )
                    current_event_idx += 1

    def _to_final_record(
        self,
        record: _PendingExpertRoutingEvent,
        batch_idx: int,
        offset: int,
        current_event_idx: int,
        ids: list[int],
        weights: list[float],
        logits: list[float] | None,
    ) -> ExpertRoutingEvent:
        step_context = TraceStepContext(
            sequence_id=record.step_context.sequence_id + batch_idx,
            token_idx=record.step_context.token_idx + offset,
            token_id=record.step_context.token_id,
        )
        event_metadata = EventMetadata(
            event_idx=current_event_idx,
            timestamp=record.event_metadata.timestamp,
        )
        return ExpertRoutingEvent(
            expert_ids=tuple(ids),
            expert_weights=tuple(weights),
            gate_logits=tuple(logits) if logits is not None else None,
            run_context=self._run_context,
            step_context=step_context,
            event_metadata=event_metadata,
            layer_identifiers=record.layer_identifiers,
        )

    def _eval_arrays_in_buffer(self, buffer: list[_PendingExpertRoutingEvent]) -> None:
        to_evaluate = [
            arr
            for item in buffer
            for arr in (item.expert_ids, item.expert_weights, item.gate_logits)
            if arr is not None
        ]
        mx.eval(*to_evaluate)
