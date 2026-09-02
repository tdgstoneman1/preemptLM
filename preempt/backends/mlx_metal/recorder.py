from __future__ import annotations

from typing import Any, Optional
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
from preempt.engine.sinks import BaseEventSink
from preempt.engine.recorder import BaseEventRecorder


@attrs.define(frozen=True)
class _BufferedEvent:
    """Internal buffer record for an unevaluated router selection.

    Stores the raw `mx.array` tensors representing a single MoE layer's
    expert routing decisions. These arrays are kept unevaluated to avoid
    blocking the forward pass and are only evaluated when flushed.
    """

    expert_ids: mx.array = field()
    expert_weights: mx.array = field()
    gate_logits: mx.array | None = field()
    step_context: TraceStepContext = field()
    event_metadata: EventMetadata = field()
    layer_identifiers: LayerIdentifiers = field()


class MoERecorder(BaseEventRecorder):
    """Records routing decisions from instrumented MLX MoE layers.

    Captures events lazily. Calls to `capture()` buffer unevaluated
    `mx.array` tensors to avoid serializing execution graph during
    the forward pass.

    Data is materialized and emitted as individual `ExpertRoutingEvent`
    records when `flush()` is called (one per batch-token pair).
    """

    _buffer: list[_BufferedEvent]
    _emitted_event_idx: int

    def __init__(self, run_context: TraceRunContext) -> None:
        super().__init__(run_context)

        # Tracks total number of per-token events written to sink. Distinct
        # from `_event_idx` (which counts `capture()` calls) as captured
        # tensors expand into multiple discrete events when materialized
        # during multi-token prefill.
        self._emitted_event_idx = 0

    def capture(
        self,
        *,
        layer_path: str,
        layer_class: str,
        block_idx: int,  # TODO verify block_idx = transformer_block
        expert_ids: mx.array,
        expert_weights: mx.array,
        gate_logits: Optional[mx.array],
    ) -> None:
        """Buffers unevaluated routing event for a single MoE layer.

        Parameters
        ----------
        layer_path : str
            Dot-separated module path of the layer being captured
        layer_class : str
            Class name of the captured layer
        block_idx : int
            Index of the MoE layer's parent transformer block
        expert_ids : mx.array
            Indices of the router-selected experts
        expert_weights : mx.array
            Normalized routing weights for the selected experts
        gate_logits : Optional[mx.array]
            Optional unnormalized, pre-softmax gate logits

        Raises
        ------
        RuntimeError
            If no `TraceStepContext` currently active (i.e. `start_trace()`
            hasn't been called, yet)
        """
        if self._step_context is None:
            raise RuntimeError(
                "`capture()` called without an active `TraceStepContext`. "
                "Call `start_trace()` before recording events."
            )
        event_metadata = EventMetadata(
            event_idx=self._event_idx, timestamp=datetime.now(UTC)
        )
        layer_identifiers = LayerIdentifiers(
            layer_path=layer_path, layer_class=layer_class, block_idx=block_idx
        )
        self._buffer.append(
            _BufferedEvent(
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
        """Evaluates buffer and writes materialized events to sink.

        Triggers deferred MLX computation, unpacks batched tensors into
        discrete per-token events, and delegates async writes to sink.

        Ensures current step context is closed upon completion, regardless
        of success.

        Parameters
        ----------
        sink : BaseEventSink
            The active event sink to write the serialized records to

        Returns
        -------
        int
            The total number of flushed `ExpertRoutingEvent` records

        Raises
        ------
        RuntimeError
            If no capture step currently active
        """
        if self._step_context is None:
            raise RuntimeError("No capture step currently active.")
        try:
            return await self._to_sink(sink)

        finally:
            self.stop_trace()

    async def _to_sink(self, sink: BaseEventSink) -> int:
        """Internal routine orchestrating evaluation, materialization,
        and synchronous I/O.

        Parameters
        ----------
        sink : BaseEventSink
            Active event sink

        Returns
        -------
        int
            The total number of events written `sink`

        """
        if not self._buffer:
            return 0

        self._eval_buffered_arrays(self._buffer)

        n_flushed = 0
        for rec in self._materialize_buffered_events(self._emitted_event_idx):
            await sink.write(rec.as_arrow_record())
            n_flushed += 1

        self._emitted_event_idx += n_flushed
        return n_flushed

    def _materialize_buffered_events(
        self, current_event_idx: int
    ) -> Generator[ExpertRoutingEvent, Any, None]:
        """Converts evaluated batched tensors into a sequence of discrete
        routing events.

        Iterates over the evaluated batch dimensions and sequence offsets to
        yield one `ExpertRoutingEvent` per token processed by the MoE router.

        Parameters
        ----------
        current_event_idx : int
            The global index counter starting value for assigning unique IDs
            to emitted records

        Yields
        ------
        ExpertRoutingEvent
            A fully materialized and context-aware routing record
        """
        # TODO refactor, too deeply nested
        for record in self._buffer:
            expert_ids = record.expert_ids.tolist()
            expert_weights = record.expert_weights.tolist()

            gate_logits = (
                record.gate_logits.tolist() if record.gate_logits is not None else None
            )
            for batch_idx, (batch_ids, batch_weights) in enumerate(  # type: ignore
                zip(expert_ids, expert_weights, strict=True)  # type: ignore
            ):
                for offset, (ids, weights) in enumerate(
                    zip(batch_ids, batch_weights, strict=True)
                ):
                    logits = (
                        gate_logits[batch_idx][offset]  # type: ignore
                        if gate_logits is not None
                        else None
                    )
                    yield self._to_final_record(
                        record,
                        batch_idx,
                        offset,
                        current_event_idx,
                        ids,
                        weights,
                        logits,  # type: ignore
                    )
                    current_event_idx += 1

    def _to_final_record(
        self,
        record: _BufferedEvent,
        batch_idx: int,
        offset: int,
        current_event_idx: int,
        ids: list[int],
        weights: list[float],
        logits: list[float] | None,
    ) -> ExpertRoutingEvent:
        """Constructs an `ExpertRoutingEvent` with full provenance and trace context.

        Parameters
        ----------
        record : _BufferedEvent
            Evaluated parent buffer record containing baseline context data
        batch_idx : int
            Batch index of the specific token
        offset : int
            Sequence token index offset
        current_event_idx : int
            The assigned global event identifier
        ids : list[int]
            Indices of the top-k experts selected by MoE router for this token
        weights : list[float]
            Normalized routing weights for the selected experts
        logits : list[float] | None
            Optional unnormalized, pre-softmax gate logits

        Returns
        -------
        ExpertRoutingEvent
            The final tracing record ready for Parquet serialization
        """
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

    def _eval_buffered_arrays(self, buffer: list[_BufferedEvent]) -> None:
        """Consolidates unevaluated expert IDs, weights, and logits across all
        captured layers in `buffer` into a flat list, and enforces execution in
        one synchronous `mx.eval()` call.

        Parameters
        ----------
        buffer : list[_BufferedEvent]
            The active list of unevaluated capture records
        """
        to_evaluate = [
            arr
            for item in buffer
            for arr in (item.expert_ids, item.expert_weights, item.gate_logits)
            if arr is not None
        ]
        mx.eval(*to_evaluate)
