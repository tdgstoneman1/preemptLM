from __future__ import annotations

from typing import Any, Optional
from collections.abc import Generator

from datetime import datetime, UTC

import attrs
from attrs import field

import mlx.core as mx

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_selection import (
    ExpertSelectionTrace,
    TraceMetadata,
    MoEBlockIdentifiers,
)
from preempt.engine.sinks import BaseTraceSink
from preempt.engine.recorder import BaseTraceRecorder


@attrs.define(frozen=True)
class _BufferedTrace:
    """Buffered record for an unevaluated router selection.

    Stores arrays representing a single MoE block's router selection,
    which are kept unevaluated to avoid blocking forward pass and are
    evaluated when flushed.
    """

    expert_ids: mx.array = field()
    softmax_weights: mx.array = field()
    gate_logits: mx.array | None = field()
    step_context: TraceStepContext = field()
    trace_metadata: TraceMetadata = field()
    moe_block_identifiers: MoEBlockIdentifiers = field()


class MlxTraceRecorder(BaseTraceRecorder):
    """Records expert selection traces from instrumented MLX MoE blocks.

    Captures events lazily. Calls to `capture()` buffer unevaluated
    `mx.array` tensors to avoid serializing execution graph during
    the forward pass.

    Trace data is materialized and written as individual `ExpertSelectionTrace`
    records when `flush()` is called (one per batch-token pair).
    """

    _buffer: list[_BufferedTrace]

    def __init__(self, run_context: TraceRunContext) -> None:
        super().__init__(run_context)

    def capture(
        self,
        *,
        layer_path: str,
        layer_class: str,
        block_idx: int,
        expert_ids: mx.array,
        softmax_weights: mx.array,
        gate_logits: Optional[mx.array],
    ) -> None:
        """Buffers unevaluated routing event for a single MoE block.

        Parameters
        ----------
        layer_path : str
            Dot-separated module path of the layer being captured
        layer_class : str
            Class name of the captured layer
        block_idx : int
            Index of the MoE block's parent transformer block
        expert_ids : mx.array
            Indices of the router-selected experts
        softmax_weights : mx.array
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
        trace_metadata = TraceMetadata(
            event_idx=self._num_captures, timestamp=datetime.now(UTC)
        )
        moe_block_identifiers = MoEBlockIdentifiers(
            path=layer_path, module_class=layer_class, transformer_block_idx=block_idx
        )
        self._buffer.append(
            _BufferedTrace(
                expert_ids=expert_ids,
                softmax_weights=softmax_weights,
                gate_logits=gate_logits,
                step_context=self._step_context,
                trace_metadata=trace_metadata,
                moe_block_identifiers=moe_block_identifiers,
            )
        )
        self._num_captures += 1

    async def flush(self, sink: BaseTraceSink) -> int:
        """Evaluates buffer and writes materialized events to sink.

        Triggers deferred MLX computation, unpacks batched tensors into
        discrete per-token events, and delegates async writes to sink.

        Ensures current step context is closed upon completion, regardless
        of success.

        Parameters
        ----------
        sink : BaseTraceSink
            The active event sink to write the serialized records to

        Returns
        -------
        int
            The total number of flushed `ExpertSelectionTrace` records

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

    async def _to_sink(self, sink: BaseTraceSink) -> int:
        if not self._buffer:
            return 0

        self._eval_buffered_arrays(self._buffer)

        n_flushed = 0
        for rec in self._materialize_buffered_events(self._num_records):
            await sink.write(rec.as_arrow_record())
            n_flushed += 1

        self._num_records += n_flushed
        return n_flushed

    def _materialize_buffered_events(
        self, current_event_idx: int
    ) -> Generator[ExpertSelectionTrace, Any, None]:
        """Converts evaluated batched arrays into a sequence of discrete
        records.

        Iterates over the batch dim and sequence offsets to yield one
        `ExpertSelectionTrace` per token processed by the MoE router.
        """
        # TODO refactor, too deeply nested
        for record in self._buffer:
            expert_ids = record.expert_ids.tolist()
            softmax_weights = record.softmax_weights.tolist()

            gate_logits = (
                record.gate_logits.tolist() if record.gate_logits is not None else None
            )
            for batch_idx, (batch_ids, batch_weights) in enumerate(  # type: ignore
                zip(expert_ids, softmax_weights, strict=True)  # type: ignore
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
        record: _BufferedTrace,
        batch_idx: int,
        offset: int,
        current_event_idx: int,
        ids: list[int],
        weights: list[float],
        logits: list[float] | None,
    ) -> ExpertSelectionTrace:
        step_context = TraceStepContext(
            sequence_id=record.step_context.sequence_id + batch_idx,
            token_idx=record.step_context.token_idx + offset,
            token_id=record.step_context.token_id,
        )
        trace_metadata = TraceMetadata(
            event_idx=current_event_idx,
            timestamp=record.trace_metadata.timestamp,
        )
        return ExpertSelectionTrace(
            expert_ids=tuple(ids),
            softmax_weights=tuple(weights),
            gate_logits=tuple(logits) if logits is not None else None,
            run_context=self._run_context,
            step_context=step_context,
            trace_metadata=trace_metadata,
            moe_block_identifiers=record.moe_block_identifiers,
        )

    def _eval_buffered_arrays(self, buffer: list[_BufferedTrace]) -> None:
        """Consolidates unevaluated expert IDs, weights, and logits across all
        captured layers in `buffer` into a flat list, and enforces execution in
        one synchronous `mx.eval()` call.
        """
        to_evaluate = [
            arr
            for item in buffer
            for arr in (item.expert_ids, item.softmax_weights, item.gate_logits)
            if arr is not None
        ]
        mx.eval(*to_evaluate)
