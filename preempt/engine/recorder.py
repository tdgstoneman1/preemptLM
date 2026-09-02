from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod

from preempt.engine.sinks import BaseEventSink

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext


class BaseEventRecorder(ABC):
    """*Abstract, do not instantiate.* Base event recorder.

    A forward pass starts tracing with `start_trace(...)`, and subsequent calls
    to `capture(...)` buffer traced events. `flush(...)` writes buffered events
    to a sink, and `stop_trace` clears the buffer and closes the trace step.

    Concrete subclasses implement `capture` and `flush` with support for specific
    backends.
    """

    _run_context: TraceRunContext
    _step_context: TraceStepContext | None
    _buffer: list[Any]
    _event_idx: int

    def __init__(self, run_context: TraceRunContext) -> None:
        self._run_context = run_context
        self._step_context = None
        self._buffer = []
        self._event_idx = 0

    def start_trace(self, step_context: TraceStepContext) -> None:
        """Starts a new trace or raises `RuntimeError` if one is already active."""

        if self._step_context is not None:
            raise RuntimeError(
                "Trace already active. Call `stop_trace()` or `flush()` to clear event "
                "buffer prior to starting a new trace."
            )
        self._step_context = step_context

    def stop_trace(self) -> None:
        """Discards any buffered events and closes the current trace."""

        self._buffer.clear()
        self._step_context = None

    @abstractmethod
    def capture(self, **kwargs) -> None:
        """Buffers one event for the active step."""
        ...

    @abstractmethod
    async def flush(self, sink: BaseEventSink) -> int:
        """Writes event buffer to `sink` and returns the number of records written."""
        ...
