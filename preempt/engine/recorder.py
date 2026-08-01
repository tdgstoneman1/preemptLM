from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod

from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext


class BaseEventRecorder(ABC):
    """*Abstract, do not instantiate.*"""

    _run_context: TraceRunContext
    _step_context: TraceStepContext | None
    _buffer: list[Any]
    _event_idx: int

    def __init__(self, run_context: TraceRunContext) -> None:
        self._run_context = run_context
        self._step_context = None
        self._buffer = []
        self._event_idx = 0

    def start_step(self, step_context: TraceStepContext) -> None:
        if self._step_context is not None:
            raise RuntimeError(
                "Trace already active. Use `end_step()` or `flush()` "
                "to clear recorder's event buffer before starting a "
                "new one."
            )
        self._step_context = step_context

    def end_step(self) -> None:
        self._buffer.clear()
        self._step_context = None

    @abstractmethod
    def capture(self, **kwargs) -> None:
        pass

    @abstractmethod
    async def flush(self, sink: BaseEventSink) -> int:
        pass
