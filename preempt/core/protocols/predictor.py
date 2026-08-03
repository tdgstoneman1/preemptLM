from __future__ import annotations

from typing import Protocol, runtime_checkable

import attrs
from attrs import field, validators

from preempt.core.identity import ExpertKey


@attrs.define(kw_only=True, frozen=True)
class PredictionRequest:
    """Observed routing at `layer_idx`; predict routing `horizon` layers ahead."""

    layer_idx: int = field(validator=validators.ge(0))
    observed: tuple[ExpertKey, ...] = field()
    horizon: int = field(validator=validators.ge(1))


@attrs.define(kw_only=True, frozen=True)
class PredictionResponse:
    predicted: tuple[ExpertKey, ...] = field()


@runtime_checkable
class PredictionHandle(Protocol):
    async def poll_until(self, deadline: float) -> PredictionResponse | None:
        """Return the prediction, or `None` if unavailable by `deadline`
        (`time.monotonic()` seconds). Callers fall back to the heuristic."""
        ...


@runtime_checkable
class Predictor(Protocol):
    def submit(self, request: PredictionRequest) -> PredictionHandle: ...
