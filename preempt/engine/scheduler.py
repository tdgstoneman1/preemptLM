from __future__ import annotations

from collections.abc import Sequence

from preempt.core.identity import ExpertKey


class AllResidentProvider:
    """`ExpertProvider` for fully-resident models: every acquire is a no-op.

    The phase-1 stub — the loop, hooks, and wrapper signature are final from
    day one; streaming (phase 3) swaps this for the real scheduler and
    nothing above it changes.
    """

    def acquire(self, keys: Sequence[ExpertKey]) -> None:
        return None
