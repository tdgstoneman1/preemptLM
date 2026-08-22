from __future__ import annotations

from collections.abc import Sequence

# TODO fix hallucinated 'row' terminology, confusing


def group_rows_by_expert(
    row_experts: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:

    rows_by_expert: dict[int, list[int]] = {}
    for row, expert in enumerate(row_experts):
        if expert < 0:
            raise ValueError()

        rows_by_expert.setdefault(expert, []).append(row)

    return tuple(
        (expert, tuple(rows_by_expert[expert])) for expert in sorted(rows_by_expert)
    )
