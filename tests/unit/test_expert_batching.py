import random

import pytest

from preempt.engine.expert_batching import group_rows_by_expert


def test_empty_input_yields_empty_output() -> None:
    assert group_rows_by_expert([]) == ()


def test_single_expert_collects_every_row() -> None:
    assert group_rows_by_expert([7, 7, 7, 7]) == ((7, (0, 1, 2, 3)),)


def test_every_row_a_distinct_expert() -> None:
    assert group_rows_by_expert([3, 1, 2, 0]) == (
        (0, (3,)),
        (1, (1,)),
        (2, (2,)),
        (3, (0,)),
    )


def test_repeated_experts_preserve_ascending_row_order() -> None:
    grouped = group_rows_by_expert([5, 2, 5, 2, 5])
    assert grouped == ((2, (1, 3)), (5, (0, 2, 4)))
    for _, rows in grouped:
        assert list(rows) == sorted(rows)


def test_experts_ascend_regardless_of_first_appearance() -> None:
    # First appearances are 9, 4, 0; output must not follow insertion order.
    grouped = group_rows_by_expert([9, 4, 0, 4, 9])
    assert [expert for expert, _ in grouped] == [0, 4, 9]


def test_every_row_appears_exactly_once_across_groups() -> None:
    rng = random.Random(20260804)
    for num_rows, num_routed_experts in (
        (1, 1),
        (8, 256),
        (64, 4),
        (1016, 256),
        (300, 300),
    ):
        row_experts = [rng.randrange(num_routed_experts) for _ in range(num_rows)]
        grouped = group_rows_by_expert(row_experts)

        emitted = [row for _, rows in grouped for row in rows]
        assert sorted(emitted) == list(range(num_rows))
        assert len(emitted) == num_rows

        # Each row lands under the expert the router actually chose for it.
        for expert, rows in grouped:
            for row in rows:
                assert row_experts[row] == expert


def test_grouping_is_a_partition_with_no_duplicate_experts() -> None:
    rng = random.Random(11)
    row_experts = [rng.randrange(16) for _ in range(200)]
    grouped = group_rows_by_expert(row_experts)

    experts = [expert for expert, _ in grouped]
    assert experts == sorted(experts)
    assert len(experts) == len(set(experts))
    assert set(experts) == set(row_experts)


def test_negative_expert_index_raises() -> None:
    with pytest.raises(ValueError):
        group_rows_by_expert([0, 1, -1])
