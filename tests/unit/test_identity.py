import pytest

from preempt.core.identity import ExpertKey


def test_key_is_hashable_and_value_equal() -> None:
    a = ExpertKey(model_fingerprint="fp", block_idx=3, expert_idx=17)
    b = ExpertKey(model_fingerprint="fp", block_idx=3, expert_idx=17)
    assert a == b
    assert len({a, b}) == 1
    assert a.variant == "all"


def test_distinct_hashes_are_distinct_keys() -> None:
    a = ExpertKey(model_fingerprint="fp1", block_idx=0, expert_idx=0)
    b = ExpertKey(model_fingerprint="fp2", block_idx=0, expert_idx=0)
    assert a != b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_fingerprint": "", "block_idx": 0, "expert_idx": 0},
        {"model_fingerprint": "fp", "block_idx": -1, "expert_idx": 0},
        {"model_fingerprint": "fp", "block_idx": 0, "expert_idx": -1},
        {"model_fingerprint": "fp", "block_idx": 0, "expert_idx": 0, "variant": ""},
    ],
)
def test_invalid_fields_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExpertKey(**kwargs)
