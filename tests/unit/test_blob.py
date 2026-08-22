import pytest

import numpy as np

from preempt.storage.blob import assemble_expert_blob, derive_tensor_specs


def make_arrays() -> dict[str, np.ndarray]:
    return {
        "w": np.arange(6, dtype=np.uint32).reshape(2, 3),
        "s": np.ones((2, 1), dtype=np.float16),
    }


def test_derive_specs_preserves_order_and_metadata() -> None:
    specs = derive_tensor_specs(make_arrays(), order=("s", "w"))

    assert [spec.name for spec in specs] == ["s", "w"]
    assert specs[0].dtype == "float16"
    assert specs[1].shape == (2, 3)
    assert specs[1].num_bytes == 24


def test_assemble_concatenates_in_spec_order() -> None:
    arrays = make_arrays()
    specs = derive_tensor_specs(arrays, order=("s", "w"))
    blob = assemble_expert_blob(arrays, specs)

    assert blob == arrays["s"].tobytes() + arrays["w"].tobytes()
    assert len(blob) == sum(spec.num_bytes for spec in specs)


def test_assemble_rejects_spec_mismatch() -> None:
    arrays = make_arrays()
    specs = derive_tensor_specs(arrays, order=("s", "w"))
    arrays["w"] = arrays["w"].astype(np.uint8)  # dtype drift

    with pytest.raises(ValueError, match="dtype"):
        assemble_expert_blob(arrays, specs)


def test_derive_specs_requires_all_names_present() -> None:
    with pytest.raises(KeyError):
        derive_tensor_specs(make_arrays(), order=("s", "missing"))
