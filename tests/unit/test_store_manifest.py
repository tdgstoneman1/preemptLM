from pathlib import Path

import pytest
from pydantic import ValidationError

from preempt.core.identity import ExpertKey
from preempt.storage.manifest import (
    ExpertBlobRecord,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)


def make_manifest() -> ExpertStoreManifest:
    specs = (
        TensorSpec(name="gate_proj.weight", dtype="uint32", shape=(4, 8), nbytes=128),
        TensorSpec(name="gate_proj.scales", dtype="float16", shape=(4, 2), nbytes=16),
    )
    return ExpertStoreManifest(
        model_id="some/model",
        model_fingerprint="fp",
        payload_encoding="mlx-affine-q4-g64",
        tensor_specs=specs,
        topology=ExpertTopology(moe_layer_idxs=(1, 3), num_experts=2, top_k=1),
        blobs=(
            ExpertBlobRecord(layer_idx=1, expert_idx=0, offset=0, length=144),
            ExpertBlobRecord(layer_idx=1, expert_idx=1, offset=4096, length=144),
        ),
    )


def test_expert_nbytes_is_sum_of_tensor_specs() -> None:
    assert make_manifest().expert_nbytes() == 144


def test_blob_length_must_match_tensor_specs() -> None:
    manifest = make_manifest()
    bad_blob = {"layer_idx": 1, "expert_idx": 0, "offset": 0, "length": 7}
    with pytest.raises(ValidationError, match="length"):
        ExpertStoreManifest.model_validate(
            manifest.model_dump() | {"blobs": [bad_blob]}
        )


def test_duplicate_blob_identity_rejected() -> None:
    manifest = make_manifest()
    blob = {"layer_idx": 1, "expert_idx": 0, "offset": 0, "length": 144}
    with pytest.raises(ValidationError, match="[Dd]uplicate"):
        ExpertStoreManifest.model_validate(
            manifest.model_dump() | {"blobs": [blob, blob]}
        )


def test_blob_index_keys_by_expert_key() -> None:
    manifest = make_manifest()
    index = manifest.blob_index()
    key = ExpertKey(model_fingerprint="fp", layer_idx=1, expert_idx=1)
    assert index[key].offset == 4096


def test_save_load_round_trip(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest.save(tmp_path)
    assert ExpertStoreManifest.load(tmp_path) == manifest
