from pathlib import Path

import pytest

from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    MANIFEST_FILENAME,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)
from preempt.storage.writer import PackedStoreWriter

SPECS = (TensorSpec(name="w", dtype="uint8", shape=(3,), nbytes=3),)
TOPOLOGY = ExpertTopology(moe_layer_idxs=(0,), num_experts=2, top_k=1)


def make_writer(tmp_path: Path, **kwargs) -> PackedStoreWriter:
    return PackedStoreWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=SPECS,
        topology=TOPOLOGY,
        alignment=8,
        **kwargs,
    )


def test_blobs_are_aligned_and_readable_at_recorded_offsets(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"bbb")
    manifest = ExpertStoreManifest.load(tmp_path / "store")

    raw = (tmp_path / "store" / EXPERTS_FILENAME).read_bytes()
    for blob, expected in zip(manifest.blobs, (b"aaa", b"bbb"), strict=True):
        assert blob.offset % 8 == 0
        assert raw[blob.offset : blob.offset + blob.length] == expected


def test_wrong_length_data_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        with pytest.raises(ValueError, match="3 bytes"):
            writer.add_expert(layer_idx=0, expert_idx=0, data=b"toolong")
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"ok!")


def test_duplicate_expert_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")


def test_existing_store_not_overwritten_by_default(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)
    with make_writer(tmp_path, overwrite=True) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"zzz")


def test_failed_overwrite_leaves_no_stale_manifest(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"bbb")
    assert (store_dir / MANIFEST_FILENAME).exists()

    with pytest.raises(RuntimeError, match="converter died"):
        with make_writer(tmp_path, overwrite=True) as writer:
            writer.add_expert(layer_idx=0, expert_idx=0, data=b"xxx")
            raise RuntimeError("converter died")

    assert not (store_dir / MANIFEST_FILENAME).exists()
    with pytest.raises(FileNotFoundError):
        ExpertStoreManifest.load(store_dir)

    with make_writer(tmp_path, overwrite=True) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"ccc")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"ddd")
    manifest = ExpertStoreManifest.load(store_dir)

    raw = (store_dir / EXPERTS_FILENAME).read_bytes()
    for blob, expected in zip(manifest.blobs, (b"ccc", b"ddd"), strict=True):
        assert raw[blob.offset : blob.offset + blob.length] == expected
