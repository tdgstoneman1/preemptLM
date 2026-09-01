from pathlib import Path

import pytest

from preempt.core.identity import TensorSpec
from preempt.core.constants import EXPERTS_FILENAME, MANIFEST_FILENAME

from preempt.expert_bank.manifest import (
    ExpertBankManifest,
    ModelMoESpec,
)
from preempt.expert_bank.writer import ExpertBankWriter

SPECS = (TensorSpec(name="w", dtype="uint8", shape=(3,), num_bytes=3),)
TOPOLOGY = ModelMoESpec(moe_block_idxs=(0,), num_routed_experts=2, top_k=1)


def make_writer(tmp_path: Path, **kwargs) -> ExpertBankWriter:
    return ExpertBankWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=SPECS,
        model_moe_spec=TOPOLOGY,
        alignment=8,
        **kwargs,
    )


def test_blobs_are_aligned_and_readable_at_recorded_offsets(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(block_idx=0, expert_idx=1, data=b"bbb")
    manifest = ExpertBankManifest.load(tmp_path / "store")

    raw = (tmp_path / "store" / EXPERTS_FILENAME).read_bytes()
    for blob, expected in zip(manifest.blobs, (b"aaa", b"bbb"), strict=True):
        assert blob.offset % 8 == 0
        assert raw[blob.offset : blob.offset + blob.length] == expected


def test_wrong_length_data_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        with pytest.raises(ValueError, match="3 bytes"):
            writer.add_expert(block_idx=0, expert_idx=0, data=b"toolong")
        writer.add_expert(block_idx=0, expert_idx=0, data=b"ok!")


def test_duplicate_expert_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")


def test_existing_expert_bank_not_overwritten_by_default(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)
    with make_writer(tmp_path, overwrite=True) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"zzz")


def test_failed_overwrite_leaves_no_stale_manifest(tmp_path: Path) -> None:
    expert_bank_dir = tmp_path / "store"
    with make_writer(tmp_path) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(block_idx=0, expert_idx=1, data=b"bbb")
    assert (expert_bank_dir / MANIFEST_FILENAME).exists()

    with pytest.raises(RuntimeError, match="converter died"):
        with make_writer(tmp_path, overwrite=True) as writer:
            writer.add_expert(block_idx=0, expert_idx=0, data=b"xxx")
            raise RuntimeError("converter died")

    assert not (expert_bank_dir / MANIFEST_FILENAME).exists()
    with pytest.raises(FileNotFoundError):
        ExpertBankManifest.load(expert_bank_dir)

    with make_writer(tmp_path, overwrite=True) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"ccc")
        writer.add_expert(block_idx=0, expert_idx=1, data=b"ddd")
    manifest = ExpertBankManifest.load(expert_bank_dir)

    raw = (expert_bank_dir / EXPERTS_FILENAME).read_bytes()
    for blob, expected in zip(manifest.blobs, (b"ccc", b"ddd"), strict=True):
        assert raw[blob.offset : blob.offset + blob.length] == expected
