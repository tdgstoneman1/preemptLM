from pathlib import Path

import pytest

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertStore, ReadPriority
from preempt.storage.manifest import (
    ExpertTopology,
    StoreCompatibilityError,
    TensorSpec,
)
from preempt.storage.packed_store import PackedExpertStore
from preempt.storage.writer import PackedStoreWriter


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    with PackedStoreWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=(TensorSpec(name="w", dtype="uint8", shape=(3,), nbytes=3),),
        topology=ExpertTopology(moe_layer_idxs=(0, 2), num_experts=2, top_k=1),
        alignment=8,
    ) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"bbb")
        writer.add_expert(layer_idx=2, expert_idx=0, data=b"ccc")
    return tmp_path / "store"


async def test_round_trips_written_blobs(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        assert isinstance(store, ExpertStore)
        key = ExpertKey(model_fingerprint="fp", layer_idx=2, expert_idx=0)
        payload = await store.read(key, ReadPriority.DEMAND)
        assert payload.data == b"ccc"
        assert payload.encoding == "test-enc"
        assert payload.key == key


async def test_key_for_round_trips_without_caller_knowing_fingerprint(
    store_dir: Path,
) -> None:
    with PackedExpertStore(store_dir) as store:
        payload = await store.read(store.key_for(0, 1), ReadPriority.DEMAND)
        assert payload.data == b"bbb"
        assert payload.key.model_fingerprint == store.fingerprint == "fp"
        assert payload.key.variant == "all"


async def test_unknown_key_raises_key_error(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        missing = ExpertKey(model_fingerprint="fp", layer_idx=9, expert_idx=9)
        with pytest.raises(KeyError):
            await store.read(missing, ReadPriority.PREFETCH)


def test_ensure_compatible_accepts_matching_topology(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        store.ensure_compatible(
            model_id="m", num_experts=2, top_k=1, moe_layer_idxs=(0, 2)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "other", "num_experts": 2, "top_k": 1, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 4, "top_k": 1, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 2, "top_k": 8, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 2, "top_k": 1, "moe_layer_idxs": (0,)},
    ],
)
def test_ensure_compatible_rejects_mismatch(store_dir: Path, kwargs: dict) -> None:
    with PackedExpertStore(store_dir) as store:
        with pytest.raises(StoreCompatibilityError):
            store.ensure_compatible(**kwargs)


async def test_read_after_close_raises(store_dir: Path) -> None:
    store = PackedExpertStore(store_dir)
    store.close()
    key = ExpertKey(model_fingerprint="fp", layer_idx=0, expert_idx=0)
    with pytest.raises(RuntimeError, match="closed"):
        await store.read(key, ReadPriority.DEMAND)
