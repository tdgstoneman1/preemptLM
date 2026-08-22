from pathlib import Path

import pytest

from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.protocols.expert_bank import IExpertBank, ReadPriority
from preempt.core.exceptions import ExpertBankCompatibilityError
from preempt.storage.manifest import (
    ModelMoESpec,
)
from preempt.storage.expert_io import ExpertBank, ExpertBankWriter

SPECS = (TensorSpec(name="w", dtype="uint8", shape=(3,), num_bytes=3),)


@pytest.fixture()
def expert_bank_dir(tmp_path: Path) -> Path:
    with ExpertBankWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=SPECS,
        model_moe_spec=ModelMoESpec(
            moe_block_idxs=(0, 2), num_routed_experts=2, top_k=1
        ),
        alignment=8,
    ) as writer:
        writer.add_expert(block_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(block_idx=0, expert_idx=1, data=b"bbb")
        writer.add_expert(block_idx=2, expert_idx=0, data=b"ccc")
    return tmp_path / "store"


async def test_round_trips_written_blobs(expert_bank_dir: Path) -> None:
    with ExpertBank(expert_bank_dir) as expert_bank:
        assert isinstance(expert_bank, IExpertBank)
        key = ExpertKey(model_fingerprint="fp", block_idx=2, expert_idx=0)
        payload = await expert_bank.read(key, ReadPriority.DEMAND)
        assert payload.data == b"ccc"
        assert payload.encoding == "test-enc"
        assert payload.key == key


async def test_payload_carries_the_tensor_layout(expert_bank_dir: Path) -> None:
    """A payload is decodable on its own: the residency reads specs off it
    rather than reaching back into the expert bank."""
    with ExpertBank(expert_bank_dir) as expert_bank:
        payload = await expert_bank.read(expert_bank.key_for(0, 0), ReadPriority.DEMAND)
        assert payload.tensor_specs == SPECS
        assert sum(spec.num_bytes for spec in payload.tensor_specs) == len(payload.data)


async def test_payloads_share_one_tensor_spec_tuple(expert_bank_dir: Path) -> None:
    """Specs are shared across an expert bank; `read` is on the demand path and must
    not rebuild them per miss."""
    with ExpertBank(expert_bank_dir) as expert_bank:
        first = await expert_bank.read(expert_bank.key_for(0, 0), ReadPriority.DEMAND)
        second = await expert_bank.read(expert_bank.key_for(0, 1), ReadPriority.DEMAND)
        assert first.tensor_specs is second.tensor_specs
        assert first.tensor_specs is expert_bank.manifest.tensor_specs


async def test_key_for_round_trips_without_caller_knowing_hash(
    expert_bank_dir: Path,
) -> None:
    with ExpertBank(expert_bank_dir) as expert_bank:
        payload = await expert_bank.read(expert_bank.key_for(0, 1), ReadPriority.DEMAND)
        assert payload.data == b"bbb"
        assert payload.key.model_fingerprint == expert_bank.model_fingerprint == "fp"
        assert payload.key.variant == "all"


async def test_unknown_key_raises_key_error(expert_bank_dir: Path) -> None:
    with ExpertBank(expert_bank_dir) as expert_bank:
        missing = ExpertKey(model_fingerprint="fp", block_idx=9, expert_idx=9)
        with pytest.raises(KeyError):
            await expert_bank.read(missing, ReadPriority.PREFETCH)


def test_check_model_compatibility_accepts_matching_topology(
    expert_bank_dir: Path,
) -> None:
    with ExpertBank(expert_bank_dir) as expert_bank:
        expert_bank.check_model_compatibility(
            model_id="m", num_routed_experts=2, top_k=1, moe_block_idxs=(0, 2)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_id": "other",
            "num_routed_experts": 2,
            "top_k": 1,
            "moe_block_idxs": (0, 2),
        },
        {
            "model_id": "m",
            "num_routed_experts": 4,
            "top_k": 1,
            "moe_block_idxs": (0, 2),
        },
        {
            "model_id": "m",
            "num_routed_experts": 2,
            "top_k": 8,
            "moe_block_idxs": (0, 2),
        },
        {"model_id": "m", "num_routed_experts": 2, "top_k": 1, "moe_block_idxs": (0,)},
    ],
)
def test_check_model_compatibility_rejects_mismatch(
    expert_bank_dir: Path, kwargs: dict
) -> None:
    with ExpertBank(expert_bank_dir) as expert_bank:
        with pytest.raises(ExpertBankCompatibilityError):
            expert_bank.check_model_compatibility(**kwargs)


async def test_read_after_close_raises(expert_bank_dir: Path) -> None:
    expert_bank = ExpertBank(expert_bank_dir)
    expert_bank.close()
    key = ExpertKey(model_fingerprint="fp", block_idx=0, expert_idx=0)
    with pytest.raises(RuntimeError, match="closed"):
        await expert_bank.read(key, ReadPriority.DEMAND)


@pytest.mark.parametrize("bypass_page_cache", [True, False])
async def test_reads_correct_bytes_regardless_of_bypass_flag(
    expert_bank_dir: Path, bypass_page_cache: bool
) -> None:
    """The page-cache-bypass flag is accepted both ways and never changes the
    bytes returned. Its actual effect (the macOS `F_NOCACHE` syscall) is
    unobservable from Python and is a no-op off Darwin, so correctness under the
    flag is all this test can assert."""
    with ExpertBank(
        expert_bank_dir, bypass_page_cache=bypass_page_cache
    ) as expert_bank:
        payload = await expert_bank.read(expert_bank.key_for(2, 0), ReadPriority.DEMAND)
        assert payload.data == b"ccc"
