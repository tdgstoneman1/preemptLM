from collections.abc import Sequence

from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.protocols.loader import IExpertLoader
from preempt.core.protocols.residency import IExpertResidency
from preempt.core.protocols.runner import IModelRunner, ITokenCodec
from preempt.core.protocols.expert_bank import ExpertPayload, IExpertBank, ReadPriority

from preempt.engine.expert_loaders import AllResidentLoader

KEY = ExpertKey(model_fingerprint="fp", block_idx=0, expert_idx=0)
SPECS = (TensorSpec(name="w", dtype="uint8", shape=(1,), num_bytes=1),)


class FakeRunner:
    def prepare(self) -> None: ...
    def step(self, tokens: Sequence[int]) -> int:
        return 1


class FakeCodec:
    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, tokens: Sequence[int]) -> str:
        return "x"


class FakeResidency:
    def install(self, key: ExpertKey, payload: ExpertPayload) -> None: ...
    def evict(self, key: ExpertKey) -> None: ...
    def is_resident(self, key: ExpertKey) -> bool:
        return True

    def resident_bytes(self) -> int:
        return 0


class FakeExpertBank:
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        return ExpertPayload(
            key=key, data=b"\x00", encoding="mlx-unquantized-f32", tensor_specs=SPECS
        )


def test_fakes_satisfy_protocols_structurally() -> None:
    assert isinstance(FakeRunner(), IModelRunner)
    assert isinstance(FakeCodec(), ITokenCodec)
    assert isinstance(FakeResidency(), IExpertResidency)
    assert isinstance(FakeExpertBank(), IExpertBank)
    assert isinstance(AllResidentLoader(), IExpertLoader)


def test_demand_orders_before_prefetch() -> None:
    assert ReadPriority.DEMAND < ReadPriority.PREFETCH


def test_all_resident_provider_acquire_is_noop() -> None:
    AllResidentLoader().load((KEY,))  # must not raise or block
