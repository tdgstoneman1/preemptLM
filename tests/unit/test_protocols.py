from collections.abc import Sequence

from preempt.core.identity import ExpertKey
from preempt.core.protocols.provider import ExpertProvider
from preempt.core.protocols.residency import ExpertResidency
from preempt.core.protocols.runner import ModelRunner, TokenCodec
from preempt.core.protocols.store import ExpertPayload, ExpertStore, ReadPriority
from preempt.engine.scheduler import AllResidentProvider

KEY = ExpertKey(model_fingerprint="fp", layer_idx=0, expert_idx=0)


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


class FakeStore:
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        return ExpertPayload(key=key, data=b"\x00", encoding="fake")


def test_fakes_satisfy_protocols_structurally() -> None:
    assert isinstance(FakeRunner(), ModelRunner)
    assert isinstance(FakeCodec(), TokenCodec)
    assert isinstance(FakeResidency(), ExpertResidency)
    assert isinstance(FakeStore(), ExpertStore)
    assert isinstance(AllResidentProvider(), ExpertProvider)


def test_demand_orders_before_prefetch() -> None:
    assert ReadPriority.DEMAND < ReadPriority.PREFETCH


def test_all_resident_provider_acquire_is_noop() -> None:
    AllResidentProvider().acquire((KEY,))  # must not raise or block
