from typing import Any
from collections.abc import Sequence, Mapping, Hashable

from preempt.core.protocols import IExpertLoader
from preempt.core.protocols import IExpertCache
from preempt.core.protocols import IModelRunner, ITokenizer
from preempt.core.enums import ReadPriority

from preempt.datamodel.identity import ExpertKey, TensorSpec

from preempt.expert_bank.blob import SerializedExpert

from preempt.engine.expert_loaders import DummyExpertLoader

KEY = ExpertKey(model_fingerprint="fp", block_idx=0, expert_idx=0)
SPECS = (TensorSpec(name="w", dtype="uint8", shape=(1,), num_bytes=1),)


class FakeRunner:
    def prepare(self) -> None: ...
    def step(self, tokens: Sequence[int]) -> int:
        return 1


class FakeTokenizer:
    eos_token_ids: set[int] | None = None

    @property
    def think_start_id(self) -> int:
        return 67

    @property
    def think_end_id(self) -> int:
        return 666

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, tokens: Sequence[int]) -> str:
        return "x"


class FakeCache:
    def __contains__(self, item) -> bool:
        return True

    def add(self, expert: SerializedExpert) -> None: ...

    def evict(self, key: ExpertKey) -> None: ...

    def size(self) -> int:
        return 0

    def get(self, key: ExpertKey) -> Mapping[str, Any]: ...


# class FakeExpertBank:
#     async def read(self, key: ExpertKey, priority: ReadPriority) -> SerializedExpert:
#         return SerializedExpert(
#             key=key, data=b"\x00", encoding="mlx-unquantized-float32", tensor_specs=SPECS
#         )


def test_fakes_satisfy_protocols_structurally() -> None:
    assert isinstance(FakeRunner(), IModelRunner)
    assert isinstance(FakeTokenizer(), ITokenizer)
    assert isinstance(FakeCache(), IExpertCache)
    assert isinstance(DummyExpertLoader(), IExpertLoader)


def test_demand_orders_before_prefetch() -> None:
    assert ReadPriority.DEMAND < ReadPriority.PREFETCH


def test_all_resident_provider_acquire_is_noop() -> None:
    DummyExpertLoader().load((KEY,))  # must not raise or block
