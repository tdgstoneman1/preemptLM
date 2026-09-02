from __future__ import annotations

import pytest

from types import new_class

from preempt.backends.mlx_metal.adapters.moe_arch_adapter import BaseMoEArchAdapter
from preempt.backends.mlx_metal.adapters.qwen3_x import Qwen3_xArchAdapter
from preempt.backends.mlx_metal.instrumented.qwen3_x_moe import InstrumentedQwen3_xMoE
from preempt.backends.mlx_metal.adapters.registry import (
    ArchClassRegistry,
    DefaultArchClassRegistry,
)


# TODO unit tests for new registry methods and name aliases
class TestArchitectureRegistry:
    def test_register_and_get(self) -> None:
        reg = new_class("mock_registry", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        arch = reg.get_arch_adapter("qwen")
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_get_unknown_raises(self) -> None:
        reg = new_class("mock_registry1", (ArchClassRegistry,))
        with pytest.raises(KeyError, match="Unknown"):
            reg.get_arch_adapter("nonexistent")

    def test_available(self) -> None:
        reg = new_class("mock_registry2", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        assert "qwen" in reg.architectures()

    def test_register_duplicate_overwrites(self) -> None:
        reg = new_class("mock_registry3", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        reg.register(
            "qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE
        )  # no error, overwrites
        assert "qwen" in reg.architectures()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = new_class("mock_registry4", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        a = reg.get_arch_adapter("qwen")
        b = reg.get_arch_adapter("qwen")
        assert a is not b

    def test_get_returns_moearchitecture(self) -> None:
        reg = new_class("mock_registry5", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        reg.register("dummy", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        arch = reg.get_arch_adapter("qwen")
        assert isinstance(arch, BaseMoEArchAdapter)

    def test_empty_registry_available(self) -> None:
        reg = new_class("mock_registry6", (ArchClassRegistry,))
        assert reg.architectures() == ()

    def test_error_message_lists_available(self) -> None:
        reg = new_class("mock_registry7", (ArchClassRegistry,))
        reg.register("qwen", Qwen3_xArchAdapter, InstrumentedQwen3_xMoE)
        with pytest.raises(KeyError, match=r"\['qwen'\]"):
            reg.get_arch_adapter("nonexistent")


class TestDefaultArchitectureRegistry:
    def test_has_qwen3_x(self) -> None:
        reg = DefaultArchClassRegistry()
        arch = reg.get_arch_adapter("qwen3-next", instantiate=True)
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_register_raises_on_frozen(self) -> None:
        reg = DefaultArchClassRegistry()
        with pytest.raises(TypeError):
            reg.register("other", Qwen3_xArchAdapter)  # type: ignore

    def test_available_includes_qwen(self) -> None:
        reg = DefaultArchClassRegistry()
        assert "qwen3-next" in reg.architectures()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = DefaultArchClassRegistry()
        a = reg.get_arch_adapter("qwen3-next", instantiate=True)
        b = reg.get_arch_adapter("qwen3-next", instantiate=True)
        c = reg.get_arch_adapter("qwen3-next", instantiate=False)

        assert a is not b
        assert b is not c
