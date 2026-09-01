from __future__ import annotations

import pytest

from preempt.backends.mlx_metal.adapters.moe_arch_adapter import MoEArchAdapter
from preempt.backends.mlx_metal.adapters.qwen3_x import Qwen3_xArchAdapter
from preempt.backends.mlx_metal.adapters.registry import (
    MoEArchAdapterRegistry,
    DefaultMoEArchAdapterRegistry,
)


class TestArchitectureRegistry:
    def test_register_and_get(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        arch = reg.get("qwen")
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_get_unknown_raises(self) -> None:
        reg = MoEArchAdapterRegistry()
        with pytest.raises(KeyError, match="Unknown"):
            reg.get("nonexistent")

    def test_available(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        assert "qwen" in reg.available()

    def test_register_duplicate_overwrites(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        reg.register("qwen", Qwen3_xArchAdapter)  # no error, overwrites
        assert "qwen" in reg.available()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        a = reg.get("qwen")
        b = reg.get("qwen")
        assert a is not b

    def test_get_returns_moearchitecture(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        arch = reg.get("qwen")
        assert isinstance(arch, MoEArchAdapter)

    def test_empty_registry_available(self) -> None:
        reg = MoEArchAdapterRegistry()
        assert reg.available() == ()

    def test_error_message_lists_available(self) -> None:
        reg = MoEArchAdapterRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        with pytest.raises(KeyError, match=r"\['qwen'\]"):
            reg.get("nonexistent")


class TestDefaultArchitectureRegistry:
    def test_has_qwen3_x(self) -> None:
        reg = DefaultMoEArchAdapterRegistry()
        arch = reg.get("qwen3-next")
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_register_raises_on_frozen(self) -> None:
        reg = DefaultMoEArchAdapterRegistry()
        with pytest.raises(AttributeError, match="frozen"):
            reg.register("other", Qwen3_xArchAdapter)

    def test_available_includes_qwen(self) -> None:
        reg = DefaultMoEArchAdapterRegistry()
        assert "qwen3-next" in reg.available()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = DefaultMoEArchAdapterRegistry()
        a = reg.get("qwen3-next")
        b = reg.get("qwen3-next")
        assert a is not b
