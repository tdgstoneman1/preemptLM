from __future__ import annotations

import pytest

from preempt.backends.mlx_metal.architecture import MoEArchAdapter
from preempt.backends.mlx_metal.adapters.qwen3_x import Qwen3_xArchAdapter
from preempt.backends.mlx_metal.registry import (
    MoEArchRegistry,
    DefaultMoEArchRegistry,
)


class TestArchitectureRegistry:
    def test_register_and_get(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        arch = reg.get("qwen")
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_get_unknown_raises(self) -> None:
        reg = MoEArchRegistry()
        with pytest.raises(KeyError, match="Unknown"):
            reg.get("nonexistent")

    def test_available(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        assert "qwen" in reg.available()

    def test_register_duplicate_overwrites(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        reg.register("qwen", Qwen3_xArchAdapter)  # no error, overwrites
        assert "qwen" in reg.available()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        a = reg.get("qwen")
        b = reg.get("qwen")
        assert a is not b

    def test_get_returns_moearchitecture(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        arch = reg.get("qwen")
        assert isinstance(arch, MoEArchAdapter)

    def test_empty_registry_available(self) -> None:
        reg = MoEArchRegistry()
        assert reg.available() == ()

    def test_error_message_lists_available(self) -> None:
        reg = MoEArchRegistry()
        reg.register("qwen", Qwen3_xArchAdapter)
        with pytest.raises(KeyError, match=r"\['qwen'\]"):
            reg.get("nonexistent")


class TestDefaultArchitectureRegistry:
    def test_has_qwen3_x(self) -> None:
        reg = DefaultMoEArchRegistry()
        arch = reg.get("qwen3-next")
        assert isinstance(arch, Qwen3_xArchAdapter)

    def test_register_raises_on_frozen(self) -> None:
        reg = DefaultMoEArchRegistry()
        with pytest.raises(AttributeError, match="frozen"):
            reg.register("other", Qwen3_xArchAdapter)

    def test_available_includes_qwen(self) -> None:
        reg = DefaultMoEArchRegistry()
        assert "qwen3-next" in reg.available()

    def test_each_get_returns_fresh_instance(self) -> None:
        reg = DefaultMoEArchRegistry()
        a = reg.get("qwen3-next")
        b = reg.get("qwen3-next")
        assert a is not b
