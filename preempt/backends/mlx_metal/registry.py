from __future__ import annotations

from typing import ClassVar, Literal, overload, NamedTuple, Never

import copy
import inspect

import mlx.nn as nn

from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5
from mlx_lm.models.qwen3_next import Model as Qwen3Next, Qwen3NextSparseMoeBlock

from .module_wrappers.base_moe_wrapper import BaseMoEWrapper
from .module_wrappers.qwen3_x_moe import Qwen3_xMoEWrapper

from .expert_bank.base_adapter import BaseMoEArchAdapter
from .expert_bank.qwen3_x import Qwen3_xArchAdapter


class _Entry(NamedTuple):
    arch_adapter_cls: type[BaseMoEArchAdapter]
    moe_wrapper_cls: type[BaseMoEWrapper]
    mlx_lm_model_cls: type[nn.Module]
    mlx_lm_moe_cls: type[nn.Module]


class ArchClassRegistry:
    """Registry mapping MoE architecture names to compatible architecture adapter
    classes (`BaseMoEArchAdapter`), MoE wrapper classes (`BaseMoEWrapper`), mlx-lm
    model classes (`nn.Module`), and mlx-lm MoE block classes (`nn.Module`).

    :Note: For built-in defaults, use the `DefaultArchClassRegistry` subclass.
    """

    _registered: ClassVar[dict[str, _Entry]] = {}
    _aliases: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls._registered = copy.deepcopy(cls._registered)
        cls._aliases = copy.deepcopy(cls._aliases)

    @classmethod
    def register(
        cls,
        name: str,
        arch_adapter_cls: type[BaseMoEArchAdapter],
        moe_wrapper_cls: type[BaseMoEWrapper],
        mlx_lm_model_cls: type[nn.Module],
        mlx_lm_moe_cls: type[nn.Module],
    ) -> None:
        """Registers architecture adapter-MoE wrapper pair under `name`.

        Parameters
        ----------
        name : str
            MoE architecture name used as a key in the registry, e.g. `'qwen3.x'`
        arch_adapter_cls : type[BaseMoEArchAdapter]
            An architecture adapter class
        moe_wrapper_cls : type[BaseMoEWrapper]
            An MoE wrapper class
        mlx_lm_model_cls: type[nn.Module]
            An mlx-lm model class
        mlx_lm_moe_cls: type[nn.Module]
            An mlx-lm MoE block class
        """
        cls._registered[name] = _Entry(
            arch_adapter_cls=arch_adapter_cls,
            moe_wrapper_cls=moe_wrapper_cls,
            mlx_lm_model_cls=mlx_lm_model_cls,
            mlx_lm_moe_cls=mlx_lm_moe_cls,
        )

    @classmethod
    def register_alias(cls, name: str, alias_for: str) -> None:
        """Links new alias a registered key.

        Parameters
        ----------
        name : str
            The new name alias
        alias_for: Optional[str]
            A key in the registry

        Raises
        ------
        ValueError
            If `alias_for` is not a known key in the registry.
        """
        if alias_for not in cls._registered:
            raise ValueError()  # TODO error msg

        cls._aliases.update({name: alias_for})

    @classmethod
    def _resolve_alias(cls, key: str) -> str:
        for k, alias_for in cls._aliases.items():
            if k == key:
                return alias_for

        return key

    @classmethod
    def _validate_entry(cls, key: str) -> _Entry:
        if (entry := cls._registered.get(key)) is None:
            raise KeyError(
                f"Unknown architecture {key!r}. Registered architectures: "
                f"{cls.architectures()!r}"
            )
        return entry

    @classmethod
    def get(
        cls, key: str
    ) -> tuple[
        type[BaseMoEArchAdapter], type[BaseMoEWrapper], type[nn.Module], type[nn.Module]
    ]:
        entry = cls._validate_entry(key)
        return (
            entry.arch_adapter_cls,
            entry.moe_wrapper_cls,
            entry.mlx_lm_model_cls,
            entry.mlx_lm_moe_cls,
        )

    @classmethod
    @overload
    def get_arch_adapter(
        cls, key: str, instantiate: Literal[True]
    ) -> BaseMoEArchAdapter: ...

    @classmethod
    @overload
    def get_arch_adapter(
        cls, key: str, instantiate: Literal[False]
    ) -> type[BaseMoEArchAdapter]: ...

    @classmethod
    def get_arch_adapter(
        cls, key: str, instantiate: bool = True
    ) -> BaseMoEArchAdapter | type[BaseMoEArchAdapter]:
        """Returns the architecture adapter class registered for `key`,
        or a new instance of it if `instantiate=True`

        Parameters
        ----------
        key : str
            Registered architecture name or alias
        instantiate : bool
            Whether to return a new instance of the registered class,
            by default True

        Returns
        -------
        BaseMoEArchAdapter | type[BaseMoEArchAdapter]
            `BaseMoEArchAdapter` instance or class

        Raises
        ------
        KeyError
            If `key` not in the registry.
        """
        key = cls._resolve_alias(key)
        entry = cls._validate_entry(key)
        cls_ = entry.arch_adapter_cls

        if instantiate:
            return cls_()
        return cls_

    @classmethod
    def get_moe_wrapper(
        cls, key: str | nn.Module | type[nn.Module]
    ) -> type[BaseMoEWrapper]:
        """Returns the MoE module wrapper class registered under `key`.

        Parameters
        ----------
        key : str
            Registered architecture name or alias, model instance, or model
            class.

        Raises
        ------
        KeyError
            If `key` not in the registry.
        """
        if isinstance(key, str):
            key = cls._resolve_alias(key)
            entry = cls._validate_entry(key)
            return entry.moe_wrapper_cls

        for v in cls._registered.values():
            if inspect.isclass(key):
                if key == v.mlx_lm_model_cls or issubclass(key, v.mlx_lm_model_cls):
                    return v.moe_wrapper_cls

            elif isinstance(key, v.mlx_lm_model_cls):
                return v.moe_wrapper_cls

        raise KeyError()  # TODO error msg

    @classmethod
    def get_moe_module_cls(
        cls, key: str | nn.Module | type[nn.Module]
    ) -> type[nn.Module]:
        """Returns the mlx-lm MoE block class registered for `key`.

        Parameters
        ----------
        key : str
            Registered architecture name or alias, model instance, or model
            class.

        Raises
        ------
        KeyError
            If `key` not in the registry.
        """
        if isinstance(key, str):
            key = cls._resolve_alias(key)
            entry = cls._validate_entry(key)
            return entry.moe_wrapper_cls

        for v in cls._registered.values():
            if inspect.isclass(key):
                if key == v.mlx_lm_model_cls or issubclass(key, v.mlx_lm_model_cls):
                    return v.mlx_lm_moe_cls

            elif isinstance(key, v.mlx_lm_model_cls):
                return v.mlx_lm_moe_cls

        raise KeyError()  # TODO error msg

    @classmethod
    def architectures(cls) -> tuple[str, ...]:
        names = set(list(cls._registered) + list(cls._aliases))
        return tuple(sorted(names))


class DefaultArchClassRegistry(ArchClassRegistry):
    """Frozen registry for architectures supported by preemptLM.

    Use for built-in defaults, otherwise use or subclass `ArchClassRegistry` for
    user-extensibility. The public `register` method is overridden to prevent
    registry mutation.
    """

    _registered: ClassVar[dict[str, _Entry]] = {
        "qwen3.x": _Entry(
            arch_adapter_cls=Qwen3_xArchAdapter,
            moe_wrapper_cls=Qwen3_xMoEWrapper,
            mlx_lm_model_cls=Qwen3_5,
            mlx_lm_moe_cls=Qwen3NextSparseMoeBlock,
        ),
        "qwen3-next": _Entry(
            arch_adapter_cls=Qwen3_xArchAdapter,
            moe_wrapper_cls=Qwen3_xMoEWrapper,
            mlx_lm_model_cls=Qwen3Next,
            mlx_lm_moe_cls=Qwen3NextSparseMoeBlock,
        ),
    }
    _aliases: ClassVar[dict[str, str]] = {"qwen3.5": "qwen3.x", "qwen3.6": "qwen3.x"}

    @classmethod
    def register(cls, **kwargs) -> Never:
        """Automatically throws to prevent mutating built-in defaults."""

        raise TypeError(
            "`DefaultArchClassRegistry` is frozen and cannot be extended. "
            "For extensible registries, use or subclass `ArchClassRegistry`,"
            "instead."
        )
