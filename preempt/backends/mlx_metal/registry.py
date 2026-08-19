from __future__ import annotations

from typing import Never

from .types import MlxMoEFactory
from .architecture import MoEArchitecture
from .architectures.defaults import V1_MOE_FACTORIES

# TODO make `register`, `get`, and `available` classmethods


class MoEArchRegistry:
    """Mutable registry mapping MoE architecture names to `MoEArchitecture`
    factories.

    :Note: For tamper-proof defaults, use the `DefaultMoEArchRegistry` subclass.
    """

    _factories: dict[str, MlxMoEFactory]

    def __init__(self) -> None:
        self._factories = dict()

    def register(self, name: str, factory: MlxMoEFactory) -> None:
        """Registers or overwrites a callable architecture factory under `name`

        Parameters
        ----------
        name : str
            MoE architecture name, e.g. `"qwen3-next"`
        factory : Callable[[], MoEArchitecture]
            No-arg callable that builds and returns a new `MoEArchitecture`
            instance
        """
        self._factories[name] = factory

    def get(self, name: str) -> MoEArchitecture:
        """Returns new instance of the `MoEArchitecture` keyed under `name`

        Parameters
        ----------
        name : str
            Architecture name as registered

        Returns
        -------
        MoEArchitecture
            New instance created with the registered factory

        Raises
        ------
        KeyError
            If `name` is not registered
        """
        try:
            factory = self._factories[name]

        except KeyError:
            raise KeyError(
                f"'{name!r}' is not a known MoE architecture. Registered "
                f"architectures:\n{sorted(self._factories.keys())}"
            ) from None

        return factory()

    def available(self) -> tuple[str, ...]:  # TODO rename to `registered`
        """Returns the sorted names of all registered architectures"""
        return tuple(sorted(self._factories))


class DefaultMoEArchRegistry(MoEArchRegistry):
    """Frozen registry with preemptLM's built-in MoE factories registered.

    Use for built-in defaults, otherwise use or subclass `MoEArchRegistry` for
    user-extensibility. The public `register` method is overridden to prevent
    registry mutation.
    """

    def __init__(self) -> None:
        super().__init__()

        self._factories = V1_MOE_FACTORIES

    def register(self, name: str, factory: MlxMoEFactory) -> Never:
        """Automatically raises `AttributeError` when called to prevent mutating
        built-in defaults.
        """
        raise AttributeError(
            "`DefaultMoEArchRegistry` is frozen and cannot be extended. "
            "For extensible registries, use or subclass `MoEArchRegistry`,"
            "instead."
        )
