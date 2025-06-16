from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any, Dict, Type, TypeVar

from .exceptions import ConfigurationError, ServiceNotFoundError

T = TypeVar("T")

__all__ = ["ServiceContainer", "global_container"]


class ServiceContainer:
    """Light-weight dependency-injection container.

    The container adheres to the *Inversion of Control* principle by
    resolving dependencies based on constructor type hints. Only singleton
    lifetimes are supported at the moment because they cover the vast
    majority of use-cases inside this application.
    """

    def __init__(self) -> None:  # noqa: D401
        self._registrations: Dict[Type[Any], Type[Any]] = {}
        self._singletons: Dict[Type[Any], Any] = {}

    # ---------------------------------------------------------------------
    # Registration helpers
    # ---------------------------------------------------------------------
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register *implementation* to be instantiated once per container."""
        if not issubclass(implementation, interface):
            raise ConfigurationError(
                f"{implementation.__name__} does not implement {interface.__name__}"
            )
        self._registrations[interface] = implementation

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Bind an already-created *instance* as singleton for *interface*."""
        if not isinstance(instance, interface):  # type: ignore[arg-type]
            raise ConfigurationError(
                f"Instance of {type(instance).__name__} does not implement {interface.__name__}"
            )
        self._singletons[interface] = instance

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def resolve(self, interface: Type[T]) -> T:
        """Return a fully-constructed singleton for *interface*."""
        if interface in self._singletons:
            return self._singletons[interface]  # type: ignore[return-value]

        if interface not in self._registrations:
            raise ServiceNotFoundError(f"Service {interface.__name__} not registered")

        implementation = self._registrations[interface]
        instance: T = self._create_instance(implementation)
        self._singletons[interface] = instance
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_instance(self, cls: Type[T]) -> T:
        """Recursively construct *cls* by resolving its annotated deps."""
        signature = inspect.signature(cls.__init__)
        kwargs: Dict[str, Any] = {}

        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            if param.annotation is inspect.Parameter.empty:
                # No type hint – assume caller will provide via default.
                if param.default is inspect.Parameter.empty:
                    raise ConfigurationError(
                        f"Cannot resolve untyped parameter '{name}' for {cls.__name__}"
                    )
                continue
            dep_type = param.annotation  # type: ignore[assignment]
            try:
                kwargs[name] = self.resolve(dep_type)
            except ServiceNotFoundError as exc:
                if param.default is not inspect.Parameter.empty:
                    kwargs[name] = param.default
                else:
                    raise ConfigurationError(
                        f"Unsatisfied dependency '{name}: {dep_type.__name__}' "
                        f"for {cls.__name__}"
                    ) from exc
        return cls(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    @property
    def registrations(self) -> MappingProxyType:
        """Return a read-only view of current registrations."""
        return MappingProxyType(self._registrations)


# Single global container instance used by legacy code paths until full
# boot-strapping is in place.

global_container = ServiceContainer() 