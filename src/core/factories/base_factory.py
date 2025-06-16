from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

T_contra = TypeVar("T_contra", contravariant=False, covariant=False)

__all__ = ["IServiceFactory"]


class IServiceFactory(ABC, Generic[T_contra]):
    """Generic factory interface following the Factory Method pattern."""

    @abstractmethod
    def create(self, provider_type: str, **kwargs: Any) -> T_contra:  # noqa: D401
        """Return a service instance for *provider_type*."""

    @abstractmethod
    def get_supported_providers(self) -> list[str]:  # noqa: D401
        """Return a list of supported provider identifiers.""" 