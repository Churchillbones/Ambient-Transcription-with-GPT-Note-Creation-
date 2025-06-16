from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["IConfigurationService"]


class IConfigurationService(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401
        """Return configuration value identified by *key* or *default*."""

    @abstractmethod
    def validate_configuration(self) -> bool:  # noqa: D401
        """Return *True* if configuration is valid and complete.""" 