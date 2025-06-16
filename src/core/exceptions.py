from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "AmbientScribeError",
    "ConfigurationError",
    "ServiceNotFoundError",
]


class AmbientScribeError(Exception):
    """Base exception for all application-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context: Dict[str, Any] = context or {}

    def __str__(self) -> str:  # pragma: no cover – thin wrapper
        return self.message


class ConfigurationError(AmbientScribeError):
    """Raised when configuration is invalid or missing."""


class ServiceNotFoundError(AmbientScribeError):
    """Raised when a requested service is not registered in the container.""" 