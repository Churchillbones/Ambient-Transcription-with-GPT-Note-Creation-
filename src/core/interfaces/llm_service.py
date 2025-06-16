from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["ILLMProvider"]


class ILLMProvider(ABC):
    """Abstract interface for large-language-model providers."""

    @abstractmethod
    async def generate_completion(self, prompt: str, **kwargs: Any) -> str:  # noqa: D401
        """Return a raw completion string for *prompt*."""

    @abstractmethod
    async def generate_note(self, transcript: str, **kwargs: Any) -> str:  # noqa: D401
        """Return a formatted clinical note from *transcript*.""" 