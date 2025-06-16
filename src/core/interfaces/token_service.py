from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

__all__ = ["ITokenService"]


class ITokenService(ABC):
    """Abstract service for token-related helpers (counting, chunking)"""

    @abstractmethod
    def count(self, text: str, model: str = "gpt-4o") -> int:  # noqa: D401
        """Return token count for *text* using *model* encoding."""

    @abstractmethod
    def chunk(self, transcript: str, max_chunk_tokens: int = 2048, model: str = "gpt-4o") -> List[str]:  # noqa: D401
        """Split *transcript* into chunks each below *max_chunk_tokens*.""" 