from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["ITranscriber"]


class ITranscriber(ABC):
    """Abstract interface for audio transcription services."""

    @abstractmethod
    async def transcribe(self, audio_path: Path, **kwargs: Any) -> str:  # noqa: D401
        """Return the transcribed text for *audio_path*."""

    @abstractmethod
    def is_supported_format(self, file_path: Path) -> bool:  # noqa: D401
        """Return *True* if *file_path* is in a format supported by the transcriber.""" 