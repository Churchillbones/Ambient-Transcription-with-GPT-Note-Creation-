from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from importlib import import_module

from core.interfaces.transcription import ITranscriber

__all__ = ["WhisperTranscriberProvider"]


class WhisperTranscriberProvider(ITranscriber):
    """Async wrapper for local WhisperTranscriber size variant."""

    def __init__(self, *, size: str = "tiny") -> None:  # noqa: D401
        mod = import_module("src.asr.whisper")
        LegacyCls = getattr(mod, "WhisperTranscriber")
        self._legacy = LegacyCls(size=size)

    async def transcribe(self, audio_path: Path, **_: Any) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._legacy.transcribe, audio_path)

    def is_supported_format(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"} 