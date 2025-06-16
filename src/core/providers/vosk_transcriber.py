from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.interfaces.transcription import ITranscriber
from importlib import import_module

__all__ = ["VoskTranscriberProvider"]


class VoskTranscriberProvider(ITranscriber):
    """Async wrapper around legacy VoskTranscriber."""

    def __init__(self, *, model_path: str | Path | None = None) -> None:  # noqa: D401
        vosk_mod = import_module("src.asr.vosk")
        LegacyCls = getattr(vosk_mod, "VoskTranscriber")
        self._legacy = LegacyCls(model_path=model_path)

    async def transcribe(self, audio_path: Path, **_: Any) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._legacy.transcribe, audio_path)

    def is_supported_format(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".wav"} 