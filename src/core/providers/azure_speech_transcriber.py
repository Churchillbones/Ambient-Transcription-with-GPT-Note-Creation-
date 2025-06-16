from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.interfaces.transcription import ITranscriber

from importlib import import_module

__all__ = ["AzureSpeechTranscriberProvider"]


class AzureSpeechTranscriberProvider(ITranscriber):
    """Async wrapper around legacy *src.asr.azure_speech.AzureSpeechTranscriber*."""

    def __init__(
        self,
        *,
        speech_key: str,
        speech_endpoint: str,
        openai_key: str | None = None,
        openai_endpoint: str | None = None,
        language: str = "en-US",
        return_raw: bool = False,
    ) -> None:
        azure_mod = import_module("src.asr.azure_speech")
        LegacyCls = getattr(azure_mod, "AzureSpeechTranscriber")
        self._legacy = LegacyCls(
            speech_key=speech_key,
            speech_endpoint=speech_endpoint,
            openai_key=openai_key,
            openai_endpoint=openai_endpoint,
            language=language,
            return_raw=return_raw,
        )

    # ------------------------------------------------------------------
    async def transcribe(self, audio_path: Path, **_: Any) -> str:  # noqa: D401
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._legacy.transcribe, audio_path)

    def is_supported_format(self, file_path: Path) -> bool:  # noqa: D401
        return file_path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"} 