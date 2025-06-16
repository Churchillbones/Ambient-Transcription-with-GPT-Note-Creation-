"""Unified entry point for ASR transcribers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Any

from core.bootstrap import container  # DI bootstrap
from core.factories.transcriber_factory import TranscriberFactory


def transcribe_audio(
    audio_path: Union[str, Path],
    model: str,
    model_path: Optional[str] = None,
    azure_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_endpoint: Optional[str] = None,
    language: Optional[str] = "en-US",
    return_raw: bool = False,
) -> str:
    """Dispatch transcription to the requested backend."""
    wav_file = Path(audio_path)
    if not wav_file.exists():
        return f"ERROR: Audio file not found: {audio_path}"

    factory = container.resolve(TranscriberFactory)

    provider_type = model.lower()
    options: dict[str, Any] = {}

    if provider_type.startswith("whisper:"):
        provider_type = "whisper"
        options["size"] = model.split(":", 1)[1]
    elif provider_type == "vosk_model" and model_path:
        provider_type = "vosk"
        options["model_path"] = model_path

    try:
        transcriber = factory.create(provider_type, **options)
    except Exception as exc:
        return f"ERROR: {exc}"

    return transcriber.transcribe(wav_file)
