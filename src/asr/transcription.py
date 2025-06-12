"""Unified entry point for ASR transcribers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..config import config, logger
from .whisper import WhisperTranscriber
from .vosk import VoskTranscriber
from .azure_speech import AzureSpeechTranscriber, AzureWhisperTranscriber


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

    model_id = model.lower()
    lang_code = language or "en-US"

    if model_id.startswith("whisper:"):
        size = model_id.split(":", 1)[1]
        try:
            transcriber = WhisperTranscriber(size)
        except ValueError as e:
            return f"ERROR: {e}"
        return transcriber.transcribe(wav_file)

    if model_id == "azure_speech":
        transcriber = AzureSpeechTranscriber(
            speech_key=azure_key or "",
            speech_endpoint=azure_endpoint or "",
            openai_key=openai_key,
            openai_endpoint=openai_endpoint,
            language=lang_code,
            return_raw=return_raw,
        )
        return transcriber.transcribe(wav_file)

    if model_id == "vosk_model":
        transcriber = VoskTranscriber(model_path or str(config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"))
        return transcriber.transcribe(wav_file)

    if model_id == "vosk_small":
        transcriber = VoskTranscriber(str(config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"))
        return transcriber.transcribe(wav_file)

    if model_id == "azure_whisper":
        transcriber = AzureWhisperTranscriber(
            api_key=openai_key or "",
            endpoint=openai_endpoint or "",
            language=lang_code,
        )
        return transcriber.transcribe(wav_file)

    return (
        f"ERROR: Unknown ASR model '{model}'. Supported: whisper:<size>, azure_speech, "
        "vosk_model, vosk_small, azure_whisper."
    )
