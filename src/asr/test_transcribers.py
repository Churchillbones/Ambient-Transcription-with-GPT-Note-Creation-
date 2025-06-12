import sys
from unittest import mock

for mod in ["dotenv", "bleach", "pyaudio", "requests", "openai"]:
    sys.modules.setdefault(mod, mock.Mock())

import asyncio
from pathlib import Path
import pytest

from .diarization import apply_speaker_diarization, generate_gpt_speaker_tags
from .transcription import transcribe_audio
from .whisper import WhisperTranscriber
from .vosk import VoskTranscriber


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    f = tmp_path / "dummy.wav"
    f.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return f


def test_apply_speaker_diarization():
    txt = "Hello. How are you? I am fine."
    out = apply_speaker_diarization(txt)
    assert out.startswith("User:")
    assert "Patient:" in out


def test_generate_gpt_speaker_tags_fallback():
    txt = "Hello there."
    result = asyncio.run(generate_gpt_speaker_tags(txt, api_key=None))
    assert result.startswith("User:")


def test_transcribe_audio_unknown_model(dummy_wav):
    res = transcribe_audio(dummy_wav, "unknown")
    assert "Unknown ASR model" in res


def test_whisper_transcriber_invalid_size():
    with pytest.raises(ValueError):
        WhisperTranscriber("badsize")


def test_vosk_transcriber_import_error(dummy_wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", None)
    vt = VoskTranscriber("/nonexistent")
    res = vt.transcribe(dummy_wav)
    assert "vosk" in res.lower()
