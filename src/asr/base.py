from pathlib import Path


class Transcriber:
    """Base class for ASR transcribers."""

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError
