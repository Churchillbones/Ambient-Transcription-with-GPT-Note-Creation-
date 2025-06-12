from abc import ABC, abstractmethod
from pathlib import Path


class Transcriber(ABC):
    """Abstract base class for ASR transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return a transcript for the provided audio file."""
        raise NotImplementedError
