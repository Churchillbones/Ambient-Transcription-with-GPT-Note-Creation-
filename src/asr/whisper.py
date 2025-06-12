from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import Transcriber
from ..config import config, logger


class WhisperTranscriber(Transcriber):
    """Transcriber using local Whisper models."""

    _VALID_SIZES = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, size: str = "tiny") -> None:
        if size not in self._VALID_SIZES:
            raise ValueError(
                f"Invalid Whisper size '{size}'. Options: {', '.join(self._VALID_SIZES)}"
            )
        self.size = size
        self._ffmpeg_checked = False

    # ------------------------------------------------------------------
    def _verify_ffmpeg(self) -> bool:
        ffmpeg_dir_path = Path(__file__).parent.parent / "ffmpeg" / "bin"
        os.environ["PATH"] = str(ffmpeg_dir_path) + os.pathsep + os.environ.get("PATH", "")
        ffmpeg_exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        ffmpeg_full_path = ffmpeg_dir_path / ffmpeg_exe
        try:
            if ffmpeg_full_path.exists():
                subprocess.run([str(ffmpeg_full_path), "-version"], capture_output=True, text=True, check=True, timeout=5)
                logger.info("FFmpeg verified using bundled binary")
                return True
            result = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True, check=False, timeout=5)
            return result.returncode == 0
        except Exception as e:  # pragma: no cover - unexpected failure
            logger.error(f"FFmpeg verification failed: {e}")
            return False

    # ------------------------------------------------------------------
    def _download_model(self, custom_dir: Path) -> None:
        import whisper  # type: ignore

        custom_dir.mkdir(parents=True, exist_ok=True)
        whisper.load_model(self.size, download_root=str(custom_dir), device=str(config.get("WHISPER_DEVICE", "cpu")))

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        if not self._ffmpeg_checked:
            self._ffmpeg_checked = True
            self._ffmpeg_available = self._verify_ffmpeg()
        if not getattr(self, "_ffmpeg_available", False):
            return "ERROR: FFmpeg not found or not functional. Please ensure FFmpeg is properly installed and accessible."

        import whisper  # type: ignore

        custom_model_dir = Path(str(config.get("WHISPER_MODELS_DIR", Path("./app_data/whisper_models"))))
        custom_model_dir.mkdir(parents=True, exist_ok=True)
        model_file_name = f"{self.size}.pt"
        model_path = custom_model_dir / model_file_name
        if not model_path.exists():
            try:
                self._download_model(custom_model_dir)
            except Exception as e:  # pragma: no cover - network / download failure
                logger.error(f"Failed to download Whisper model {self.size}: {e}")
                return f"ERROR: Failed to download Whisper model: {e}"

        device_to_use = str(config.get("WHISPER_DEVICE", "cpu"))
        try:
            model = whisper.load_model(name=self.size, download_root=str(custom_model_dir), device=device_to_use)
            result = model.transcribe(
                str(audio_path), language="en", fp16=False if device_to_use == "cpu" else True
            )
            return result.get("text", "").strip()
        except Exception as e:  # pragma: no cover - runtime failure
            logger.error(f"Local Whisper recognition failed (size={self.size}): {e}")
            return f"ERROR: Local Whisper transcription failed: {e}"
