from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests

from dataclasses import dataclass

from .base import Transcriber
from ..config import config, logger


@dataclass
class VoskTranscriber(Transcriber):
    """Transcriber using the Vosk engine."""

    model_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        self.model_path = self.model_path or config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"

    # ------------------------------------------------------------------
    def _download_small_model(self, target_dir: Path) -> None:
        model_name = "vosk-model-small-en-us-0.15"
        final_model_path = target_dir / model_name
        if final_model_path.exists() and any(final_model_path.iterdir()):
            return

        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = target_dir / f"{model_name}.zip"
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Vosk model {model_name} from {url} to {zip_path}")
        with requests.get(url, timeout=120, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        if not final_model_path.exists():
            raise FileNotFoundError(f"Vosk model dir {final_model_path} not found after extraction.")

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Optional[str]:
        if not self.model_path.exists() or not any(self.model_path.iterdir()):
            logger.warning(f"Vosk model directory not found or empty: {self.model_path}")
            default_path = config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"
            if self.model_path == default_path:
                try:
                    self._download_small_model(config["MODEL_DIR"])
                except Exception as e:  # pragma: no cover - network failure
                    logger.error(f"Failed to download default Vosk model: {e}")
                    return f"ERROR: Failed to download default Vosk model: {e}"
                if not self.model_path.exists() or not any(self.model_path.iterdir()):
                    return f"ERROR: Default Vosk model download attempted but failed at {self.model_path}"
            else:
                return f"ERROR: Vosk model not found at {self.model_path}."
        return None

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
        except ImportError:
            return "ERROR: 'vosk' library not installed. Please install it (e.g., pip install vosk)."

        err = self._ensure_model()
        if err:
            return err

        try:
            logger.info(f"Loading Vosk model from: {self.model_path}")
            model = Model(str(self.model_path))
            sample_rate = int(config.get("RATE", 16000))
            rec = KaldiRecognizer(model, sample_rate)
            rec.SetWords(True)
            with open(audio_path, "rb") as wf:
                while True:
                    data = wf.read(4000 * 2)
                    if not data:
                        break
                    rec.AcceptWaveform(data)
            final_result = json.loads(rec.FinalResult())
            return final_result.get("text", "").strip()
        except Exception as e:  # pragma: no cover - runtime failure
            logger.error(f"Vosk recognition failed (model: {self.model_path}): {e}")
            return f"ERROR: Vosk transcription failed: {e}"
