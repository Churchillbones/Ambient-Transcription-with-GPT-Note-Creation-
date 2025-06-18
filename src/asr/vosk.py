from __future__ import annotations

import json
import os
import zipfile
import logging
from pathlib import Path
from typing import Optional

import requests

from dataclasses import dataclass

from .base import Transcriber
from ..core.container import global_container
from ..core.interfaces.config_service import IConfigurationService

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


@dataclass
class VoskTranscriber(Transcriber):
    """Transcriber using the Vosk engine."""

    model_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        # Get configuration from DI container
        try:
            config_service = global_container.resolve(IConfigurationService)
            base_dir = config_service.get("base_dir", Path("./app_data"))
        except Exception:
            # Fallback if DI not fully initialized
            base_dir = Path("./app_data")
        
        model_dir = base_dir / "models"
        self.model_path = self.model_path or model_dir / "vosk-model-small-en-us-0.15"

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
            
            # Get model directory from configuration
            try:
                config_service = global_container.resolve(IConfigurationService)
                base_dir = config_service.get("base_dir", Path("./app_data"))
            except Exception:
                base_dir = Path("./app_data")
            
            model_dir = base_dir / "models"
            default_path = model_dir / "vosk-model-small-en-us-0.15"
            
            if self.model_path == default_path:
                try:
                    self._download_small_model(model_dir)
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

        # Ensure audio is in correct format for Vosk (16kHz mono WAV)
        try:
            from ..audio.audio_processing import convert_to_wav
            logger.info(f"Converting audio file: {audio_path}")
            converted_path = convert_to_wav(audio_path)
            audio_path = Path(converted_path)
            logger.info(f"Audio converted to: {audio_path}")
        except Exception as e:
            logger.error(f"Audio conversion failed, trying original file: {e}")

        err = self._ensure_model()
        if err:
            logger.error(f"Model validation failed: {err}")
            return err

        try:
            logger.info(f"Loading Vosk model from: {self.model_path}")
            model = Model(str(self.model_path))
            
            # Get sample rate from configuration with fallback
            sample_rate = 16000  # Default sample rate for Vosk
            try:
                config_service = global_container.resolve(IConfigurationService)
                sample_rate = config_service.get("rate", 16000)
            except Exception:
                pass
            
            logger.info(f"Using sample rate: {sample_rate}")
            rec = KaldiRecognizer(model, sample_rate)
            rec.SetWords(True)
            
            try:
                # Use wave module to properly read WAV file
                import wave
                logger.info(f"Opening WAV file: {audio_path}")
                
                with wave.open(str(audio_path), 'rb') as wf:
                    # Verify format
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    frames = wf.getnframes()
                    logger.info(f"WAV format: {channels} channels, {rate}Hz, {frames} frames")
                    
                    if channels != 1:
                        logger.warning(f"Audio has {channels} channels, expected 1 (mono)")
                    if rate != sample_rate:
                        logger.warning(f"Audio sample rate is {rate}, expected {sample_rate}")
                    
                    # Read and process audio frames
                    total_frames_processed = 0
                    while True:
                        audio_frames = wf.readframes(4000)
                        if not audio_frames:
                            break
                        rec.AcceptWaveform(audio_frames)
                        total_frames_processed += len(audio_frames)
                    
                    logger.info(f"Processed {total_frames_processed} audio bytes")
                
                logger.info("Getting final result from Vosk...")
                final_result = json.loads(rec.FinalResult())
                logger.info(f"Raw Vosk result: {final_result}")
                transcript = final_result.get("text", "").strip()
                logger.info(f"Vosk transcription result: '{transcript}' (length: {len(transcript)})")
                return transcript
                
            except Exception as wave_error:
                logger.error(f"WAV file processing error: {wave_error}")
                return f"ERROR: WAV file processing failed: {wave_error}"
                
        except Exception as e:  # pragma: no cover - runtime failure
            logger.error(f"Vosk recognition failed (model: {self.model_path}): {e}")
            return f"ERROR: Vosk transcription failed: {e}"
