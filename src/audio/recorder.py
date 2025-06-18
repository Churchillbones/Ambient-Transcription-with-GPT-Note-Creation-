"""
Three-state audio recorder for Streamlit

    ▶  start()   – begin capture (spawns a background thread)
    ⏸  pause()   – temporary halt (non-blocking)
    ▶  resume()  – continue after pause
    ■  stop()    – stop & write a 16 kHz mono 16-bit WAV; returns Path

NEW: optional on_chunk callback allows live ASR.
"""

from __future__ import annotations
import datetime, threading, wave
import logging
from pathlib import Path
from typing import List, Optional, Callable

import pyaudio

from ..core.container import global_container
from ..core.interfaces.config_service import IConfigurationService

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


def _get_audio_config():
    """Helper to get audio configuration from DI container with fallbacks."""
    try:
        config_service = global_container.resolve(IConfigurationService)
        base_dir = config_service.get("base_dir", Path("./app_data"))
        return {
            "format_str": "paInt16",  # 16-bit PCM
            "channels": 1,            # Mono
            "rate": 16000,           # 16 kHz
            "chunk": 1024,           # Default chunk size
            "cache_dir": base_dir / "cache",
        }
    except Exception:
        # Fallback values if DI not available
        return {
            "format_str": "paInt16",
            "channels": 1,
            "rate": 16000,
            "chunk": 1024,
            "cache_dir": Path("./app_data/cache"),
        }


class StreamRecorder:
    """Thread-based audio grabber with pause / resume and live-chunk hook."""

    def __init__(self, on_chunk: Callable[[bytes], None] | None = None) -> None:
        self._audio:   Optional[pyaudio.PyAudio]  = None
        self._stream:  Optional[pyaudio.Stream]   = None
        self._thread:  Optional[threading.Thread] = None
        self._frames:  List[bytes]                = []
        self._running = False
        self._paused  = False
        self._on_chunk = on_chunk                 # NEW — callback per CHUNK

    # ── user controls ──────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:           # ignore double-starts
            return
        
        audio_config = _get_audio_config()
        # Ensure cache directory exists
        audio_config["cache_dir"].mkdir(parents=True, exist_ok=True)
        
        self._audio  = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=getattr(pyaudio, audio_config["format_str"]),
            channels=audio_config["channels"],
            rate=audio_config["rate"],
            input=True,
            frames_per_buffer=audio_config["chunk"],
        )
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Recorder ▶ started.")

    def pause(self) -> None:
        self._paused = True
        logger.debug("Recorder ⏸ paused.")

    def resume(self) -> None:
        if self._running:
            self._paused = False
            logger.debug("Recorder ▶ resumed.")

    def stop(self) -> Path:
        if not self._running:
            raise RuntimeError("Recorder not running.")
        self._running = False
        self._thread.join()

        # tidy up PortAudio resources
        self._stream.stop_stream(); self._stream.close()
        self._audio.terminate()

        # write WAV
        audio_config = _get_audio_config()
        out = (audio_config["cache_dir"] /
               f"rec_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav")
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(audio_config["channels"])
            wf.setsampwidth(self._audio.get_sample_size(
                getattr(pyaudio, audio_config["format_str"])))
            wf.setframerate(audio_config["rate"])
            wf.writeframes(b"".join(self._frames))

        # reset all internal state
        self._frames.clear()
        self._thread = self._stream = self._audio = None
        logger.info(f"Recorder ■ stopped → {out}")
        return out

    # ── internal capture loop ──────────────────────────────────────────────
    def _loop(self) -> None:
        audio_config = _get_audio_config()
        while self._running:
            if self._paused:
                continue
            try:
                data = self._stream.read(
                    audio_config["chunk"], exception_on_overflow=False)
            except Exception as e:
                logger.warning(f"Recorder read error: {e}")
                continue

            self._frames.append(data)
            if self._on_chunk and not self._paused:
                try:
                    self._on_chunk(data)
                except Exception as e:
                    logger.debug(f"on_chunk callback error (ignored): {e}")
