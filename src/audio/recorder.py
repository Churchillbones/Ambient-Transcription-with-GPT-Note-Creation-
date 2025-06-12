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
from pathlib import Path
from typing import List, Optional, Callable

import pyaudio

from ..config import config, logger          # repo-local imports


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
        self._audio  = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=getattr(pyaudio, config["FORMAT_STR"]),
            channels=config["CHANNELS"],
            rate=config["RATE"],
            input=True,
            frames_per_buffer=config["CHUNK"],
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
        out = (config["CACHE_DIR"] /
               f"rec_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav")
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(config["CHANNELS"])
            wf.setsampwidth(self._audio.get_sample_size(
                getattr(pyaudio, config["FORMAT_STR"])))
            wf.setframerate(config["RATE"])
            wf.writeframes(b"".join(self._frames))

        # reset all internal state
        self._frames.clear()
        self._thread = self._stream = self._audio = None
        logger.info(f"Recorder ■ stopped → {out}")
        return out

    # ── internal capture loop ──────────────────────────────────────────────
    def _loop(self) -> None:
        while self._running:
            if self._paused:
                continue
            try:
                data = self._stream.read(
                    config["CHUNK"], exception_on_overflow=False)
            except Exception as e:
                logger.warning(f"Recorder read error: {e}")
                continue

            self._frames.append(data)
            if self._on_chunk and not self._paused:
                try:
                    self._on_chunk(data)
                except Exception as e:
                    logger.debug(f"on_chunk callback error (ignored): {e}")
