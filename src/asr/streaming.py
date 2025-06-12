from __future__ import annotations

import io
import json
import queue
import time
import wave
from dataclasses import dataclass, field
from typing import List

from ..config import config, logger


def load_vosk_model(model_path: str):
    """Return (and cache) a Vosk Model instance."""
    from vosk import Model

    cache = getattr(load_vosk_model, "_cache", {})
    if model_path not in cache:
        cache[model_path] = Model(model_path)
        setattr(load_vosk_model, "_cache", cache)
    return cache[model_path]


@dataclass
class VoskStreamingHandler:
    """Streaming callback for Vosk."""

    model_path: str
    update_queue: queue.Queue
    rec: object = field(init=False)
    transcriptions: List[str] = field(default_factory=list)
    last_final_text: str = ""
    start_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        from vosk import KaldiRecognizer

        self.rec = KaldiRecognizer(load_vosk_model(self.model_path), config["RATE"])
        self.rec.SetWords(True)

    def __call__(self, chunk: bytes) -> None:
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        if self.rec.AcceptWaveform(chunk):
            result = json.loads(self.rec.Result())
            final_text = result.get("text", "").strip()
            words_info = result.get("result", []) if final_text else []
            if final_text and final_text != self.last_final_text:
                self.last_final_text = final_text
                self.transcriptions.append(final_text)
                update = {
                    "text": " ".join(self.transcriptions),
                    "words_info": words_info,
                    "is_final": True,
                    "elapsed": elapsed_str,
                    "partial": "",
                }
                self.update_queue.put(update)
        else:
            partial_result = json.loads(self.rec.PartialResult())
            partial = partial_result.get("partial", "").strip()
            if partial:
                update = {
                    "text": " ".join(self.transcriptions),
                    "words_info": [],
                    "is_final": False,
                    "elapsed": elapsed_str,
                    "partial": partial,
                }
                self.update_queue.put(update)


@dataclass
class WhisperStreamingHandler:
    """Streaming callback for local Whisper models."""

    model_size: str
    update_queue: queue.Queue
    model: object = field(init=False)
    buf: List[bytes] = field(default_factory=list)
    last_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    full_transcription: str = ""

    def __post_init__(self) -> None:
        import whisper

        self.model = whisper.load_model(self.model_size)

    def __call__(self, chunk: bytes) -> None:
        import numpy as np

        self.buf.append(chunk)
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        if time.time() - self.last_time < 3:
            update = {
                "text": self.full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": "..." if self.full_transcription else "",
                "processing": False,
            }
            self.update_queue.put(update)
            return
        self.last_time = time.time()
        update = {
            "text": self.full_transcription,
            "words_info": [],
            "is_final": False,
            "elapsed": elapsed_str,
            "partial": "Processing audio...",
            "processing": True,
        }
        self.update_queue.put(update)
        audio_buffer = b"".join(self.buf)
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self.model.transcribe(
                audio_np, language="en", fp16=False, suppress_tokens=None
            )
            txt = result["text"].strip()
            if txt:
                self.full_transcription = txt
                update = {
                    "text": self.full_transcription,
                    "words_info": [],
                    "is_final": True,
                    "elapsed": elapsed_str,
                    "partial": "",
                    "processing": False,
                }
                self.update_queue.put(update)
        except Exception as e:  # pragma: no cover - runtime failure
            logger.error(f"Whisper transcription error: {e}")
            update = {
                "text": self.full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": f"Whisper Error: {str(e)[:50]}...",
                "processing": False,
            }
            self.update_queue.put(update)
        max_buf_len_bytes = int(10 * config["RATE"] * 2)
        if len(audio_buffer) > max_buf_len_bytes:
            start_byte = len(audio_buffer) - max_buf_len_bytes
            new_buf = audio_buffer[start_byte:]
            chunk_size = config["CHUNK"] * 2
            self.buf = [new_buf[i : i + chunk_size] for i in range(0, len(new_buf), chunk_size)]


@dataclass
class AzureSpeechStreamingHandler:
    """Streaming callback for Azure Speech service."""

    api_key: str
    endpoint: str
    update_queue: queue.Queue
    buf: List[bytes] = field(default_factory=list)
    last_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    chunk_duration: int = 45
    full_transcription: str = ""

    def __call__(self, chunk: bytes) -> None:
        import requests

        self.buf.append(chunk)
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        audio_buffer = b"".join(self.buf)
        samples_per_chunk = int(config["RATE"] * self.chunk_duration)
        audio_samples = len(audio_buffer) // 2
        if audio_samples >= samples_per_chunk or (time.time() - self.last_time >= 3):
            self.last_time = time.time()
            update = {
                "text": self.full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": "Processing audio...",
                "processing": True,
            }
            self.update_queue.put(update)
            try:
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(config["CHANNELS"])
                    wf.setsampwidth(2)
                    wf.setframerate(config["RATE"])
                    wf.writeframes(audio_buffer)
                request_url = f"{self.endpoint}/speech/recognition/conversation/cognitiveservices/v1"
                headers = {"api-key": self.api_key, "Content-Type": "audio/wav"}
                params = {"language": "en-US"}
                resp = requests.post(
                    request_url,
                    headers=headers,
                    params=params,
                    data=wav_buffer.getvalue(),
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("RecognitionStatus") == "Success":
                        text = result.get("DisplayText", "").strip()
                        if text:
                            self.full_transcription = (self.full_transcription + " " + text).strip()
                            update = {
                                "text": self.full_transcription,
                                "words_info": [],
                                "is_final": True,
                                "elapsed": elapsed_str,
                                "partial": "",
                                "processing": False,
                            }
                            self.update_queue.put(update)
                else:
                    logger.error(
                        f"Azure Speech API error: {resp.status_code} - {resp.text}"
                    )
            except Exception as e:  # pragma: no cover - runtime failure
                logger.error(f"Azure Speech transcription error: {e}")
                update = {
                    "text": self.full_transcription,
                    "words_info": [],
                    "is_final": False,
                    "elapsed": elapsed_str,
                    "partial": f"Azure Error: {str(e)[:50]}...",
                    "processing": False,
                }
                self.update_queue.put(update)
            if audio_samples >= samples_per_chunk:
                excess_samples = audio_samples - samples_per_chunk
                new_buf_bytes = audio_buffer[-excess_samples * 2 :]
                chunk_size = config["CHUNK"] * 2
                self.buf = [
                    new_buf_bytes[i : i + chunk_size]
                    for i in range(0, len(new_buf_bytes), chunk_size)
                ]
            else:
                self.buf = []

