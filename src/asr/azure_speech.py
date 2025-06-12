from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Optional

import requests

from .base import Transcriber
from ..config import config, logger


class AzureSpeechTranscriber(Transcriber):
    """Transcriber using Azure Speech service with optional OpenAI post-processing."""

    def __init__(
        self,
        speech_key: str,
        speech_endpoint: str,
        openai_key: Optional[str] = None,
        openai_endpoint: Optional[str] = None,
        language: str = "en-US",
        return_raw: bool = False,
    ) -> None:
        self.speech_key = speech_key
        self.speech_endpoint = speech_endpoint
        self.openai_key = openai_key
        self.openai_endpoint = openai_endpoint
        self.language = language or "en-US"
        self.return_raw = return_raw

    # ------------------------------------------------------------------
    def _post_process(self, transcript: str) -> str:
        if not self.openai_key or not self.openai_endpoint or config.get("SKIP_OPENAI_SUMMARIZATION", False):
            return transcript
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception:  # pragma: no cover - openai not installed
            logger.warning("OpenAI SDK not installed. Skipping post-processing.")
            return transcript
        try:
            client = AzureOpenAI(
                api_key=self.openai_key,
                api_version=str(config.get("API_VERSION")),
                azure_endpoint=self.openai_endpoint,
            )
            system_prompt = (
                "Refine this raw audio transcript for clarity and medical context. "
                "If it seems like a summary already, return it as is or improve its structure slightly."
            )
            chat_resp = client.chat.completions.create(
                model=str(config.get("MODEL_NAME")),
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}],
                max_tokens=int(len(transcript) * 1.2) + 300,
            )
            return chat_resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover - API failure
            logger.error(f"Azure OpenAI post-processing error: {e}")
            return transcript

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        if not self.speech_key or not self.speech_endpoint:
            return "ERROR: Azure Speech requires API key and endpoint for transcription."
        try:
            transcript_parts = []
            with wave.open(str(audio_path), "rb") as wf:
                channels, sampwidth, framerate, nframes = (
                    wf.getnchannels(),
                    wf.getsampwidth(),
                    wf.getframerate(),
                    wf.getnframes(),
                )
                max_chunk_s = 45
                frames_per_chunk = int(framerate * max_chunk_s)
                num_chunks = (nframes + frames_per_chunk - 1) // frames_per_chunk
                for i in range(num_chunks):
                    wf.setpos(i * frames_per_chunk)
                    chunk_frames = wf.readframes(frames_per_chunk)
                    if not chunk_frames:
                        continue
                    with io.BytesIO() as chunk_io, wave.open(chunk_io, "wb") as chunk_w:
                        chunk_w.setnchannels(channels)
                        chunk_w.setsampwidth(sampwidth)
                        chunk_w.setframerate(framerate)
                        chunk_w.writeframes(chunk_frames)
                        chunk_data = chunk_io.getvalue()
                    url = f"{self.speech_endpoint.rstrip('/')}/speech/recognition/conversation/cognitiveservices/v1"
                    headers = {"api-key": self.speech_key, "Content-Type": "audio/wav"}
                    params = {"language": self.language}
                    resp = requests.post(url, headers=headers, params=params, data=chunk_data, timeout=60)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        if res_json.get("RecognitionStatus") == "Success":
                            text = res_json.get("DisplayText", "").strip()
                            if text:
                                transcript_parts.append(text)
                    else:
                        err_text = f"Azure Speech API error (Chunk {i+1}): {resp.status_code} - {resp.text[:200]}"
                        logger.error(err_text)
                        if "language" in resp.text.lower():
                            return f"ERROR: Invalid language '{self.language}' for Azure Speech."
                        return err_text
            combined_transcript = " ".join(transcript_parts).strip()
            if not combined_transcript:
                return "NOTE: Azure Speech generated empty transcript."
            if self.return_raw:
                return combined_transcript
            return self._post_process(combined_transcript)
        except Exception as e:  # pragma: no cover - runtime failure
            logger.error(f"Azure Speech pipeline error: {e}")
            return f"ERROR: Azure Speech pipeline failed: {e}"


class AzureWhisperTranscriber(Transcriber):
    """Transcriber using Azure OpenAI Whisper deployment."""

    def __init__(self, api_key: str, endpoint: str, language: str = "en-US") -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.language = language or "en-US"

    def transcribe(self, audio_path: Path) -> str:
        if not self.api_key or not self.endpoint:
            return "ERROR: Azure Whisper (OpenAI SDK) requires 'openai_key' and 'openai_endpoint' for Azure OpenAI service."
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception:  # pragma: no cover - openai not installed
            return "ERROR: OpenAI SDK not installed."
        try:
            client = AzureOpenAI(api_key=self.api_key, api_version=str(config.get("API_VERSION")), azure_endpoint=self.endpoint)
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=str(config.get("AZURE_WHISPER_DEPLOYMENT_NAME", "whisper-1")),
                    file=fh,
                    response_format="text",
                    language=self.language.split("-")[0] if self.language else "en",
                )
            return str(resp).strip()
        except Exception as e:  # pragma: no cover - API failure
            logger.error(f"Azure Whisper (OpenAI SDK) error: {e}")
            return f"ERROR: Azure Whisper (OpenAI SDK) failed: {e}"
