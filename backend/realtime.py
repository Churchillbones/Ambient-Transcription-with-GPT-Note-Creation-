from __future__ import annotations

import json, os
import logging
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from vosk import Model, KaldiRecognizer  # type: ignore

from src.core.container import global_container
from src.core.interfaces.config_service import IConfigurationService

__all__ = ["router"]

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")

SAMPLE_RATE = 16000  # Hz – expected by small-english model

# ---------------------------------------------------------------------------
# Lazy-load global Vosk model once to avoid re-loading per connection
# Get configuration from DI container
try:
    config_service = global_container.resolve(IConfigurationService)
    base_dir = config_service.get("base_dir", Path("./app_data"))
except Exception:
    # Fallback if DI not fully initialized
    base_dir = Path("./app_data")

MODELS_BASE = base_dir / "models"
default_folder = os.getenv("DEFAULT_VOSK_MODEL", "small-english")
MODEL_DIR = MODELS_BASE / default_folder

# If configured folder missing, fall back to the first dir present
if not MODEL_DIR.exists():
    candidates = [p for p in MODELS_BASE.iterdir() if p.is_dir()]
    if candidates:
        MODEL_DIR = candidates[0]
        logger.warning(
            f"Configured Vosk model '{default_folder}' not found; falling back to '{MODEL_DIR.name}'"
        )
    else:
        raise RuntimeError("No Vosk models found in app_data/models – please add one.")

logger.info("Loading Vosk model for real-time WebSocket endpoint …")
_VOSK_MODEL = Model(str(MODEL_DIR))
logger.info("Vosk model loaded.")

router = APIRouter()


@router.websocket("/ws/vosk")
async def websocket_vosk(ws: WebSocket):
    """Real-time speech-to-text over WebSocket.

    The browser must stream raw 16-bit little-endian mono PCM at 16-kHz.
    Messages sent back:
        { "type": "partial", "text": "..." }
        { "type": "final",   "text": "..." }
    """
    await ws.accept()
    # Switch model if specified in query (?model=name)
    model_param = ws.query_params.get("model")
    if model_param and model_param != "default":
        # Expect model directories under app_data/models/<model_param>
        candidate_dir = Path("app_data/models") / model_param
        if not candidate_dir.exists():
            await ws.close(code=4404, reason=f"Model '{model_param}' not found")
            return
        recognizer_model = Model(str(candidate_dir))
    else:
        recognizer_model = _VOSK_MODEL

    recognizer = KaldiRecognizer(recognizer_model, SAMPLE_RATE)
    recognizer.SetWords(True)

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            if recognizer.AcceptWaveform(audio_bytes):
                res = json.loads(recognizer.Result())
                await ws.send_json({
                    "type": "final", 
                    "text": res.get("text", ""),
                    "result": res.get("result", [])
                })
            else:
                res = json.loads(recognizer.PartialResult())
                await ws.send_json({"type": "partial", "text": res.get("partial", "")})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/vosk")
    except Exception as exc:  # pragma: no cover
        logger.error(f"WebSocket /ws/vosk error: {exc}")
        await ws.close(code=1011) 