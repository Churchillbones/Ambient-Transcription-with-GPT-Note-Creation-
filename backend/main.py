from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile
import asyncio
import logging

# ensure DI bootstrap
import core.bootstrap  # noqa: F401

from src.asr.transcription import transcribe_audio
from src.llm.llm_integration import generate_note_router
from src.llm.prompts import load_prompt_templates

# Realtime ASR
from backend.realtime import router as realtime_router

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")

app = FastAPI(title="Ambient Scribe API")

# Define allowed origins
origins = [
    "http://localhost:4200",  # Angular frontend
    "http://127.0.0.1:4200", # Alternative for localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Important for WebSockets and some auth flows
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebSocket router
app.include_router(realtime_router)

class NoteRequest(BaseModel):
    transcript: str
    template: str = ""
    api_key: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    model: str | None = None
    use_local: bool = False
    local_model: str = ""
    patient_data: dict | None = None
    use_agent_pipeline: bool = False
    agent_settings: dict | None = None

@app.get("/templates")
def list_templates():
    return load_prompt_templates()

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model: str = Form("vosk"),
    language: str = Form("en-US"),
    model_path: str | None = Form(None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    logger.info(f"Transcribe request: model={model}, model_path={model_path}, file={file.filename}")
    transcript_obj = transcribe_audio(tmp_path, model, model_path=model_path, language=language)
    if asyncio.iscoroutine(transcript_obj):
        transcript = await transcript_obj
    else:
        transcript = transcript_obj
    
    logger.info(f"Transcription result: '{transcript}' (length: {len(transcript)})")
    Path(tmp_path).unlink(missing_ok=True)
    return {"transcript": transcript}

@app.post("/notes")
async def generate_note_endpoint(req: NoteRequest):
    note, _ = await generate_note_router(
        req.transcript,
        api_key=req.api_key,
        azure_endpoint=req.endpoint,
        azure_api_version=req.api_version,
        azure_model_name=req.model,
        prompt_template=req.template,
        use_local=req.use_local,
        local_model=req.local_model,
        patient_data=req.patient_data,
        use_agent_pipeline=req.use_agent_pipeline,
        agent_settings=req.agent_settings,
        progress_callback=None,
    )
    return {"note": note}
