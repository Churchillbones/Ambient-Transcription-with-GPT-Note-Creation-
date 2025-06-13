from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum

class ASREngine(str, Enum):
    VOSK = "vosk"
    WHISPER = "whisper"
    AZURE_SPEECH = "azure_speech"
    AZURE_WHISPER = "azure_whisper"

class TranscriptionMode(str, Enum):
    TRADITIONAL = "traditional"
    REALTIME = "realtime"

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"

class PatientData(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    ehr_data: Optional[str] = None
    consent_given: bool = False
    consent_timestamp: Optional[str] = None

class TranscriptionRequest(BaseModel):
    audio_file: Optional[str] = None  # Base64 encoded for upload
    asr_engine: ASREngine = ASREngine.VOSK
    language: str = "en-US"
    model_path: Optional[str] = None
    use_encryption: bool = True

class TranscriptionResponse(BaseModel):
    transcript: str
    duration_seconds: float
    word_count: int
    confidence_score: Optional[float] = None
    speaker_stats: Optional[Dict[str, int]] = None
    processing_time: float

class NoteGenerationRequest(BaseModel):
    transcript: str
    template_name: str = "SOAP"
    patient_data: Optional[PatientData] = None
    use_azure: bool = True
    use_local_model: bool = False
    local_model_name: Optional[str] = None
    use_agent_pipeline: bool = False
    agent_settings: Optional[Dict[str, Any]] = None

class NoteGenerationResponse(BaseModel):
    note: str
    metadata: Dict[str, Any]
    template_used: str
    processing_time: float
    token_count: Optional[int] = None

class ModelComparisonRequest(BaseModel):
    audio_file: str  # Base64 encoded
    models: List[ASREngine]
    language: str = "en-US"

class ConfigurationUpdate(BaseModel):
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_endpoint: Optional[str] = None
    local_model_url: Optional[str] = None
    encryption_enabled: Optional[bool] = None

class RecordingSession(BaseModel):
    session_id: str
    status: str  # "recording", "paused", "stopped"
    duration_seconds: float
    start_time: str
    transcript_preview: Optional[str] = None
