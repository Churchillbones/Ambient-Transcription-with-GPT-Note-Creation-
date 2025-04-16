import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables and configure settings
load_dotenv()

config = {
    "AZURE_API_KEY": os.getenv("AZURE_API_KEY"),
    "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT", "https://vac20oaispd201.openai.azure.us/"),
    "MODEL_NAME": os.getenv("MODEL_NAME", "gpt-4o"),
    "API_VERSION": os.getenv("API_VERSION", "2024-02-01"),
    "LOCAL_MODEL_API_URL": os.getenv("LOCAL_MODEL_API_URL", "http://localhost:8000/generate_note"),
    "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",

    # Audio settings
    "CHUNK": 1024,
    "FORMAT_STR": "paInt16", # Storing format name, PyAudio constant will be resolved later if needed
    "CHANNELS": 1,
    "RATE": 16000,

    # Directory setup
    "BASE_DIR": Path("./app_data"),
    "LOCAL_MODELS_DIR": Path("./local_llm_models"),

    # Whisper settings
    "USE_WHISPER": os.getenv("USE_WHISPER", "False").lower() == "true",
    "WHISPER_MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "tiny"),  # Use tiny or base for CPU-only systems
    "WHISPER_MODELS_DIR": Path("./app_data/whisper_models"),
    "WHISPER_DEVICE": "cpu"  # Force CPU usage
}

# Set up derived paths
config["MODEL_DIR"] = config["BASE_DIR"] / "models"
config["KEY_DIR"] = config["BASE_DIR"] / "keys"
config["LOG_DIR"] = config["BASE_DIR"] / "logs"
config["CACHE_DIR"] = config["BASE_DIR"] / "cache"
config["NOTES_DIR"] = config["BASE_DIR"] / "notes"
config["PROMPT_STORE"] = config["BASE_DIR"] / "prompt_templates.json"

# Set FFmpeg path
config["FFMPEG_PATH"] = Path("./ffmpeg/ffmpeg-2025-03-20-git-76f09ab647-essentials_build/bin/ffmpeg.exe") if os.name == 'nt' else Path("./ffmpeg/ffmpeg-2025-03-20-git-76f09ab647-essentials_build/bin/ffmpeg")

# Create directories in one go
for directory in [config["MODEL_DIR"], config["KEY_DIR"], config["LOG_DIR"],
                 config["CACHE_DIR"], config["NOTES_DIR"], config["WHISPER_MODELS_DIR"]]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
log_level = logging.DEBUG if config["DEBUG_MODE"] else logging.INFO
logging.basicConfig(
    filename=config["LOG_DIR"] / "app.log",
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("medical_transcription")

# Resolve PyAudio format constant dynamically if needed elsewhere
# Example:
# import pyaudio
# config["FORMAT"] = getattr(pyaudio, config["FORMAT_STR"])

logger.info("Configuration loaded.")
