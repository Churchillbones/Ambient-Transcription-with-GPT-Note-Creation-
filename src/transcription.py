import os
import json
import wave
import time
from pathlib import Path
from typing import Any, Dict
import streamlit as st # For caching and UI feedback

# Disable CUDA/GPU globally for PyTorch (important for Whisper on CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Conditional imports for Vosk and Whisper
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from .config import config, logger
from .utils import get_file_hash # Import from utils

# Global variable for model caches
vosk_model_cache: Dict[str, Any] = {}
whisper_model_cache: Dict[str, Any] = {}

# --- Vosk Model Management ---
@st.cache_resource # Cache Vosk model resource
def load_vosk_model(model_path: str) -> Any:
    """Load a Vosk ASR model, with caching."""
    if not VOSK_AVAILABLE:
        msg = "Vosk library not installed. Cannot load Vosk model."
        logger.error(msg)
        st.error(msg)
        raise ImportError(msg)

    global vosk_model_cache
    if model_path in vosk_model_cache:
        logger.debug(f"Vosk model cache hit for: {model_path}")
        return vosk_model_cache[model_path]

    model_path_obj = Path(model_path)
    if not model_path_obj.exists() or not model_path_obj.is_dir():
        msg = f"Vosk model directory not found or invalid at {model_path}."
        logger.error(msg)
        st.error(f"{msg} Please ensure the model is downloaded and extracted correctly.")
        raise FileNotFoundError(msg)

    try:
        logger.info(f"Loading Vosk model from {model_path}...")
        model = Model(str(model_path_obj))
        vosk_model_cache[model_path] = model
        logger.info(f"Vosk model loaded and cached: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Vosk model from {model_path}: {e}")
        st.error(f"Error loading Vosk model: {e}")
        raise

# --- Whisper Model Management ---
@st.cache_resource # Cache Whisper model resource
def load_whisper_model(model_size: str = "tiny") -> Any:
    """Load a Whisper model with caching, ensuring CPU usage."""
    if not WHISPER_AVAILABLE:
        msg = "Whisper library or PyTorch not installed. Cannot load Whisper model."
        logger.error(msg)
        st.error(msg)
        raise ImportError(msg)

    global whisper_model_cache
    cache_key = f"{model_size}_cpu" # Include device in key
    if cache_key in whisper_model_cache:
        logger.debug(f"Whisper model cache hit for: {cache_key}")
        return whisper_model_cache[cache_key]

    try:
        # Double-check CUDA availability and log status
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}. Whisper will use CPU regardless.")

        # Force CPU device
        device = torch.device("cpu")

        # Performance optimization for CPU
        num_threads = max(4, os.cpu_count() or 4) # Use at least 4 threads
        torch.set_num_threads(num_threads)
        logger.info(f"PyTorch using {torch.get_num_threads()} CPU threads for Whisper.")

        # Load model on CPU explicitly
        logger.info(f"Loading Whisper {model_size} model onto CPU...")
        # Specify download root if needed, defaults usually work
        model = whisper.load_model(
            model_size,
            device=device,
            download_root=str(config["WHISPER_MODELS_DIR"]) # Use configured dir
        )

        # Verify the model is on CPU
        if hasattr(model, 'device'):
             logger.info(f"Whisper model loaded on device: {model.device}")
             if model.device.type != 'cpu':
                  logger.warning(f"Model unexpectedly loaded on {model.device.type}, attempting to move to CPU.")
                  model = model.to(device)
                  logger.info(f"Model moved to device: {model.device}")
        else:
             # Check parameters if device attribute isn't present
             try:
                  param_device = next(model.parameters()).device
                  logger.info(f"Checked model parameter device: {param_device}")
                  if param_device.type != 'cpu':
                       logger.warning(f"Model parameters unexpectedly on {param_device.type}, attempting to move to CPU.")
                       model = model.to(device)
                       logger.info(f"Model parameters moved to device: {next(model.parameters()).device}")
             except Exception as e:
                  logger.warning(f"Could not verify model parameter device: {e}")


        whisper_model_cache[cache_key] = model
        logger.info(f"Whisper {model_size} model loaded and cached successfully on CPU.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_size}': {e}")
        st.error(f"Error loading Whisper model: {e}")
        raise

# --- Transcription Functions ---
@st.cache_data(show_spinner=False) # Cache transcription results based on hash
def transcribe_with_vosk(audio_path: str, model_path: str) -> str:
    """Transcribe audio using Vosk ASR, with caching based on file hash."""
    if not VOSK_AVAILABLE:
        return "ERROR: Vosk library not available."

    # Check cache first
    file_hash = get_file_hash(audio_path)
    if not file_hash: return "ERROR: Could not hash input file." # Handle hash failure

    # Use a model-specific cache file name
    model_name = Path(model_path).name
    cache_file = config["CACHE_DIR"] / f"vosk_{file_hash}_{model_name}.txt"

    if cache_file.exists():
        logger.info(f"Vosk cache hit for {audio_path} using model {model_name}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read Vosk cache file {cache_file}: {e}")

    logger.info(f"Vosk cache miss for {audio_path}. Transcribing with model {model_name}...")
    try:
        # Load model (uses resource cache)
        model = load_vosk_model(model_path)

        # Set up recognizer
        rec = KaldiRecognizer(model, config["RATE"])
        rec.SetWords(True) # Enable word timestamps if needed later

        # Process audio file
        with wave.open(audio_path, 'rb') as wf:
            # Verify WAV format matches config (optional but good practice)
            if wf.getframerate() != config["RATE"] or wf.getnchannels() != config["CHANNELS"]:
                 logger.warning(f"Input WAV {audio_path} format ({wf.getframerate()} Hz, {wf.getnchannels()} ch) "
                                f"doesn't match config ({config['RATE']} Hz, {config['CHANNELS']} ch). Results may be affected.")

            while True:
                data = wf.readframes(config["CHUNK"])
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data) # Process chunk

        # Get final result
        result_json = rec.FinalResult()
        result = json.loads(result_json)
        transcription = result.get('text', '')

        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            logger.info(f"Vosk transcription cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to write Vosk cache file {cache_file}: {e}")

        logger.info(f"Vosk transcription successful for {audio_path}")
        return transcription

    except FileNotFoundError:
         logger.error(f"Audio file not found for Vosk transcription: {audio_path}")
         return f"ERROR: Audio file not found: {Path(audio_path).name}"
    except wave.Error as e:
         logger.error(f"Invalid WAV file for Vosk transcription {audio_path}: {e}")
         return f"ERROR: Invalid WAV file: {Path(audio_path).name}"
    except Exception as e:
        logger.error(f"Vosk transcription failed for {audio_path}: {e}")
        # Avoid showing generic exception details directly in UI output
        return f"ERROR: Vosk transcription failed for {Path(audio_path).name}."

@st.cache_data(show_spinner=False) # Cache transcription results based on hash and model size
def transcribe_with_whisper(audio_path: str, model_size: str = "tiny") -> str:
    """Transcribe audio using Whisper ASR on CPU, with caching."""
    if not WHISPER_AVAILABLE:
        return "ERROR: Whisper library not available."

    # Check cache first
    file_hash = get_file_hash(audio_path)
    if not file_hash: return "ERROR: Could not hash input file."

    cache_file = config["CACHE_DIR"] / f"whisper_{file_hash}_{model_size}.txt"
    if cache_file.exists():
        logger.info(f"Whisper cache hit for {audio_path} using model {model_size}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read Whisper cache file {cache_file}: {e}")

    logger.info(f"Whisper cache miss for {audio_path}. Transcribing with {model_size} model on CPU...")
    try:
        # Load model (uses resource cache, ensures CPU)
        model = load_whisper_model(model_size)

        # Transcribe with optimized settings for CPU
        # Reference: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
        options = {
            "language": "en",       # Specify English if known, helps speed up
            "task": "transcribe",
            "fp16": False,          # Must be False for CPU
            "beam_size": 3,         # Smaller beam size is faster on CPU
            "without_timestamps": True # Faster if timestamps aren't needed
            # "condition_on_previous_text": False # Can sometimes help/hinder
        }
        decode_options = {k: v for k, v in options.items() if k in whisper.DecodingOptions.__annotations__}


        logger.info(f"Starting Whisper transcription (model={model_size}, device=cpu)...")
        start_time = time.time()

        # Run transcription
        result = model.transcribe(audio_path, **decode_options)
        transcription = result["text"]

        processing_time = time.time() - start_time
        logger.info(f"Whisper transcription completed in {processing_time:.2f} seconds (model={model_size}, device=cpu)")

        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            logger.info(f"Whisper transcription cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to write Whisper cache file {cache_file}: {e}")

        return transcription
    except FileNotFoundError:
         logger.error(f"Audio file not found for Whisper transcription: {audio_path}")
         return f"ERROR: Audio file not found: {Path(audio_path).name}"
    except Exception as e:
        logger.error(f"Whisper transcription failed for {audio_path} (model={model_size}): {e}")
        # Consider checking for specific errors like out-of-memory if possible
        return f"ERROR: Whisper transcription failed for {Path(audio_path).name}."


def transcribe_audio(audio_path: str, model_info: str) -> str:
    """
    Unified function to transcribe using either Vosk or Whisper based on model_info.
    model_info format:
        - For Vosk: Full path to the Vosk model directory.
        - For Whisper: "whisper:<model_size>" (e.g., "whisper:tiny", "whisper:base").
    """
    logger.info(f"Transcription requested for '{Path(audio_path).name}' using engine info: '{model_info}'")
    if model_info.startswith("whisper:"):
        # Extract Whisper model size
        parts = model_info.split(":")
        if len(parts) == 2 and parts[1] in ["tiny", "base"]: # Add other valid sizes if needed
            model_size = parts[1]
            logger.info(f"Using Whisper engine with model size: {model_size}")
            return transcribe_with_whisper(audio_path, model_size)
        else:
            err_msg = f"Invalid Whisper model info format: {model_info}. Expected 'whisper:size'."
            logger.error(err_msg)
            return f"ERROR: {err_msg}"
    elif Path(model_info).is_dir(): # Check if it looks like a Vosk path
        logger.info(f"Using Vosk engine with model path: {model_info}")
        return transcribe_with_vosk(audio_path, model_info)
    else:
        err_msg = f"Invalid model info: {model_info}. Not a valid Whisper format or Vosk directory path."
        logger.error(err_msg)
        return f"ERROR: {err_msg}"
