"""
audio_processing.py
Recording helpers + real-time transcription router
"""

from __future__ import annotations
import os, wave, datetime, json, subprocess, io, time, queue
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any

import streamlit as st
import pyaudio

from ..config import config, logger
from ..utils import audio_stream
from .recorder import StreamRecorder        # NEW
from ..asr.transcription import transcribe_audio  # reuse existing unified helper

# ---------------------------------------------------------------------------
# ADD 1:  fallback model loader if user hasn't implemented it elsewhere
def load_vosk_model(model_path: str):
    """Return (and cache) a Vosk Model instance."""
    from vosk import Model
    if not hasattr(load_vosk_model, "_cache"):
        load_vosk_model._cache = {}
    cache = load_vosk_model._cache          # type: ignore
    if model_path not in cache:
        cache[model_path] = Model(model_path)
    return cache[model_path]

# ---------------------------------------------------------------------------
# ADD 2:  streaming callbacks for live UI updates with enhanced visuals
def live_vosk_callback(model_path: str, update_queue: queue.Queue):
    """
    Return a per-chunk handler that puts partial/final Vosk text updates
    into the provided queue.
    """
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(load_vosk_model(model_path), config["RATE"])
    rec.SetWords(True)
    
    # Keep track of accumulated transcriptions
    transcriptions = []
    last_final_text = ""
    start_time = time.time()
    
    def _handler(chunk: bytes):
        nonlocal last_final_text, start_time
        import json
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        
        if rec.AcceptWaveform(chunk):
            # Final result
            result = json.loads(rec.Result())
            final_text = result.get("text", "").strip()
            
            # Get word-level confidence if available
            words_info = []
            if "result" in result and final_text:
                words_info = result.get("result", [])
            
            if final_text and final_text != last_final_text:
                last_final_text = final_text
                transcriptions.append(final_text)
                full_text = " ".join(transcriptions)
                
                # Create a dictionary with all the information to update the UI
                update_data = {
                    "text": full_text,
                    "words_info": words_info,
                    "is_final": True,
                    "elapsed": elapsed_str,
                    "partial": "",
                }
                update_queue.put(update_data) # Use queue
        else:
            # Partial result
            partial_result = json.loads(rec.PartialResult())
            partial = partial_result.get("partial", "").strip()
            
            if partial:
                # Show accumulated text + partial
                full_text = " ".join(transcriptions)
                
                # Create a dictionary with all the information to update the UI
                update_data = {
                    "text": full_text,
                    "words_info": [],
                    "is_final": False,
                    "elapsed": elapsed_str,
                    "partial": partial,
                }
                update_queue.put(update_data) # Use queue

    return _handler


def live_whisper_callback(model_size: str, update_queue: queue.Queue):
    """
    CPU streaming for Whisper: buffer audio and re-run transcription every ~3s.
    Puts updates into the provided queue.
    """
    import whisper
    import time
    import io
    import wave
    import numpy as np # Import numpy for audio conversion

    model = whisper.load_model(model_size)
    buf = []
    last_time = time.time()
    start_time = time.time()
    
    # Keep track of full transcription
    full_transcription = ""
    
    def _handler(chunk: bytes):
        nonlocal last_time, full_transcription, buf, start_time
        
        # Add current chunk to buffer
        buf.append(chunk)
        
        # Calculate elapsed time for display
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        
        # Only process every 3 seconds to avoid overloading CPU
        if time.time() - last_time < 3:
            # Send periodic updates even if not processing, to keep elapsed time fresh
            update_data = {
                "text": full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": "..." if full_transcription else "", # Show ellipsis if waiting
                "processing": False # Not actively processing yet
            }
            update_queue.put(update_data) # Use queue
            return
            
        last_time = time.time()
        
        # Signal that processing is happening
        update_data = {
            "text": full_transcription,
            "words_info": [],
            "is_final": False,
            "elapsed": elapsed_str,
            "partial": "Processing audio...",
            "processing": True
        }
        update_queue.put(update_data) # Use queue
        
        # Convert the buffer of bytes into a NumPy array suitable for Whisper
        audio_buffer = b"".join(buf)
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe the audio buffer
        try:
            # Add suppress_tokens=None to avoid potential issues with default suppression
            result = model.transcribe(audio_np, language="en", fp16=False, suppress_tokens=None) # Use numpy array, disable fp16 for CPU
            txt = result["text"].strip()
            
            if txt:
                # Whisper processes the whole buffer, so replace the transcription
                # rather than appending. This avoids duplicates.
                # For longer sessions, a more sophisticated segment handling might be needed.
                full_transcription = txt
                
                # Update the UI with completed processing
                update_data = {
                    "text": full_transcription,
                    "words_info": [], # Whisper doesn't easily provide word confidence here
                    "is_final": True, # Considered final for this chunk
                    "elapsed": elapsed_str,
                    "partial": "",
                    "processing": False
                }
                update_queue.put(update_data) # Use queue
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True) # Log full traceback
            # Update UI with error
            update_data = {
                "text": full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": f"Whisper Error: {str(e)[:50]}...",
                "processing": False
            }
            update_queue.put(update_data) # Use queue
        
        # Keep a sliding window of ~10 seconds of audio (adjust as needed)
        # This prevents memory from growing unbounded while maintaining context
        # Whisper works best with slightly longer chunks anyway
        max_buf_len_bytes = int(10 * config["RATE"] * 2) # 10 seconds * sample_rate * bytes_per_sample
        current_buf_len_bytes = len(audio_buffer)
        
        if current_buf_len_bytes > max_buf_len_bytes:
            # Keep only the most recent bytes corresponding to max_buf_len_bytes
            start_byte = current_buf_len_bytes - max_buf_len_bytes
            new_buf_bytes = audio_buffer[start_byte:]
            # Recalculate buf (list of chunks) from the truncated bytes
            chunk_size = config["CHUNK"] * 2 # Bytes per chunk (paInt16)
            buf = [new_buf_bytes[i:i + chunk_size] for i in range(0, len(new_buf_bytes), chunk_size)]
    
    return _handler

def live_azure_callback(api_key: str, endpoint: str, update_queue: queue.Queue):
    """
    Azure Speech callback that chunks audio into 45-second segments and processes them.
    Puts updates into the provided queue.
    """
    import requests
    import time
    import io
    import wave
    import numpy as np
    
    buf = []
    last_time = time.time()
    start_time = time.time()
    chunk_duration = 45  # Duration in seconds for each chunk
    samples_per_chunk = int(config["RATE"] * chunk_duration)  # Number of samples per 45 seconds
    
    # Keep track of full transcription
    full_transcription = ""
    
    def _handler(chunk: bytes):
        nonlocal last_time, full_transcription, buf, start_time
        
        # Add current chunk to buffer
        buf.append(chunk)
        
        # Calculate elapsed time for display
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        
        # Convert buffer to samples to check length
        audio_buffer = b"".join(buf)
        audio_samples = len(audio_buffer) // 2  # 2 bytes per sample for 16-bit audio
        
        # Process if we have 45 seconds of audio or if 3 seconds passed since last processing
        if audio_samples >= samples_per_chunk or (time.time() - last_time >= 3):
            last_time = time.time()
            
            # Signal that processing is happening
            update_data = {
                "text": full_transcription,
                "words_info": [],
                "is_final": False,
                "elapsed": elapsed_str,
                "partial": "Processing audio...",
                "processing": True
            }
            update_queue.put(update_data)
            
            try:
                # Create WAV data in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(config["CHANNELS"])
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(config["RATE"])
                    wf.writeframes(audio_buffer)
                
                # Prepare the request
                request_url = f"{endpoint}/speech/recognition/conversation/cognitiveservices/v1"
                headers = {
                    'api-key': api_key,
                    'Content-Type': 'audio/wav'
                }
                
                # Create parameters object with proper language parameter
                params = {
                    'language': 'en-US'  # Explicitly set language to en-US
                }
                
                # Send request to Azure with language parameter
                response = requests.post(
                    request_url,
                    headers=headers,
                    params=params,
                    data=wav_buffer.getvalue()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("RecognitionStatus") == "Success":
                        text = result.get("DisplayText", "").strip()
                        if text:
                            full_transcription = (full_transcription + " " + text).strip()
                            
                            # Update the UI with completed processing
                            update_data = {
                                "text": full_transcription,
                                "words_info": [],
                                "is_final": True,
                                "elapsed": elapsed_str,
                                "partial": "",
                                "processing": False
                            }
                            update_queue.put(update_data)
                else:
                    logger.error(f"Azure Speech API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Azure Speech transcription error: {e}", exc_info=True)
                update_data = {
                    "text": full_transcription,
                    "words_info": [],
                    "is_final": False,
                    "elapsed": elapsed_str,
                    "partial": f"Azure Error: {str(e)[:50]}...",
                    "processing": False
                }
                update_queue.put(update_data)
            
            # Keep only the most recent audio that hasn't been processed
            if audio_samples >= samples_per_chunk:
                excess_samples = audio_samples - samples_per_chunk
                new_buf_bytes = audio_buffer[-excess_samples * 2:]  # Keep the excess audio
                chunk_size = config["CHUNK"] * 2  # Bytes per chunk
                buf = [new_buf_bytes[i:i + chunk_size] for i in range(0, len(new_buf_bytes), chunk_size)]
            else:
                buf = []  # Clear buffer if we processed due to time
    
    return _handler

# ---------------------------------------------------------------------------
# Helper functions for transcript formatting
def format_transcript_with_confidence(text, partial="", words_info=None):
    """
    Format transcript with confidence highlighting.
    
    Args:
        text: The accumulated final transcript text
        partial: The current partial recognition (not yet finalized)
        words_info: List of word dictionaries with confidence scores
        
    Returns:
        Formatted HTML for display
    """
    import html
    
    # Format the main text
    formatted_text = html.escape(text)
    
    # Add partial text with different styling
    if partial:
        partial_html = f'<span style="color: #999; font-style: italic;">{html.escape(partial)}</span>'
        if formatted_text:
            formatted_html = f'{formatted_text} {partial_html}'
        else:
            formatted_html = partial_html
    else:
        formatted_html = formatted_text
    
    # If we have word-level confidence, enhance the display
    if words_info and len(words_info) > 0:
        # This would be a more complex implementation to highlight words based on confidence
        # For now we'll just add a simple indicator for low confidence words
        low_confidence_words = []
        for word_info in words_info:
            if word_info.get("conf", 1.0) < 0.5:  # Threshold for low confidence
                low_confidence_words.append(word_info.get("word", ""))
        
        if low_confidence_words:
            formatted_html += '<div style="color: #ff6b6b; font-size: 0.8em; margin-top: 5px;">'
            formatted_html += 'Low confidence words: ' + ', '.join(html.escape(w) for w in low_confidence_words)
            formatted_html += '</div>'
    
    return formatted_html

def format_elapsed_time(start_time, current_time=None):
    """Format elapsed time as MM:SS."""
    import time
    
    if current_time is None:
        current_time = time.time()
    
    elapsed = current_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    return f"{minutes:02d}:{seconds:02d}"

# ---------------------------------------------------------------------------
# Universal helper: convert any audio container to 16-kHz mono 16-bit WAV
def convert_to_wav(in_path: str | Path) -> str:
    """
    Convert MP3/M4A/etc. to WAV (16-kHz, mono, PCM-s16le) using ffmpeg.
    Returns the output path; if input is already .wav, returns unchanged path.
    """
    in_path = Path(in_path)
    if in_path.suffix.lower() == ".wav":
        return str(in_path)

    out_path = str(in_path.with_suffix(".wav"))
    
    # Use explicit FFmpeg path from our directory
    ffmpeg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "ffmpeg", "bin", "ffmpeg.exe")
    
    if not os.path.exists(ffmpeg_path):
        logger.error(f"FFmpeg not found at expected path: {ffmpeg_path}")
        raise RuntimeError(f"FFmpeg not found at {ffmpeg_path}")
    
    logger.info(f"Using FFmpeg at: {ffmpeg_path}")

    cmd = [
        ffmpeg_path, "-y", "-i", str(in_path),
        "-ac", "1",         # mono
        "-ar", str(config["RATE"]),  # 16 kHz
        "-sample_fmt", "s16",
        out_path,
    ]
    try:
        import subprocess
        subprocess.run(cmd, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"ffmpeg conversion ? {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr.decode(errors='ignore')}")
        raise RuntimeError("Audio conversion failed - see logs.") from e
