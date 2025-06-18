"""
audio_processing.py
Recording helpers + real-time transcription router
"""

from __future__ import annotations
import os, subprocess, time
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any

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
            "rate": 16000,  # Default sample rate
            "ffmpeg_path": None,  # Will be determined later
            "base_dir": base_dir,
        }
    except Exception:
        # Fallback values if DI not available
        return {
            "rate": 16000,
            "ffmpeg_path": None,
            "base_dir": Path("./app_data"),
        }


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
    Convert any audio format to WAV (16-kHz, mono, PCM-s16le) using ffmpeg.
    Always verifies the file is proper WAV format, even if extension is .wav.
    """
    in_path = Path(in_path)
    
    audio_config = _get_audio_config()
    
    # Check if file is actually a proper WAV file
    is_proper_wav = False
    if in_path.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(in_path), 'rb') as wf:
                # If we can read it as WAV and it's the right format, keep it
                if wf.getnchannels() == 1 and wf.getframerate() == int(audio_config["rate"]):
                    is_proper_wav = True
                    logger.info(f"File is already proper WAV format: {in_path}")
        except Exception as e:
            logger.info(f"File appears to be non-WAV despite .wav extension: {e}")
    
    if is_proper_wav:
        return str(in_path)

    # Create unique output filename to avoid conflicts
    import tempfile
    temp_dir = Path(in_path).parent
    out_path = str(temp_dir / f"converted_{Path(in_path).stem}.wav")
    
    # Determine FFmpeg path
    ffmpeg_path = ""
    if os.name == "nt":  # Windows
        ffmpeg_path = "./ffmpeg/bin/ffmpeg.exe"
    else:
        ffmpeg_path = "./ffmpeg/bin/ffmpeg"
    
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ffmpeg", "bin", "ffmpeg.exe" if os.name == "nt" else "ffmpeg",
        )

    if not os.path.exists(ffmpeg_path):
        logger.error(f"FFmpeg not found at expected path: {ffmpeg_path}")
        raise RuntimeError(f"FFmpeg not found at {ffmpeg_path}")
    
    logger.info(f"Using FFmpeg at: {ffmpeg_path}")

    cmd = [
        ffmpeg_path, "-y", "-i", str(in_path),
        "-ac", "1",         # mono
        "-ar", str(audio_config["rate"]),  # 16 kHz
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
