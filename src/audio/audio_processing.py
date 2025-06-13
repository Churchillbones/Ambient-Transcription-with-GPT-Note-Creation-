"""
audio_processing.py
Recording helpers + real-time transcription router
"""

from __future__ import annotations
import os, subprocess, time
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any

import pyaudio

from ..config import config, logger

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
