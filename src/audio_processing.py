import pyaudio
import wave
import json
import datetime
import subprocess
from pathlib import Path
from typing import Tuple
import streamlit as st # Needed for UI elements like progress bar

from .config import config, logger
from .utils import audio_stream # Import from utils module

# --- Audio Recording and Processing ---

def record_audio(duration: int) -> Tuple[str, bytes]:
    """Record audio for the specified duration and return the path and raw data."""
    frames = []
    total_iterations = int(config["RATE"] / config["CHUNK"] * duration)

    # Resolve PyAudio format constant
    try:
        audio_format = getattr(pyaudio, config["FORMAT_STR"])
    except AttributeError:
        logger.error(f"Invalid PyAudio format string in config: {config['FORMAT_STR']}")
        st.error(f"Invalid PyAudio format: {config['FORMAT_STR']}")
        return "", b""

    p = None # Initialize p to None
    try:
        p = pyaudio.PyAudio() # Initialize PyAudio here
        progress_bar = st.progress(0)
        time_text = st.empty()
        time_text.text(f"Time remaining: {duration:.1f} seconds")

        with audio_stream(p, close_pyaudio=False) as (stream, _): # Pass p, don't terminate it yet
            logger.info(f"Starting recording for {duration} seconds...")
            for i in range(total_iterations):
                try:
                    data = stream.read(config["CHUNK"], exception_on_overflow=False)
                    frames.append(data)

                    # Update progress
                    progress = (i + 1) / total_iterations
                    progress_bar.progress(int(progress * 100))
                    remaining = duration - (duration * progress)
                    time_text.text(f"Time remaining: {remaining:.1f} seconds")
                except IOError as e:
                    # Handle potential input overflow errors if they occur despite exception_on_overflow=False
                    logger.warning(f"Audio read warning (overflow?): {e}")
                except Exception as e:
                    logger.error(f"Error reading audio chunk: {e}")
                    st.error(f"Error during recording: {e}")
                    # Decide if we should stop recording or continue
                    break # Stop recording on error

        logger.info("Recording finished.")
        audio_data = b''.join(frames)

        # Generate a filename and save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        output_path = config["CACHE_DIR"] / filename
        output_path_str = str(output_path)

        try:
            with wave.open(output_path_str, 'wb') as wf:
                wf.setnchannels(config["CHANNELS"])
                wf.setsampwidth(p.get_sample_size(audio_format)) # Use resolved format
                wf.setframerate(config["RATE"])
                wf.writeframes(audio_data)
            logger.info(f"Recorded audio saved to {output_path_str}")

            # Verify the WAV file
            with wave.open(output_path_str, 'rb') as test_wf:
                test_frames = test_wf.readframes(1)
            logger.debug(f"Successfully verified recorded WAV file: {output_path_str}")

            return output_path_str, audio_data

        except wave.Error as e:
            logger.error(f"Failed to write or verify WAV file {output_path_str}: {e}")
            st.error(f"Failed to save recording: {e}")
            return "", b""
        except Exception as e:
            logger.error(f"Unexpected error saving WAV {output_path_str}: {e}")
            st.error(f"Failed to save recording: {e}")
            return "", b""

    except Exception as e:
        logger.error(f"Error during recording setup or process: {e}")
        st.error(f"Recording failed: {e}")
        return "", b""
    finally:
        if p:
            p.terminate() # Terminate PyAudio instance here
            logger.debug("PyAudio terminated after recording.")


def real_time_transcribe(duration: int, model_path: str) -> Tuple[str, bytes]:
    """
    Record audio with real-time transcription using Vosk.
    Returns the full transcription and the raw audio data.
    """
    # This function now depends on transcription module, import dynamically or pass recognizer
    # For now, assume transcription module exists and provides necessary functions/classes
    try:
        from .transcription import load_vosk_model # Assuming this exists
        from vosk import KaldiRecognizer
    except ImportError:
        logger.error("Transcription module or Vosk not available for real_time_transcribe.")
        st.error("Real-time transcription components not found.")
        return "", b""

    frames = []
    total_iterations = int(config["RATE"] / config["CHUNK"] * duration)

    # Resolve PyAudio format constant
    try:
        audio_format = getattr(pyaudio, config["FORMAT_STR"])
    except AttributeError:
        logger.error(f"Invalid PyAudio format string in config: {config['FORMAT_STR']}")
        st.error(f"Invalid PyAudio format: {config['FORMAT_STR']}")
        return "", b""

    # Initialize Vosk model for real-time transcription
    rec = None
    try:
        model = load_vosk_model(model_path) # Assumes model_path is for Vosk
        rec = KaldiRecognizer(model, config["RATE"])
        rec.SetWords(True)
        logger.info("Vosk recognizer initialized for real-time transcription.")
    except Exception as e:
        st.error(f"Failed to initialize transcription model: {str(e)}")
        logger.error(f"Vosk model initialization failed: {str(e)}")
        return "", b""

    # Set up UI elements
    progress_bar = st.progress(0)
    time_text = st.empty()
    time_text.text(f"Time remaining: {duration:.1f} seconds")
    transcription_area = st.empty()
    transcription_area.markdown("### Real-time Transcription\n...") # Initial text

    # Set up complete transcription tracking
    full_transcription = ""

    # Initialize PyAudio
    p = None
    try:
        p = pyaudio.PyAudio()
        with audio_stream(p, close_pyaudio=False) as (stream, _):
            logger.info(f"Starting real-time recording/transcription for {duration} seconds...")
            for i in range(total_iterations):
                try:
                    # Read audio chunk
                    data = stream.read(config["CHUNK"], exception_on_overflow=False)
                    frames.append(data)

                    # Process for real-time transcription
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        partial_text = result.get("text", "")
                        if partial_text:
                            # Append recognized segment with a space
                            full_transcription = (full_transcription + " " + partial_text).strip()
                            # Update display with the latest full transcription
                            transcription_area.markdown(f"### Real-time Transcription\n{full_transcription}")
                    else:
                        # Show partial results if available
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            # Display current full + partial
                            transcription_area.markdown(f"### Real-time Transcription\n{full_transcription} _{partial_text}_")

                    # Update progress
                    progress = (i + 1) / total_iterations
                    progress_bar.progress(int(progress * 100))
                    remaining = duration - (duration * progress)
                    time_text.text(f"Time remaining: {remaining:.1f} seconds")

                except IOError as e:
                    logger.warning(f"Audio read warning during real-time (overflow?): {e}")
                except Exception as e:
                    logger.error(f"Error in real-time transcription loop: {e}")
                    # Optionally break or continue
                    break

        # Get final transcription segment
        final_result = json.loads(rec.FinalResult())
        final_text = final_result.get("text", "")
        if final_text:
            full_transcription = (full_transcription + " " + final_text).strip()
        logger.info("Real-time transcription finished.")
        transcription_area.markdown(f"### Real-time Transcription\n{full_transcription}") # Final update

        # Create a WAV file from the recorded frames
        audio_data = b''.join(frames)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        output_path = config["CACHE_DIR"] / filename
        output_path_str = str(output_path)

        try:
            with wave.open(output_path_str, 'wb') as wf:
                wf.setnchannels(config["CHANNELS"])
                wf.setsampwidth(p.get_sample_size(audio_format))
                wf.setframerate(config["RATE"])
                wf.writeframes(audio_data)
            logger.info(f"Recorded audio with real-time transcription saved to {output_path_str}")

            # Verify the WAV file
            with wave.open(output_path_str, 'rb') as test_wf:
                test_frames = test_wf.readframes(1)
            logger.debug(f"Successfully verified recorded WAV file: {output_path_str}")

            return full_transcription.strip(), audio_data

        except wave.Error as e:
            logger.error(f"Failed to write or verify WAV file {output_path_str} after real-time: {e}")
            st.error(f"Failed to save recording: {e}")
            return full_transcription.strip(), b"" # Return transcription even if save fails
        except Exception as e:
            logger.error(f"Unexpected error saving WAV {output_path_str} after real-time: {e}")
            st.error(f"Failed to save recording: {e}")
            return full_transcription.strip(), b""

    except Exception as e:
        logger.error(f"Error during real-time transcription setup or process: {e}")
        st.error(f"Real-time transcription failed: {e}")
        return "", b""
    finally:
        if p:
            p.terminate()
            logger.debug("PyAudio terminated after real-time transcription.")


def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format with proper settings using FFmpeg."""
    output_path = str(Path(input_path).with_suffix('.wav'))
    # Ensure output is in cache dir to avoid permission issues if input is elsewhere
    output_filename = Path(output_path).name
    output_path = str(config["CACHE_DIR"] / output_filename)

    try:
        # Use bundled FFmpeg if available, otherwise try system PATH
        ffmpeg_path = config["FFMPEG_PATH"]
        if not ffmpeg_path.exists():
            logger.warning(f"Bundled FFmpeg not found at {ffmpeg_path}, trying system 'ffmpeg'")
            ffmpeg_command = "ffmpeg"
        else:
            ffmpeg_command = str(ffmpeg_path)
            logger.debug(f"Using FFmpeg from: {ffmpeg_command}")

        # FFmpeg command arguments
        cmd = [
            ffmpeg_command,
            "-i", input_path,          # Input file
            "-ar", str(config["RATE"]), # Audio sample rate
            "-ac", str(config["CHANNELS"]),# Audio channels
            "-acodec", "pcm_s16le",    # Codec for WAV (signed 16-bit little-endian)
            "-y",                      # Overwrite output file if it exists
            output_path                # Output file path
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        # Run FFmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"FFmpeg stdout: {result.stdout}")
        logger.debug(f"FFmpeg stderr: {result.stderr}") # FFmpeg often logs info to stderr

        # Verify the output WAV file is valid
        try:
            with wave.open(output_path, 'rb') as test_wf:
                n_frames = test_wf.getnframes()
                if n_frames == 0:
                     logger.warning(f"Converted WAV file {output_path} appears empty.")
                else:
                     logger.debug(f"Successfully verified converted WAV file: {output_path} ({n_frames} frames)")
            return output_path
        except wave.Error as e:
            logger.error(f"Converted WAV file {output_path} validation failed: {e}")
            raise # Re-raise wave error
        except FileNotFoundError:
             logger.error(f"Converted WAV file {output_path} not found after conversion.")
             raise

    except FileNotFoundError:
         logger.error(f"FFmpeg command '{ffmpeg_command}' not found. Ensure FFmpeg is installed and in PATH or configured correctly.")
         raise RuntimeError("FFmpeg not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio conversion failed. FFmpeg command failed with exit code {e.returncode}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        logger.error(f"FFmpeg stdout: {e.stdout}")
        # Attempt to clean up potentially incomplete output file
        if Path(output_path).exists():
            try: Path(output_path).unlink()
            except OSError: pass
        raise RuntimeError(f"Audio conversion failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during audio conversion: {e}")
        # Attempt to clean up potentially incomplete output file
        if Path(output_path).exists():
            try: Path(output_path).unlink()
            except OSError: pass
        raise


def resample_audio(input_path: str, output_path: str, rate: int = 16000, channels: int = 1):
    """Resample audio using bundled FFmpeg (Simplified version of convert_to_wav)."""
    try:
        # Use bundled FFmpeg if available, otherwise try system PATH
        ffmpeg_path = config["FFMPEG_PATH"]
        if not ffmpeg_path.exists():
            logger.warning(f"Bundled FFmpeg not found at {ffmpeg_path}, trying system 'ffmpeg'")
            ffmpeg_command = "ffmpeg"
        else:
            ffmpeg_command = str(ffmpeg_path)

        # FFmpeg command arguments
        cmd = [
            ffmpeg_command,
            "-i", input_path,
            "-ar", str(rate),
            "-ac", str(channels),
            "-y",
            output_path
        ]

        logger.info(f"Running FFmpeg resample command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"FFmpeg resample stderr: {result.stderr}")
        logger.info(f"Audio resampled successfully to {output_path}")

    except FileNotFoundError:
         logger.error(f"FFmpeg command '{ffmpeg_command}' not found for resampling.")
         raise RuntimeError("FFmpeg not found for resampling.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio resampling failed. FFmpeg command failed with exit code {e.returncode}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Audio resampling failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during audio resampling: {e}")
        raise
