"""
transcription.py
Adaptive ASR pipeline with support for Azure Whisper, dynamic local Whisper sizes (tiny/base/medium), and Vosk.
"""
from __future__ import annotations
import os, json, io, tarfile, requests, sys, subprocess 
from pathlib import Path
from typing import Literal, Union, Optional
import zipfile 

from ..config import config, MODEL_DIR, logger

ffmpeg_dir_path = Path(__file__).parent.parent / "ffmpeg" / "bin"
os.environ["PATH"] = str(ffmpeg_dir_path) + os.pathsep + os.environ.get("PATH", "")
logger.info(f"Prepended FFmpeg directory to PATH: {ffmpeg_dir_path}")


def verify_ffmpeg():
    try:
        ffmpeg_exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        ffmpeg_full_path = ffmpeg_dir_path / ffmpeg_exe
        
        if not ffmpeg_full_path.exists():
            logger.debug(f"FFmpeg executable not found at expected path: {ffmpeg_full_path}. Trying system PATH.")
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False, timeout=5)
            if result.returncode == 0:
                logger.info(f"FFmpeg found in system PATH.")
                return True
            logger.error(f"FFmpeg also not found in system PATH. Subprocess error: {result.stderr or result.stdout}")
            return False

        result = subprocess.run([str(ffmpeg_full_path), "-version"], capture_output=True, text=True, check=True, timeout=5)
        logger.info(f"FFmpeg verified successfully at {ffmpeg_full_path}.")
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg verification timed out.")
        return False
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"FFmpeg check failed with specific path: {e}. Trying system PATH as fallback.")
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=5)
            logger.info(f"FFmpeg found in system PATH after initial check failed.")
            return True
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg verification timed out on system PATH.")
            return False
        except (subprocess.SubprocessError, FileNotFoundError) as e_fallback:
            logger.error(f"FFmpeg check failed on system PATH as well: {e_fallback}")
            return False

ffmpeg_available = verify_ffmpeg()

def _download_whisper_model(model_size: str, custom_dir: Path) -> None:
    import whisper # type: ignore
    custom_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading/verifying Whisper {model_size} model to {custom_dir}...")
    try:
        whisper.load_model(model_size, download_root=str(custom_dir), device=str(config.get("WHISPER_DEVICE", "cpu")))
        logger.info(f"Successfully downloaded/verified Whisper {model_size} model in {custom_dir}")
    except Exception as e:
        logger.error(f"Failed to download/load Whisper model {model_size}: {e}", exc_info=True)
        raise

def _local_whisper_dynamic(wav: Path, size: str) -> str:
    import whisper # type: ignore
    if not ffmpeg_available:
        return "ERROR: FFmpeg not found or not functional. Please ensure FFmpeg is properly installed and accessible."
    
    try:
        custom_model_dir = Path(str(config.get("WHISPER_MODELS_DIR", Path("./app_data/whisper_models"))))
        custom_model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file_name = f"{size}.pt"
        if not (custom_model_dir / model_file_name).exists():
             _download_whisper_model(size, custom_model_dir)
        
        device_to_use = str(config.get("WHISPER_DEVICE", "cpu"))
        logger.info(f"Loading Local Whisper model {size} from {custom_model_dir} (device: {device_to_use})")
        model = whisper.load_model(name=size, download_root=str(custom_model_dir), device=device_to_use)
            
        result = model.transcribe(
            str(wav), 
            language="en", 
            fp16=False if device_to_use == "cpu" else True
        )
        return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Local Whisper recognition failed (size={size}): {e}", exc_info=True)
        return f"ERROR: Local Whisper transcription failed: {str(e)}"

def _download_vosk_small_en_us(target_dir: Path) -> None:
    model_name = "vosk-model-small-en-us-0.15"
    final_model_path = target_dir / model_name
    
    if final_model_path.exists() and any(final_model_path.iterdir()):
        logger.info(f"Vosk model {model_name} already exists at {final_model_path}.")
        return

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = target_dir / f"{model_name}.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Vosk model {model_name} from {url} to {zip_path}")
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16): f.write(chunk)
        
        logger.info(f"Extracting {zip_path} to {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(target_dir)
        
        if not final_model_path.exists():
             raise FileNotFoundError(f"Vosk model dir {final_model_path} not found after extraction.")
        logger.info(f"Successfully downloaded and extracted Vosk model to {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to download or extract Vosk model {model_name}: {e}", exc_info=True)
        raise
    finally:
        if zip_path.exists(): os.remove(zip_path)


def _vosk_transcribe(wav: Path, model_path_str: str) -> str:
    try:
        from vosk import Model, KaldiRecognizer # type: ignore
    except ImportError:
        return "ERROR: 'vosk' library not installed. Please install it (e.g., pip install vosk)."

    model_path_obj = Path(model_path_str)
    if not model_path_obj.exists() or not model_path_obj.is_dir() or not any(model_path_obj.iterdir()):
        logger.warning(f"Vosk model directory not found or empty: {model_path_str}")
        if model_path_str == str(config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"):
            logger.info("Attempting to download default Vosk small English model...")
            try: _download_vosk_small_en_us(Path(str(config["MODEL_DIR"])))
            except Exception as e_dl: return f"ERROR: Failed to download default Vosk model: {str(e_dl)}"
            if not model_path_obj.exists() or not any(model_path_obj.iterdir()): 
                return f"ERROR: Default Vosk model download attempted but failed at {model_path_str}"
        else: return f"ERROR: Vosk model not found at {model_path_str}."
            
    try:
        logger.info(f"Loading Vosk model from: {model_path_str}")
        model = Model(model_path_str)
        sample_rate = int(config.get("RATE", 16000))
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)

        with open(wav, "rb") as wf:
            while True:
                data = wf.read(4000 * 2) 
                if not data: break
                rec.AcceptWaveform(data)
        
        final_result = json.loads(rec.FinalResult())
        return final_result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Vosk recognition failed (model: {model_path_str}): {e}", exc_info=True)
        return f"ERROR: Vosk transcription failed: {str(e)}"


def _azure_speech(
    wav_path: Path,
    speech_api_key: str, speech_endpoint: str, 
    openai_api_key: Optional[str], openai_endpoint: Optional[str], 
    language: str = "en-US",
    return_raw_transcript: bool = False # New parameter
) -> str:
    import wave, io 
    try:
        from openai import AzureOpenAI 
    except ImportError:
        logger.warning("OpenAI SDK not installed. Post-processing step in _azure_speech will be skipped.")
        AzureOpenAI = None # type: ignore

    try:
        logger.info(f"Processing with Azure Speech: {wav_path} (Lang: {language}, Endpoint: {speech_endpoint}, Raw: {return_raw_transcript})")
        if not speech_api_key or not speech_endpoint:
            return "ERROR: Azure Speech requires API key and endpoint for transcription."
        
        transcript_parts = []
        with wave.open(str(wav_path), 'rb') as wf:
            channels, sampwidth, framerate, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            logger.info(f"Audio: {nframes/framerate:.2f}s, {framerate}Hz, {channels}ch, {sampwidth*8}-bit")

            max_chunk_s = 45; frames_per_chunk = int(framerate * max_chunk_s)
            num_chunks = (nframes + frames_per_chunk - 1) // frames_per_chunk
            logger.info(f"Processing in {num_chunks} chunk(s) of max {max_chunk_s}s.")
            
            for i in range(num_chunks):
                wf.setpos(i * frames_per_chunk)
                chunk_frames = wf.readframes(frames_per_chunk)
                if not chunk_frames: continue

                with io.BytesIO() as chunk_io, wave.open(chunk_io, 'wb') as chunk_w:
                    chunk_w.setnchannels(channels); chunk_w.setsampwidth(sampwidth); chunk_w.setframerate(framerate)
                    chunk_w.writeframes(chunk_frames)
                    chunk_data = chunk_io.getvalue()

                url = f"{speech_endpoint.rstrip('/')}/speech/recognition/conversation/cognitiveservices/v1"
                headers = {'api-key': speech_api_key, 'Content-Type': 'audio/wav'}
                params = {'language': language}
                
                logger.debug(f"Sending chunk {i+1}/{num_chunks} ({len(chunk_data)} bytes) to {url} (lang: {language})")
                resp = requests.post(url, headers=headers, params=params, data=chunk_data, timeout=60)

                if resp.status_code == 200:
                    res_json = resp.json()
                    if res_json.get("RecognitionStatus") == "Success":
                        text = res_json.get("DisplayText", "").strip()
                        if text: transcript_parts.append(text)
                        logger.debug(f"Chunk {i+1} success. Text: '{text[:30]}...'")
                    else: logger.warning(f"Chunk {i+1} status not Success: {res_json.get('RecognitionStatus')}. Resp: {res_json}")
                else:
                    err_text = f"Azure Speech API error (Chunk {i+1}): {resp.status_code} - {resp.text[:200]}"
                    logger.error(err_text)
                    if "language" in resp.text.lower(): return f"ERROR: Invalid language '{language}' for Azure Speech."
                    return err_text 

            combined_transcript = " ".join(transcript_parts).strip()
            if not combined_transcript: return "NOTE: Azure Speech generated empty transcript."
            logger.info(f"Azure Speech transcript success, len: {len(combined_transcript)} chars.")
            
            if return_raw_transcript: # Check the new flag
                return combined_transcript

            if AzureOpenAI and not config.get("SKIP_OPENAI_SUMMARIZATION", False) and openai_api_key and openai_endpoint:
                logger.info("Sending to Azure OpenAI for post-processing...")
                try:
                    client = AzureOpenAI(api_key=openai_api_key, api_version=str(config.get("API_VERSION")), azure_endpoint=openai_endpoint)
                    system_prompt = "Refine this raw audio transcript for clarity and medical context. If it seems like a summary already, return it as is or improve its structure slightly." # Modified prompt
                    chat_resp = client.chat.completions.create(
                        model=str(config.get("MODEL_NAME")), messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": combined_transcript}],
                        max_tokens=int(len(combined_transcript) * 1.2) + 300 # Increased allowance
                    )
                    return chat_resp.choices[0].message.content.strip()
                except Exception as openai_err: logger.error(f"Azure OpenAI post-proc error: {openai_err}", exc_info=True)
            return combined_transcript
    except Exception as e:
        logger.error(f"Azure Speech pipeline error: {e}", exc_info=True)
        return f"ERROR: Azure Speech pipeline failed: {str(e)}"


def transcribe_audio(
    audio_path: Union[str, Path],
    model: str, 
    model_path: Optional[str] = None, 
    azure_key: Optional[str] = None, 
    azure_endpoint: Optional[str] = None, 
    openai_key: Optional[str] = None, 
    openai_endpoint: Optional[str] = None,
    language: Optional[str] = "en-US",
    return_raw: bool = False # New parameter, default to False for backward compatibility
) -> str:
    wav_file = Path(audio_path)
    if not wav_file.exists(): return f"ERROR: Audio file not found: {audio_path}"

    model_id = model.lower()
    lang_code = language or "en-US" 

    if model_id.startswith("whisper:"):
        size = model_id.split(":", 1)[1]
        if size not in ["tiny", "base", "small", "medium", "large"]:
             return f"ERROR: Invalid Whisper size '{size}'. Options: tiny, base, small, medium, large."
        return _local_whisper_dynamic(wav_file, size)
    
    elif model_id == "azure_speech":
        if not azure_key or not azure_endpoint: return "ERROR: Azure Speech requires 'azure_key' and 'azure_endpoint' for Speech service."
        return _azure_speech(wav_file, azure_key, azure_endpoint, openai_key, openai_endpoint, lang_code, return_raw_transcript=return_raw) # Pass return_raw

    elif model_id == "vosk_model":
        path_to_vosk = model_path or str(config["MODEL_DIR"] / "vosk-model-small-en-us-0.15") 
        return _vosk_transcribe(wav_file, path_to_vosk)
    
    elif model_id == "vosk_small": 
        return _vosk_transcribe(wav_file, str(config["MODEL_DIR"] / "vosk-model-small-en-us-0.15"))

    elif model_id == "azure_whisper": 
        if not openai_key or not openai_endpoint: return "ERROR: Azure Whisper (OpenAI SDK) requires 'openai_key' and 'openai_endpoint' for Azure OpenAI service."
        try:
            from openai import AzureOpenAI 
            client = AzureOpenAI(api_key=openai_key, api_version=str(config.get("API_VERSION")), azure_endpoint=openai_endpoint)
            with open(wav_file, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=str(config.get("AZURE_WHISPER_DEPLOYMENT_NAME", "whisper-1")), file=fh,
                    response_format="text", language=lang_code.split('-')[0] if lang_code else "en" 
                )
            return str(resp).strip() 
        except Exception as e:
            logger.error(f"Azure Whisper (OpenAI SDK) error: {e}", exc_info=True)
            return f"ERROR: Azure Whisper (OpenAI SDK) failed: {str(e)}"

    return f"ERROR: Unknown ASR model '{model}'. Supported: whisper:<size>, azure_speech, vosk_model, vosk_small, azure_whisper."