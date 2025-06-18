import json
import wave
import tempfile
import logging
from pathlib import Path
from typing import Tuple
from contextlib import contextmanager
from functools import lru_cache
from cryptography.fernet import Fernet

from .core.container import global_container
from .core.interfaces.config_service import IConfigurationService

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


def _get_encryption_config():
    """Helper to get encryption configuration from DI container with fallbacks."""
    try:
        config_service = global_container.resolve(IConfigurationService)
        base_dir = config_service.get("base_dir", Path("./app_data"))
        return {
            "key_dir": base_dir / "keys",
            "cache_dir": base_dir / "cache",
        }
    except Exception:
        # Fallback values if DI not available
        return {
            "key_dir": Path("./app_data/keys"),
            "cache_dir": Path("./app_data/cache"),
        }


# --- Encryption Functions ---
@lru_cache(maxsize=None)
def get_encryption_key() -> bytes:
    """Get or generate an encryption key."""
    enc_config = _get_encryption_config()
    # Ensure the key directory exists
    enc_config["key_dir"].mkdir(parents=True, exist_ok=True)
    
    key_file = enc_config["key_dir"] / "encryption_key.bin"
    if not key_file.exists():
        key = Fernet.generate_key()
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info("Generated new encryption key")
            return key
        except Exception as e:
            logger.error(f"Failed to write new encryption key: {e}")
            raise
    try:
        with open(key_file, 'rb') as f:
            key = f.read()
            # Basic validation: Fernet keys are base64 encoded and 44 bytes long
            if len(key) != 44:
                 logger.warning(f"Encryption key file {key_file} has unexpected length: {len(key)}")
                 # Optionally raise an error or attempt to regenerate
                 # raise ValueError("Invalid key length found.")
            return key
    except Exception as e:
        logger.error(f"Failed to read encryption key: {e}")
        raise

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt binary data."""
    try:
        return Fernet(key).encrypt(data)
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise ValueError("Encryption process failed.")


def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt binary data."""
    try:
        return Fernet(key).decrypt(encrypted_data)
    except Exception as e:
        # Log the specific Fernet error if possible, but don't expose details to user
        logger.error(f"Decryption failed: {e}")
        # Raise a generic error to avoid leaking information
        raise ValueError("Failed to decrypt data. Key may be invalid or data corrupted.")


def encrypt_wav_file(wav_path: str, key: bytes) -> Tuple[str, bool]:
    """
    Encrypt a WAV file using a custom format with metadata.
    Returns the path to the encrypted file and a success flag.
    """
    try:
        # Get WAV file properties before encryption
        with wave.open(wav_path, 'rb') as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)
            if nframes == 0 or len(frames) == 0:
                logger.warning(f"Input WAV file {wav_path} appears empty.")
                # Decide how to handle empty files, e.g., return failure or encrypt an empty payload
                # return wav_path, False # Example: treat as failure

        # Encrypt only the audio frames
        encrypted_frames = encrypt_data(frames, key)

        # Store metadata alongside encrypted data
        metadata = {
            "channels": channels,
            "sampwidth": sampwidth,
            "framerate": framerate,
            "nframes": nframes, # Store original frame count
            "encrypted": True
        }

        # Save as custom format '.enc'
        enc_path = str(Path(wav_path).with_suffix('.enc'))
        with open(enc_path, 'wb') as f:
            # Magic identifier (4 bytes)
            f.write(b'MENC')
            # Version (4 bytes, little-endian)
            f.write((1).to_bytes(4, byteorder='little'))
            # Metadata length (4 bytes, little-endian) + JSON metadata
            metadata_json = json.dumps(metadata).encode('utf-8')
            f.write(len(metadata_json).to_bytes(4, byteorder='little'))
            f.write(metadata_json)
            # Encrypted audio frames
            f.write(encrypted_frames)

        logger.info(f"Encrypted audio saved to {enc_path}")
        return enc_path, True
    except wave.Error as e:
        logger.error(f"Error reading WAV file {wav_path}: {e}")
        return wav_path, False
    except Exception as e:
        logger.error(f"Encryption failed for {wav_path}: {e}")
        return wav_path, False


def decrypt_to_wav(enc_path: str, key: bytes) -> Tuple[str, bool]:
    """
    Decrypt an encrypted audio file (.enc) and reconstruct a valid WAV file.
    Returns the path to the temporary decrypted WAV file and a success flag.
    """
    temp_path = ""
    try:
        with open(enc_path, 'rb') as f:
            # Read and verify magic identifier
            magic = f.read(4)
            if magic != b'MENC':
                raise ValueError("Not a valid encrypted audio file (invalid magic number)")

            # Read version
            version = int.from_bytes(f.read(4), byteorder='little')
            if version != 1:
                raise ValueError(f"Unsupported encryption version: {version}")

            # Read metadata
            metadata_len = int.from_bytes(f.read(4), byteorder='little')
            metadata_json = f.read(metadata_len).decode('utf-8')
            metadata = json.loads(metadata_json)

            # Read encrypted frames
            encrypted_frames = f.read()

        # Decrypt the audio frames
        frames = decrypt_data(encrypted_frames, key) # Handles its own exceptions

        # Create a temporary WAV file
        enc_config = _get_encryption_config()
        # Ensure cache directory exists
        enc_config["cache_dir"].mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=enc_config["cache_dir"]) as tmp:
            temp_path = tmp.name

        # Write a proper WAV file with the decrypted frames
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(metadata["channels"])
            wf.setsampwidth(metadata["sampwidth"])
            wf.setframerate(metadata["framerate"])
            # Use nframes from metadata if available, otherwise calculate
            nframes_meta = metadata.get("nframes")
            if nframes_meta is not None:
                 wf.setnframes(nframes_meta)
            wf.writeframes(frames)

        # Verify the WAV file is valid (optional but recommended)
        try:
            with wave.open(temp_path, 'rb') as test_wf:
                test_frames_read = test_wf.readframes(1) # Try reading one frame
                logger.debug(f"Successfully verified decrypted WAV file: {temp_path}")
        except wave.Error as e:
            logger.error(f"Decrypted WAV file validation failed: {e}")
            # Clean up invalid temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise ValueError("Failed to create a valid WAV file from decrypted data")

        logger.info(f"Decrypted audio saved to temporary file: {temp_path}")
        return temp_path, True

    except FileNotFoundError:
        logger.error(f"Encrypted file not found: {enc_path}")
        return "", False
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode metadata from {enc_path}: {e}")
        return "", False
    except ValueError as e: # Catch specific errors from decrypt_data or validation
        logger.error(f"Decryption or validation error for {enc_path}: {e}")
        # Clean up potentially created temp file
        if temp_path and Path(temp_path).exists():
             try: Path(temp_path).unlink()
             except OSError: pass
        return "", False
    except Exception as e:
        logger.error(f"General decryption failed for {enc_path}: {e}")
        # Clean up potentially created temp file
        if temp_path and Path(temp_path).exists():
             try: Path(temp_path).unlink()
             except OSError: pass
        return "", False


@contextmanager
def secure_audio_processing(audio_path: str, use_encryption: bool):
    """
    Context manager to handle encryption/decryption workflow for audio processing.
    Yields the path to use for processing (original or decrypted temp) and handles cleanup.
    """
    temp_paths_to_clean = []
    processing_path = audio_path
    original_encrypted_path = "" # Keep track of the .enc file if created

    try:
        if use_encryption:
            logger.debug(f"Secure processing requested for: {audio_path}")
            key = get_encryption_key() # Can raise exceptions

            # 1. Encrypt the original WAV file
            enc_path, encryption_success = encrypt_wav_file(audio_path, key)
            if not encryption_success:
                # If encryption fails, warn and proceed with original unencrypted path
                logger.warning(f"Failed to encrypt {Path(audio_path).name}. Processing unencrypted audio.")
                # No need to decrypt, yield original path
                yield audio_path
                return # Exit context manager early

            original_encrypted_path = enc_path # Store path to potentially keep
            logger.info(f"Successfully encrypted to: {enc_path}")

            # 2. Decrypt the .enc file to a temporary WAV for processing
            decrypted_temp_path, decryption_success = decrypt_to_wav(enc_path, key)
            if not decryption_success:
                # If decryption fails (unexpectedly, as we just encrypted it), warn and fallback
                logger.warning("Decryption failed after encryption. Processing unencrypted audio.")
                logger.error(f"Decryption failed for {enc_path} immediately after creation. Using original path {audio_path}.")
                # Yield original path as fallback
                yield audio_path
                return # Exit context manager early

            # If decryption succeeded, use the temp path for processing
            processing_path = decrypted_temp_path
            temp_paths_to_clean.append(decrypted_temp_path) # Mark for cleanup
            logger.debug(f"Using temporary decrypted path for processing: {processing_path}")
            yield processing_path # Yield the path to the temporary decrypted WAV

        else:
            # No encryption requested, just yield the original path
            logger.debug(f"No encryption requested for: {audio_path}. Using original path.")
            yield audio_path

    except Exception as e:
        # Catch any other exceptions during the setup phase (e.g., key generation)
        logger.error(f"Error during secure audio processing setup for {audio_path}: {e}")
        logger.error(f"An error occurred during secure audio setup: {e}")
        # In case of error during setup, yield the original path as a last resort? Or raise?
        # Raising might be safer to prevent processing potentially compromised data.
        raise # Re-raise the exception to halt processing

    finally:
        # Cleanup: Remove temporary decrypted WAV files
        for path in temp_paths_to_clean:
            try:
                if path and Path(path).exists():
                    Path(path).unlink()
                    logger.debug(f"Removed temporary decrypted file: {path}")
            except Exception as e:
                logger.error(f"Failed to remove temp file {path}: {e}")

        # Decision: Keep or remove the original .wav file after creating .enc?
        # If encryption was successful and requested, we might want to remove the original .wav
        # This depends on the desired workflow (e.g., always keep encrypted, remove original)
        # Current implementation keeps the original .wav alongside the .enc
        # To remove original wav after successful encryption:
        # if use_encryption and encryption_success and Path(audio_path).exists() and audio_path != original_encrypted_path:
        #     try:
        #         Path(audio_path).unlink()
        #         logger.info(f"Removed original WAV file after encryption: {audio_path}")
        #     except Exception as e:
        #         logger.error(f"Failed to remove original WAV file {audio_path}: {e}")
        pass # Keep original WAV for now
