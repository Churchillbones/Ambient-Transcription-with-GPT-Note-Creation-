import os
import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
import bleach
import pyaudio
from .config import config, logger # Import config and logger from the same package

# --- Utility Functions ---
def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent XSS attacks."""
    return bleach.clean(user_input, tags=[], strip=True)

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file to use as a cache key."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found for hashing: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return ""

@contextmanager
def audio_stream(p=None, close_pyaudio=True):
    """Context manager for handling PyAudio streams."""
    audio_interface = None
    stream = None
    try:
        # Resolve PyAudio format constant
        try:
            audio_format = getattr(pyaudio, config["FORMAT_STR"])
        except AttributeError:
            logger.error(f"Invalid PyAudio format string in config: {config['FORMAT_STR']}")
            raise ValueError(f"Invalid PyAudio format: {config['FORMAT_STR']}")

        if p is None:
            audio_interface = pyaudio.PyAudio()
        else:
            audio_interface = p

        stream = audio_interface.open(
            format=audio_format,
            channels=config["CHANNELS"],
            rate=config["RATE"],
            input=True,
            frames_per_buffer=config["CHUNK"]
        )
        logger.debug("Audio stream opened successfully.")
        yield stream, audio_interface # Yield both stream and interface
    except Exception as e:
        logger.error(f"Failed to open audio stream: {e}")
        # Ensure resources are cleaned up even if opening fails partially
        if stream:
            try:
                stream.stop_stream()
                stream.close()
                logger.debug("Closed stream after error during opening.")
            except Exception as close_err:
                logger.error(f"Error closing stream after initial error: {close_err}")
        if audio_interface and close_pyaudio:
            try:
                audio_interface.terminate()
                logger.debug("Terminated PyAudio interface after error.")
            except Exception as term_err:
                logger.error(f"Error terminating PyAudio interface after error: {term_err}")
        raise # Re-raise the original exception
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
                logger.debug("Audio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
        if audio_interface and close_pyaudio:
            try:
                audio_interface.terminate()
                logger.debug("PyAudio interface terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio interface: {e}")


# --- Resource Monitoring ---
def monitor_resources():
    """Monitor system resources during transcription."""
    try:
        import psutil

        process = psutil.Process(os.getpid())

        # Get baseline - use interval=None for immediate reading
        baseline_cpu = process.cpu_percent(interval=None)
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"Resource monitor baseline: CPU={baseline_cpu}%, Memory={baseline_memory:.2f} MB")

        measurements = []

        # Take measurements
        def measure():
            try:
                # Use interval=None for non-blocking measurement
                cpu = process.cpu_percent(interval=None)
                memory = process.memory_info().rss / 1024 / 1024  # MB
                measurements.append((cpu, memory))
                logger.debug(f"Resource measurement: CPU={cpu}%, Memory={memory:.2f} MB")
            except psutil.NoSuchProcess:
                logger.warning("Process not found during resource measurement.")
            except Exception as e:
                logger.error(f"Error during resource measurement: {e}")


        # Return a function to get results
        def get_results():
            if not measurements:
                logger.debug("No resource measurements recorded.")
                return {"cpu_avg": 0, "memory_avg": 0, "peak_memory": baseline_memory} # Return baseline if no measurements

            cpu_values = [m[0] for m in measurements]
            memory_values = [m[1] for m in measurements]

            results = {
                "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "memory_avg": sum(memory_values) / len(memory_values) if memory_values else baseline_memory,
                "peak_memory": max(memory_values) if memory_values else baseline_memory
            }
            logger.debug(f"Resource results: Avg CPU={results['cpu_avg']:.2f}%, Avg Mem={results['memory_avg']:.2f} MB, Peak Mem={results['peak_memory']:.2f} MB")
            return results

        return measure, get_results

    except ImportError:
        logger.warning("psutil not installed. Resource monitoring disabled.")
        # Dummy functions if psutil not available
        def dummy_measure():
            pass

        def dummy_results():
            return {"cpu_avg": 0, "memory_avg": 0, "peak_memory": 0}

        return dummy_measure, dummy_results
    except Exception as e:
        logger.error(f"Failed to initialize resource monitor: {e}")
        # Return dummy functions in case of other errors
        def dummy_measure():
            pass
        def dummy_results():
            return {"cpu_avg": 0, "memory_avg": 0, "peak_memory": 0}
        return dummy_measure, dummy_results
