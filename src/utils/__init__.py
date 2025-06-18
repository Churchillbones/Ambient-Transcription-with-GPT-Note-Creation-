import os
import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
import bleach
import pyaudio
from typing import List, Dict, Any, Union
try:
    import numpy as np
except ImportError:
    np = None

from ..core.container import global_container
from ..core.interfaces.config_service import IConfigurationService

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


def _get_audio_config():
    """Helper to get audio configuration from DI container with fallbacks."""
    try:
        config_service = global_container.resolve(IConfigurationService)
        return {
            "format_str": "paInt16",  # 16-bit PCM
            "channels": 1,            # Mono
            "rate": 16000,           # 16 kHz
            "chunk": 1024,           # Default chunk size
        }
    except Exception:
        # Fallback values if DI not available
        return {
            "format_str": "paInt16",
            "channels": 1,
            "rate": 16000,
            "chunk": 1024,
        }


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
        # Get audio configuration
        audio_config = _get_audio_config()
        
        # Resolve PyAudio format constant
        try:
            audio_format = getattr(pyaudio, audio_config["format_str"])
        except AttributeError:
            logger.error(f"Invalid PyAudio format string in config: {audio_config['format_str']}")
            raise ValueError(f"Invalid PyAudio format: {audio_config['format_str']}")

        if p is None:
            audio_interface = pyaudio.PyAudio()
        else:
            audio_interface = p

        stream = audio_interface.open(
            format=audio_format,
            channels=audio_config["channels"],
            rate=audio_config["rate"],
            input=True,
            frames_per_buffer=audio_config["chunk"]
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

# --- Embedding Utilities ---
try:
    from ..llm.embedding_service import EmbeddingService
except ImportError:
    logger.warning("EmbeddingService module not found. Embedding features will be unavailable.")
    EmbeddingService = None

def get_embedding_service(api_key: str = None, endpoint: str = None, api_version: str = "2025-01-01-preview", verify_ssl: bool = False):
    """Get properly configured embedding service instance"""
    if EmbeddingService is None:
        logger.error("EmbeddingService module not available")
        return None
        
    if not api_key:
        # Try to get from DI configuration service
        try:
            config_service = global_container.resolve(IConfigurationService)
            api_key = config_service.get("azure.api_key", "")
        except Exception:
            api_key = ""
    
    if not endpoint:
        # Use default VA endpoint
        endpoint = "https://va-east-apim.devtest.spd.vaec.va.gov/openai/deployments/text-embedding-3-large/embeddings"
    
    return EmbeddingService(api_key, endpoint, api_version, verify_ssl)

def semantic_chunking(text: str, max_tokens: int = 1000, min_chunk_size: int = 200) -> List[str]:
    """Chunk text into semantic units rather than arbitrary splits"""
    # 1. Split into sentences or paragraphs
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 2. Create chunks that respect semantic boundaries
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Estimate token count - roughly 4 chars per token
        sentence_length = len(sentence) // 4
        
        # If adding this sentence would exceed max_tokens, start a new chunk
        if current_length + sentence_length > max_tokens and current_length >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def find_similar_chunks(chunks: List[str], embedding_service, 
                        target_index: int, top_k: int = 2) -> List[int]:
    """Find chunks most similar to the target chunk"""
    if not embedding_service:
        logger.warning("Embedding service not available, cannot find similar chunks")
        return []
        
    # Get embeddings for all chunks
    embeddings = embedding_service.get_batch_embeddings(chunks)
    
    # Calculate similarity between target and all other chunks
    target_embedding = embeddings[target_index]
    similarities = []
    
    for i, emb in enumerate(embeddings):
        if i == target_index:
            continue  # Skip self
        sim = embedding_service.cosine_similarity(target_embedding, emb)
        similarities.append((i, sim))
    
    # Sort by similarity and get top_k indices
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in similarities[:top_k]]

def cluster_by_topic(chunks: List[str], embedding_service, 
                     num_clusters: int = 5) -> Dict[int, List[int]]:
    """Cluster chunks by topic using embeddings"""
    if not embedding_service or not np:
        logger.warning("Embedding service or numpy not available, cannot cluster chunks")
        return {}
        
    # Get embeddings for all chunks
    embeddings = embedding_service.get_batch_embeddings(chunks)
    
    # Use KMeans clustering
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=min(num_clusters, len(chunks)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group chunk indices by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters
