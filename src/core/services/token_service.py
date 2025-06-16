from __future__ import annotations

"""Token management helpers abstracted behind *ITokenService*.

Phase-2 extraction of logic formerly in *src/llm/token_management.py*.
"""

from typing import List

try:
    import tiktoken
except ImportError:  # pragma: no cover – optional dependency
    tiktoken = None  # type: ignore

from ..exceptions import ConfigurationError
from ..interfaces.token_service import ITokenService

__all__ = ["OpenAITokenService"]


class OpenAITokenService(ITokenService):
    """Token counting & chunking based on OpenAI `tiktoken` encodings."""

    def count(self, text: str, model: str = "gpt-4o") -> int:  # noqa: D401
        if not text:
            return 0
        if tiktoken is None:  # Fallback heuristic when library missing
            return int(len(text.split()) * 1.3)
        try:
            return len(tiktoken.encoding_for_model(model).encode(text))
        except Exception:
            try:
                return len(tiktoken.get_encoding("cl100k_base").encode(text))
            except Exception:
                return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    def chunk(self, transcript: str, max_chunk_tokens: int = 2048, model: str = "gpt-4o") -> List[str]:  # noqa: D401
        if not transcript:
            return []

        tokens = self.count(transcript, model)
        if tokens <= max_chunk_tokens:
            return [transcript]

        chunks: List[str] = []
        sentences = transcript.split(". ")
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            if sentence and not sentence.endswith("."):
                sentence += "."

            sent_tokens = self.count(sentence, model)

            if sent_tokens > max_chunk_tokens:
                # Fallback: split overly long sentence by words
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk, current_tokens = "", 0
                
                # Split sentence word by word, ensuring each chunk stays under limit
                words = sentence.split()
                word_buffer = []
                for word in words:
                    test_chunk = " ".join(word_buffer + [word])
                    if self.count(test_chunk, model) > max_chunk_tokens:
                        if word_buffer:  # Commit current buffer if not empty
                            chunks.append(" ".join(word_buffer))
                            word_buffer = [word]
                        else:  # Single word exceeds limit - force include it
                            chunks.append(word)
                            word_buffer = []
                    else:
                        word_buffer.append(word)
                
                if word_buffer:
                    current_chunk = " ".join(word_buffer)
                    current_tokens = self.count(current_chunk, model)
                continue

            # Check if adding this sentence would exceed the limit
            test_chunk = (current_chunk + " " + sentence).strip()
            test_tokens = self.count(test_chunk, model)
            
            if test_tokens > max_chunk_tokens:
                # Current chunk is full, start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sent_tokens
            else:
                # Add sentence to current chunk
                current_chunk = test_chunk
                current_tokens = test_tokens

        # Handle final chunk with safety check
        if current_chunk:
            final_tokens = self.count(current_chunk, model)
            if final_tokens > max_chunk_tokens:
                # Emergency word-by-word split
                words = current_chunk.split()
                word_buffer = []
                for word in words:
                    test_chunk = " ".join(word_buffer + [word])
                    if self.count(test_chunk, model) > max_chunk_tokens:
                        if word_buffer:
                            chunks.append(" ".join(word_buffer))
                            word_buffer = [word]
                        else:
                            chunks.append(word)
                            word_buffer = []
                    else:
                        word_buffer.append(word)
                if word_buffer:
                    chunks.append(" ".join(word_buffer))
            else:
                chunks.append(current_chunk.strip())
        
        return chunks


# ------------------------------------------------------------------
# Container registration helper (import-side-effect)
# ------------------------------------------------------------------

from ..container import global_container as _container  # noqa: E402  pylint: disable=wrong-import-position

try:
    # Register singleton only if not already bound (bootstrapping idempotent)
    from ..interfaces.token_service import ITokenService as _Iface  # noqa: E402

    if _Iface not in _container.registrations:
        _container.register_instance(_Iface, OpenAITokenService())
except ConfigurationError:  # pragma: no cover – circular/duplicate registration
    pass 