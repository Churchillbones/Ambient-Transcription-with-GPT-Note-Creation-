from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from ..exceptions import ConfigurationError, ServiceNotFoundError
from ..interfaces.transcription import ITranscriber
from .base_factory import IServiceFactory

# No direct imports of ASR modules here to avoid circular dependencies –
# they are imported lazily inside _register_default_providers()

__all__ = ["TranscriberFactory"]


class TranscriberFactory(IServiceFactory[ITranscriber]):
    """Factory responsible for constructing *ITranscriber* implementations.

    New providers can be registered dynamically via :py:meth:`register_provider`.
    """

    def __init__(self) -> None:  # noqa: D401
        self._providers: Dict[str, Type[ITranscriber]] = {}
        self._register_default_providers()

    # ------------------------------------------------------------------
    # IServiceFactory implementation
    # ------------------------------------------------------------------
    def create(self, provider_type: str, **kwargs: Any) -> ITranscriber:  # noqa: D401
        provider_type = provider_type.lower()
        if provider_type not in self._providers:
            raise ServiceNotFoundError(
                f"Transcriber provider '{provider_type}' not supported. "
                f"Available: {', '.join(self._providers)}"
            )

        provider_cls = self._providers[provider_type]
        return self._instantiate(provider_type, provider_cls, kwargs)

    def get_supported_providers(self) -> list[str]:  # noqa: D401
        return sorted(self._providers.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_provider(self, provider_type: str, cls: Type[ITranscriber]) -> None:
        """Register a new provider class at runtime."""
        if provider_type in self._providers:
            raise ConfigurationError(f"Provider '{provider_type}' is already registered")
        if not issubclass(cls, ITranscriber):
            raise ConfigurationError("Custom provider must implement ITranscriber")
        self._providers[provider_type] = cls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_default_providers(self) -> None:
        """Register the built-in Vosk, Whisper, and Azure speech providers."""
        # Import lazily to avoid circular-import issues when bootstrap
        # itself relies on this factory during application import time.
        from importlib import import_module

        _Vosk = import_module("src.asr.vosk").VoskTranscriber
        _Whisper = import_module("src.asr.whisper").WhisperTranscriber
        _AzureSpeech = import_module("src.asr.azure_speech").AzureSpeechTranscriber

        self._providers.update(
            {
                "vosk": _Vosk,
                "whisper": _Whisper,
                "azure_speech": _AzureSpeech,
            }
        )

    def _instantiate(
        self,
        provider_type: str,
        cls: Type[ITranscriber],
        options: Dict[str, Any],
    ) -> ITranscriber:
        """Instantiate *cls* with provider-specific fallbacks."""
        if provider_type == "vosk":
            model_path = options.get("model_path")
            if model_path is not None and not Path(model_path).exists():
                raise ConfigurationError(f"Vosk model path not found: {model_path}")
            return cls(model_path=model_path)  # type: ignore[arg-type]

        if provider_type == "whisper":
            size = options.get("size", "tiny")
            return cls(size=size)  # type: ignore[arg-type]

        if provider_type == "azure_speech":
            return cls(**options)  # type: ignore[arg-type]

        # Default: forward all options transparently
        return cls(**options)  # type: ignore[arg-type] 