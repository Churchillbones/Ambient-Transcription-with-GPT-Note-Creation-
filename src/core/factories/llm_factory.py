from __future__ import annotations

from typing import Any, Dict, Type

from ..exceptions import ConfigurationError, ServiceNotFoundError
from ..interfaces.llm_service import ILLMProvider
from .base_factory import IServiceFactory

from ..providers.azure_openai_provider import AzureOpenAIProvider
from ..providers.ollama_provider import OllamaProvider
from ..providers.local_llm_provider import LocalLLMProvider
from ..providers.openai_provider import OpenAIProvider

__all__ = ["LLMProviderFactory"]


class LLMProviderFactory(IServiceFactory[ILLMProvider]):
    """Factory for constructing *ILLMProvider* implementations."""

    def __init__(self) -> None:  # noqa: D401
        self._providers: Dict[str, Type[ILLMProvider]] = {
            "azure_openai": AzureOpenAIProvider,
            "ollama": OllamaProvider,
            "local": LocalLLMProvider,
            "openai": OpenAIProvider,
        }

    # ------------------------------------------------------------------
    def create(self, provider_type: str, **kwargs: Any) -> ILLMProvider:  # noqa: D401
        provider_type = provider_type.lower()
        if provider_type not in self._providers:
            raise ServiceNotFoundError(
                f"LLM provider '{provider_type}' not supported. Available: {', '.join(self._providers)}"
            )
        provider_cls = self._providers[provider_type]
        try:
            return provider_cls(**kwargs)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigurationError(f"Invalid arguments for provider '{provider_type}': {exc}") from exc

    def get_supported_providers(self) -> list[str]:  # noqa: D401
        return sorted(self._providers.keys())

    # ------------------------------------------------------------------
    def register_provider(self, provider_type: str, cls: Type[ILLMProvider]) -> None:
        if not issubclass(cls, ILLMProvider):
            raise ConfigurationError("Custom provider must implement ILLMProvider")
        self._providers[provider_type] = cls 