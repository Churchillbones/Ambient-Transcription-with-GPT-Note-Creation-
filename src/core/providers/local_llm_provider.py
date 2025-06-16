from __future__ import annotations

import asyncio
from typing import Any

import requests

from ..exceptions import ConfigurationError
from ..interfaces.llm_service import ILLMProvider
from ..container import global_container
from ..interfaces.config_service import IConfigurationService

__all__ = ["LocalLLMProvider"]


class LocalLLMProvider(ILLMProvider):
    """Generic HTTP provider hitting a configurable local LLM endpoint."""

    def __init__(self, *, endpoint: str | None = None, model: str | None = None) -> None:
        cfg_service = None
        try:
            cfg_service = global_container.resolve(IConfigurationService)
        except Exception:
            pass
        self._endpoint = (endpoint or (cfg_service.get("local_model_api_url") if cfg_service else None)) or "http://localhost:8000/generate_note"
        self._model = model or "gemma3-4b"

    # ------------------------------------------------------------------
    async def generate_completion(self, prompt: str, **_: Any) -> str:  # noqa: D401
        return await _post_json(self._endpoint, {"prompt": prompt, "model": self._model})

    async def generate_note(self, transcript: str, **kwargs: Any) -> str:  # noqa: D401
        return await self.generate_completion(transcript)


async def _post_json(url: str, payload: dict[str, Any]) -> str:
    def _post() -> str:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise ConfigurationError(f"Local LLM request failed {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("note", "")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _post) 