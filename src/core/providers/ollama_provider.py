from __future__ import annotations

import asyncio
from typing import Any

import requests

from ..exceptions import ConfigurationError
from ..interfaces.llm_service import ILLMProvider

__all__ = ["OllamaProvider"]


class OllamaProvider(ILLMProvider):
    """Provider that calls a local Ollama HTTP API (via the bridge or direct)."""

    def __init__(self, *, base_url: str = "http://localhost:11434", model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    # ------------------------------------------------------------------
    async def generate_completion(self, prompt: str, **_: Any) -> str:  # noqa: D401
        payload = {"model": self._model, "prompt": prompt, "stream": False}
        return await _run_req(self._base_url + "/api/generate", payload)

    async def generate_note(self, transcript: str, **kwargs: Any) -> str:  # noqa: D401
        # Delegate to /generate_note endpoint on our Flask bridge if present
        endpoint = self._base_url + "/generate_note"
        payload = {"model": self._model, "prompt": transcript}
        return await _run_req(endpoint, payload)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

async def _run_req(url: str, json_payload: dict[str, Any]) -> str:
    def _post() -> str:
        resp = requests.post(url, json=json_payload, timeout=120)
        if resp.status_code != 200:
            raise ConfigurationError(f"LLM request failed {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data.get("note") or data.get("response") or data.get("cleaned_text") or ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _post) 