from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from openai import AzureOpenAI, OpenAIError  # Stub class in repo for offline tests

from ..exceptions import ConfigurationError
from ..interfaces.llm_service import ILLMProvider

__all__ = ["AzureOpenAIProvider"]


class AzureOpenAIProvider(ILLMProvider):
    """LLM provider backed by Azure OpenAI Chat Completion API."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model_name: str,
        api_version: str = "2024-02-15-preview",
    ) -> None:
        if not api_key or not endpoint or not model_name:
            raise ConfigurationError("AzureOpenAIProvider requires api_key, endpoint, and model_name")
        self._client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self._model_name = model_name

    # ------------------------------------------------------------------
    async def generate_completion(self, prompt: str, **_: Any) -> str:  # noqa: D401
        return await _run_in_executor(self._chat_completion, prompt)

    async def generate_note(self, transcript: str, **kwargs: Any) -> str:  # noqa: D401
        template: str = kwargs.get("template", "")
        prompt = template.format(transcript=transcript) if template else transcript
        return await self.generate_completion(prompt)

    # ------------------------------------------------------------------
    def _chat_completion(self, prompt: str) -> str:
        try:
            completion = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content  # type: ignore[index]
        except OpenAIError as exc:  # pragma: no cover – network failures
            raise ConfigurationError(f"Azure OpenAI error: {exc}") from exc


# ------------------------------------------------------------------
# Helper to run sync SDK in default loop executor
# ------------------------------------------------------------------

def _run_in_executor(func, *args):  # type: ignore[no-any-unbound]
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args) 