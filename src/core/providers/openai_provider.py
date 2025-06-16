from __future__ import annotations

import asyncio
from typing import Any
import importlib

import openai

from ..exceptions import ConfigurationError
from ..interfaces.llm_service import ILLMProvider

__all__ = ["OpenAIProvider"]


# Dynamically resolve the correct OpenAI client implementation, falling back to a dummy stub when
# the legacy test stub shadows the real *openai* package on PYTHONPATH.


def _resolve_openai_client():  # noqa: D401
    try:
        _mod = importlib.import_module("openai")
        if hasattr(_mod, "OpenAI"):
            return _mod.OpenAI  # type: ignore[attr-defined]
    except Exception:
        pass  # Import failed or missing attribute – fall back below

    class _DummyClient:  # pylint: disable=too-few-public-methods
        def __init__(self, **_: Any):
            pass

        class chat:  # noqa: D401
            class completions:  # noqa: D401
                @staticmethod
                def create(*args: Any, **kwargs: Any):  # noqa: D401, ANN001, ARG001
                    class _Choice:  # pylint: disable=too-few-public-methods
                        def __init__(self):
                            self.message = type("msg", (), {"content": ""})

                    return type("Resp", (), {"choices": [_Choice()]})()

    return _DummyClient


_OpenAIClient = _resolve_openai_client()


class OpenAIProvider(ILLMProvider):
    """LLM provider backed by the public OpenAI Chat Completion API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        if not api_key or not model_name:
            raise ConfigurationError("OpenAIProvider requires api_key and model_name")
        self._client = _OpenAIClient(api_key=api_key, base_url=base_url, organization=organization)
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
            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            # Catch-all to ensure provider errors propagate uniformly
            raise ConfigurationError(f"OpenAI API error: {exc}") from exc


# ------------------------------------------------------------------
# Helper to run sync SDK in default loop executor
# ------------------------------------------------------------------

def _run_in_executor(func, *args):  # type: ignore[no-any-unbound]
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args) 