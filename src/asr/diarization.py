from __future__ import annotations

import asyncio
import re
import logging
from typing import Optional

from ..core.container import global_container
from ..core.interfaces.config_service import IConfigurationService
from ..utils import sanitize_input
from core.bootstrap import container
from core.factories.llm_factory import LLMProviderFactory
from core.exceptions import ConfigurationError

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


class Diarizer:
    """Utility class for adding speaker tags to transcripts."""

    def apply(self, transcript: str) -> str:
        """Apply simple alternating speaker diarization (User/Patient)."""
        if not transcript:
            return ""
        lines = re.split(r"(?<=[.!?])\s+", transcript)
        processed = []
        speaker_idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            speaker = "User" if speaker_idx % 2 == 0 else "Patient"
            processed.append(f"{speaker}: {line}")
            if len(line.split()) > 2:
                speaker_idx += 1
        return "\n".join(processed)

    async def gpt_speaker_tags(
        self,
        transcript: str,
        api_key: Optional[str],
        *,
        endpoint: Optional[str] = None,
        api_ver: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Generate more accurate speaker tags using Azure OpenAI."""
        if not api_key:
            logger.warning("Azure API key not provided. Falling back to basic diarization.")
            return self.apply(transcript)
        if not transcript:
            logger.warning("Empty transcript provided for GPT diarization.")
            return ""
        sanitized = sanitize_input(transcript)
        if not sanitized:
            logger.warning("Transcript became empty after sanitization.")
            return ""
        
        # Get configuration from DI container
        try:
            config_service = global_container.resolve(IConfigurationService)
            default_endpoint = config_service.get("azure.endpoint")
            default_model = config_service.get("azure.model_name", "gpt-4o")
            default_api_version = config_service.get("azure.api_version", "2024-02-15-preview")
        except Exception:
            # Fallback defaults if DI not available
            default_endpoint = None
            default_model = "gpt-4o"
            default_api_version = "2024-02-15-preview"
        
        try:
            llm_factory = container.resolve(LLMProviderFactory)
            provider = llm_factory.create(
                "azure_openai",
                api_key=api_key,
                endpoint=endpoint or default_endpoint,
                model_name=model_name or default_model,
                api_version=api_ver or default_api_version,
            )
            prompt = (
                "System: You are an expert medical transcript editor. Your task is to accurately assign speaker roles (User or Patient) "
                "to each utterance in the provided clinical conversation transcript. Maintain the original wording. "
                "Format each utterance clearly as 'Speaker: Text'. If unsure, use 'Unknown Speaker:'.\n\n"
                "TRANSCRIPT:\n" + sanitized
            )
            tagged = await provider.generate_completion(prompt)
            if "User:" not in tagged and "Patient:" not in tagged:
                logger.warning("GPT output missing expected speaker tags. Falling back to basic diarization.")
                return self.apply(transcript)
            return tagged.strip()
        except ConfigurationError as e:
            logger.error(f"Provider configuration error during speaker tagging: {e}")
            return self.apply(transcript)
        except Exception as e:
            logger.error(f"Unexpected error generating speaker tags: {e}")
            return self.apply(transcript)


def apply_speaker_diarization(transcript: str) -> str:
    return Diarizer().apply(transcript)


aSYNC_DEPRECATED = "This function is deprecated; use Diarizer.gpt_speaker_tags instead."

async def generate_gpt_speaker_tags(
    transcript: str,
    api_key: Optional[str],
    endpoint: Optional[str] = None,
    api_ver: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    logger.warning(aSYNC_DEPRECATED)
    return await Diarizer().gpt_speaker_tags(transcript, api_key, endpoint=endpoint, api_ver=api_ver, model_name=model_name)
