import asyncio
from typing import Optional
from openai import AzureOpenAI, OpenAIError # Import specific error for better handling

from ..config import config, logger
from ..utils import sanitize_input # Import from utils

# --- Speaker Diarization ---

def apply_speaker_diarization(transcript: str) -> str:
    """Apply simple alternating speaker diarization (User/Patient)."""
    if not transcript:
        return ""

    # Split by sentence endings more robustly
    import re
    lines = re.split(r'(?<=[.!?])\s+', transcript) # Split after sentence-ending punctuation

    processed_lines = []
    current_speaker_index = 0 # Start with User

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Simple heuristic: Assume speaker changes unless line is very short (e.g., "Okay.")
        # This is very basic and likely needs improvement based on real data.
        # A better approach might involve looking for keywords or using a more sophisticated model.
        # For now, stick to simple alternation.

        speaker = "User" if current_speaker_index % 2 == 0 else "Patient"
        processed_lines.append(f"{speaker}: {line}")

        # Only increment speaker index if the line seems substantial enough
        if len(line.split()) > 2: # Arbitrary threshold
             current_speaker_index += 1


    return "\n".join(processed_lines)

async def generate_gpt_speaker_tags(transcript: str, api_key: Optional[str], 
                                   endpoint: Optional[str] = None, api_ver: Optional[str] = None, 
                                   model_name: Optional[str] = None) -> str:
    """Use Azure OpenAI (GPT) to generate more accurate speaker tags, or fall back to basic diarization if no API key."""
    if not api_key:
        logger.warning("Azure API key not provided. Falling back to basic diarization.")
        return apply_speaker_diarization(transcript)
    if not transcript:
        logger.warning("Empty transcript provided for GPT diarization.")
        return ""

    sanitized_transcript = sanitize_input(transcript)
    if not sanitized_transcript:
        logger.warning("Transcript became empty after sanitization.")
        return ""

    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_ver or config["API_VERSION"],
            azure_endpoint=endpoint or config["AZURE_ENDPOINT"],
            timeout=30.0 # Add a timeout
        )

        # Improved prompt for better speaker tagging
        prompt = (
            "System: You are an expert medical transcript editor. Your task is to accurately assign speaker roles (User or Patient) "
            "to each utterance in the provided clinical conversation transcript. Maintain the original wording. "
            "Format each utterance clearly as 'Speaker: Text'. If unsure, use 'Unknown Speaker:'.\n\n"
            "TRANSCRIPT:\n"
            f"{sanitized_transcript}"
        )

        logger.info(f"Requesting speaker tagging from Azure OpenAI model: {model_name or config['MODEL_NAME']}")
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model_name or config["MODEL_NAME"],
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2, # Lower temperature for more deterministic output
                max_tokens=int(len(sanitized_transcript.split()) * 1.5) + 100 # Estimate output tokens
            )
        )

        tagged_transcript = response.choices[0].message.content
        logger.info("Successfully received speaker tags from Azure OpenAI.")
        # Basic validation: Check if the output contains expected speaker tags
        if "User:" not in tagged_transcript and "Patient:" not in tagged_transcript:
             logger.warning("GPT output did not contain expected 'User:' or 'Patient:' tags. Returning original with basic diarization.")
             return apply_speaker_diarization(transcript) # Fallback

        return tagged_transcript.strip()

    except OpenAIError as e:
        logger.error(f"Azure OpenAI API error during speaker tagging: {e}")
        # Fallback to basic diarization on API errors
        return apply_speaker_diarization(transcript)
    except Exception as e:
        logger.error(f"Unexpected error generating speaker tags: {e}")
        # Fallback to basic diarization on other errors
        return apply_speaker_diarization(transcript)
