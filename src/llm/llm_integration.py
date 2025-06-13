import asyncio
import requests
from typing import Optional, Dict, Tuple, List
from openai import AzureOpenAI, OpenAIError

from ..config import config, logger
from ..utils import sanitize_input
from .token_management import (
    count_tokens,
    generate_note_from_chunked_transcript,
    generate_note_with_two_stage_summarization
)
from .llm_agent_enhanced import generate_note_agent_based as perform_agent_generation
# Ensure this is placed with other . (relative) imports


async def generate_note_router(
    transcript: str,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_model_name: Optional[str] = None,
    prompt_template: str = "",
    use_local: bool = False, # For traditional local model path
    local_model: str = "",   # For traditional local model path
    patient_data: Optional[Dict] = None,
    use_agent_pipeline: bool = False, # Flag from UI to enable agent pipeline
    agent_settings: Optional[Dict[str, Any]] = None,  # Settings for the agent pipeline from UI
    progress_callback: Optional[Any] = None # Pass progress_callback for agent pipeline UI updates
) -> Tuple[str, Dict[str, Any]]:
    """
    Routes note generation to either the traditional pipeline or the new agent-based pipeline.
    The agent-based pipeline is used if 'use_agent_pipeline' is True and not 'use_local'.
    Fallback to traditional if agent pipeline fails.
    """
    logger.info(f"generate_note_router called. use_agent_pipeline: {use_agent_pipeline}, use_local: {use_local}")

    # Metadata to be returned along with the note
    metadata_result: Dict[str, Any] = {
        "pipeline_type_attempted": "agent" if use_agent_pipeline and not use_local else "traditional",
        "agent_based_processing_used": False, # Default to false, set true if agent pipeline runs
        "fallback_triggered": False,
        "fallback_reason": None
    }

    if use_agent_pipeline and not use_local:
        logger.info("Attempting agent-based note generation.")
        if not api_key or not azure_endpoint or not azure_api_version or not azure_model_name:
            logger.error("Agent pipeline selected, but Azure credentials or model name are missing.")
            metadata_result["fallback_triggered"] = True
            metadata_result["fallback_reason"] = "Missing Azure credentials or model name for agent pipeline."
            # Proceed to fallback section by not returning here
        else:
            try:
                # Call the core agent generation function from llm_agent_enhanced.py
                note_text, agent_metadata = await perform_agent_generation(
                    transcript=transcript,
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    azure_model_name=azure_model_name,
                    prompt_template=prompt_template,
                    patient_data=patient_data,
                    agent_settings=agent_settings or {},
                    progress_callback=progress_callback
                )

                # Combine metadata from agent pipeline with router's metadata
                metadata_result.update(agent_metadata)
                # Ensure agent_based_processing_used is correctly set from agent_metadata or explicitly
                metadata_result["agent_based_processing_used"] = agent_metadata.get("agent_based_processing_used", True)

                logger.info("Agent-based note generation successful.")
                return note_text, metadata_result

            except Exception as e:
                logger.error(f"Agent-based pipeline failed: {e}. Falling back to traditional method.", exc_info=True)
                metadata_result["fallback_triggered"] = True
                metadata_result["fallback_reason"] = f"Agent pipeline error: {str(e)}"
                # Proceed to fallback (traditional method) which is handled after this if-block

    # Fallback to traditional method or if traditional was chosen initially
    # This block is reached if:
    # 1. use_agent_pipeline was False
    # 2. use_local was True
    # 3. Agent pipeline was selected but Azure credentials were missing (and fallback_triggered was set)
    # 4. Agent pipeline was attempted and an exception occurred (and fallback_triggered was set)

    logger.info("Using traditional note generation method (either as primary choice or as fallback).")
    # Update pipeline_type_attempted if it was 'agent' but failed leading to fallback
    if metadata_result["pipeline_type_attempted"] == "agent" and metadata_result["fallback_triggered"]:
         logger.info(f"Fallback to traditional method was triggered. Reason: {metadata_result['fallback_reason']}")

    metadata_result["pipeline_type_attempted"] = "traditional" # Set/confirm as traditional path

    # Call the existing traditional 'generate_note' function from this file
    traditional_note_text = await generate_note(
        transcript=transcript,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_model_name=azure_model_name,
        prompt_template=prompt_template,
        use_local=use_local,
        local_model=local_model,
        patient_data=patient_data
    )

    metadata_result["agent_based_processing_used"] = False # Explicitly false for traditional
    metadata_result["traditional_note_details"] = {"note_length": len(traditional_note_text)} # Example detail

    logger.info("Traditional note generation complete.")
    return traditional_note_text, metadata_result

# --- Transcription Cleanup ---
async def clean_transcription(
    transcript: str,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_model_name: Optional[str] = None,
    use_local: bool = False,
    local_model: str = "",
    highlight_terms: bool = True
) -> str:
    """
    Clean up transcription using AI to correct medical spelling, punctuation,
    and optionally highlight important medical terms.

    Args:
        transcript: Raw transcription text
        api_key: API key for Azure OpenAI (if using Azure)
        azure_endpoint: Azure endpoint URL
        azure_api_version: Azure API version
        azure_model_name: Azure model deployment name
        use_local: Whether to use local LLM instead of Azure
        local_model: Specific local model to use
        highlight_terms: Whether to highlight important medical terms

    Returns:
        Cleaned transcription text with potential highlights
    """
    if not transcript:
        logger.warning("clean_transcription called with empty transcript.")
        return "ERROR: No transcript provided for cleanup."

    sanitized_transcript = sanitize_input(transcript)
    if not sanitized_transcript:
        logger.warning("Transcript became empty after sanitization in clean_transcription.")
        return "ERROR: Transcript content is invalid."

    # Prepare the cleanup prompt
    if highlight_terms:
        prompt = (
            "Clean up this medical transcription by:\n"
            "1. Fixing spelling and grammar errors\n"
            "2. Correcting medical terminology\n"
            "3. Improving punctuation and formatting\n"
            "4. Highlighting important medical terms with **asterisks** around them\n\n"
            "Only make necessary changes to improve accuracy and readability. "
            "Keep the original meaning and speaker information intact.\n\n"
            "ORIGINAL TRANSCRIPTION:\n"
            f"{sanitized_transcript}"
        )
    else:
        prompt = (
            "Clean up this medical transcription by:\n"
            "1. Fixing spelling and grammar errors\n"
            "2. Correcting medical terminology\n"
            "3. Improving punctuation and formatting\n\n"
            "Only make necessary changes to improve accuracy and readability. "
            "Keep the original meaning and speaker information intact.\n\n"
            "ORIGINAL TRANSCRIPTION:\n"
            f"{sanitized_transcript}"
        )

    logger.debug(f"Transcription cleanup prompt (first 100 chars): {prompt[:100]}...")

    try:
        if use_local:
            # --- Use local LLM model API ---
            if not config["LOCAL_MODEL_API_URL"]:
                logger.error("Local LLM API URL is not configured.")
                return sanitized_transcript  # Fall back to original transcript

            # Modify the endpoint to differentiate from note generation
            cleanup_endpoint = f"{config['LOCAL_MODEL_API_URL'].rstrip('/')}/cleanup"

            request_payload = {"prompt": prompt}
            if local_model:
                request_payload["model"] = local_model
                logger.info(f"Sending cleanup request to local LLM: {local_model}")
            else:
                logger.info("Sending cleanup request to default local LLM model")

            try:
                # Use asyncio.to_thread for the blocking requests call
                response = await asyncio.to_thread(
                    lambda: requests.post(
                        cleanup_endpoint,
                        json=request_payload,
                        timeout=30  # Shorter timeout for cleanup vs full note generation
                    )
                )
                response.raise_for_status()

                response_data = response.json()
                cleaned_text = response_data.get("cleaned_text", "")

                if not cleaned_text:
                    logger.warning("Local LLM returned empty cleaned text.")
                    return sanitized_transcript  # Fall back to original

                logger.info("Successfully received cleaned transcription from local LLM.")
                return cleaned_text

            except requests.exceptions.Timeout:
                logger.error("Request to local LLM for cleanup timed out.")
                return sanitized_transcript  # Fall back to original
            except requests.exceptions.RequestException as e:
                logger.error(f"Local LLM cleanup request failed: {e}")
                return sanitized_transcript  # Fall back to original

        else:
            # --- Use Azure OpenAI ---
            if not api_key:
                logger.warning("Azure API key not provided for transcription cleanup.")
                return sanitized_transcript  # Fall back to original
            # Check other required params passed down
            if not azure_endpoint or not azure_api_version or not azure_model_name:
                 logger.warning("Azure endpoint, version, or model name missing for cleanup.")
                 return sanitized_transcript

            client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version, # Use passed param
                azure_endpoint=azure_endpoint, # Use passed param
                timeout=30.0  # Shorter timeout for cleanup
            )

            logger.info(f"Requesting transcription cleanup from Azure OpenAI model: {azure_model_name}")
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=azure_model_name, # Use passed param
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.3,  # Lower temperature for accuracy
                    max_tokens=int(len(prompt.split()) * 1.2)  # Should be similar length to input
                )
            )

            cleaned_text = response.choices[0].message.content
            if not cleaned_text:
                logger.warning("Azure OpenAI returned empty cleaned text.")
                return sanitized_transcript  # Fall back to original

            logger.info("Successfully received cleaned transcription from Azure OpenAI.")
            return cleaned_text.strip()

    except OpenAIError as e:
        logger.error(f"Azure OpenAI API error during transcription cleanup: {e}")
        return sanitized_transcript  # Fall back to original
    except Exception as e:
        logger.error(f"Unexpected error in transcription cleanup: {e}")
        return sanitized_transcript  # Fall back to original

# --- Note Generation ---
async def generate_note(
    transcript: str,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_model_name: Optional[str] = None,
    prompt_template: str = "",
    use_local: bool = False,
    local_model: str = "",
    patient_data: Optional[Dict] = None
) -> str:
    """Generate a structured note from transcript using either Azure or local model."""
    if not transcript:
        logger.warning("generate_note called with empty transcript.")
        return "Error: No transcript provided for note generation."

    sanitized_transcript = sanitize_input(transcript)
    if not sanitized_transcript:
        logger.warning("Transcript became empty after sanitization in generate_note.")
        return "Error: Transcript content is invalid."

    # Determine the final prompt, incorporating patient data if available
    final_prompt_template = prompt_template or "Create a clinical note from the following transcription."

    # Construct patient info string safely
    patient_info_lines = []
    if patient_data:
        if patient_data.get("name"):
            patient_info_lines.append(f"Name: {sanitize_input(patient_data['name'])}") # Sanitize patient name
        if patient_data.get("ehr_data"):
            # Sanitize EHR data more carefully if needed, depending on expected content
            patient_info_lines.append(f"\nEHR DATA:\n{sanitize_input(patient_data['ehr_data'])}") # Basic sanitize

    if patient_info_lines:
        patient_info_str = "\n".join(patient_info_lines)
        # Check if the template already has a placeholder for transcription
        if "{transcription}" in final_prompt_template:
             # Inject patient info before the transcription placeholder or at a suitable point
             # This might need refinement based on typical template structures
             parts = final_prompt_template.split("{transcription}", 1)
             prompt = f"{parts[0]}\n\nPATIENT INFORMATION:\n{patient_info_str}\n\nENCOUNTER TRANSCRIPTION:\n{{transcription}}{parts[1]}"
             prompt = prompt.format(transcription=sanitized_transcript)
        else:
             # Append patient info and transcription if no placeholder exists
             prompt = f"{final_prompt_template}\n\nPATIENT INFORMATION:\n{patient_info_str}\n\nENCOUNTER TRANSCRIPTION:\n{sanitized_transcript}"
    else:
        # Use standard prompt formatting if no patient data
        if "{transcription}" in final_prompt_template:
             prompt = final_prompt_template.format(transcription=sanitized_transcript)
        else:
             prompt = f"{final_prompt_template}\n\n{sanitized_transcript}"


    logger.debug(f"Final prompt for note generation (first 100 chars): {prompt[:100]}...")
    try:
        if use_local:
            # --- Use local LLM model API ---
            if not config["LOCAL_MODEL_API_URL"]:
                 logger.error("Local LLM API URL is not configured.")
                 return "Error: Local LLM endpoint not configured."

            request_payload = {"prompt": prompt}
            if local_model:
                request_payload["model"] = local_model
                logger.info(f"Sending request to local LLM: {local_model} at {config['LOCAL_MODEL_API_URL']}")
            else:
                 logger.info(f"Sending request to local LLM at {config['LOCAL_MODEL_API_URL']} (default model)")


            try:
                # Use asyncio.to_thread for the blocking requests call
                response = await asyncio.to_thread(
                    lambda: requests.post(
                        config["LOCAL_MODEL_API_URL"],
                        json=request_payload,
                        timeout=60 # Increased timeout for potentially slower local models
                    )
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                response_data = response.json()
                note = response_data.get("note", "")
                if not note:
                     logger.warning("Local model returned an empty note.")
                     return "Note generation result was empty."
                logger.info("Successfully received note from local LLM.")
                return note

            except requests.exceptions.Timeout:
                logger.error("Request to local LLM timed out.")
                return "Error: Request to local model timed out."
            except requests.exceptions.RequestException as e:
                logger.error(f"Local LLM request failed: {e}")
                error_detail = str(e)
                try:
                    # Try to get more detail from response if available
                    error_detail = response.text
                except NameError: # response might not be defined if connection failed early
                    pass
                return f"Error communicating with local model: {error_detail[:200]}" # Limit error length

        else:
            # --- Use Azure OpenAI ---
            if not api_key:
                logger.error("Azure API key is required but not provided for note generation.")
                return "Error: API key is required for Azure OpenAI."
            # Check other required params passed down
            if not azure_endpoint or not azure_api_version or not azure_model_name:
                 logger.error("Azure endpoint, version, or model name missing for note generation.")
                 return "Error: Azure configuration incomplete."
                 
            # Get token count for transcript to determine approach
            transcript_tokens = count_tokens(sanitized_transcript, "gpt-4o")
            logger.info(f"Transcript token count: {transcript_tokens}")
            
            # Use token management strategies for large transcripts
            if transcript_tokens > 2500:                # Choose which approach to use based on configuration
                token_management_approach = config.get("TOKEN_MANAGEMENT_APPROACH", "chunking")
                
                if token_management_approach == "chunking":
                    logger.info(f"🧩 TOKEN MANAGEMENT: Using chunk processing for large transcript ({transcript_tokens} tokens)")
                    # Use synchronous function with asyncio.to_thread
                    note = await asyncio.to_thread(
                        lambda: generate_note_from_chunked_transcript(
                            transcript=sanitized_transcript,
                            prompt_template=final_prompt_template,
                            azure_endpoint=azure_endpoint,
                            azure_api_key=api_key,
                            deployment_name=azure_model_name,
                            api_version=azure_api_version,
                            model="gpt-4o"
                        )
                    )
                    return note
                else:
                    logger.info(f"📝 TOKEN MANAGEMENT: Using two-stage summarization for large transcript ({transcript_tokens} tokens)")
                    # Use synchronous function with asyncio.to_thread
                    note = await asyncio.to_thread(
                        lambda: generate_note_with_two_stage_summarization(
                            transcript=sanitized_transcript,
                            prompt_template=final_prompt_template,
                            azure_endpoint=azure_endpoint,
                            azure_api_key=api_key,
                            deployment_name=azure_model_name,
                            api_version=azure_api_version,
                            model="gpt-4o"
                        )
                    )
                    return note
            
            # For smaller transcripts, use the standard approach
            client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                timeout=60.0 # Increased timeout
            )

            logger.info(f"Requesting note generation from Azure OpenAI model: {azure_model_name}")
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=azure_model_name,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.3,
                    max_tokens=3000  # Safe token limit for smaller transcripts
                )
            )

            note = response.choices[0].message.content
            if not note:
                 logger.warning("Azure OpenAI returned an empty note.")
                 return "Note generation result was empty."

            logger.info("Successfully received note from Azure OpenAI.")
            return note.strip()

    except OpenAIError as e:
        logger.error(f"Azure OpenAI API error during note generation: {e}")
        return f"Error generating note via Azure: {e}"
    except Exception as e:
        logger.error(f"Unexpected error generating note: {e}")
        return f"Error generating note: An unexpected error occurred."

# --- Diarization ---
# Placeholder for actual diarization logic
def apply_speaker_diarization(transcript: str) -> str:
    """Applies basic speaker diarization based on simple patterns."""
    # Very basic example: Assume new speaker on double newline
    lines = transcript.split('\n')
    output_lines = []
    speaker_id = 1
    for i, line in enumerate(lines):
        trimmed_line = line.strip()
        if not trimmed_line: # Skip empty lines
            continue
        # Basic logic: Assume new speaker if previous line was also contentful
        # This is highly inaccurate and just a placeholder
        prefix = f"Speaker {speaker_id}: "
        if i > 0 and lines[i-1].strip(): # If previous line wasn't empty
             # Simple toggle for demo purposes
             speaker_id = 2 if speaker_id == 1 else 1
             prefix = f"Speaker {speaker_id}: "

        output_lines.append(prefix + trimmed_line)

    # Fallback if no lines were processed
    if not output_lines:
         return f"Speaker 1: {transcript}" # Default if split fails

    return "\n".join(output_lines)


async def generate_gpt_speaker_tags(
    transcript: str,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_model_name: Optional[str] = None
) -> str:
    """Use GPT to identify speakers and tag the transcript."""
    if not transcript:
        logger.warning("generate_gpt_speaker_tags called with empty transcript.")
        return "ERROR: No transcript provided for speaker tagging."
    if not api_key:
        logger.warning("Azure API key not provided for GPT speaker tagging. Applying basic diarization.")
        return apply_speaker_diarization(transcript)
    # Check other required params passed down
    if not azure_endpoint or not azure_api_version or not azure_model_name:
         logger.warning("Azure endpoint, version, or model name missing for GPT speaker tagging. Applying basic diarization.")
         return apply_speaker_diarization(transcript)


    sanitized_transcript = sanitize_input(transcript)
    if not sanitized_transcript:
        logger.warning("Transcript became empty after sanitization in generate_gpt_speaker_tags.")
        return "ERROR: Transcript content is invalid."

    prompt = (
        "Analyze the following medical conversation transcript and identify the speakers "
        "(e.g., 'User', 'Patient', 'Nurse'). Prefix each utterance with the identified speaker label "
        "followed by a colon (e.g., 'User: How are you feeling today?'). Maintain the original transcript content accurately.\n\n"
        "TRANSCRIPT:\n"
        f"{sanitized_transcript}"
    )

    logger.debug("Requesting GPT speaker tagging from Azure OpenAI.")

    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=azure_api_version, # Use passed param
            azure_endpoint=azure_endpoint, # Use passed param
            timeout=45.0 # Moderate timeout for tagging
        )

        # Use asyncio.to_thread for the blocking SDK call
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=azure_model_name, # Use passed param
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2, # Low temperature for factual tagging
                max_tokens=int(len(prompt.split()) * 1.3) # Allow slight expansion for tags
            )
        )

        tagged_text = response.choices[0].message.content
        if not tagged_text:
            logger.warning("GPT speaker tagging returned empty result. Applying basic diarization.")
            return apply_speaker_diarization(transcript) # Fallback

        logger.info("Successfully received GPT speaker tags from Azure OpenAI.")
        return tagged_text.strip()

    except OpenAIError as e:
        logger.error(f"Azure OpenAI API error during GPT speaker tagging: {e}. Applying basic diarization.")
        return apply_speaker_diarization(transcript) # Fallback
    except Exception as e:
        logger.error(f"Unexpected error in GPT speaker tagging: {e}. Applying basic diarization.")
        return apply_speaker_diarization(transcript) # Fallback
