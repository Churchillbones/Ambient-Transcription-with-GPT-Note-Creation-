import asyncio
import requests
from typing import Optional, Dict
from openai import AzureOpenAI, OpenAIError

from .config import config, logger
from .utils import sanitize_input

# --- Note Generation ---
async def generate_note(
    transcript: str,
    api_key: Optional[str] = None,
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

            client = AzureOpenAI(
                api_key=api_key,
                api_version=config["API_VERSION"],
                azure_endpoint=config["AZURE_ENDPOINT"],
                timeout=60.0 # Increased timeout
            )

            logger.info(f"Requesting note generation from Azure OpenAI model: {config['MODEL_NAME']}")
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=config["MODEL_NAME"],
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.3, # Adjust temperature for clinical note generation
                    max_tokens=max(1000, int(len(prompt.split()) * 1.5)) # Generous token limit
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
