"""
token_management.py
Handles token management for OpenAI API calls to stay within token limits.

This module provides two approaches for handling large transcriptions that might exceed
the token limits of the OpenAI API:

1. Chunk Processing (generate_note_from_chunked_transcript):
   - Splits the transcript into manageable chunks
   - Processes each chunk independently
   - Combines the results into a cohesive note

2. Two-Stage Summarization (generate_note_with_two_stage_summarization):
   - First extracts key information from transcript sections
   - Then combines these summaries into a cohesive note

The approach can be configured using the TOKEN_MANAGEMENT_APPROACH setting in config.py
or the TOKEN_MANAGEMENT_APPROACH environment variable ('chunking' or 'two_stage').

Both approaches use the tiktoken library for accurate token counting.
"""

import asyncio
from typing import List, Dict, Any, Optional

from core.bootstrap import container
from core.factories.llm_factory import LLMProviderFactory
from core.exceptions import ConfigurationError
from core.interfaces.token_service import ITokenService

from ..config import config, logger

_token_service: ITokenService = container.resolve(ITokenService)

def count_tokens(text: str, model: str = "gpt-4o") -> int:  # noqa: D401
    """Forward to configured *ITokenService*."""
    return _token_service.count(text, model)

def chunk_transcript(transcript: str, max_chunk_tokens: int = 2048, model: str = "gpt-4o") -> List[str]:  # noqa: D401
    """Forward to configured *ITokenService*."""
    return _token_service.chunk(transcript, max_chunk_tokens, model)

def generate_note_from_chunked_transcript(transcript: str, prompt_template: str, 
                                          azure_endpoint: str, azure_api_key: str,
                                          deployment_name: str, api_version: str,
                                          model: str = "gpt-4o") -> str:
    """
    Generate a clinical note from a transcript while handling token limits.
    
    Args:
        transcript: The full transcript text
        prompt_template: Template for the note generation prompt
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_key: Azure OpenAI API key
        deployment_name: Azure deployment name for the model
        api_version: Azure API version
        model: Model name for token counting
        
    Returns:
        Generated clinical note
    """
    # Lazily create provider once for all completions in this helper
    provider = container.resolve(LLMProviderFactory).create(
        "azure_openai",
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        model_name=deployment_name,
        api_version=api_version,
    )

    def _complete(prompt: str) -> str:
        try:
            return asyncio.run(provider.generate_completion(prompt))
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Provider completion failed: {exc}") from exc
    
    # Estimate tokens in prompt template (to reserve space)
    prompt_template_tokens = count_tokens(prompt_template, model)
    
    # Determine max tokens per chunk for input (leaving room for completion)
    # Reserve at least 1000 tokens for response per chunk
    reserve_tokens = 1000
    # Adjust for prompt template size
    max_chunk_tokens = 4096 - reserve_tokens - prompt_template_tokens
    
    # Split transcript into manageable chunks
    chunks = chunk_transcript(transcript, max_chunk_tokens, model)
    
    if not chunks:
        return "Error: No transcript content to process."
    
    logger.info(f"Split transcript into {len(chunks)} chunks for processing")
    
    # For single chunk, process normally
    if len(chunks) == 1:
        try:
            prompt = prompt_template.replace("{transcript}", chunks[0])
            return _complete(prompt)
        except Exception as e:
            logger.error(f"Error generating note: {str(e)}")
            return f"Error generating note: {str(e)}"
    
    # For multiple chunks, process each and combine results
    chunk_notes = []
    chunk_prompt = prompt_template + "\nNote: This is part {part_num} of {total_parts} of the transcript. Focus on extracting key information from this section only."
    
    for i, chunk in enumerate(chunks):
        try:
            prompt = chunk_prompt.replace("{transcript}", chunk)
            prompt = prompt.replace("{part_num}", str(i+1))
            prompt = prompt.replace("{total_parts}", str(len(chunks)))
            
            chunk_notes.append(_complete(prompt))
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            chunk_notes.append(f"[Error processing chunk {i+1}: {str(e)}]")
    
    # Final pass to combine the chunk notes
    try:
        combined_notes = "\n\n".join(chunk_notes)
        integration_prompt = f"""You have been given multiple sections of clinical notes derived from a longer transcript.
Please integrate these sections into one cohesive clinical note, removing any redundancy:

{combined_notes}

Please provide a single, integrated clinical note from all these sections."""

        return _complete(integration_prompt)
    except Exception as e:
        logger.error(f"Error in final integration: {str(e)}")
        # Fall back to joining the chunks if integration fails
        return "\n\n--- SECTION BREAK ---\n\n".join(chunk_notes)

def generate_note_with_two_stage_summarization(transcript: str, prompt_template: str,
                                              azure_endpoint: str, azure_api_key: str,
                                              deployment_name: str, api_version: str,
                                              model: str = "gpt-4o") -> str:
    """
    Generate a clinical note using two-stage summarization to handle token limits.
    
    First stage: Generate summaries of transcript sections
    Second stage: Create final note from these summaries
    
    Args:
        transcript: The full transcript text
        prompt_template: Template for the note generation prompt
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_key: Azure OpenAI API key
        deployment_name: Azure deployment name for the model
        api_version: Azure API version
        model: Model name for token counting
        
    Returns:
        Generated clinical note
    """
    # Lazily create provider once for all completions in this helper
    provider = container.resolve(LLMProviderFactory).create(
        "azure_openai",
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        model_name=deployment_name,
        api_version=api_version,
    )

    _complete = lambda p: asyncio.run(provider.generate_completion(p))  # type: ignore
    
    # If transcript is short enough, process normally
    if count_tokens(transcript, model) < 2500:
        try:
            prompt = prompt_template.replace("{transcript}", transcript)
            return _complete(prompt)
        except Exception as e:
            logger.error(f"Error generating note: {str(e)}")
            return f"Error generating note: {str(e)}"
    
    # For longer transcripts, use two-stage approach
    
    # STAGE 1: Break transcript into sections and summarize each
    
    # Determine chunk size for first stage (more chunks than first approach)
    max_chunk_tokens = 2000  # Smaller chunks for more focused summaries
    chunks = chunk_transcript(transcript, max_chunk_tokens, model)
    
    logger.info(f"First stage: Split transcript into {len(chunks)} sections")
    
    # Process each section to extract key information
    section_summaries = []
    summary_prompt = """Extract the key medical information from this transcript section, including:
- Chief complaints
- Symptoms discussed
- Relevant patient history mentioned
- Assessment points
- Treatment plans or recommendations
- Follow-up instructions

TRANSCRIPT SECTION:
{section}

Provide a concise summary with the most important clinical details only."""
    
    for i, chunk in enumerate(chunks):
        try:
            prompt = summary_prompt.replace("{section}", chunk)
            section_summaries.append(_complete(prompt))
        except Exception as e:
            logger.error(f"Error summarizing section {i+1}: {str(e)}")
            # Include a marker but also some of the original text to not lose information
            section_summaries.append(f"[Error summarizing section. Original text excerpt: {chunk[:200]}...]")
    
    # STAGE 2: Generate final note from the summaries
    combined_summaries = "\n\n--- SECTION SUMMARY ---\n\n".join(section_summaries)
    
    try:
        # Use the original prompt template but replace transcript with our summaries
        final_prompt = f"""You are generating a clinical note based on summaries extracted from a patient encounter.
These summaries contain the key information from the full transcript.

{prompt_template.replace("{transcript}", "the summarized encounter information")}

SUMMARIZED ENCOUNTER INFORMATION:
{combined_summaries}

Create a complete, well-structured clinical note based on these summaries."""

        return _complete(final_prompt)
    except Exception as e:
        logger.error(f"Error in final note generation: {str(e)}")
        return f"Error generating final note: {str(e)}\n\nSummarized content:\n{combined_summaries}"

def generate_coding_and_review(
    note: str,
    azure_endpoint: str,
    azure_api_key: str,
    deployment_name: str,
    api_version: str,
    model: str = "gpt-4o"
) -> str:
    """
    Use GPT to generate E/M, ICD-10, SNOMED, CPT, risk, SDOH, alternate dx, and payer review based on a clinical note.
    The prompt is tailored for insurance-defensible coding and payer review.
    """
    if not note:
        return "No note provided for coding analysis."
    prompt = f"""
E/M Coding: on separate lines recommend insurance-defensible e/m coding based on documented a) mdm and b) time.  Briefly summarize justification for each as specifically documented in the note.   Time coding for established patients: total 10-19 minutes = 99212, 20-29 minutes = 99213, 30-39 minutes = 99214, and 40-54 minutes = 99215.  If greater than 54 minutes, code 99417 may be added for each additional fifteen minutes.  For new patients, time coding is 15-29 minutes = 99202, 30-44 minutes = 99203, 45-59 minutes = 99204, and 60-74 minutes = 99205. If greater than 74 minutes, code 99417 may be added for each additional fifteen minutes. 3) Recommend icd-10 and snomed codes, placing both on the same line for each separate condition. Include BOTH codes AND DESCRIPTIVE TEXT FOR EACH.  Ensure in each each match of icd-10 and snomed codes that the identifying text for icd-10 matches the text for snomed.  Specifically identify and note conditions considered high-risk for coding. Include only icd-10 and snomed codes that are mentioned in the text of the note, NOT in an inserted problem list.  Include items from the problem list ONLY if they are ALSO discussed in the note.  4) Recommend CPT codes for any noted procedures.  For EKGs please specifically include codes for test performance and provider reading.  Specifically evaluate then include, only when clearly relevant, potential CPT procedure modifiers (25, 59, 76 or 77).  Specifically document why such modifiers are relevant to this encounter.  Especially ensure CPT modifier 59 is identified if procedures were performed at different sites during the encounter.  5) Identify comorbidities and risk adjustment codes.  6) Review and identify any relevant social determinants of health codes (Z55-Z65). 7) Summarize at least four alternate diagnoses considered and explain why they weren't actioned.  8) Review your responses 1-7 from a payer perspective, noting whether a payer might dispute e/m or diagnosis codes due to insufficient documentation in the note.  Also suggest what might be improved in the note to counter the payer's perspective.\n\nCLINICAL NOTE:\n{note}\n"""
    provider = container.resolve(LLMProviderFactory).create(
        "azure_openai",
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        model_name=deployment_name,
        api_version=api_version,
    )
    try:
        return _complete(prompt)
    except Exception as e:
        logger.error(f"Error generating coding analysis: {str(e)}")
        return f"Error generating coding analysis: {str(e)}"
