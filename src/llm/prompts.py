import json
import os
import logging
from typing import Dict
from functools import lru_cache
from pathlib import Path

from ..core.container import global_container
from ..core.interfaces.config_service import IConfigurationService

# Setup logging using the standard Python logging module
logger = logging.getLogger("ambient_scribe")


def _get_prompt_store_path():
    """Helper to get prompt store path from DI container with fallback."""
    try:
        config_service = global_container.resolve(IConfigurationService)
        base_dir = config_service.get("base_dir", Path("./app_data"))
        return base_dir / "prompt_templates.json"
    except Exception:
        # Fallback if DI not available
        return Path("./app_data/prompt_templates.json")


# --- Prompt Template Management ---

DEFAULT_TEMPLATES = {
    "SOAP": "Please structure the following transcription into a SOAP note.",
    "Discharge Summary": "Please structure the following transcription into a Discharge Summary.",
    "Operative Note": "Please structure the following transcription into an Operative Note.",
    "PrimaryCare": (
        "You are a medical scribe tasked with accurately documenting primary care visits. "
        "Your documentation should clearly distinguish speakers and provide a narrative-style SOAP note including:\n"
        "Subjective: Describe the patient's chief complaints, symptoms, and relevant medical history using narrative sentences. "
        "Objective: Detail clinical observations, vital signs, and examination findings clearly. "
        "Assessment: Provide concise clinical impressions and diagnoses. "
        "Plan: Outline treatments, medications, tests, referrals, and follow-up clearly and concisely."
    ),
    "Biopsychosocial": (
        "You are a mental health clinician. Please structure the following diagnostic assessment "
        "into a biopsychosocial format. Include sections for Biological, Psychological, and Social factors."
    )
}

TEMPLATE_SUGGESTIONS = {
    "SOAP Note": (
        "You are a medical scribe. Create a comprehensive SOAP note from the following transcription.\n"
        "Structure it with clear sections for:\n"
        "- Subjective: Patient's symptoms, complaints, and history as described\n"
        "- Objective: All physical findings, vital signs, and test results mentioned\n"
        "- Assessment: Clear diagnosis or differential diagnoses\n"
        "- Plan: All treatment recommendations, medications, follow-ups\n\n"
        "Use medical terminology appropriately. Format with proper headings and bullet points.\n\n"
        "TRANSCRIPTION:\n{transcription}" # Added placeholder
    ),
    "Psychiatric Assessment": (
        "Create a comprehensive psychiatric assessment note from the following transcription.\n"
        "Include sections for:\n"
        "- Chief Complaint\n"
        "- History of Present Illness\n"
        "- Psychiatric History\n"
        "- Medical History\n"
        "- Substance Use\n"
        "- Mental Status Examination\n"
        "- Risk Assessment\n"
        "- DSM-5 Diagnosis\n"
        "- Treatment Plan\n\n"
        "Use professional psychiatric terminology and format with clear headings.\n\n"
        "TRANSCRIPTION:\n{transcription}" # Added placeholder
    ),
    "Physical Therapy Evaluation": (
        "Create a detailed physical therapy evaluation note from the following transcription.\n"
        "Include:\n"
        "- Patient Demographics and Referral Information\n"
        "- Subjective Report (pain, limitations, goals)\n"
        "- Objective Measures (ROM, strength, special tests)\n"
        "- Assessment (clinical impressions, functional limitations)\n"
        "- Plan (frequency, duration, interventions, HEP)\n"
        "- Short and Long-term Goals (measurable and time-bound)\n\n"
        "Use proper PT terminology and clear formatting.\n\n"
        "TRANSCRIPTION:\n{transcription}" # Added placeholder
    ),
    "Discharge Summary": (
        "Create a comprehensive hospital discharge summary from this transcription.\n"
        "Include:\n"
        "- Admission Information (date, reason, diagnosis)\n"
        "- Hospital Course (chronological summary of stay)\n"
        "- Procedures Performed\n"
        "- Consultations\n"
        "- Discharge Diagnosis (primary and secondary)\n"
        "- Discharge Medications (with doses and schedule)\n"
        "- Follow-up Instructions\n"
        "- Pending Results\n\n"
        "Format with clear headings and maintain medical accuracy.\n\n"
        "TRANSCRIPTION:\n{transcription}" # Added placeholder
    ),
    "Well-Child Visit": (
        "Create a pediatric well-child visit note from this transcription.\n"
        "Include:\n"
        "- Demographics and Vital Signs\n"
        "- Growth Parameters (with percentiles)\n"
        "- Development Assessment\n"
        "- Nutrition and Sleep Assessment\n"
        "- Review of Systems\n"
        "- Physical Examination (by system)\n"
        "- Immunizations Given/Due\n"
        "- Anticipatory Guidance\n"
        "- Plan/Recommendations\n\n"
        "Use age-appropriate developmental milestones and screening recommendations.\n\n"
        "TRANSCRIPTION:\n{transcription}" # Added placeholder
    )
}


@lru_cache(maxsize=None)
def load_prompt_templates() -> Dict[str, str]:
    """Load prompt templates from JSON file, falling back to defaults."""
    prompt_file = _get_prompt_store_path()
    
    # Ensure the parent directory exists
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not prompt_file.exists():
        logger.info(f"Prompt template file not found at {prompt_file}. Creating with defaults.")
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_TEMPLATES, f, indent=4)
            return DEFAULT_TEMPLATES
        except Exception as e:
            logger.error(f"Failed to create default prompt template file: {e}")
            return DEFAULT_TEMPLATES # Return defaults in memory if file creation fails

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
            # Optionally merge with defaults to ensure core templates exist?
            # merged_templates = {**DEFAULT_TEMPLATES, **templates}
            # return merged_templates
            logger.info(f"Loaded {len(templates)} prompt templates from {prompt_file}")
            return templates
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from prompt template file {prompt_file}: {e}")
        return DEFAULT_TEMPLATES # Fallback to defaults on decode error
    except Exception as e:
        logger.error(f"Failed to load prompt templates from {prompt_file}: {e}")
        return DEFAULT_TEMPLATES # Fallback to defaults on other errors


def save_custom_template(name: str, template: str) -> bool:
    """Save or update a custom template to the templates store."""
    if not name or not template:
        logger.warning("Attempted to save template with empty name or content.")
        return False

    prompt_file = _get_prompt_store_path()
    success = False
    try:
        # Load existing templates (use load function to handle file creation/errors)
        templates = load_prompt_templates()

        # Add/update template
        templates[name] = template

        # Save updated templates
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=4) # Use indent for readability

        logger.info(f"Saved/Updated custom template: '{name}' in {prompt_file}")
        success = True
    except Exception as e:
        logger.error(f"Failed to save template '{name}' to {prompt_file}: {e}")
        success = False
    
    if success:
        load_prompt_templates.clear() # Clear the cache after a successful save

    return success


def load_template_suggestions() -> Dict[str, str]:
    """Load predefined template suggestions."""
    # Currently returns a hardcoded dict, could be loaded from a file if needed
    return TEMPLATE_SUGGESTIONS

# Function to load only custom templates (excluding defaults if needed)
# def load_custom_templates() -> Dict[str, str]:
#     """Load only user-saved custom templates."""
#     all_templates = load_prompt_templates()
#     custom_templates = {k: v for k, v in all_templates.items() if k not in DEFAULT_TEMPLATES}
#     return custom_templates
