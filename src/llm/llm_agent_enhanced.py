"""
Focused Multi-Agent System for Medical Note Generation
Implements specialized agents and iterative refinement for quality improvement
"""

import asyncio
import json
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass # Removed 'field' as it's not used in this version
from enum import Enum
from core.bootstrap import container
from core.factories.llm_factory import LLMProviderFactory
from core.exceptions import ConfigurationError
import time
import re # Import re for JSON extraction

# Attempt to import from existing modules
try:
    from ..config import config, logger
    from ..utils import sanitize_input
except ImportError:
    # Fallback for direct script execution or different project structures
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import config, logger # type: ignore
    from utils import sanitize_input # type: ignore


@dataclass
class NoteSection: # Not used in the provided code, but kept from the user's last snippet
    """Represents a section of a medical note"""
    name: str
    content: str
    confidence: float
    issues: List[str]

class SpecializedAgent(Enum):
    """Specialized agents for medical note processing"""
    TRANSCRIPTION_CLEANER = "transcription_cleaner"
    MEDICAL_EXTRACTOR = "medical_information_extractor"
    CLINICAL_WRITER = "clinical_note_writer"
    QUALITY_REVIEWER = "note_quality_reviewer"

class MedicalNoteAgentPipeline:
    """Simplified multi-agent pipeline focused on note quality"""

    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, model_name: str):
        self._provider = container.resolve(LLMProviderFactory).create(
            "azure_openai",
            api_key=api_key,
            endpoint=azure_endpoint,
            model_name=model_name,
            api_version=api_version,
        )
        self.model_name = model_name
        self.agents = self._initialize_agents() # Corrected: call the method

    def _initialize_agents(self) -> Dict[SpecializedAgent, Dict[str, str]]:
        """Initialize agent prompts and configurations"""
        return {
            SpecializedAgent.TRANSCRIPTION_CLEANER: {
                "role": "Medical Transcription Specialist",
                "prompt": """You are a medical transcription specialist with expertise in:
                - Correcting medical terminology and drug names
                - Identifying and labeling speakers (Doctor, Patient, Nurse)
                - Fixing grammar while preserving clinical meaning
                - Expanding medical abbreviations appropriately

                Clean the transcription while maintaining complete accuracy of clinical information.
                Format with clear speaker labels and paragraph breaks."""
            },
            SpecializedAgent.MEDICAL_EXTRACTOR: {
                "role": "Clinical Information Analyst",
                "prompt": """You are a clinical information analyst. Extract and organize:

                REQUIRED ELEMENTS:
                - Chief Complaint
                - History of Present Illness (HPI) with all OLDCARTS elements
                - Past Medical History
                - Medications (name, dose, frequency, indication)
                - Allergies and reactions
                - Vital Signs with specific values
                - Physical Exam findings by system
                - Assessment with differential diagnosis
                - Plan with specific actions

                Identify any missing critical information and note it clearly.
                Output ONLY structured JSON with keys for each element. Example: {"chief_complaint": "...", "medications": [{"name": "Lisinopril", "dose": "10mg", "frequency": "daily", "indication": "HTN"}] ...} """
            },
            SpecializedAgent.CLINICAL_WRITER: {
                "role": "Clinical Documentation Specialist",
                "prompt": """You are a clinical documentation specialist who creates professional medical notes.
                Transform extracted information into a well-structured clinical note that:
                - Follows standard medical documentation formats (e.g., SOAP, Consultation Note) based on the provided TEMPLATE name.
                - Uses appropriate medical terminology
                - Maintains narrative flow and readability
                - Includes all required sections for billing compliance
                - Demonstrates medical decision-making clearly
                Output ONLY the generated note text."""
            },
            SpecializedAgent.QUALITY_REVIEWER: {
                "role": "Documentation Quality Reviewer",
                "prompt": """You are a medical documentation quality reviewer. Evaluate the provided medical note against the original extracted data for:

                CLINICAL ACCURACY:
                - Consistency of information between note and original data.
                - Appropriate medical terminology.
                - Logical clinical reasoning if evident.

                COMPLETENESS:
                - All relevant extracted information included in the note.
                - Sufficient detail for the encounter type.

                FORMAT & CLARITY:
                - Adherence to specified template structure.
                - Readability and professional tone.

                Provide feedback as structured JSON with keys: "quality_score" (0-100), "issues_found" (list of strings), "suggestions_for_improvement" (list of strings), and "refined_note" (string, only if significant improvements are made, otherwise the original note text).
                If the note is good (score >= 90), "refined_note" can be the same as the input note.
                Example JSON output: {"quality_score": 85, "issues_found": ["HPI lacks detail on symptom progression."], "suggestions_for_improvement": ["Elaborate on the timeline of symptoms in HPI."], "refined_note": "The improved note text..."}"""
            }
        }

    async def _call_agent(self, agent: SpecializedAgent, input_text: str,
                         context: Optional[str] = None, is_json_output_expected: bool = False) -> str: # Added is_json_output_expected
        """Call a specific agent with input and optional context"""
        agent_config = self.agents[agent]

        messages = [
            {"role": "system", "content": agent_config["prompt"]}
        ]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nInput:\n{input_text}"}) # Clearer separation
        else:
            messages.append({"role": "user", "content": input_text})

        try:
            logger.debug(
                "Calling agent %s. Model: %s. JSON expected: %s",
                agent.value,
                self.model_name,
                is_json_output_expected,
            )

            # Combine system + user messages into a single prompt
            prompt_parts = [f"System: {messages[0]['content']}"] + [m['content'] for m in messages[1:]]
            prompt = "\n\n".join(prompt_parts)

            raw_response_text = await self._provider.generate_completion(prompt)

            logger.debug(f"Agent {agent.value} raw response length: {len(raw_response_text)}")

            if is_json_output_expected:
                # Attempt to extract JSON from the response if markdown or other text is present
                # Regex to find JSON object or array, possibly wrapped in markdown code blocks
                match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```|\s*(\{.*\}|\[.*\])\s*', raw_response_text, re.DOTALL | re.MULTILINE)
                if match:
                    json_str = match.group(1) or match.group(2) # Prioritize ```json ... ```
                    try:
                        # Validate it's actual JSON before returning
                        json.loads(json_str) # Test parsing
                        logger.info(f"Successfully extracted JSON from {agent.value} response.")
                        return json_str
                    except json.JSONDecodeError as e:
                        logger.error(f"Extracted string for {agent.value} is not valid JSON: {json_str[:200]}... Error: {e}. Falling back to raw response.")
                        # Fallback or raise error, for now, returning raw text if JSON parsing of extracted part fails
                        # This might happen if the agent's output is malformed JSON even after extraction.
                        # Depending on strictness, you might want to raise an error here.
                        # return raw_response_text # Or raise ValueError("Invalid JSON from agent")
                        raise ValueError(f"Agent {agent.value} response was not valid JSON despite extraction attempt. Content: {json_str[:200]}")

                else: # No clear JSON block found via regex
                    # If response_format={"type": "json_object"} was used, the raw_response_text should ideally be JSON already.
                    if response_format_param and response_format_param["type"] == "json_object":
                        try:
                            json.loads(raw_response_text) # Test parsing
                            logger.info(f"Successfully parsed raw response as JSON from {agent.value} (response_format used).")
                            return raw_response_text
                        except json.JSONDecodeError:
                            logger.error(f"No JSON object found in {agent.value} response where one was expected (response_format used). Raw response: {raw_response_text[:500]}")
                            raise ValueError(f"Agent {agent.value} did not return a valid JSON object as per response_format. Raw: {raw_response_text[:200]}")
                    else: # No response_format used, and regex failed.
                        logger.warning(f"No JSON object found via regex in {agent.value} response and response_format not used. Raw response: {raw_response_text[:500]}")
                        # Heuristic: if it looks like JSON, try parsing. Otherwise, raise error or return raw.
                        if raw_response_text.strip().startswith("{") and raw_response_text.strip().endswith("}"):
                            try:
                                json.loads(raw_response_text)
                                return raw_response_text
                            except json.JSONDecodeError:
                                pass # Fall through to raise error
                        raise ValueError(f"Agent {agent.value} did not return a detectable JSON object. Raw: {raw_response_text[:200]}")

            return raw_response_text # Return raw if not JSON expected

        except ConfigurationError as e:
            logger.error(f"Provider configuration error for agent {agent.value}: {e}")
            raise
        except Exception as e:
            logger.error(f"Agent {agent.value} call failed: {e}", exc_info=True)
            raise # Re-raise the original exception

    async def clean_and_structure_transcript(self, raw_transcript: str) -> Tuple[str, Dict[str, Any]]:
        """Clean transcript and identify speakers using TRANSCRIPTION_CLEANER agent."""
        logger.info("Starting transcript cleaning with TRANSCRIPTION_CLEANER agent.")
        cleaned_transcript_text = await self._call_agent(
            SpecializedAgent.TRANSCRIPTION_CLEANER,
            f"Clean and format this medical transcription, identifying speakers (e.g., Doctor:, Patient:):\n\n{raw_transcript}"
        )
        # Basic speaker statistics (can be made more robust)
        speaker_stats = {
            "doctor_utterances": cleaned_transcript_text.lower().count("doctor:"),
            "patient_utterances": cleaned_transcript_text.lower().count("patient:"),
            "nurse_utterances": cleaned_transcript_text.lower().count("nurse:"),
            "total_words": len(cleaned_transcript_text.split())
        }
        logger.info(f"Transcript cleaning complete. Word count: {speaker_stats['total_words']}.")
        return cleaned_transcript_text, speaker_stats

    async def extract_medical_information(self, cleaned_transcript: str) -> Dict[str, Any]:
        """Extract structured medical information using MEDICAL_EXTRACTOR agent."""
        logger.info("Extracting medical information with MEDICAL_EXTRACTOR agent.")
        extraction_prompt = f"""
        Extract all relevant medical information from the following clinical encounter transcript.
        Adhere strictly to the JSON output format with the specified REQUIRED ELEMENTS.

        Transcript:
        ---
        {cleaned_transcript}
        ---

        Output ONLY the JSON object.
        """
        json_result_str = await self._call_agent(
            SpecializedAgent.MEDICAL_EXTRACTOR,
            extraction_prompt,
            is_json_output_expected=True
        )
        try:
            extracted_data = json.loads(json_result_str)
            # Add metadata about completeness
            required_sections = [
                "Chief Complaint", "History of Present Illness", "Past Medical History",
                "Medications", "Allergies", "Vital Signs", "Physical Exam", "Assessment", "Plan"
            ] # Match keys expected from prompt

            # Normalize keys for comparison (e.g. handle case variations if necessary, though prompt is specific)
            # For simplicity, assuming agent returns keys as specified.
            completeness_map = {
                section: bool(extracted_data.get(section)) # Check if key exists and has some value
                for section in required_sections
            }
            completeness_score = sum(completeness_map.values()) / len(completeness_map) * 100 if completeness_map else 0

            extracted_data["_metadata"] = {
                "completeness_map": completeness_map,
                "completeness_score": round(completeness_score, 2)
            }
            logger.info(f"Medical information extraction successful. Completeness: {completeness_score:.2f}%.")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse medical extraction JSON from MEDICAL_EXTRACTOR: {e}. Raw response: {json_result_str[:500]}", exc_info=True)
            return { # Return a structured error
                "error": "Failed to parse extracted data as JSON.",
                "raw_extraction_output": json_result_str,
                "_metadata": {"completeness_score": 0, "completeness_map": {s: False for s in required_sections}}
            }

    async def generate_clinical_note(self, extracted_data: Dict[str, Any],
                                   template_name: str, patient_data: Optional[Dict] = None) -> str:
        """Generate clinical note from extracted data using CLINICAL_WRITER agent."""
        logger.info(f"Generating clinical note with CLINICAL_WRITER agent. Template: {template_name}")

        context_lines = []
        if patient_data:
            if patient_data.get("name"): context_lines.append(f"Patient Name: {patient_data['name']}")
            if patient_data.get("age"): context_lines.append(f"Patient Age: {patient_data['age']}")
            if patient_data.get("gender"): context_lines.append(f"Patient Gender: {patient_data['gender']}")
            # Add other relevant patient_data fields to context if needed

        context_str = "\n".join(context_lines) if context_lines else None

        # Ensure _metadata is not passed to the writer LLM if it's sensitive or verbose
        data_to_write = {k: v for k, v in extracted_data.items() if k != "_metadata"}

        generation_prompt = f"""
        Generate a professional medical note using the specified TEMPLATE structure and the EXTRACTED INFORMATION provided below.

        TEMPLATE: "{template_name}"

        EXTRACTED INFORMATION:
        {json.dumps(data_to_write, indent=2)}
        """
        note_text = await self._call_agent(
            SpecializedAgent.CLINICAL_WRITER,
            generation_prompt,
            context=context_str
        )
        logger.info(f"Clinical note generation complete. Note length: {len(note_text)}.")
        return note_text

    async def review_and_refine_note(self, current_note_text: str, original_extracted_data: Dict[str, Any],
                                   encounter_type: str = "general_encounter") -> Tuple[str, Dict[str, Any]]:
        """Review note quality using QUALITY_REVIEWER and refine if indicated by agent."""
        logger.info(f"Reviewing note (type: {encounter_type}) with QUALITY_REVIEWER agent.")

        # Ensure _metadata is not passed to the reviewer LLM
        data_for_review = {k: v for k, v in original_extracted_data.items() if k != "_metadata"}

        review_prompt = f"""
        Review the following medical note (encounter type: '{encounter_type}') for quality, accuracy, and completeness, comparing it against the original extracted data.

        MEDICAL NOTE TEXT:
        ---
        {current_note_text}
        ---

        ORIGINAL EXTRACTED DATA (for comparison):
        ---
        {json.dumps(data_for_review, indent=2)}
        ---

        Provide your review STRICTLY as a JSON object with keys: "quality_score" (0-100), "issues_found" (list of strings), "suggestions_for_improvement" (list of strings), and "refined_note" (string containing the full text of an improved note if changes are warranted, otherwise the original note text).
        """
        json_review_result_str = await self._call_agent(
            SpecializedAgent.QUALITY_REVIEWER,
            review_prompt,
            is_json_output_expected=True
        )
        try:
            review_output = json.loads(json_review_result_str)

            # Basic validation of review output structure
            if not all(k in review_output for k in ["quality_score", "issues_found", "suggestions_for_improvement", "refined_note"]):
                logger.warning(f"QUALITY_REVIEWER output missing expected keys. Raw: {json_review_result_str[:300]}")
                # Fallback: assume no refinement, low score
                return current_note_text, {
                    "quality_score": review_output.get("quality_score", 0), # try to get score if present
                    "issues_found": review_output.get("issues_found", ["Review output structure error."]),
                    "suggestions_for_improvement": review_output.get("suggestions_for_improvement", []),
                    "refined_note_provided": False, # Explicitly state no refined note due to structure issue
                    "was_refined": False,
                    "reviewer_feedback_raw": json_review_result_str
                }

            refined_note_text = review_output["refined_note"]
            was_refined = bool(refined_note_text and refined_note_text.strip() != current_note_text.strip() and len(refined_note_text) > 0.8 * len(current_note_text) ) # Heuristic: refined if different and not empty/trivial

            review_metadata = {
                "quality_score": review_output.get("quality_score", 0),
                "issues_found": review_output.get("issues_found", []),
                "suggestions_for_improvement": review_output.get("suggestions_for_improvement", []),
                "refined_note_provided": bool(refined_note_text and refined_note_text.strip()),
                "was_refined": was_refined, # Based on whether the refined_note is different and substantial
                "review_timestamp": time.time(),
                "original_length_before_review": len(current_note_text),
                "refined_length_after_review": len(refined_note_text) if was_refined else len(current_note_text)
            }
            logger.info(f"Note review complete. Score: {review_metadata['quality_score']}. Was refined: {was_refined}.")
            return refined_note_text if was_refined else current_note_text, review_metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON review from QUALITY_REVIEWER: {e}. Raw response: {json_review_result_str[:500]}", exc_info=True)
            # Fallback: return original note, indicate review failure
            return current_note_text, {
                "quality_score": 0, "error": "Failed to parse review output as JSON.",
                "issues_found": ["Review output parsing error."], "suggestions_for_improvement": [],
                "was_refined": False, "reviewer_feedback_raw": json_review_result_str
            }


    async def generate_note_with_agents(self, transcript: str, template_name: str, # Renamed from template to template_name for clarity
                                      patient_data: Optional[Dict] = None,
                                      max_refinement_iterations: int = 1, # Default to 1 refinement attempt
                                      progress_callback: Optional[Any] = None # Added progress_callback
                                      ) -> Tuple[str, Dict[str, Any]]:
        """Complete pipeline for agent-based medical note generation with iterative refinement."""

        overall_start_time = time.time()
        pipeline_metadata: Dict[str, Any] = {
            "pipeline_version": "agent_v1.1_focused", # Version update
            "stages": [],
            "final_status": "started",
            "max_refinement_iterations_config": max_refinement_iterations
        }

        def _log_stage_progress(stage_name: str, status: str, details: Optional[str] = None):
            message = f"🤖 [Agent] {stage_name}: {status}"
            if details: message += f" ({details})"
            if progress_callback:
                if status == "started": progress_callback.info(message)
                elif status == "completed": progress_callback.success(message)
                elif status == "failed": progress_callback.error(message)
                elif status == "warning": progress_callback.warning(message)
            logger.info(message)

        current_note = "" # Initialize current_note

        try:
            # --- Stage 1: Clean and Structure Transcript ---
            stage_start_time = time.time()
            _log_stage_progress("Transcript Cleaning", "started")
            cleaned_transcript, speaker_stats = await self.clean_and_structure_transcript(transcript)
            pipeline_metadata["stages"].append({
                "name": "Transcript Cleaning", "duration": time.time() - stage_start_time,
                "status": "completed", "details": speaker_stats
            })
            _log_stage_progress("Transcript Cleaning", "completed", f"Word count: {speaker_stats.get('total_words',0)}")

            # --- Stage 2: Extract Medical Information ---
            stage_start_time = time.time()
            _log_stage_progress("Medical Information Extraction", "started")
            extracted_data = await self.extract_medical_information(cleaned_transcript)
            completeness_score = extracted_data.get("_metadata", {}).get("completeness_score", 0)
            pipeline_metadata["stages"].append({
                "name": "Medical Information Extraction", "duration": time.time() - stage_start_time,
                "status": "completed" if "error" not in extracted_data else "failed",
                "details": {"completeness_score": completeness_score, "error": extracted_data.get("error")}
            })
            if "error" in extracted_data:
                _log_stage_progress("Medical Information Extraction", "failed", extracted_data["error"])
                raise ValueError(f"Medical Information Extraction failed: {extracted_data['error']}")
            _log_stage_progress("Medical Information Extraction", "completed", f"Completeness: {completeness_score:.2f}%")


            # --- Stage 3: Generate Initial Clinical Note ---
            stage_start_time = time.time()
            _log_stage_progress("Initial Note Generation", "started")
            current_note = await self.generate_clinical_note(extracted_data, template_name, patient_data)
            pipeline_metadata["stages"].append({
                "name": "Initial Note Generation", "duration": time.time() - stage_start_time,
                "status": "completed", "details": {"initial_note_length": len(current_note)}
            })
            _log_stage_progress("Initial Note Generation", "completed", f"Length: {len(current_note)}")


            # --- Stage 4: Iterative Refinement ---
            refinement_summary = {"iterations_done": 0, "final_score_after_refinement": 0, "history": []}
            encounter_type_for_review = self._determine_encounter_type(template_name)

            for i in range(max_refinement_iterations):
                _log_stage_progress(f"Refinement Iteration {i+1}/{max_refinement_iterations}", "started")
                iteration_start_time = time.time()

                refined_note_candidate, review_metadata = await self.review_and_refine_note(
                    current_note, extracted_data, encounter_type_for_review
                )

                iteration_details = {
                    "iteration_num": i + 1,
                    "duration": time.time() - iteration_start_time,
                    **review_metadata # Add all keys from review_metadata
                }
                refinement_summary["history"].append(iteration_details)
                refinement_summary["iterations_done"] = i + 1
                refinement_summary["final_score_after_refinement"] = review_metadata.get("quality_score", 0)

                if review_metadata.get("error"):
                    _log_stage_progress(f"Refinement Iteration {i+1}", "failed", review_metadata["error"])
                    # Decide if to break or continue with unrefined note
                    break

                _log_stage_progress(f"Refinement Iteration {i+1}", "completed", f"Score: {review_metadata.get('quality_score',0)}, Refined: {review_metadata.get('was_refined', False)}")

                current_note = refined_note_candidate # Update current note with the (potentially) refined one

                if not review_metadata.get("was_refined", False) and review_metadata.get("quality_score", 0) >= 90: # Configurable threshold
                    logger.info(f"Note approved at iteration {i+1} with score {review_metadata.get('quality_score',0)}. No further refinement needed.")
                    break

            pipeline_metadata["stages"].append({"name": "Refinement Process", "details": refinement_summary})
            pipeline_metadata["final_status"] = "completed_successfully"
            if progress_callback: progress_callback.success("✅ [Agent] Note generation pipeline completed successfully!")


        except ValueError as ve: # Catch specific errors from pipeline stages
            logger.error(f"Agent pipeline failed due to ValueError: {ve}", exc_info=True)
            pipeline_metadata["final_status"] = "failed_value_error"
            pipeline_metadata["error_details"] = str(ve)
            if progress_callback: progress_callback.error(f"❌ [Agent] Pipeline Error: {ve}")
            # current_note might be empty or partially complete, or an error message
            if not current_note: current_note = f"Pipeline Error: {str(ve)}"
        except TimeoutError as te:
            logger.error(f"Agent pipeline timed out: {te}", exc_info=True)
            pipeline_metadata["final_status"] = "failed_timeout"
            pipeline_metadata["error_details"] = str(te)
            if progress_callback: progress_callback.error(f"❌ [Agent] Pipeline Timeout: {te}")
            current_note = f"Pipeline Timeout: {str(te)}"
        except Exception as e:
            logger.error(f"Unexpected error in agent pipeline: {e}", exc_info=True)
            pipeline_metadata["final_status"] = "failed_unexpected_exception"
            pipeline_metadata["error_details"] = str(e)
            if progress_callback: progress_callback.error(f"❌ [Agent] Unexpected Pipeline Error: {e}")
            current_note = f"Unexpected Pipeline Error: {str(e)}"
        finally:
            total_duration = time.time() - overall_start_time
            pipeline_metadata["total_duration_seconds"] = round(total_duration, 2)
            pipeline_metadata["final_note_length"] = len(current_note)
            # Add key outputs to metadata for easier access by the caller
            pipeline_metadata.setdefault("stages_summary", {}) # Ensure it exists
            for stage in pipeline_metadata["stages"]:
                stage_name_key = stage["name"].lower().replace(" ", "_")
                pipeline_metadata["stages_summary"][stage_name_key] = stage.get("details", {"duration": stage.get("duration")})


            logger.info(f"Agent pipeline finished. Status: {pipeline_metadata['final_status']}. Total duration: {total_duration:.2f}s")

        return current_note, pipeline_metadata


    def _determine_encounter_type(self, template_name: str) -> str: # Renamed from template to template_name
        """Determine encounter type from template name for review context."""
        template_lower = template_name.lower()
        if "discharge" in template_lower: return "Discharge Summary"
        if "operative" in template_lower or "op note" in template_lower: return "Operative Note"
        if "soap" in template_lower or "progress" in template_lower: return "Progress Note"
        if "consult" in template_lower: return "Consultation Note"
        if "psychiatric" in template_lower or "mental health" in template_lower: return "Psychiatric Evaluation"
        if "procedure" in template_lower: return "Procedure Note"
        if "initial" in template_lower or "new patient" in template_lower: return "Initial Evaluation"
        logger.debug(f"Determined encounter type as '{template_lower}' based on template name: '{template_name}'")
        return "General Medical Encounter" # A sensible default


# This is the main function to be imported and called by llm_integration.py or app.py
async def generate_note_agent_based(
    transcript: str,
    api_key: str, # Made non-optional
    azure_endpoint: str, # Made non-optional
    azure_api_version: str, # Made non-optional
    azure_model_name: str, # Made non-optional (Azure Deployment Name)
    prompt_template: str, # Name/description of the note template (e.g., "SOAP Note")
    # use_local: bool = False, # This logic should be handled by the caller in llm_integration.py
    # local_model: str = "", # Same as above
    patient_data: Optional[Dict] = None,
    # use_agent_pipeline: bool = True, # This is implied if this function is called
    agent_settings: Optional[Dict[str, Any]] = None, # For settings like max_refinements
    progress_callback: Optional[Any] = None # For Streamlit UI updates
) -> Tuple[str, Dict[str, Any]]:
    """
    Main entry point for generating a medical note using the agent-based pipeline.
    This function will be called from llm_integration.py.
    """
    logger.info(f"generate_note_agent_based invoked. Model: {azure_model_name}, Template: {prompt_template}")

    if not transcript or not transcript.strip():
        logger.warning("Agent-based generation called with empty transcript.")
        return "Error: Transcript is empty.", {"error": "Empty transcript", "pipeline_status": "input_error"}

    if not all([api_key, azure_endpoint, azure_api_version, azure_model_name]):
        logger.error("Agent-based generation called with missing Azure OpenAI credentials or model name.")
        return "Error: Azure OpenAI API credentials and model name are required.", {"error": "Missing API credentials or model", "pipeline_status": "config_error"}

    try:
        pipeline = MedicalNoteAgentPipeline(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            model_name=azure_model_name
        )

        # Extract agent-specific settings
        current_agent_config = agent_settings or {}
        max_refinements = current_agent_config.get("max_refinements", 1) # Default from issue was 2, then 1

        note, metadata = await pipeline.generate_note_with_agents(
            transcript=sanitize_input(transcript), # Sanitize transcript before passing
            template_name=prompt_template or "General Medical Note", # Provide a default template name
            patient_data=patient_data,
            max_refinement_iterations=max_refinements,
            progress_callback=progress_callback
        )

        metadata["agent_based_processing_used"] = True # Add flag for clarity in logs/UI
        logger.info(f"Agent-based note generation completed. Final status: {metadata.get('final_status')}")
        return note, metadata

    except Exception as e:
        logger.error(f"Critical error during agent-based note generation entry point: {e}", exc_info=True)
        if progress_callback: progress_callback.error(f"❌ [Agent] Critical system error: {e}")
        return f"Critical Error in Agent System: {str(e)}", {
            "error": str(e),
            "pipeline_status": "critical_system_failure",
            "agent_based_processing_used": True
        }

# Example usage for standalone testing (if this file is run directly)
if __name__ == "__main__":
    import os # For os.name and path operations
    import sys # For sys.path and sys.version_info

    # Mock Streamlit progress callback for testing
    class MockProgressCallback:
        def _log(self, level: str, message: str): print(f"UI MOCK [{level.upper()}]: {message}")
        def text(self, message: str): self._log("text", message) # Older streamlit attribute
        def info(self, message: str): self._log("info", message)
        def success(self, message: str): self._log("success", message)
        def warning(self, message: str): self._log("warning", message)
        def error(self, message: str): self._log("error", message)

    async def main_test():
        logger.info("--- Starting Standalone Test of llm_agent_enhanced.py ---")

        try:
            # These should be set in your config.py or environment variables via config.py
            test_api_key = config.AZURE_OPENAI_API_KEY
            test_azure_endpoint = config.AZURE_OPENAI_ENDPOINT
            test_api_version = config.API_VERSION
            test_model_name = config.MODEL_NAME # e.g., your gpt-4 deployment name
        except AttributeError as e: # More specific exception
            print(f"Error loading test configuration from config.py: {e}. Ensure all required attributes (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, MODEL_NAME) are set.")
            return
        except Exception as e: # Catch other potential errors during config loading
            print(f"An unexpected error occurred while loading test configuration: {e}")
            return


        if not all([test_api_key, test_azure_endpoint, test_api_version, test_model_name]):
            print("Error: Azure OpenAI credentials or model name are missing (likely from config.py). Cannot run test.")
            return

        print(f"Test Config: Endpoint='{test_azure_endpoint}', Model='{test_model_name}'")

        # Corrected sample transcript to be a valid string for direct embedding
        sample_transcript = """
        Patient Name: Jane Doe. Age: 52. Female.
        Chief Complaint: Persistent cough and shortness of breath for 3 weeks.
        History of Present Illness: Cough is productive, yellow sputum. SOB worse on exertion. Denies fever, chills.
        Past Medical History: Hypertension (HTN), Type 2 Diabetes Mellitus (DM2).
        Medications: Lisinopril 20mg daily, Metformin 1000mg BID.
        Allergies: Penicillin - causes rash.
        Social History: Non-smoker. Occasional alcohol.
        Review of Systems: Cardiovascular: No chest pain, no palpitations. Respiratory: As per HPI. GI: No nausea, vomiting.
        Physical Exam: Vitals: BP 145/88, HR 78, RR 20, Temp 37.1C, SpO2 96% RA. General: NAD. Lungs: Few crackles at right base. Heart: RRR, no murmurs.
        Assessment: Likely community-acquired pneumonia.
        Plan: Chest X-ray. Sputum culture. Start Amoxicillin-clavulanate 875/125 mg BID for 7 days. Follow up in 3 days.
        """
        sample_patient_data = {"name": "Jane Doe", "age": "52", "gender": "Female"}
        sample_template_name = "SOAP Note" # Changed from prompt_template to template_name

        mock_progress = MockProgressCallback()

        print("\n--- Running generate_note_agent_based (main test function) ---")
        try:
            generated_note, metadata = await generate_note_agent_based(
                transcript=sample_transcript,
                api_key=test_api_key,
                azure_endpoint=test_azure_endpoint,
                azure_api_version=test_api_version,
                azure_model_name=test_model_name,
                prompt_template=sample_template_name, # Pass template_name
                patient_data=sample_patient_data,
                agent_settings={"max_refinements": 1},
                progress_callback=mock_progress
            )

            print("\n--- FINAL GENERATED NOTE ---")
            print(generated_note)
            print("\n--- FINAL METADATA ---")
            # Pretty print metadata for readability
            print(json.dumps(metadata, indent=2, default=str)) # Use default=str for any non-serializable data like timestamps

        except Exception as e:
            print(f"\n--- Test Execution Failed ---")
            logger.error(f"An error occurred during the standalone test execution: {e}", exc_info=True)

    # Windows asyncio policy fix if needed
    if os.name == 'nt':
        if sys.version_info >= (3, 8) and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
            pass # Proactor is fine for Python 3.8+
        else:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_test())
