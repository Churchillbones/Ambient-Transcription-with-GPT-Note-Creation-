# Apply PyTorch-Streamlit patch before importing Streamlit
import os
# Set environment variables for PyTorch CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Import the patch before anything else
import torch_streamlit_patch

# Now it's safe to import Streamlit
import streamlit as st

# Import necessary components from the new src package
from src.config import logger # Import logger early for potential use during imports
try:
    from src.ui_components import (
        render_sidebar,
        render_recording_section,
        render_upload_section,
        render_view_transcription_section,
        render_model_comparison_section
    )
    # Import config if needed directly, though most access should be within modules
    # from src.config import config
except ImportError as e:
    # Log the error and display a user-friendly message in Streamlit
    logger.error(f"Failed to import necessary modules from 'src': {e}", exc_info=True)
    st.error(f"Critical Error: Failed to load application components from the 'src' directory: {e}\n"
             "Please ensure all files in the 'src' directory exist and are correctly structured.")
    # Optionally, halt execution if core components are missing
    st.stop()
except Exception as e:
    # Catch any other unexpected errors during import
    logger.error(f"Unexpected error during module import: {e}", exc_info=True)
    st.error(f"An unexpected error occurred while loading the application: {e}")
    st.stop()


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Medical Transcription App", layout="wide")
    st.title("🩺 Medical Transcription & Note Generation App")

    # Initialize session state for patient data if not exists (moved here for safety)
    if "current_patient" not in st.session_state:
        st.session_state["current_patient"] = {"name": "", "ehr_data": ""}
    # Initialize session state for custom template text if not exists
    if "custom_template_text" not in st.session_state:
        st.session_state["custom_template_text"] = ""
    # Initialize session state for selected template name
    if "selected_template_name" not in st.session_state:
         # We need to load templates here to set a default, or handle it in render_custom_template
         st.session_state["selected_template_name"] = "✨ Create New Template" # Default


    logger.info("Application started.")

    try:
        # Get configuration from sidebar
        # This function now resides in ui_components
        # Returns: (api_key, endpoint, api_version, model_name, asr_model_info, use_local_llm, local_llm_model, use_encryption, language, use_agent_pipeline)
        (azure_api_key, azure_endpoint, azure_api_version, azure_model_name, 
         asr_model_info, use_local_llm, local_llm_model, use_encryption, language,
         use_agent_pipeline) = render_sidebar() # Added use_agent_pipeline
         
        agent_settings = {} # Default to empty dict
        if use_agent_pipeline and not use_local_llm:
            # Ensure render_agent_settings is imported if it's not already automatically available
            # (it should be, as render_sidebar is from the same module)
            from src.ui_components import render_agent_settings
            agent_settings = render_agent_settings()

        # Create tabs for different functionalities
        tab_titles = ["Record Audio", "Upload Audio", "View Transcription", "Compare Notes", "Model Comparison"]
        # Use st.tabs for a cleaner look
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles) # Added tab5
        
        with tab1:
            st.header("Record & Generate Note")
            # Pass all relevant config down
            render_recording_section(
                azure_api_key, azure_endpoint, azure_api_version, azure_model_name,
                asr_model_info, use_local_llm, local_llm_model, use_encryption,
                language,
                use_agent_pipeline, # New
                agent_settings      # New
            )
            
        with tab2:
            st.header("Upload & Generate Note")
            # Pass all relevant config down
            render_upload_section(
                azure_api_key, azure_endpoint, azure_api_version, azure_model_name,
                asr_model_info, use_local_llm, local_llm_model, use_encryption,
                language,
                use_agent_pipeline, # New
                agent_settings      # New
            ) # Upload section has its own encryption toggle
            
        with tab3:
            st.header("View Transcription")
            # Pass necessary config for ASR and diarization
            render_view_transcription_section(
                azure_api_key, azure_endpoint, azure_api_version, azure_model_name,
                asr_model_info, language
            )

        with tab4: # This was previously Model Comparison, now it's Compare Notes
            st.header("⚖️ Compare Note Generation Pipelines")
            st.write("This section allows for a side-by-side comparison of notes generated by the traditional pipeline and the new agent-based pipeline.")

            # Import the comparison view function
            from src.ui_components import render_comparison_view

            # Placeholder for transcript input - e.g., file uploader or select existing
            st.subheader("1. Provide Transcript for Comparison")
            comparison_transcript_text = st.text_area("Enter or paste transcript here:", height=150, key="compare_transcript_input")
            # Or allow upload:
            # uploaded_comp_file = st.file_uploader("Upload audio/text for comparison", type=["wav", "mp3", "m4a", "txt"], key="compare_file_uploader")

            st.subheader("2. Generate and Compare")
            if st.button("Run Comparison on Above Transcript", key="run_full_comparison_button"):
                if comparison_transcript_text and comparison_transcript_text.strip():
                    st.info("Comparison logic to be implemented: This would involve running the provided transcript through both the traditional and agent-based pipelines, then displaying the results below.")

                    # --- Placeholder for actual dual pipeline execution ---
                    # Example of how it might look (conceptual):
                    #
                    # with st.spinner("Generating traditional note..."):
                    #     # Call traditional pipeline (e.g., generate_note_router with use_agent_pipeline=False)
                    #     # traditional_note, traditional_metadata = asyncio.run(generate_note_router(...params_for_traditional...))
                    #     traditional_note = "This is a placeholder traditional note for comparison."
                    #     traditional_metadata = {"total_duration_seconds": 1.5, "pipeline_type_attempted": "traditional"}
                    #     st.success("Traditional note generated.")

                    # with st.spinner("Generating agent-based note..."):
                    #     # Call agent pipeline (e.g., generate_note_router with use_agent_pipeline=True)
                    #     # agent_note, agent_metadata = asyncio.run(generate_note_router(...params_for_agent...))
                    #     agent_note = "This is a placeholder agent-based note, which is often more detailed."
                    #     agent_metadata = {"total_duration_seconds": 10.2, "pipeline_type_attempted": "agent", "agent_based_processing_used": True, "stages_summary": {"refinement_process": {"details": {"final_score_after_refinement": 92}}}}
                    #     st.success("Agent-based note generated.")

                    # if traditional_note and agent_note:
                    #     render_comparison_view(
                    #         traditional_note,
                    #         agent_note,
                    #         traditional_metadata,
                    #         agent_metadata
                    #     )
                    # else:
                    #     st.error("Could not generate one or both notes for comparison.")
                    # --- End Placeholder ---

                    # For now, just show some dummy data using the render_comparison_view
                    dummy_trad_note = "Traditional Note Example: Patient presents with cough. Plan: Amox."
                    dummy_agent_note = "Agent-Based Note Example: Patient, a 45-year-old male, presents with a persistent, non-productive cough for three weeks. On examination, lungs are clear. Assessment: Viral URI. Plan: Recommend supportive care, hydration, and rest. Follow up if symptoms worsen or do not resolve in 7-10 days."
                    dummy_trad_meta = {"total_duration_seconds": 1.2, "pipeline_type_attempted": "traditional"}
                    dummy_agent_meta = {
                        "total_duration_seconds": 8.7,
                        "pipeline_type_attempted": "agent",
                        "agent_based_processing_used": True,
                        "stages_summary": {
                            "medical_information_extraction": {"details": {"completeness_score": 95}},
                            "refinement_process": {"details": {"iterations_done": 1, "final_score_after_refinement": 92}}
                        }
                    }
                    render_comparison_view(dummy_trad_note, dummy_agent_note, dummy_trad_meta, dummy_agent_meta)

                else:
                    st.warning("Please provide a transcript in the text area above to run the comparison.")
            else:
                st.write("Click the button above to process the transcript and see the comparison (currently uses placeholder data).")

        with tab5: # This was previously tab4
            st.header("Compare ASR Models")
            # The call to render_model_comparison_section remains the same, just in a new tab variable
            render_model_comparison_section(asr_model_info, language)

    except FileNotFoundError as e:
         logger.error(f"File not found error in main app: {e}")
         st.error(f"Error: A required file or directory was not found: {e}")
    except ImportError as e:
         logger.error(f"Import error in main app: {e}")
         st.error(f"Error: Failed to import a necessary component: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main application flow: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        st.warning("Please check the logs for more details or restart the application.")


if __name__ == "__main__":
    # Ensure necessary directories exist (config module does this, but double-check)
    # try:
    #     from src.config import config as app_config
    #     for dir_key in ["MODEL_DIR", "KEY_DIR", "LOG_DIR", "CACHE_DIR", "NOTES_DIR", "WHISPER_MODELS_DIR"]:
    #         if dir_key in app_config:
    #             app_config[dir_key].mkdir(parents=True, exist_ok=True)
    # except Exception as e:
    #     st.error(f"Failed to ensure directories on startup: {e}")

    main()