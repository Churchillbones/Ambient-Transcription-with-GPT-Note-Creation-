import streamlit as st
import os

# Set environment variables for PyTorch CPU usage *before* importing anything that might use it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

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
        api_key, model_info, use_local, local_model, use_encryption = render_sidebar()

        # Create tabs for different functionalities
        tab_titles = ["Record Audio", "Upload Audio", "View Transcription", "Model Comparison"]
        # Use st.tabs for a cleaner look
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            st.header("Record & Generate Note")
            render_recording_section(api_key, model_info, use_local, local_model, use_encryption)

        with tab2:
            st.header("Upload & Generate Note")
            # Pass use_encryption from sidebar? Upload section has its own toggle now.
            # Let's assume the upload section's toggle is the primary control for uploads.
            render_upload_section(api_key, model_info, use_local, local_model)

        with tab3:
            st.header("View Transcription")
            # Pass use_encryption from sidebar? View section has its own toggle now.
            render_view_transcription_section(api_key, model_info)

        with tab4:
            st.header("Compare ASR Models")
            render_model_comparison_section()

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
