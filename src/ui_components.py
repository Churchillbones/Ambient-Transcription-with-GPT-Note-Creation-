import streamlit as st
import asyncio
import tempfile
import datetime
import time
import os
import wave
from pathlib import Path
from typing import Tuple, Dict, Optional

# Import necessary functions from other modules within the src package
from .config import config, logger
from .utils import sanitize_input, get_file_hash # Import necessary utils
from .encryption import secure_audio_processing # Import encryption context manager
from .audio_processing import record_audio, real_time_transcribe, convert_to_wav
from .transcription import transcribe_audio, transcribe_with_vosk, transcribe_with_whisper # Import unified and specific functions
from .diarization import apply_speaker_diarization, generate_gpt_speaker_tags
from .llm_integration import generate_note
from .prompts import load_prompt_templates, save_custom_template, load_template_suggestions

# --- UI Components ---

def render_sidebar() -> Tuple[str, str, bool, str, bool]:
    """Render sidebar and return key configuration values."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input
        st.markdown("#### Azure OpenAI Credentials")
        azure_api_key = st.text_input(
            "API Key",
            key="chatbot_api_key",
            type="password",
            value=config.get("AZURE_API_KEY", ""), # Use .get for safety
            help="Enter your Azure OpenAI API Key"
        )
        # Optionally add Endpoint and Version inputs if they need to be user-configurable
        # config["AZURE_ENDPOINT"] = st.text_input("Azure Endpoint", value=config.get("AZURE_ENDPOINT", ""))
        # config["API_VERSION"] = st.text_input("API Version", value=config.get("API_VERSION", ""))

        # Model selection for Note Generation
        st.markdown("#### Note Generation Model")
        note_creation_source = st.selectbox(
            "Select Model Source",
            ["Azure OpenAI (GPT-4)", "Local LLM Model"],
            key="note_model_source",
            help="Choose between cloud-based Azure OpenAI or a locally running model via API."
        )
        use_local = note_creation_source == "Local LLM Model"

        # Local model selection if applicable
        local_model = ""
        if use_local:
            local_models_dir = config["LOCAL_MODELS_DIR"]
            if local_models_dir.exists() and local_models_dir.is_dir():
                try:
                    local_models = [d.name for d in local_models_dir.iterdir() if d.is_dir()]
                    if local_models:
                        local_model = st.selectbox(
                            "Select Local LLM Model",
                            local_models,
                            key="local_llm_select",
                            help="Select a model available in your local_llm_models directory."
                        )
                    else:
                        st.warning("No subdirectories found in local_llm_models folder.")
                except Exception as e:
                    st.error(f"Error reading local models directory: {e}")
            else:
                st.warning(f"Local LLM models folder not found or invalid: {local_models_dir}")

        # ASR Engine Selection
        st.markdown("#### Transcription (ASR) Engine")
        asr_engine = st.selectbox(
            "Choose ASR Engine:",
            ["Vosk", "Whisper"],
            key="asr_engine_select",
            help="Vosk is generally faster for real-time. Whisper (CPU) can be more accurate offline but is slower."
        )

        # Model selection based on ASR engine
        model_path_or_info = "" # This will hold Vosk path or Whisper info string
        whisper_model_size = "tiny" # Default

        if asr_engine == "Vosk":
            st.markdown("##### Select Vosk Model")
            model_dir = config["MODEL_DIR"]
            if not model_dir.exists() or not model_dir.is_dir():
                 st.error(f"Vosk models directory not found: {model_dir}")
                 model_dirs = []
            else:
                 try:
                      model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                 except Exception as e:
                      st.error(f"Error reading Vosk models directory: {e}")
                      model_dirs = []

            if not model_dirs:
                st.warning("No Vosk models found. Please download a model into the 'app_data/models' directory.")
                model_path_or_info = "" # Ensure it's empty if no models
            else:
                model_names = [d.name for d in model_dirs]
                try:
                    # Use index=0 for default if list is not empty
                    default_index = 0 if model_names else 0
                    selected_model_name = st.selectbox(
                        "Choose a Vosk model:",
                        model_names,
                        index=default_index,
                        key="vosk_model_select"
                    )
                    model_path_or_info = str(model_dir / selected_model_name)
                except Exception as e:
                     st.error(f"Error setting up Vosk model selection: {e}")
                     model_path_or_info = ""

        else:  # Whisper
            st.markdown("##### Select Whisper Model Size")
            whisper_model_size = st.selectbox(
                "Choose a Whisper model size:",
                ["tiny", "base"], # Only offer CPU-compatible sizes easily
                key="whisper_size_select",
                help="'tiny' is fastest, 'base' is more accurate but slower. Both run on CPU."
            )
            # Set model_path_or_info to the special Whisper format
            model_path_or_info = f"whisper:{whisper_model_size}"
            # Update config dict if needed elsewhere, though transcribe_audio handles the format string
            config["USE_WHISPER"] = True
            config["WHISPER_MODEL_SIZE"] = whisper_model_size

        # Encryption toggle
        st.markdown("#### Security")
        use_encryption = st.checkbox("Encrypt recordings on save", value=True,
                                    key="encrypt_toggle",
                                    help="Securely store audio files using encryption. Requires a key file.")

        # Display selected model info for confirmation
        st.markdown("---")
        st.caption(f"ASR Model: {model_path_or_info}")
        st.caption(f"Note Model: {'Local: ' + local_model if use_local else 'Azure OpenAI'}")
        st.caption(f"Encryption: {'Enabled' if use_encryption else 'Disabled'}")

        # Return the necessary config values
        return azure_api_key, model_path_or_info, use_local, local_model, use_encryption


def render_patient_data_section(key_prefix: str):
    """Render a section for inputting patient information and EHR data."""
    # Remove the key from the expander itself
    with st.expander("📋 Patient Information (Optional)", expanded=False):
        st.markdown("Enter patient details to potentially enhance note generation context.")

        # Initialize session state for patient data if not exists
        if "current_patient" not in st.session_state:
            st.session_state["current_patient"] = {"name": "", "ehr_data": ""}

        # Create columns for patient info and EHR data
        col1, col2 = st.columns([2, 1])

        with col1:
            # Basic patient information
            name = st.text_input(
                "Patient Name",
                value=st.session_state["current_patient"].get("name", ""),
                key=f"{key_prefix}_patient_name_input", # Apply prefix
                help="Enter the patient's full name (optional)"
            )

        with col2:
            # Option to clear patient data
            if st.button("Clear Patient Data", key=f"{key_prefix}_clear_patient_data_button"): # Apply prefix
                st.session_state["current_patient"] = {"name": "", "ehr_data": ""}
                st.success("Patient data cleared.")
                # No rerun needed, just clear state

        # Text area for pasting EHR data
        st.markdown("##### Patient EHR Data")
        st.markdown("Paste relevant information from the EHR system (optional):")

        ehr_data = st.text_area(
            "EHR Data",
            height=150,
            value=st.session_state["current_patient"].get("ehr_data", ""),
            key=f"{key_prefix}_ehr_data_input", # Apply prefix
            help="Copy and paste relevant patient history, medications, allergies, etc."
        )

        # Save button (updates session state)
        if st.button("Save Patient Data", key=f"{key_prefix}_save_patient_button"): # Apply prefix
            # Update session state with potentially sanitized data
            st.session_state["current_patient"]["name"] = sanitize_input(name)
            st.session_state["current_patient"]["ehr_data"] = sanitize_input(ehr_data) # Basic sanitize
            st.session_state["current_patient"]["timestamp"] = datetime.datetime.now().isoformat()
            st.success("Patient data updated in session.")

    # Return the current patient data from session state
    return st.session_state["current_patient"]


def render_custom_template_section(key_prefix: str):
    """Render the UI section for custom template input and management."""
    st.subheader("📝 Note Generation Template")
    st.markdown("Define instructions for the AI to structure the note.")

    # Template selector
    template_options = ["✨ Create New Template"] # Use an emoji or prefix
    custom_templates = load_prompt_templates() # Load all saved templates

    # Add default templates if they aren't already saved by the user under the same name
    # This logic might need refinement depending on whether defaults should be editable/overwritable
    # for name in DEFAULT_TEMPLATES:
    #     if name not in custom_templates:
    #         template_options.append(f"[Default] {name}") # Mark defaults

    if custom_templates:
        # Sort custom templates alphabetically for easier selection
        template_options.extend(sorted(list(custom_templates.keys())))

    # Use session state to manage the selected template and avoid reruns changing it
    if "selected_template_name" not in st.session_state:
         st.session_state["selected_template_name"] = template_options[0] # Default to 'Create New'

    selected_template_display = st.selectbox(
        "Select or Create Template:",
        template_options,
        key=f"{key_prefix}_template_selector", # Apply prefix
        index=template_options.index(st.session_state["selected_template_name"]) # Set index based on state
    )
    st.session_state["selected_template_name"] = selected_template_display # Update state on change

    # Extract the actual template name (without prefixes if used)
    selected_template_name = selected_template_display
    # if selected_template_name.startswith("[Default] "):
    #     selected_template_name = selected_template_name[len("[Default] "):]


    # Template suggestions expander
    with st.expander("💡 View Template Suggestions"):
        suggestions = load_template_suggestions()
        suggestion_choice = st.selectbox(
            "Select a suggested template:",
            list(suggestions.keys()),
            key=f"{key_prefix}_suggestion_selector" # Apply prefix
            )

        if st.button("Use This Suggestion", key=f"{key_prefix}_use_suggestion_button"): # Apply prefix
            # Load the suggestion text into the editor and set state
            st.session_state["custom_template_text"] = suggestions[suggestion_choice]
            st.session_state["selected_template_name"] = "✨ Create New Template" # Reset selector
            st.experimental_rerun() # Rerun to update the text area

    # Template editor text area
    # Use session state to preserve edits across interactions
    if "custom_template_text" not in st.session_state:
        st.session_state["custom_template_text"] = ""

    # Load template text if a saved template is selected and hasn't been loaded yet
    if selected_template_name != "✨ Create New Template":
         # Check if this template's text needs loading into the state
         if "loaded_template_name" not in st.session_state or st.session_state["loaded_template_name"] != selected_template_name:
              if selected_template_name in custom_templates:
                   st.session_state["custom_template_text"] = custom_templates[selected_template_name]
                   st.session_state["loaded_template_name"] = selected_template_name # Mark as loaded
              # Handle default templates if they were prefixed and selectable
              # elif selected_template_name in DEFAULT_TEMPLATES:
              #      st.session_state["custom_template_text"] = DEFAULT_TEMPLATES[selected_template_name]
              #      st.session_state["loaded_template_name"] = selected_template_name
    elif st.session_state["selected_template_name"] == "✨ Create New Template":
         # If user switches back to 'Create New', clear the loaded marker
         if "loaded_template_name" in st.session_state:
              del st.session_state["loaded_template_name"]
              # Optionally clear the text area or keep the last content?
              # st.session_state["custom_template_text"] = ""


    template_text = st.text_area(
        "Template Instructions:",
        value=st.session_state["custom_template_text"],
        height=200,
        key=f"{key_prefix}_template_editor_area", # Apply prefix
        help="Provide detailed instructions for the AI. Use {transcription} where the transcript should be inserted."
    )
    # Update session state with any edits made by the user
    st.session_state["custom_template_text"] = template_text

    # Save template section
    st.markdown("##### Save Template")
    col1, col2 = st.columns([2, 1])
    with col1:
        # Default the name input to the selected template if editing, or empty if new
        default_save_name = ""
        if selected_template_name != "✨ Create New Template":
             default_save_name = selected_template_name

        template_name_to_save = st.text_input(
            "Save as:",
            value=default_save_name,
            key=f"{key_prefix}_template_save_name", # Apply prefix
            help="Enter a name to save this template."
        )
    with col2:
        # Add vertical space using markdown or empty columns if needed
        st.markdown(" ") # Creates a little space
        st.markdown(" ") # Creates a little space
        if st.button("Save Template", key=f"{key_prefix}_save_template_button"): # Apply prefix
            if template_name_to_save and template_text:
                # Sanitize template name slightly (e.g., remove leading/trailing spaces)
                clean_name = template_name_to_save.strip()
                if not clean_name:
                     st.warning("Please enter a valid template name.")
                elif save_custom_template(clean_name, template_text):
                    st.success(f"Template '{clean_name}' saved successfully!")
                    # Update state to reflect the saved template is now selected
                    st.session_state["selected_template_name"] = clean_name
                    st.session_state["loaded_template_name"] = clean_name
                    # Force rerun to update selector and potentially disable "Create New" state
                    st.experimental_rerun()
                else:
                    st.error("Failed to save template.")
            elif not template_name_to_save:
                 st.warning("Please enter a name for the template.")
            else: # Not template_text
                 st.warning("Template content cannot be empty.")


    # Return the current template text from the editor
    return template_text


def render_recording_section(api_key, model_info, use_local, local_model, use_encryption):
    """Render the Record Audio section."""

    # Get patient data (optional), passing a unique prefix
    patient_data = render_patient_data_section(key_prefix="record")

    # Get custom template, passing a unique prefix
    custom_template = render_custom_template_section(key_prefix="record")

    st.subheader("🎤 Record New Audio")

    # Recording settings
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Recording Duration (seconds)", value=30, min_value=5, max_value=600, step=5, key="rec_duration")
    with col2:
        # Real-time is only available with Vosk
        real_time_available = isinstance(model_info, str) and not model_info.startswith("whisper:")
        real_time = st.toggle(
            "Real-time Transcription",
            value=False,
            key="real_time_toggle",
            help="Show transcription as you speak (only available with Vosk models)",
            disabled=not real_time_available
        )
        if not real_time_available and real_time:
            st.info("Real-time transcription requires a Vosk model to be selected in the sidebar.")
            # Force real_time off if Vosk isn't selected
            real_time = False

    # Record button
    if st.button("Start Recording", key="start_recording_button"):
        if not model_info:
            st.error("Please select a valid ASR model in the sidebar.")
            return # Stop processing if no model selected

        # --- Recording Logic ---
        audio_path = ""
        transcription = ""
        audio_data = b""

        if real_time:
            # --- Real-time Transcription Mode (Vosk only) ---
            st.info("Starting real-time transcription...")
            try:
                # real_time_transcribe handles UI updates (progress, text area)
                transcription, audio_data = real_time_transcribe(duration, model_info) # model_info is Vosk path here

                if not transcription and not audio_data:
                     st.error("Real-time transcription failed to start or produced no data.")
                     return # Stop if function failed severely

                # Need to save the audio_data to a file path for secure_audio_processing
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_audio_path = config["CACHE_DIR"] / f"realtime_rec_{timestamp}.wav"
                try:
                     # Need PyAudio to get sample width if not passed back
                     import pyaudio
                     p = pyaudio.PyAudio()
                     sample_width = p.get_sample_size(getattr(pyaudio, config["FORMAT_STR"]))
                     p.terminate()

                     with wave.open(str(temp_audio_path), 'wb') as wf:
                          wf.setnchannels(config["CHANNELS"])
                          wf.setsampwidth(sample_width)
                          wf.setframerate(config["RATE"])
                          wf.writeframes(audio_data)
                     audio_path = str(temp_audio_path)
                     logger.info(f"Saved real-time audio data to {audio_path}")
                except Exception as e:
                     logger.error(f"Failed to save audio data after real-time transcription: {e}")
                     st.error("Failed to save audio data after real-time transcription.")
                     # Proceed with transcription if available, but warn about missing audio file
                     audio_path = "" # Indicate file saving failed

            except Exception as e:
                st.error(f"An error occurred during real-time transcription: {e}")
                logger.error(f"Real-time transcription process error: {e}")
                return # Stop processing

        else:
            # --- Standard Record-then-Transcribe Mode ---
            st.info("Starting recording...")
            try:
                # record_audio handles UI updates (progress bar) and saves the file
                audio_path, audio_data = record_audio(duration)
                if not audio_path:
                     st.error("Recording failed. Please check logs.")
                     return # Stop if recording failed

                st.success(f"Recording saved: {Path(audio_path).name}")

                # Transcribe after recording
                with st.spinner("Transcribing audio..."):
                    # Use the unified transcribe_audio function
                    transcription = transcribe_audio(audio_path, model_info)

                if not transcription or transcription.startswith("ERROR"):
                    st.error(f"Transcription failed: {transcription}")
                    # Keep audio_path for potential manual processing or debugging
                else:
                    st.markdown("### Raw Transcription")
                    st.text_area("Transcription Result:", transcription, height=150, key="raw_transcription_output")

            except Exception as e:
                st.error(f"An error occurred during recording or transcription: {e}")
                logger.error(f"Record/Transcribe process error: {e}")
                return # Stop processing


        # --- Post-Transcription Processing (Common for both modes) ---
        if not transcription or transcription.startswith("ERROR:"):
             st.warning("Cannot proceed with note generation due to transcription failure.")
             # Clean up audio file if it exists and wasn't encrypted? Optional.
             return

        # Use secure_audio_processing for potential encryption *after* recording/transcription
        # This ensures the file exists before trying to encrypt it.
        processed_audio_path_for_log = "N/A (Encryption Disabled or Failed)"
        if audio_path: # Only proceed if audio was successfully saved
             try:
                  with secure_audio_processing(audio_path, use_encryption) as processing_path:
                       # The context manager handles encryption/decryption/cleanup.
                       # We don't actually *use* processing_path here, as transcription is done.
                       # We mainly use it to trigger the encryption of the original audio_path if requested.
                       if use_encryption:
                            # Check if the .enc file was created (logic inside secure_audio_processing)
                            enc_path = Path(audio_path).with_suffix('.enc')
                            if enc_path.exists():
                                 st.info(f"Encrypted recording saved: {enc_path.name}")
                                 processed_audio_path_for_log = str(enc_path)
                            else:
                                 # This case happens if encryption failed within the context manager
                                 st.warning("Audio encryption failed. Original recording remains unencrypted.")
                                 processed_audio_path_for_log = f"{audio_path} (Encryption Failed)"
                       else:
                            processed_audio_path_for_log = f"{audio_path} (Unencrypted)"
             except Exception as e:
                  st.error(f"Error during secure audio handling: {e}")
                  logger.error(f"Secure audio processing context failed: {e}")
                  # Log that the original might still exist unencrypted
                  processed_audio_path_for_log = f"{audio_path} (Secure Handling Error)"

        logger.info(f"Final audio state: {processed_audio_path_for_log}")


        # --- Diarization ---
        diarized_transcription = ""
        with st.spinner("Adding speaker labels..."):
            # Use Azure OpenAI for diarization if key is provided, otherwise basic
            diarized_transcription = asyncio.run(generate_gpt_speaker_tags(transcription, api_key))

        if diarized_transcription and diarized_transcription != transcription: # Show only if different
            st.markdown("### Diarized Transcription")
            st.text_area("Speakers:", diarized_transcription, height=150, key="diarized_transcription_output")
        else:
            # If GPT diarization failed or wasn't used, the raw transcription is already shown
            logger.info("Diarization did not produce a different result or was skipped.")
            diarized_transcription = transcription # Use raw transcription for note gen if diarization failed


        # --- Note Generation ---
        if not custom_template:
            st.warning("Please provide a template in the 'Note Generation Template' section to generate a structured note.")
        elif not api_key and not use_local:
             st.warning("Please provide an Azure API key or select a local model in the sidebar for note generation.")
        else:
            with st.spinner("Generating medical note..."):
                note = asyncio.run(generate_note(
                    diarized_transcription, # Use potentially diarized text
                    api_key,
                    custom_template,
                    use_local,
                    local_model,
                    patient_data # Pass patient data
                ))

                if note and not note.startswith("Error:"):
                    st.subheader("📄 Structured Note Output")
                    st.text_area("Generated Note:", note, height=300, key="generated_note_output")

                    # --- Edit and Export ---
                    st.markdown("##### Review and Edit Note")
                    edited_note = st.text_area("Edit the note as needed:", value=note, height=200, key="edited_note_area")

                    col1_exp, col2_exp = st.columns(2)
                    with col1_exp:
                        # Simple copy-paste button using JavaScript hack (may have browser limitations)
                        # Consider using a dedicated library like streamlit-clipboard if needed
                        if st.button("Copy to Clipboard", key="copy_note_button"):
                             # Basic JS injection - might not work in all environments/browsers
                             st.info("Note content ready to be pasted (Ctrl+C/Cmd+C if button doesn't work).")
                             # st.code(edited_note) # Show code block to easily copy

                    with col2_exp:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="Download Note (.txt)",
                            data=edited_note.encode('utf-8'), # Encode to bytes
                            file_name=f"medical_note_{timestamp}.txt",
                            mime="text/plain",
                            key="download_note_button"
                        )
                elif note: # Handle notes that start with "Error:"
                     st.error(f"Note generation failed: {note}")
                else: # Handle empty note case
                     st.error("Note generation returned an empty result.")


def render_upload_section(api_key, model_info, use_local, local_model):
    """Render the Upload Audio section."""

    # Get patient data (optional), passing a unique prefix
    patient_data = render_patient_data_section(key_prefix="upload")

    # Get custom template, passing a unique prefix
    custom_template = render_custom_template_section(key_prefix="upload")

    st.subheader("⬆️ Upload Existing Audio File")

    # File uploader
    uploaded_audio = st.file_uploader(
        "Select an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"], # Added aac
        key="audio_uploader"
    )

    # Encryption toggle for uploaded files
    use_encryption_upload = st.checkbox("Encrypt uploaded audio on save", value=True,
                                     key="encrypt_upload_toggle",
                                     help="Securely store an encrypted copy of this audio file after processing.")

    # Additional notes input (optional)
    st.markdown("##### Add Additional Clinician Notes (Optional)")
    clinician_notes = st.text_area("Type or paste additional notes here:", height=100, key="clinician_notes_area")

    # Upload notes file (optional)
    uploaded_notes_file = st.file_uploader(
        "Or upload a text file (.txt) with notes",
        type=["txt"],
        key="notes_uploader"
    )

    if uploaded_audio is None:
        st.info("Upload an audio file to begin processing.")
        return

    # Process button
    if st.button("Process Uploaded Audio", key="process_upload_button"):
        if not model_info:
            st.error("Please select a valid ASR model in the sidebar.")
            return
        if not custom_template:
             st.warning("Please provide a template in the 'Note Generation Template' section to generate a structured note.")
             # Allow transcription without template? Maybe add separate button later. For now, require template.
             # return
        if not api_key and not use_local:
             st.warning("Please provide an Azure API key or select a local model in the sidebar for note generation.")
             # Allow transcription without LLM?
             # return


        # --- File Handling and Conversion ---
        temp_dir = Path(tempfile.gettempdir())
        input_path = temp_dir / uploaded_audio.name
        final_audio_path = "" # Path to the final WAV file used for transcription
        original_input_deleted = False

        try:
            # Save uploaded file temporarily
            with open(input_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            logger.info(f"Uploaded file saved temporarily to: {input_path}")

            # Convert to WAV if necessary
            if not input_path.suffix.lower() == '.wav':
                with st.spinner(f"Converting {input_path.suffix} to WAV..."):
                    try:
                        final_audio_path = convert_to_wav(str(input_path))
                        logger.info(f"Converted audio to WAV: {final_audio_path}")
                        # Clean up original non-WAV temp file
                        input_path.unlink()
                        original_input_deleted = True
                    except Exception as e:
                        st.error(f"Audio conversion failed: {e}")
                        logger.error(f"Conversion failed for {input_path}: {e}")
                        return # Stop processing
            else:
                # If already WAV, use its path (might still be in temp)
                final_audio_path = str(input_path)
                logger.info(f"Input file is already WAV: {final_audio_path}")


            # --- Transcription ---
            transcription = ""
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(final_audio_path, model_info)

            if not transcription or transcription.startswith("ERROR"):
                st.error(f"Transcription failed: {transcription}")
                return # Stop processing
            else:
                st.markdown("### Raw Transcription")
                st.text_area("Transcription Result:", transcription, height=150, key="raw_transcription_upload_output")


            # --- Process Additional Notes ---
            additional_text = clinician_notes.strip()
            if uploaded_notes_file is not None:
                try:
                    notes_content = uploaded_notes_file.read().decode("utf-8")
                    additional_text += "\n" + notes_content.strip()
                    logger.info("Loaded additional notes from uploaded file.")
                except Exception as e:
                    st.error(f"Failed to read uploaded notes file: {e}")
                    logger.error(f"Error reading notes file {uploaded_notes_file.name}: {e}")
            additional_text = sanitize_input(additional_text)


            # --- Diarization ---
            diarized_transcription = ""
            with st.spinner("Adding speaker labels..."):
                diarized_transcription = asyncio.run(generate_gpt_speaker_tags(transcription, api_key))

            # Combine with additional notes if provided
            final_text_for_note = diarized_transcription
            if additional_text:
                final_text_for_note += f"\n\n--- Additional Clinician Notes ---\n{additional_text}"
                logger.info("Appended additional clinician notes.")

            if final_text_for_note != transcription: # Show if different from raw
                st.markdown("### Diarized Transcription with Notes")
                st.text_area("Combined Text:", final_text_for_note, height=150, key="diarized_upload_output")


            # --- Secure Processing (Encryption) ---
            # Apply encryption to the final WAV file if requested
            processed_audio_path_for_log = "N/A (Encryption Disabled or Failed)"
            try:
                 with secure_audio_processing(final_audio_path, use_encryption_upload) as processing_path:
                      # Context manager handles encryption of final_audio_path
                      if use_encryption_upload:
                           enc_path = Path(final_audio_path).with_suffix('.enc')
                           if enc_path.exists():
                                st.info(f"Encrypted audio saved: {enc_path.name}")
                                processed_audio_path_for_log = str(enc_path)
                           else:
                                st.warning("Audio encryption failed. Original WAV remains unencrypted.")
                                processed_audio_path_for_log = f"{final_audio_path} (Encryption Failed)"
                      else:
                           processed_audio_path_for_log = f"{final_audio_path} (Unencrypted)"
            except Exception as e:
                 st.error(f"Error during secure audio handling: {e}")
                 logger.error(f"Secure audio processing context failed for upload: {e}")
                 processed_audio_path_for_log = f"{final_audio_path} (Secure Handling Error)"

            logger.info(f"Final audio state for upload: {processed_audio_path_for_log}")


            # --- Note Generation ---
            if not custom_template:
                 st.warning("No template provided. Skipping note generation.")
            elif not api_key and not use_local:
                 st.warning("No Azure API key or local model selected. Skipping note generation.")
            else:
                with st.spinner("Generating medical note..."):
                    note = asyncio.run(generate_note(
                        final_text_for_note, # Use combined text
                        api_key,
                        custom_template,
                        use_local,
                        local_model,
                        patient_data # Pass patient data
                    ))

                    if note and not note.startswith("Error:"):
                        st.subheader("📄 Structured Note Output")
                        st.text_area("Generated Note:", note, height=300, key="generated_note_upload_output")

                        # --- Edit and Export ---
                        st.markdown("##### Review and Edit Note")
                        edited_note = st.text_area("Edit the note as needed:", value=note, height=200, key="edited_note_upload_area")

                        col1_exp, col2_exp = st.columns(2)
                        with col1_exp:
                            if st.button("Copy to Clipboard", key="copy_note_upload_button"):
                                st.info("Note content ready to be pasted.")
                                # st.code(edited_note)
                        with col2_exp:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="Download Note (.txt)",
                                data=edited_note.encode('utf-8'),
                                file_name=f"medical_note_{timestamp}.txt",
                                mime="text/plain",
                                key="download_note_upload_button"
                            )
                    elif note:
                         st.error(f"Note generation failed: {note}")
                    else:
                         st.error("Note generation returned an empty result.")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            logger.error(f"Error processing uploaded file {uploaded_audio.name}: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Clean up the initial temporary file if it wasn't the final WAV
            if input_path.exists() and input_path != Path(final_audio_path):
                try:
                    input_path.unlink()
                    logger.debug(f"Removed initial temp file: {input_path}")
                except Exception as e:
                    logger.error(f"Failed to remove initial temp file {input_path}: {e}")
            # Clean up the final WAV file *unless* it was encrypted (context manager handles temp decrypted file)
            # If encryption was off, or failed, the final_audio_path might still be the .wav in cache.
            # If encryption was ON and SUCCEEDED, secure_audio_processing should have created .enc
            # and we might want to delete the .wav. For now, let's keep the final .wav if it exists
            # unless encryption was explicitly successful.
            # This cleanup logic needs careful review based on desired persistence.
            # Let's keep the final WAV for now for simplicity, assuming cache might be cleared later.
            # if final_audio_path and Path(final_audio_path).exists() and use_encryption_upload and Path(final_audio_path).with_suffix('.enc').exists():
            #      try:
            #           Path(final_audio_path).unlink()
            #           logger.debug(f"Removed final WAV after successful encryption: {final_audio_path}")
            #      except Exception as e:
            #           logger.error(f"Failed to remove final WAV {final_audio_path}: {e}")
            pass


def render_view_transcription_section(api_key, model_info):
    """Render the View Transcription section."""

    # Get patient data (optional, maybe less relevant here but keep consistent)
    # patient_data = render_patient_data_section() # Commented out for transcription-only view

    st.subheader("📄 View Transcription Only")
    st.markdown("Upload an audio file to view its transcription and speaker labels.")

    # File uploader
    uploaded_audio = st.file_uploader(
        "Select an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        key="transcribe_only_uploader"
    )

    # Encryption toggle (less critical here, but maybe user wants to store encrypted copy)
    use_encryption_view = st.checkbox("Encrypt audio on save (optional)", value=False,
                                    key="encrypt_view_toggle",
                                    help="Store an encrypted copy of this audio file.")

    if uploaded_audio is None:
        st.info("Upload an audio file to view its transcription.")
        return

    # Process button
    if st.button("Transcribe Audio File", key="transcribe_only_button"):
        if not model_info:
            st.error("Please select a valid ASR model in the sidebar.")
            return

        # --- File Handling and Conversion ---
        temp_dir = Path(tempfile.gettempdir())
        input_path = temp_dir / uploaded_audio.name
        final_audio_path = ""
        original_input_deleted = False

        try:
            # Save uploaded file temporarily
            with open(input_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            logger.info(f"Transcribe-only file saved temporarily to: {input_path}")

            # Convert to WAV if necessary
            if not input_path.suffix.lower() == '.wav':
                with st.spinner(f"Converting {input_path.suffix} to WAV..."):
                    try:
                        final_audio_path = convert_to_wav(str(input_path))
                        logger.info(f"Converted audio to WAV: {final_audio_path}")
                        input_path.unlink()
                        original_input_deleted = True
                    except Exception as e:
                        st.error(f"Audio conversion failed: {e}")
                        logger.error(f"Conversion failed for {input_path}: {e}")
                        return
            else:
                final_audio_path = str(input_path)
                logger.info(f"Input file is already WAV: {final_audio_path}")

            # --- Transcription ---
            transcription = ""
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(final_audio_path, model_info)

            if not transcription or transcription.startswith("ERROR"):
                st.error(f"Transcription failed: {transcription}")
                return
            else:
                st.markdown("### Raw Transcription")
                st.text_area("Transcription Result:", transcription, height=200, key="raw_transcription_view_output")

            # --- Diarization ---
            diarized_transcription = ""
            with st.spinner("Adding speaker labels..."):
                diarized_transcription = asyncio.run(generate_gpt_speaker_tags(transcription, api_key))

            if diarized_transcription and diarized_transcription != transcription:
                st.markdown("### Transcription with Speaker Labels")
                st.text_area("Speakers:", diarized_transcription, height=200, key="diarized_transcription_view_output")
                # Use diarized version for export if available
                export_text = diarized_transcription
            else:
                st.info("Basic speaker labels applied or GPT tagging skipped/failed.")
                # Apply basic if GPT failed or no key
                basic_diarized = apply_speaker_diarization(transcription)
                st.markdown("### Basic Speaker Labels")
                st.text_area("Basic Speakers:", basic_diarized, height=200, key="basic_diarized_view_output")
                export_text = basic_diarized # Export basic version

            # --- Secure Processing (Encryption) ---
            processed_audio_path_for_log = "N/A (Encryption Disabled or Failed)"
            try:
                 with secure_audio_processing(final_audio_path, use_encryption_view) as processing_path:
                      if use_encryption_view:
                           enc_path = Path(final_audio_path).with_suffix('.enc')
                           if enc_path.exists():
                                st.info(f"Encrypted audio saved: {enc_path.name}")
                                processed_audio_path_for_log = str(enc_path)
                           else:
                                st.warning("Audio encryption failed. Original WAV remains unencrypted.")
                                processed_audio_path_for_log = f"{final_audio_path} (Encryption Failed)"
                      else:
                           processed_audio_path_for_log = f"{final_audio_path} (Unencrypted)"
            except Exception as e:
                 st.error(f"Error during secure audio handling: {e}")
                 logger.error(f"Secure audio processing context failed for view: {e}")
                 processed_audio_path_for_log = f"{final_audio_path} (Secure Handling Error)"

            logger.info(f"Final audio state for view: {processed_audio_path_for_log}")


            # --- Export Options ---
            st.markdown("---")
            col1_exp, col2_exp = st.columns(2)
            with col1_exp:
                if st.button("Copy Transcription", key="copy_transcription_view_button"):
                    st.info("Transcription ready to be pasted.")
                    # st.code(export_text)
            with col2_exp:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Transcription (.txt)",
                    data=export_text.encode('utf-8'),
                    file_name=f"transcription_{timestamp}.txt",
                    mime="text/plain",
                    key="download_transcription_view_button"
                )

        except Exception as e:
            st.error(f"An unexpected error occurred during transcription viewing: {e}")
            logger.error(f"Error processing file for viewing {uploaded_audio.name}: {e}", exc_info=True)
        finally:
            # --- Cleanup --- (Similar to upload section)
            if input_path.exists() and input_path != Path(final_audio_path):
                try:
                    input_path.unlink()
                    logger.debug(f"Removed initial temp file (view): {input_path}")
                except Exception as e:
                    logger.error(f"Failed to remove initial temp file {input_path} (view): {e}")
            # Keep final WAV for now
            pass


def render_model_comparison_section():
    """Allow testing different ASR models on the same audio."""
    st.header("🧪 ASR Model Comparison")
    st.markdown("Upload an audio file to compare transcription results between different engines and models.")

    # Upload audio file for testing
    uploaded_audio = st.file_uploader(
        "Upload audio for comparison",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        key="compare_uploader"
    )

    if uploaded_audio is None:
        st.info("Upload an audio file to start the comparison.")
        return

    # --- Model Selection for Comparison ---
    st.subheader("Select Models to Compare")
    col1, col2 = st.columns(2)

    vosk_model_to_test = None
    with col1:
        st.markdown("##### Vosk Models")
        vosk_enabled = st.checkbox("Test Vosk", value=True, key="compare_vosk_enabled")
        if vosk_enabled:
            model_dir = config["MODEL_DIR"]
            if not model_dir.exists() or not model_dir.is_dir():
                 st.warning(f"Vosk models directory not found: {model_dir}")
                 model_dirs = []
            else:
                 try:
                      model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                 except Exception as e:
                      st.warning(f"Error reading Vosk models directory: {e}")
                      model_dirs = []

            if not model_dirs:
                st.warning("No Vosk models found to compare.")
                vosk_enabled = False # Disable if none found
            else:
                model_names = [d.name for d in model_dirs]
                try:
                    selected_vosk_model_name = st.selectbox(
                        "Choose Vosk model:",
                        model_names,
                        key="compare_vosk_select"
                        )
                    vosk_model_to_test = str(model_dir / selected_vosk_model_name)
                except Exception as e:
                     st.error(f"Error setting up Vosk model selection: {e}")
                     vosk_enabled = False


    whisper_sizes_to_test = []
    with col2:
        st.markdown("##### Whisper Models (CPU)")
        whisper_enabled = st.checkbox("Test Whisper", value=True, key="compare_whisper_enabled")
        if whisper_enabled:
            whisper_sizes_to_test = st.multiselect(
                "Select Whisper model sizes:",
                ["tiny", "base"], # Only offer CPU sizes
                default=["tiny", "base"],
                key="compare_whisper_select"
            )
            if not whisper_sizes_to_test:
                 whisper_enabled = False # Disable if none selected


    # Performance metrics toggle
    # collect_metrics = st.checkbox("Collect performance metrics (CPU/Memory)", value=False, key="compare_metrics")

    # Run comparison button
    if st.button("Run Comparison", key="run_comparison_button"):
        if not vosk_enabled and not whisper_enabled:
             st.warning("Please enable at least one ASR engine (Vosk or Whisper) to compare.")
             return

        # --- File Handling ---
        temp_dir = Path(tempfile.gettempdir())
        input_path = temp_dir / f"compare_{uploaded_audio.name}"
        final_audio_path = ""
        results = [] # Store results: {"Model": name, "Time (s)": time, "Transcription": text}

        try:
            # Save uploaded file temporarily
            with open(input_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            logger.info(f"Comparison file saved temporarily to: {input_path}")

            # Convert to WAV if necessary
            if not input_path.suffix.lower() == '.wav':
                with st.spinner(f"Converting {input_path.suffix} to WAV..."):
                    try:
                        final_audio_path = convert_to_wav(str(input_path))
                        logger.info(f"Converted comparison audio to WAV: {final_audio_path}")
                        input_path.unlink() # Clean up original non-WAV temp
                    except Exception as e:
                        st.error(f"Audio conversion failed: {e}")
                        logger.error(f"Conversion failed for comparison file {input_path}: {e}")
                        return
            else:
                final_audio_path = str(input_path)
                logger.info(f"Comparison input file is already WAV: {final_audio_path}")

            # Display audio player for reference
            st.subheader("Original Audio")
            try:
                with open(final_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                 st.error(f"Failed to load audio player: {e}")


            # --- Run Transcriptions ---
            st.subheader("Transcription Results")

            # Run Vosk if selected
            if vosk_enabled and vosk_model_to_test:
                model_name = Path(vosk_model_to_test).name
                with st.spinner(f"Transcribing with Vosk ({model_name})..."):
                    start_time = time.time()
                    # Use the specific function to ensure caching works if desired
                    vosk_transcription = transcribe_with_vosk(final_audio_path, vosk_model_to_test)
                    vosk_time = time.time() - start_time
                    results.append({
                        "Model": f"Vosk ({model_name})",
                        "Time (s)": round(vosk_time, 2),
                        "Transcription": vosk_transcription
                    })
                    logger.info(f"Vosk comparison done ({model_name}): {vosk_time:.2f}s")

            # Run Whisper for each selected model size
            for model_size in whisper_sizes_to_test:
                 with st.spinner(f"Transcribing with Whisper {model_size} (CPU)..."):
                    start_time = time.time()
                    # Use the specific function
                    whisper_transcription = transcribe_with_whisper(final_audio_path, model_size)
                    whisper_time = time.time() - start_time
                    results.append({
                        "Model": f"Whisper {model_size} (CPU)",
                        "Time (s)": round(whisper_time, 2),
                        "Transcription": whisper_transcription
                    })
                    logger.info(f"Whisper comparison done ({model_size}): {whisper_time:.2f}s")

            # --- Display Results ---
            if not results:
                 st.warning("No models were selected or ran successfully.")
                 return

            # Display each result in a text area
            for result in results:
                 st.markdown(f"##### {result['Model']} ({result['Time (s)']} seconds)")
                 st.text_area(
                      f"Result_{result['Model'].replace(' ','_')}", # Unique key for text area
                      result["Transcription"],
                      height=150,
                      key=f"compare_output_{result['Model']}"
                      )

            # --- Performance Table ---
            st.subheader("Performance Summary")
            performance_data = [
                 {"Model": r["Model"], "Time (s)": r["Time (s)"], "Word Count": len(r["Transcription"].split())}
                 for r in results if not r["Transcription"].startswith("ERROR") # Exclude errors from summary
            ]
            if performance_data:
                 st.table(performance_data)
            else:
                 st.warning("No successful transcriptions to summarize performance.")


        except Exception as e:
            st.error(f"An unexpected error occurred during comparison: {e}")
            logger.error(f"Error during model comparison: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Clean up the final WAV file used for comparison
            if final_audio_path and Path(final_audio_path).exists():
                try:
                    Path(final_audio_path).unlink()
                    logger.debug(f"Removed comparison WAV file: {final_audio_path}")
                except Exception as e:
                    logger.error(f"Failed to remove comparison WAV file {final_audio_path}: {e}")
            # Also clean up original temp if conversion failed mid-way and it wasn't deleted
            if input_path.exists() and input_path != Path(final_audio_path):
                 try:
                      input_path.unlink()
                      logger.debug(f"Cleaned up initial comparison temp file: {input_path}")
                 except Exception as e:
                      logger.error(f"Failed to clean up initial comparison temp file {input_path}: {e}")
