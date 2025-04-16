# Ambient Transcription with GPT Note Creation 🩺

## Description

A Streamlit web application designed for medical professionals. It facilitates the recording or uploading of patient audio encounters, transcribes the audio using various Automatic Speech Recognition (ASR) models (including local Vosk, OpenAI API via Azure, local Whisper models, and local LLMs via Ollama), and generates structured clinical notes based on customizable templates using GPT models (via Azure OpenAI). The app features optional encryption for sensitive data and allows for comparison between different transcription models.

## Features

*   **Audio Recording:** Record audio directly within the application.
*   **Audio Upload:** Upload existing audio files (e.g., WAV).
*   **Transcription:**
    *   Utilizes Vosk for local, offline transcription.
    *   Integrates with Azure OpenAI API for cloud-based transcription (Whisper) and note generation (GPT).
    *   Supports local Whisper models (`.pt` files).
    *   Supports other local Large Language Models (LLMs) via an Ollama bridge for transcription/note generation.
*   **Note Generation:** Creates clinical notes from transcriptions using GPT models (via Azure OpenAI) and predefined or custom templates.
*   **Model Comparison:** Allows side-by-side comparison of results from different ASR models.
*   **Encryption:** Optional encryption/decryption for audio files and potentially transcriptions using `cryptography`.
*   **Configuration:** Easy configuration of API keys, model selection (local vs. API), and encryption settings via the sidebar and `.env` file.

## Prerequisites

*   **Python:** Version 3.8 or newer. The setup script checks for this.
*   **pip:** Python package installer.
*   **FFmpeg:** Required for audio format handling. The setup script can attempt to download and install it if `ffmpeg-installer.bat` is present, or you can install it manually from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and ensure it's accessible or placed in the `ffmpeg` directory.
*   **(Optional) Ollama:** Required if using non-GPT local LLMs. Needs to be installed and running separately. [Link to Ollama setup guide if available]
*   **(Optional) Vosk Models:** Required for Vosk transcription. The setup script can download a small English model (`vosk-model-small-en-us-0.15`) automatically. You can download other models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and place them in the `app_data/models/` directory (e.g., `app_data/models/vosk-model-en-us-0.22`).
*   **(Optional) Local Whisper Models:** Required for local Whisper transcription. Download model files (e.g., `tiny.pt`, `base.pt`) and place them in `app_data/whisper_models/`.

## Installation & Setup

The `setup.bat` script automates most of the setup process.

1.  **Clone/Download:** Get the project source code.
    ```bash
    # Clone the repository
    git clone https://github.com/Churchillbones/Ambient-Transcription-with-GPT-Note-Creation-
    # Navigate to the project directory
    cd Ambient-Transcription-with-GPT-Note-Creation-
    ```
2.  **Navigate:** Open a terminal or command prompt **as Administrator** in the project's root directory. The setup script requires admin privileges.
3.  **Run Setup Script:** Execute the setup batch file.
    ```bash
    setup.bat
    ```
    This script will:
    *   Check for Python 3.8+.
    *   Create a Python virtual environment named `venv`.
    *   Activate the virtual environment.
    *   Install required Python packages from `requirements.txt`.
    *   Create necessary directories (`app_data`, `local_llm_models`, etc.).
    *   Check for FFmpeg and optionally download/install it.
    *   Check for Vosk models and optionally download a default small English model.
    *   Create a template `.env` file if one doesn't exist.
    *   Attempt to launch the application (`streamlit run app.py`).

    *Note:* If `setup.bat` fails during dependency installation (e.g., PyAudio), you might need to install system prerequisites manually (like PortAudio) or use alternative installation methods mentioned in the script's output.

## Configuration

1.  **Environment Variables:** After running `setup.bat` once, a `.env` file should exist in the project root. Edit this file to add your credentials and settings:
    ```dotenv
    # Azure OpenAI API settings
    AZURE_API_KEY=YOUR_AZURE_OPENAI_API_KEY
    AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
    MODEL_NAME=gpt-4o # Or your desired Azure OpenAI deployment name

    # Local model settings (optional)
    LOCAL_MODEL_API_URL=http://localhost:8000/generate_note # URL for Ollama bridge or similar

    # Debug settings
    DEBUG_MODE=False # Set to True for more verbose logging
    ```
2.  **Application Settings:** Further configuration (like selecting specific models, toggling encryption) can often be done directly in the application's sidebar when it's running.

## Usage

1.  **Prerequisites:** Ensure any necessary external services (like Ollama) are running and prerequisites (like FFmpeg) are installed.
2.  **Activate Environment:** Open a terminal in the project root and activate the virtual environment:
    ```bash
    .\venv\Scripts\activate
    ```
3.  **Start the App:** Run the application using the `Start_app.bat` script (assuming it just runs `streamlit run app.py` within the venv) or directly with Streamlit:
    ```bash
    streamlit run app.py
    ```
    *(Note: You might need to read `Start_app.bat` to confirm its exact function if it differs from `setup.bat`'s final step).*
4.  **Access:** Open your web browser and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`).

## Key Dependencies

*   Streamlit: Web application framework.
*   PyAudio/wave: Audio recording and handling.
*   Vosk: Offline speech recognition.
*   OpenAI: Azure OpenAI API client.
*   Cryptography: Data encryption.
*   Flask: Used for the Ollama bridge component.
*   python-dotenv: Loading environment variables from `.env`.
*   Bleach: HTML sanitization.
