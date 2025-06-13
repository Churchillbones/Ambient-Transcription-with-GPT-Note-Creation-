# Ambient Transcription with GPT Note Creation 🩺

## Description

This project provides an Angular based frontend and a FastAPI backend. The application allows medical professionals to upload or record audio encounters, transcribe the audio using various Automatic Speech Recognition (ASR) models, and generate structured clinical notes using GPT models via Azure OpenAI. Optional encryption of recordings is supported and different transcription models can be compared.

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
*   **FFmpeg:** Required for audio format handling. 
    - Download from [GitHub Codex FFmpeg Release](https://github.com/GyanD/codexffmpeg/releases/tag/2025-04-14-git-3b2a9410ef)
    - Download the appropriate zip file for your system (e.g., `ffmpeg-2025-04-14-git-3b2a9410ef-essentials_build.zip`)
    - Extract the contents to a folder named `ffmpeg` in the project root directory
    - Ensure that the path `ffmpeg\bin\ffmpeg.exe` exists after extraction
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
    *   Attempt to launch the backend API (`uvicorn backend.main:app --reload`).

    *Note:* If `setup.bat` fails during dependency installation (e.g., PyAudio), you might need to install system prerequisites manually (like PortAudio) or use alternative installation methods mentioned in the script's output.

## Batch Scripts for Windows Users

This repository includes two batch scripts for Windows users to simplify setup and execution:

### `setup.bat`

A comprehensive setup script that:
- Requests administrator privileges if needed
- Verifies Python 3.8+ is installed
- Creates and configures a Python virtual environment
- Installs all dependencies from requirements.txt
- Sets up directories for the application
- Offers to download and configure FFmpeg if needed
- Offers to download a basic Vosk model if none are present
- Creates a template .env file if one doesn't exist
- Launches the application for first-time setup

To use:
```bash
setup.bat
```

### `Start_app.bat`

A streamlined script to start the application that:
- Checks if the Ollama service is running (if you're using local models)
- Updates local model information
- Starts the Ollama API bridge in a separate terminal
- Launches the main application

To use:
```bash
Start_app.bat
```

### Manual Setup (Non-Windows Users)

If you're not using Windows or prefer manual setup:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p app_data/models app_data/keys app_data/logs app_data/cache app_data/notes local_llm_models
   ```

4. Install FFmpeg manually from https://ffmpeg.org/download.html

5. Download a Vosk model (optional):
   - Download from https://alphacephei.com/vosk/models
   - Extract to app_data/models directory

6. Create a .env file with your configuration

7. Start Ollama service (if using)

8. Run the Ollama bridge:
   ```bash
   python ollama_bridge.py
   ```

9. In a separate terminal, start the backend API:
   ```bash
   uvicorn backend.main:app --reload
   ```
10. Start the Angular frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

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
3.  **Start the Backend:** Run the FastAPI server:
    ```bash
    uvicorn backend.main:app --reload
    ```
4.  **Start the Frontend:** In the `frontend` folder run:
    ```bash
    npm install
    npm start
    ```
    The Angular development server runs on `http://localhost:4200`.

## Key Dependencies

*   Angular: Frontend framework.
*   FastAPI/Uvicorn: Backend API server.
*   PyAudio/wave: Audio recording and handling.
*   Vosk: Offline speech recognition.
*   OpenAI: Azure OpenAI API client.
*   Cryptography: Data encryption.
*   Flask: Used for the Ollama bridge component.
*   python-dotenv: Loading environment variables from `.env`.
*   Bleach: HTML sanitization.
