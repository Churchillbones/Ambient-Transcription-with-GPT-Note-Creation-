@echo off
SETLOCAL EnableDelayedExpansion

:: Self-elevate to admin if not already running as admin
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% NEQ 0 (
    echo Requesting administrative privileges...
    goto UACPrompt
) else (
    goto GotAdmin
)

:UACPrompt
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
"%temp%\getadmin.vbs"
exit /B

:GotAdmin
if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
cd /d "%~dp0"

:: Medical Transcription App Setup and Run Script
echo ====================================================
echo Medical Transcription App - Setup and Run
echo ====================================================
echo Current directory: %CD%

:: Check if Python is installed
echo [INFO] Checking for Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version is 3.8 or newer
for /f "tokens=2" %%I in ('python --version 2^>^&1') do (
    set PYTHON_VER=%%I
)
echo [INFO] Python version: %PYTHON_VER%
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% LSS 3 (
    echo [ERROR] Python 3.8+ is required. You have Python %PYTHON_VER%.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    if %MAJOR% EQU 3 (
        if %MINOR% LSS 8 (
            echo [ERROR] Python 3.8+ is required. You have Python %PYTHON_VER%.
            echo Please install Python 3.8 or newer from https://www.python.org/downloads/
            pause
            exit /b 1
        )
    )
)

:: Ensure we're in the right directory (where the batch file lives)
cd /d "%~dp0"
echo [INFO] Working directory set to: %CD%

:: Check if requirements.txt exists
echo [INFO] Checking for requirements.txt...
if exist "%~dp0requirements.txt" (
    echo [INFO] Found requirements.txt at: %~dp0requirements.txt
    copy "%~dp0requirements.txt" .\requirements.txt >nul
) else (
    echo [ERROR] requirements.txt not found in %~dp0
    echo Creating a basic requirements.txt file...
    (
        echo fastapi>=0.110
        echo uvicorn>=0.23
        echo pyaudio>=0.2.13
        echo wave
        echo vosk>=0.3.45
        echo cryptography>=41.0.0
        echo bleach>=6.0.0
        echo requests>=2.31.0
        echo openai>=1.3.0
        echo python-dotenv>=1.0.0
    ) > requirements.txt
    echo [INFO] Created requirements.txt with basic dependencies
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure you have the venv module: pip install virtualenv
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Install requirements
echo [INFO] Installing requirements from: %CD%\requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some requirements may not have installed correctly.
    echo This may be due to missing system dependencies like portaudio.
    echo.
    echo For PyAudio issues on Windows, try: pip install pipwin, then pipwin install pyaudio
    echo.
    choice /C YN /M "Continue anyway?"
    if !ERRORLEVEL! EQU 2 (
        echo [INFO] Exiting setup.
        pause
        exit /b 1
    )
)

:: Create necessary directories
echo [INFO] Creating application directories...
if not exist "app_data" mkdir app_data
if not exist "app_data\models" mkdir app_data\models
if not exist "app_data\keys" mkdir app_data\keys
if not exist "app_data\logs" mkdir app_data\logs
if not exist "app_data\cache" mkdir app_data\cache
if not exist "app_data\notes" mkdir app_data\notes
if not exist "local_llm_models" mkdir local_llm_models

:: Check for FFmpeg
if not exist "ffmpeg\bin\ffmpeg.exe" (
    echo [WARNING] FFmpeg not found in the local directory.
    echo You need FFmpeg for audio processing features to work properly.
    echo Please download FFmpeg from https://github.com/GyanD/codexffmpeg/releases/tag/2025-04-14-git-3b2a9410ef
    echo and extract it to the ffmpeg folder.
    echo.
    echo See the README.md for detailed instructions.
    echo.
    choice /C YN /M "Continue without FFmpeg?"
    if !ERRORLEVEL! EQU 2 (
        echo [INFO] Exiting setup. Please install FFmpeg before continuing.
        pause
        exit /b 1
    )
)

:: Check for Vosk models
if not exist "app_data\models\*.*" (
    echo [WARNING] No Vosk models found in app_data\models
    echo You need at least one Vosk model to use the transcription features.
    echo.
    choice /C YN /M "Do you want to download a small Vosk model now? (300MB)"
    if !ERRORLEVEL! EQU 1 (
        echo [INFO] Downloading small Vosk model (vosk-model-small-en-us-0.15)...
        echo This may take a few minutes depending on your internet connection.
        
        :: Create temp directory for download
        if not exist "temp" mkdir temp
        
        :: Download the model using PowerShell
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip' -OutFile 'temp\vosk-model.zip'}"
        
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to download Vosk model.
            echo Please download a model manually from https://alphacephei.com/vosk/models
            echo and extract it to app_data\models
        ) else (
            echo [INFO] Extracting Vosk model...
            powershell -Command "& {Expand-Archive -Path 'temp\vosk-model.zip' -DestinationPath 'app_data\models'}"
            if %ERRORLEVEL% NEQ 0 (
                echo [ERROR] Failed to extract Vosk model.
            ) else (
                echo [INFO] Vosk model downloaded and extracted successfully.
                :: Rename the folder to something simpler
                move "app_data\models\vosk-model-small-en-us-0.15" "app_data\models\small-english"
            )
            del "temp\vosk-model.zip"
            rmdir "temp"
        )
    ) else (
        echo [INFO] Skipping model download.
        echo Please download a model manually from https://alphacephei.com/vosk/models
        echo and extract it to app_data\models
    )
)

:: Check for Whisper models
if not exist "app_data\whisper_models\*.pt" (
    echo [WARNING] No Whisper models found in app_data\whisper_models
    echo You need at least one Whisper model for Whisper transcription features.
    echo.
    choice /C YN /M "Do you want to download the tiny Whisper model now? (150MB)"
    if !ERRORLEVEL! EQU 1 (
        echo [INFO] Downloading tiny Whisper model...
        echo This may take a few minutes depending on your internet connection.
        
        :: Create whisper_models directory and temp directory
        if not exist "app_data\whisper_models" mkdir app_data\whisper_models
        if not exist "temp" mkdir temp
        
        :: Download the model using PowerShell
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.pt' -OutFile 'app_data\whisper_models\tiny.pt'}"
        
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to download Whisper model.
            echo Please download models manually from https://github.com/openai/whisper/blob/main/model-card.md
            echo and place them in app_data\whisper_models
        ) else (
            echo [INFO] Whisper tiny model downloaded successfully.
        )
        if exist "temp" rmdir "temp"
    ) else (
        echo [INFO] Skipping Whisper model download.
        echo Please download models manually from https://github.com/openai/whisper/blob/main/model-card.md
        echo and place them in app_data\whisper_models
    )
)

:: Check if app.py exists, if not create a reminder
if not exist "app.py" (
    echo [ERROR] app.py not found in the current directory: %CD%
    echo.
    echo Please ensure your main Python application file is named 'app.py'
    echo and is located in the same directory as this batch file.
    pause
    exit /b 1
)

:: Check if .env file exists, create a template if not
if not exist ".env" (
    echo [INFO] Creating template .env file...
    (
        echo # Azure OpenAI API settings
        echo AZURE_API_KEY=
        echo AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
        echo MODEL_NAME=gpt-4o
        echo 
        echo # Local model settings (optional)
        echo LOCAL_MODEL_API_URL=http://localhost:8000/generate_note
        echo 
        echo # Debug settings
        echo DEBUG_MODE=False
    ) > .env
    echo [INFO] Created template .env file. Please edit it with your API keys.
)

:: Run the application
echo [INFO] Starting Backend API...
uvicorn backend.main:app --reload

:: If the server exits with an error, pause to show the error message
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application exited with an error.
    pause
    exit /b 1
)

ENDLOCAL