@echo off
SETLOCAL EnableDelayedExpansion

echo ====================================================
echo Starting Medical Transcription App with Ollama
echo ====================================================

echo [INFO] Checking Ollama service...
curl http://localhost:11434/api/tags > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama service is not running.
    echo Please start Ollama first with 'ollama serve'
    pause
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

echo [INFO] Starting Ollama API bridge...
start "Ollama Bridge" cmd /c "call venv\Scripts\activate.bat && python ollama_bridge.py"

echo [INFO] Waiting for API bridge to start...
timeout /t 3 > nul

echo [INFO] Starting Backend API...
uvicorn backend.main:app --reload

ENDLOCAL