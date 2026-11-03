@echo off
title Velaris Dependency Setup v1.1.0
echo Setting up Velaris Motion Environment...
echo.

:: Check if python exists
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

:: Create Virtual Environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate and Install
echo [INFO] Activating venv and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo [INFO] Running Pre-flight Diagnostics...
python preflight.py

echo.
echo Setup Complete! You can now launch the engine using ..\run_velaris.bat
pause
