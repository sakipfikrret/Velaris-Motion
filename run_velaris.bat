@echo off
title Velaris AI Controller
cls
echo ===================================================
echo   Velaris AI Engine
echo ===================================================

cd /d "%~dp0\backend"

:: 1. CHECK FOR PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b
)

:: 2. CHECK/CREATE VENV
if not exist "venv" (
    echo [INFO] First time setup... Creating environment...
    python -m venv venv
    
    echo [INFO] Installing GPU Drivers and Dependencies...
    :: Install PyTorch with CUDA 12.1 support
    venv\Scripts\pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    :: Install other requirements
    venv\Scripts\pip install -r requirements.txt
)

:: 3. START SERVER
cls
echo ===================================================
echo   Velaris Engine is STARTING...
echo ===================================================
echo.
venv\Scripts\python.exe api.py
pause
