@echo off
title Velaris GPU Force Installer (CUDA 12.1)
cls
echo ===================================================
echo   Velaris GPU Force Installer (CUDA 12.1 / RTX 4060)
echo ===================================================
echo.
echo This script ensures PyTorch is installed with NVIDIA CUDA 12.1 acceleration.
echo.

cd /d "%~dp0\backend"

:: Check for python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org.
    pause
    exit /b 1
)

:: Ensure venv exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

echo.
echo 1. Uninstalling existing CPU torch packages...
venv\Scripts\pip uninstall -y torch torchvision torchaudio

echo.
echo 2. Installing CUDA 12.4 PyTorch 2.6 binaries (~2.5 GB)...
venv\Scripts\pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo.
echo 3. Installing Velaris requirements...
venv\Scripts\pip install -r requirements.txt

echo.
echo 4. Running Pre-flight Diagnostics...
venv\Scripts\python.exe preflight.py

echo.
echo If all checks passed above, run_velaris.bat is ready to start!
pause
