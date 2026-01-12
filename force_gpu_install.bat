@echo off
title Veo-X GPU FORCE INSTALL
echo ===================================================
echo   Veo-X GPU Force Upgrader (RTX 4060)
echo ===================================================
echo.
echo We detected "PyTorch CPU" version.
echo We are now going to DELETE it and force "PyTorch GPU" version.
echo.
echo 1. Cleaning old installation...
cd /d "%~dp0\backend"
venv\Scripts\pip uninstall -y torch torchvision torchaudio

echo.
echo 2. Installing CUDA 12.1 version (Force Re-download)...
echo This WILL take some time (~2.5 GB). Please be patient.
echo.
venv\Scripts\pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

echo.
echo 3. Verifying Installation...
venv\Scripts\python.exe debug_gpu.py

echo.
echo If you see "True" above, it worked!
pause
