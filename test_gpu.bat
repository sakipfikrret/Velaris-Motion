@echo off
title Velaris Hardware Diagnostics
cls
cd /d "%~dp0\backend"

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe preflight.py
) else (
    python preflight.py
)
pause
