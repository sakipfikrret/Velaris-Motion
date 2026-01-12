@echo off
cd /d "%~dp0\backend"
venv\Scripts\python.exe debug_gpu.py
pause
