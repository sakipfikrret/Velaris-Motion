@echo off
title Velaris Motion Engine v1.1.0
cls
echo ===================================================
echo   Velaris Motion Engine v1.1.0
echo   Deterministic Local ML Runtime (RTX 4060 / CUDA)
echo ===================================================
echo.

cd /d "%~dp0\backend"

:: 1. CHECK FOR PYTHON
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)

:: 2. DETERMINE PYTHON EXECUTABLE (venv or system)
set "PY_EXE=python"
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
) else (
    :: Check if venv should be created
    if not exist "venv" (
        echo [INFO] First time setup: Virtual environment not found.
        echo [INFO] You can use system Python or create venv.
        echo [INFO] Running preflight check with system Python...
    )
)

:: 3. RUN PRE-FLIGHT COMPATIBILITY CHECK
echo [INFO] Executing Velaris pre-flight diagnostics...
%PY_EXE% preflight.py
if %errorlevel% neq 0 (
    echo.
    echo [WARN] Pre-flight diagnostics reported critical issues or missing packages.
    echo If dependencies are missing, run install_dependencies.bat or force_gpu_install.bat.
    echo.
    set /p PROCEED="Do you want to attempt starting the engine anyway? (y/N): "
    if /i not "%PROCEED%"=="y" (
        pause
        exit /b 1
    )
)

:: 4. START ENGINE SERVER
cls
echo ===================================================
echo   Velaris Engine v1.1.0 is STARTING...
echo   Listening on: http://127.0.0.1:8000
echo   Press CTRL+C to stop the engine.
echo ===================================================
echo.

%PY_EXE% api.py
pause
