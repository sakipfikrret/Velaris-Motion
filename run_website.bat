@echo off
title Velaris Motion UI v1.1.0
echo ===================================================
echo   Starting Velaris Motion Web Interface
echo ===================================================
echo.

:: Ensure we are in the script's directory
cd /d "%~dp0"

:: Check if node_modules exists, install if missing
if not exist "node_modules" (
    echo [INFO] node_modules not detected. Installing dependencies via npm...
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] npm install failed. Please check your Node.js and network setup.
        pause
        exit /b 1
    )
)

echo [INFO] Starting Next.js Dev Server...
echo [TIP] Open http://localhost:3000 in your browser once ready.
echo.

call npm run dev

pause
