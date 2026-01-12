@echo off
title Veo-X Website
echo ===================================================
echo   Starting Veo-X Web Interface
echo ===================================================

:: Ensure we are in the script's directory
cd /d "%~dp0"

echo [INFO] Starting Next.js Dev Server...
echo [TIP] Open http://localhost:3000 in your browser once ready.
echo.

npm run dev

pause
