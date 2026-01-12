@echo off
echo Setting up Veo-X Environment...

:: Check if python exists
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.10+ and add to PATH.
    pause
    exit /b
)

:: Create Virtual Enviroment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate and Install
echo Activating venv and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup Complete! You can now run start_backend.bat
pause
