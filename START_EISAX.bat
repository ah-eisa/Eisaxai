@echo off
title EisaX AI Launcher
echo.
echo   ========================================
echo     EisaX AI - Magic Launcher
echo   ========================================
echo.

cd /d "%~dp0"

:: Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo   [OK] Virtual environment activated
) else (
    echo   [!] No venv found, using system Python
)

echo.
echo   Starting everything...
echo.

python run.py

pause
