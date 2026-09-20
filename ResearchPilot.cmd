@echo off
rem Start ResearchPilot with a visible window (double-click).
rem The window shows errors; Ctrl+C or the Quit button stops the server.
setlocal
cd /d "%~dp0"
title ResearchPilot

if exist ".venv\Scripts\python.exe" (
    set "RP_PYTHON=.venv\Scripts\python.exe"
) else (
    set "RP_PYTHON=python"
)

echo Starting ResearchPilot, the browser will open shortly ...
"%RP_PYTHON%" main.py --web --port 8000
if errorlevel 1 (
    echo.
    echo Start failed. Is ResearchPilot already running?
    pause
)
