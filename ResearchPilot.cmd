@echo off
rem ResearchPilot mit sichtbarem Fenster starten (Doppelklick).
rem Das Fenster zeigt Fehler an; Strg+C oder der "Beenden"-Button stoppt den Server.
setlocal
cd /d "%~dp0"
title ResearchPilot

if exist ".venv\Scripts\python.exe" (
    set "RP_PYTHON=.venv\Scripts\python.exe"
) else (
    set "RP_PYTHON=python"
)

echo ResearchPilot startet, der Browser oeffnet sich gleich ...
"%RP_PYTHON%" main.py --web --port 8000
if errorlevel 1 (
    echo.
    echo Start fehlgeschlagen. Laeuft ResearchPilot vielleicht schon?
    pause
)
