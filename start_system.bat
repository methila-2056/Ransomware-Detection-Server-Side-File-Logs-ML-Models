@echo off
title Ransomware Detection System
cd /d "%~dp0"
echo ==========================================
echo   RANSOMWARE DETECTION SYSTEM
echo   Server-Side File Operation Monitoring
echo ==========================================
echo.
echo Installing dependencies (if needed)...
pip install -r requirements.txt
echo.
echo Starting server at http://localhost:5000
echo.
python app.py
pause
