@echo off
title Ransomware Detection - Model Training
cd /d "%~dp0"
echo ==========================================
echo   TRAINING ML MODELS
echo ==========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Training all 5 models...
echo This may take several minutes with grid search.
echo.
python ml_engine.py
echo.
echo Model training complete.
pause
