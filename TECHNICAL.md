# Ransomware Detection System

This document provides a detailed technical overview of the Ransomware Detection System.

## Architecture

The system is built on four core modules:

### 1. Database Layer (`database.py`)
SQLite-based storage managing four tables:
- **file_operations**: 1-second file operation windows with 5 features (nc, nw, nr, nm, nu)
- **detection_alerts**: ML detection events with XGBoost probability and confidence
- **attack_sessions**: Completed attack history with detection delays
- **system_stats**: Aggregate runtime statistics

### 2. ML Engine (`ml_engine.py`)
Trains and evaluates 5 classifiers:
- Random Forest
- Support Vector Machine (with probability calibration)
- Decision Tree
- AdaBoost
- XGBoost

Each model uses GridSearchCV hyperparameter optimization focused on maximizing recall (sensitivity).

### 3. File Operation Simulator (`simulation.py`)
Generates realistic server-side file operation patterns:
- 4 user profiles (Secretary, IT Admin, CEO, Remote Worker)
- 5 ransomware families (Ryuk, WannaCry, NotPetya, Lockbit, TeslaCrypt)
- Attack ramp-up effects
- Mixed benign noise during attacks

### 4. Real Folder Monitor (`real_monitor.py`)
Uses Watchdog (Windows ReadDirectoryChangesW) to observe:
- File creation (nc)
- File modification/write (nw)
- File read/open (nr)
- File rename/move (nm)
- File deletion (nu)

## Data Pipeline

1. Tick generation (simulated or real events)
2. Feature extraction: [nc, nw, nr, nm, nu] per 1-second window
3. ML inference across all 5 models
4. XGBoost primary detection decision
5. WebSocket emission to dashboard
6. Periodic SQLite logging

## Detection Flow

- **Simulation mode**: Has ground-truth labels for proper TP/FP/FN tracking
- **Real monitor mode**: Live detection without mislabeling (tracked separately)

## Web Dashboard

The dashboard (`templates/index.html` + `static/`) provides:
- Live operation counts per second
- XGBoost probability gauge with threat levels
- Real-time operation timeline chart
- Model comparison table
- Confusion matrices (live + test)
- Runtime metrics and detection rate
- Alert log with file details
- Event feed
