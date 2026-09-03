# Ransomware Detection System

Real-time ransomware detection using server-side file operation logs and machine learning.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Demo Mode vs Real Monitoring

The system has two operational modes:

### 1. Simulation Mode
- Simulates realistic server-side file operations for an SME environment
- 4 user profiles: Secretary, IT Admin, CEO, Remote Worker
- 5 ransomware families attack patterns
- Has ground-truth labels for proper TP/FP/FN tracking

### 2. Real Monitor Mode
- Monitors actual user folders (Desktop, Downloads) via watchdog
- Windows kernel-level ReadDirectoryChangesW notifications
- Completely passive - never modifies files
- No ground-truth labels (real detections tracked separately)

## File Operation Features

Each 1-second window captures 5 operation types:

| Feature | Operation | Description |
|---------|-----------|-------------|
| nc | Create | New file creation |
| nw | Write | File modification |
| nr | Read | File read/access |
| nm | Rename | File rename/move |
| nu | Delete | File unlinking |

## ML Models

Five ML models are trained for detection:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Decision Tree (DT)
- AdaBoost (ADA)
- XGBoost (XGB) - primary detector

All models optimized for recall (sensitivity) via GridSearchCV.
