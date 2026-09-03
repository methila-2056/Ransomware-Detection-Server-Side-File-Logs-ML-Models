# Project Structure

```
RANSOMWARE DETECTION/
│
├── app.py                  # Main Flask + SocketIO server
├── database.py             # SQLite database layer
├── ml_engine.py            # ML training & inference engine
├── real_monitor.py         # Real folder watchdog monitor
├── simulation.py           # File operation simulator
├── requirements.txt        # Python dependencies
│
├── models/                 # Pre-trained models (5-feature)
│   ├── ransomware_rf.joblib
│   ├── ransomware_xgb.joblib
│   ├── ransomware_svm.joblib
│   ├── ransomware_dt.joblib
│   ├── ransomware_ada.joblib
│   ├── ransomware_scaler.joblib
│   └── ransomware_metrics.joblib
│
├── models_backup_3feature/ # Legacy 3-feature backup models
│   ├── ransomware_rf.joblib
│   ├── ransomware_xgb.joblib
│   ├── ransomware_svm.joblib
│   ├── ransomware_dt.joblib
│   ├── ransomware_ada.joblib
│   ├── ransomware_scaler.joblib
│   └── ransomware_metrics.joblib
│
├── templates/
│   └── index.html          # Main dashboard SPA
│
└── static/
    ├── css/
    │   └── style.css       # Cyber-themed CSS
    └── js/
        └── dashboard.js    # Frontend logic
```
