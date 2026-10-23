# Ransomware Detection System

[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-red.svg)](https://xgboost.readthedocs.io/)

> Real-time ransomware detection from server-side file operation telemetry using ensemble machine learning models.

## Overview

This system implements a real-time ransomware detection framework based on server-side file operation logs. By analyzing per-second aggregates of file operations (`nc`, `nw`, `nr`, `nm`, `nu`), the platform leverages multiple supervised machine learning models to identify suspicious activity indicative of ransomware behavior.

The implementation follows the methodology described in:

**Aranyi, G., Miseta, T., & Szucs, V. (2026).** "Ransomware detection based on server-side file operation logs using machine learning." *Journal on Information Security*, 2026:8.

## Key Features

- **Ensemble ML Pipeline** – Six production-ready models (Random Forest, SVM, Decision Tree, AdaBoost, XGBoost, Logistic Regression) with standardized preprocessing and calibration.
- **Real-Time Detection** – Live inference over streaming file-operation windows with low-latency WebSocket updates.
- **Dual Monitoring Modes** – Simulated attack generation for testing and watchdog-based real filesystem monitoring for deployment.
- **Interactive Dashboard** – Flask + Socket.IO web interface with live metrics, predictions per model, alert stream, and confusion-matrix tracking.
- **Feature Set** – Uses five operation counters per 1-second window: `nc` (creates), `nw` (writes), `nr` (reads), `nm` (moves/renames), `nu` (updates).
- **Attack Session Logging** – Persists detection sessions, alerts, and telemetry to SQLite for analysis and reporting.
- **Research Artifacts** – Includes a full research pipeline (`ml/`), ablations, robustness studies, benchmarks, and manuscript tooling.

## Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/methila-2056/Ransomware-Detection-Server-Side-File-Logs-ML-Models.git
cd Ransomware-Detection-Server-Side-File-Logs-ML-Models

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the web server
python app.py
```

Navigate to [http://localhost:5000](http://localhost:5000) to access the live dashboard.

### Training Models

Pre-trained models are included in `models/`. To retrain from scratch:

```bash
# Windows
train_models.bat

# Or directly
python ml_engine.py
```

## Architecture

```text
┌─────────────┐  WebSocket/HTTP  ┌─────────────────────────────────┐
│   Browser    │◄────────────────►│  Flask + Flask-SocketIO Server │
│   Dashboard  │                  │           (app.py)              │
└─────────────┘                  └─────────────┬─────────────┘
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        │                      │                      │
                   simulation.py        real_monitor.py         ml_engine.py
                        │                      │                      │
                      events              watchdog events      6-model ensemble
                        │                      │                      │
                        └──────────┬───────────┴──────────┬───────────┘
                                   │                      │
                                database.py           config.py
                                   │                      │
                                SQLite DB            Settings
```

## Project Structure

| Directory/File | Description |
|---|---|
| `app.py` | Main Flask application, WebSocket handlers, and inference pipeline |
| `ml_engine.py` | Production ML engine with 6-model training, loading, and prediction |
| `ml/` | Research pipeline (training, evaluation, scaling, ensembling, statistics) |
| `simulation.py` | Ransomware attack simulation with realistic behavioral patterns |
| `real_monitor.py` | Watchdog-based real-time filesystem monitoring |
| `database.py` | SQLite persistence for telemetry, alerts, and attack sessions |
| `models/` | Serialized models, scalers, and metrics artifacts |
| `templates/` & `static/` | Web dashboard UI and client-side logic |
| `tests/` | Comprehensive test suite (unit, integration, security) |
| `reports/`, `results/` | Research outputs, metrics, and visualizations |

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Flask 3.1, Flask-SocketIO, Python Socket.IO |
| Machine Learning | scikit-learn 1.6.1, XGBoost 2.1.3, NumPy, Pandas |
| Real-time Monitoring | Watchdog 6.0 |
| Database | SQLite (built-in) |
| Frontend | Vanilla JavaScript, HTML5, CSS3 |
| Deployment | Docker, Gunicorn/Eventlet |

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, submitting pull requests, and coding standards.

## Security

For security considerations and responsible disclosure, please review [SECURITY.md](SECURITY.md).

## License

This project is released for **academic research use** only. See [LICENSE](LICENSE) for full terms.

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@article{aranyi2026ransomware,
  title={Ransomware detection based on server-side file operation logs using machine learning},
  author={Aranyi, Gabor and Miseta, Tamas and Szucs, Viktor},
  journal={Journal on Information Security},
  year={2026},
  volume={2026},
  number={8}
}
```
