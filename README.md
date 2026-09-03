# Ransomware Detection System

Real-time ransomware detection using server-side file operation logs and machine learning models.

## Based on Research Paper

**Aranyi, G., Miseta, T., & Szucs, V. (2026)** - "Ransomware detection based on server-side file operation logs using machine learning" - Journal on Information Security, 2026:8

## Features

- **5 ML Models**: Random Forest, SVM, Decision Tree, AdaBoost, XGBoost
- **Real-time Monitoring**: Watchdog-based filesystem monitoring
- **Simulation Mode**: Realistic ransomware attack simulation
- **Live Dashboard**: WebSocket-powered real-time visualization
- **5 Operation Types**: Create, Write, Read, Rename, Delete per 1-second window

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

## Architecture

```
Browser (SPA) <--WebSocket--> Flask+SocketIO Server
                                    |
            +----------+-----------+-----------+
            |          |           |           |
     simulation.py  real_monitor.py  ml_engine.py  database.py
```

## License

Academic Research Use
