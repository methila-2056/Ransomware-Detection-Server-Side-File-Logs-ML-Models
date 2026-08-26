"""
Ransomware Detection System - Configuration
Centralized configuration for all system parameters.
"""
import os

# Server
HOST = "0.0.0.0"
PORT = 5000
SECRET_KEY = os.environ.get("SECRET_KEY", "ransomware-detection-2026")
DEBUG = False

# ML Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PREFIX = "ransomware_"
GRID_SEARCH = False
TRAINING_SAMPLES = 8000

# Simulation
ATTACK_INTERVAL_MIN = 12
ATTACK_INTERVAL_MAX = 25
DEFAULT_SPEED = 1.0
MAX_TICK_HISTORY = 120

# Monitoring
HOME = os.path.expanduser("~")
DEFAULT_FOLDERS = [
    os.path.join(HOME, "Desktop"),
    os.path.join(HOME, "Downloads"),
]
MONITOR_INTERVAL = 1.0

# Alert
MAX_ALERT_LOG = 50
ALERT_PROBABILITY_THRESHOLD = 0.5

# Database
DB_PATH = os.path.join(os.path.dirname(__file__), "ransomware_detection.db")
DB_LOG_INTERVAL = 10
