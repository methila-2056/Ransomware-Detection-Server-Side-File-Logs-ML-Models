"""
Ransomware Detection System - Main Application

Flask + SocketIO server that orchestrates:
- File operation simulation (demo mode)
- Real folder monitoring (Desktop + Downloads via watchdog)
- ML model inference on both modes
- Real-time dashboard updates
- Alert management

Based on Aranyi et al. (2026) paper:
"Ransomware detection based on server-side file operation logs using machine learning"
"""

import logging
import time
import threading
from functools import wraps

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from simulation import FileOperationSimulator, generate_training_data
from ml_engine import MLEngine, prepare_training_data
from database import Database
from real_monitor import RealFolderMonitor
import config

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO),
                    format=config.LOG_FORMAT)
log = logging.getLogger("ransomware.app")

# ──────────────────────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
limiter = Limiter(get_remote_address, app=app, default_limits=[config.RATE_LIMIT_DEFAULT])


# ──────────────────────────────────────────────────────────────
# Security Helpers
# ──────────────────────────────────────────────────────────────

def require_api_key(f):
    """Decorator: skip auth when API_KEY is empty (dev mode)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if config.API_KEY:
            provided = request.headers.get("X-API-Key", "")
            if provided != config.API_KEY:
                return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


def _validate_speed(speed):
    """Validate and clamp simulation speed."""
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return 1.0
    return max(0.1, min(10.0, speed))


def _validate_family(family):
    """Validate a ransomware family name or return None."""
    if family is None:
        return None
    if not isinstance(family, str):
        return None
    allowed = set(RansomwareFamilies.FAMILIES.keys()) if False else None
    from simulation import RansomwareFamilies
    allowed = set(RansomwareFamilies.FAMILIES.keys())
    return family if family in allowed else None


# ──────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────

simulator = FileOperationSimulator(
    attack_interval_range=(config.ATTACK_INTERVAL_MIN, config.ATTACK_INTERVAL_MAX)
)
real_monitor = RealFolderMonitor()
ml_engine = MLEngine()
db = Database()

simulation_thread = None
simulation_running = False
monitoring_thread = None
monitoring_running = False
active_mode = "idle"  # "idle", "simulation", "monitoring"

tick_history = []
alert_log = []
model_comparison = []
_previous_tick_was_attack = False

# Running metrics (shared between modes)
metrics = {
    "total_ticks": 0,
    "total_attacks": 0,
    "total_detections": 0,
    "total_fp": 0,
    "total_fn": 0,
    "current_streak": 0,
}


# ──────────────────────────────────────────────────────────────
# ML Model Initialization
# ──────────────────────────────────────────────────────────────

def init_ml_models():
    """Load or train ML models."""
    global model_comparison

    log.info("=" * 60)
    log.info("  RANSOMWARE DETECTION SYSTEM")
    log.info("  ML Model Initialization")
    log.info("=" * 60)

    if ml_engine.load_models():
        log.info("Loaded pre-trained models successfully.")
        model_comparison = ml_engine.get_model_comparison()
        return True

    log.info("No saved models found. Training new models...")
    log.info("Generating synthetic training data...")

    data = generate_training_data(num_samples=config.TRAINING_SAMPLES)
    X, y = prepare_training_data(data)

    log.info("Training data: %d samples", X.shape[0])
    log.info("Class distribution: Benign=%d, Attack=%d", sum(y == 0), sum(y == 1))

    results = ml_engine.train(X, y, use_grid_search=config.GRID_SEARCH)
    ml_engine.save_models()
    model_comparison = ml_engine.get_model_comparison()

    log.info("Training complete. Models saved.")
    log.info("=" * 60)
    return True


# ──────────────────────────────────────────────────────────────
# Shared Processing Loop
# ──────────────────────────────────────────────────────────────

def process_tick(tick):
    """
    Process a single tick through ML inference and emit to clients.
    Shared between simulation and real monitoring modes.
    """
    global metrics, tick_history, alert_log, _previous_tick_was_attack

    # Run ML inference
    features = [tick["nc"], tick["nr"], tick["nu"]]
    predictions = ml_engine.predict(features)

    # Check XGBoost prediction
    xgb_pred = predictions.get("xgb", {})
    is_detected = xgb_pred.get("prediction", 0) == 1
    prob = xgb_pred.get("probability", 0)

    # Update metrics
    metrics["total_ticks"] += 1
    if tick.get("is_attack", False):
        metrics["total_attacks"] += 1
        if is_detected:
            metrics["total_detections"] += 1
            metrics["current_streak"] += 1
        else:
            metrics["total_fn"] += 1
            metrics["current_streak"] = 0
    else:
        if is_detected:
            metrics["total_fp"] += 1
        metrics["current_streak"] = 0

    # Build tick data for frontend
    tick_data = {
        "timestamp": tick.get("timestamp", metrics["total_ticks"]),
        "nc": tick["nc"],
        "nr": tick["nr"],
        "nu": tick["nu"],
        "att": tick.get("att", 0),
        "user": tick.get("user", "Real System"),
        "family": tick.get("family"),
        "is_attack": tick.get("is_attack", False),
        "source": tick.get("source", "simulation"),
        "predictions": predictions,
        "file_events": tick.get("file_events", []),
    }

    tick_history.append(tick_data)
    if len(tick_history) > config.MAX_TICK_HISTORY:
        tick_history = tick_history[-config.MAX_TICK_HISTORY:]

    # Log to database periodically
    if metrics["total_ticks"] % config.DB_LOG_INTERVAL == 0:
        db.log_file_operation(tick)

    # Build alert entry if detection occurred
    alert_entry = None
    if is_detected and prob > config.ALERT_PROBABILITY_THRESHOLD:
        if metrics["current_streak"] == 1 or tick.get("source") == "real":
            alert_entry = {
                "timestamp": tick_data["timestamp"],
                "type": "attack_detected",
                "family": tick.get("family"),
                "probability": prob,
                "confidence": xgb_pred.get("confidence", 0),
                "source": tick.get("source", "simulation"),
                "message": _build_alert_message(tick, prob, xgb_pred),
            }
            alert_log.insert(0, alert_entry)
            if len(alert_log) > config.MAX_ALERT_LOG:
                alert_log = alert_log[:config.MAX_ALERT_LOG]
            db.log_detection_alert(tick, predictions)

    # Emit to all connected clients
    socketio.emit("tick_update", {
        "tick": tick_data,
        "metrics": metrics,
        "mode": active_mode,
        "alert": alert_entry,
        "live_cm": {
            "tp": metrics["total_detections"],
            "fp": metrics["total_fp"],
            "tn": metrics["total_ticks"] - metrics["total_attacks"] - metrics["total_fp"],
            "fn": metrics["total_fn"],
        },
    })

    # Detect attack end and persist session
    is_currently_attack = tick.get("is_attack", False)
    if _previous_tick_was_attack and not is_currently_attack:
        attacks = simulator.state.attack_history
        if attacks:
            last = attacks[-1]
            db.log_attack_session({
                "attack_id": last.get("attack_id"),
                "family": last.get("family"),
                "start_second": last.get("start_second"),
                "end_second": tick_data["timestamp"],
                "duration": last.get("duration"),
                "detected_by": "XGBoost" if metrics["current_streak"] > 0 else "None",
                "detection_delay": 0,
            })
    _previous_tick_was_attack = is_currently_attack

    # Periodic database cleanup
    if metrics["total_ticks"] % 500 == 0 and metrics["total_ticks"] > 0:
        db.clear_old_data(keep_last_n=2000)


def _build_alert_message(tick, prob, xgb_pred):
    """Build a human-readable alert message with file details."""
    source = tick.get("source", "simulation")
    family = tick.get("family")
    nc, nr, nu = tick["nc"], tick["nr"], tick["nu"]
    file_events = tick.get("file_events", [])

    if source == "real":
        base_msg = (f"Suspicious activity on your system! "
                    f"nc={nc} nr={nr} nu={nu} "
                    f"(XGB: {prob:.1%})")
        if file_events:
            creates = [e["path"] for e in file_events if e["type"] == "create"][:3]
            renames = [e["path"] for e in file_events if e["type"] == "rename"][:3]
            deletes = [e["path"] for e in file_events if e["type"] == "delete"][:3]
            details = []
            if creates:
                details.append(f"Created: {', '.join(creates)}")
            if renames:
                details.append(f"Renamed: {', '.join(renames)}")
            if deletes:
                details.append(f"Deleted: {', '.join(deletes)}")
            if details:
                base_msg += " | " + " | ".join(details)
        return base_msg
    else:
        family_str = family if family else "Unknown"
        return (f"Ransomware detected: {family_str} "
                f"(XGB probability: {prob:.1%})")


# ──────────────────────────────────────────────────────────────
# Simulation Loop
# ──────────────────────────────────────────────────────────────

def simulation_loop():
    """Simulation mode loop running in background thread."""
    global simulation_running

    simulator.start()
    log.info("Simulation started.")

    while simulation_running:
        tick = simulator.generate_next_tick()
        process_tick(tick)

        speed = simulator.state.simulation_speed
        time.sleep(1.0 / speed)

    log.info("Simulation stopped.")


# ──────────────────────────────────────────────────────────────
# Real Monitoring Loop
# ──────────────────────────────────────────────────────────────

def monitoring_loop():
    """Real monitoring mode loop running in background thread."""
    global monitoring_running

    log.info("Real-time monitoring started.")

    while monitoring_running:
        tick = real_monitor.get_tick()
        tick["timestamp"] = metrics["total_ticks"] + 1
        process_tick(tick)
        time.sleep(1.0)

    log.info("Real-time monitoring stopped.")


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
@limiter.exempt
def api_status():
    """Get current system status."""
    return jsonify({
        "active_mode": active_mode,
        "simulation_running": simulation_running,
        "monitoring_running": monitoring_running,
        "models_trained": ml_engine.is_trained,
        "metrics": metrics,
        "sim_state": simulator.get_state(),
        "monitor_state": real_monitor.get_status(),
        "model_comparison": model_comparison,
    })


@app.route("/api/history")
@limiter.exempt
def api_history():
    """Get tick history for charts."""
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(100, limit))
    return jsonify({
        "history": tick_history,
        "alerts": alert_log[:limit],
    })


@app.route("/api/models")
@limiter.exempt
def api_models():
    """Get model comparison data."""
    return jsonify({
        "models": model_comparison,
    })


@app.route("/api/folders")
@limiter.exempt
def api_folders():
    """Get available monitoring folders."""
    return jsonify({
        "folders": real_monitor.get_default_folders(),
    })


@app.route("/api/monitor_events")
@require_api_key
def api_monitor_events():
    """Get recent file events from real monitoring."""
    return jsonify({
        "events": real_monitor.get_recent_events(),
    })


@app.route("/api/stats")
@require_api_key
def api_stats():
    """Get database statistics."""
    return jsonify({
        "db_stats": db.get_stats(),
        "recent_alerts": db.get_recent_alerts(10),
        "attack_sessions": db.get_attack_sessions(10),
    })


@app.route("/api/export")
@require_api_key
def api_export():
    """Export full run data as a downloadable JSON file."""
    import json as _json
    payload = {
        "metrics": metrics,
        "tick_history": tick_history,
        "alerts": alert_log[:config.MAX_ALERT_LOG],
        "model_comparison": model_comparison,
        "db_stats": db.get_stats(),
        "db_alerts": db.get_recent_alerts(100),
        "db_sessions": db.get_attack_sessions(100),
    }
    data = _json.dumps(payload, indent=2, default=str)
    return Response(
        data,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=run_data.json"},
    )


# ──────────────────────────────────────────────────────────────
# SocketIO Events
# ──────────────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    """Handle new client connection."""
    log.debug("Client connected")
    emit("initial_data", {
        "history": tick_history,
        "alerts": alert_log[:20],
        "metrics": metrics,
        "sim_state": simulator.get_state(),
        "monitor_state": real_monitor.get_status(),
        "model_comparison": model_comparison,
        "models_trained": ml_engine.is_trained,
        "active_mode": active_mode,
        "folders": real_monitor.get_default_folders(),
        "live_cm": {
            "tp": metrics["total_detections"],
            "fp": metrics["total_fp"],
            "tn": metrics["total_ticks"] - metrics["total_attacks"] - metrics["total_fp"],
            "fn": metrics["total_fn"],
        },
    })


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    log.debug("Client disconnected")


# ── Simulation Events ──

@socketio.on("start_simulation")
def handle_start_simulation():
    """Start the simulation mode."""
    global simulation_running, simulation_thread, active_mode

    if monitoring_running:
        handle_stop_monitoring()

    if simulation_running:
        emit("mode_status", {"status": "already_running", "mode": "simulation"})
        return

    simulation_running = True
    active_mode = "simulation"
    simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
    simulation_thread.start()
    emit("mode_status", {"status": "started", "mode": "simulation"})


@socketio.on("stop_simulation")
def handle_stop_simulation():
    """Stop the simulation mode."""
    global simulation_running, active_mode

    simulation_running = False
    simulator.stop()
    if active_mode == "simulation":
        active_mode = "idle"
    emit("mode_status", {"status": "stopped", "mode": "idle"})


# ── Real Monitoring Events ──

@socketio.on("start_monitoring")
def handle_start_monitoring(data=None):
    """Start real folder monitoring mode."""
    global monitoring_running, monitoring_thread, active_mode

    if simulation_running:
        handle_stop_simulation()

    if monitoring_running:
        emit("mode_status", {"status": "already_running", "mode": "monitoring"})
        return

    selected_folders = []
    if data and isinstance(data, dict) and "folders" in data:
        raw = data["folders"]
        if isinstance(raw, list):
            selected_folders = [f for f in raw
                                if isinstance(f, dict) and f.get("selected") and f.get("path")]
            selected_folders = [f["path"] for f in selected_folders]

    if not selected_folders:
        selected_folders = None

    global real_monitor
    real_monitor = RealFolderMonitor(folders=selected_folders)

    result = real_monitor.start()

    if result["success"]:
        monitoring_running = True
        active_mode = "monitoring"
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        emit("mode_status", {
            "status": "started",
            "mode": "monitoring",
            "folders": result["folders"],
        })
    else:
        emit("mode_status", {
            "status": "error",
            "message": result["message"],
        })


@socketio.on("stop_monitoring")
def handle_stop_monitoring():
    """Stop real folder monitoring mode."""
    global monitoring_running, active_mode

    monitoring_running = False
    real_monitor.stop()
    if active_mode == "monitoring":
        active_mode = "idle"
    emit("mode_status", {"status": "stopped", "mode": "idle"})


# ── Common Events ──

@socketio.on("force_attack")
def handle_force_attack(data=None):
    """Force an immediate attack (simulation mode only)."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        data = {}
    family = _validate_family(data.get("family"))
    success = simulator.force_attack(family)
    emit("attack_forced", {"success": success, "family": family})


@socketio.on("set_speed")
def handle_set_speed(data):
    """Set simulation speed."""
    if not isinstance(data, dict):
        data = {}
    speed = _validate_speed(data.get("speed", 1.0))
    simulator.set_speed(speed)
    emit("speed_updated", {"speed": speed})


@socketio.on("reset_metrics")
def handle_reset_metrics():
    """Reset all metrics."""
    global metrics, tick_history, alert_log, _previous_tick_was_attack
    metrics = {
        "total_ticks": 0,
        "total_attacks": 0,
        "total_detections": 0,
        "total_fp": 0,
        "total_fn": 0,
        "current_streak": 0,
    }
    tick_history = []
    alert_log = []
    _previous_tick_was_attack = False
    emit("metrics_reset", {"status": "reset"})


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  RANSOMWARE DETECTION SYSTEM")
    log.info("  Based on Aranyi et al. (2026)")
    log.info("  Server-Side File Operation Monitoring with ML")
    log.info("=" * 60)

    init_ml_models()

    folders = real_monitor.get_default_folders()
    log.info("Available monitoring folders:")
    for f in folders:
        status = "READY" if f["exists"] else "NOT FOUND"
        marker = " [DEFAULT]" if f["selected"] else ""
        log.info("  %-12s %-50s [%s]%s", f["name"], f["path"], status, marker)

    log.info("Starting web server at http://localhost:%d", config.PORT)

    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        allow_unsafe_werkzeug=True,
    )
