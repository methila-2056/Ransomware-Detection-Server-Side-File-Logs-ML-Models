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

import os
import sys
import time
import threading
import json
from datetime import datetime

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from simulation import FileOperationSimulator, generate_training_data
from ml_engine import MLEngine, prepare_training_data
from database import Database
from real_monitor import RealFolderMonitor

# ──────────────────────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "ransomware-detection-2026"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ──────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────

simulator = FileOperationSimulator(attack_interval_range=(12, 25))
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

    print("=" * 60)
    print("  RANSOMWARE DETECTION SYSTEM")
    print("  ML Model Initialization")
    print("=" * 60)

    if ml_engine.load_models():
        print("Loaded pre-trained models successfully.")
        model_comparison = ml_engine.get_model_comparison()
        return True

    print("No saved models found. Training new models...")

    # Prefer the central server-side log (paper trains on logged operations).
    logged = db.get_training_data(limit=8000)
    if len(logged) >= 200:
        print(f"Training from {len(logged)} logged server-side operations...")
        data = logged
    else:
        print(f"Only {len(logged)} logged operations found. Generating synthetic training data...")
        data = generate_training_data(num_samples=8000)

    X, y = prepare_training_data(data)

    print(f"Training data: {X.shape[0]} samples")
    print(f"Class distribution: Benign={sum(y == 0)}, Attack={sum(y == 1)}")
    print()

    results = ml_engine.train(X, y, use_grid_search=True)
    ml_engine.save_models()
    model_comparison = ml_engine.get_model_comparison()

    print()
    print("Training complete. Models saved.")
    print("=" * 60)
    return True


# ──────────────────────────────────────────────────────────────
# Shared Processing Loop
# ──────────────────────────────────────────────────────────────

def process_tick(tick):
    """
    Process a single tick through ML inference and emit to clients.
    Shared between simulation and real monitoring modes.
    """
    global metrics, tick_history, alert_log

    # Run ML inference
    features = [tick["nc"], tick["nw"], tick["nr"], tick["nm"], tick["nu"]]
    predictions = ml_engine.predict(features)

    # Check XGBoost prediction
    xgb_pred = predictions.get("xgb", {})
    is_detected = xgb_pred.get("prediction", 0) == 1
    prob = xgb_pred.get("probability", 0)

    # Update metrics
    metrics["total_ticks"] += 1
    is_real = tick.get("source", "simulation") == "real"
    if is_real:
        if is_detected:
            metrics["total_detections"] += 1
    elif tick.get("is_attack", False):
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
        "timestamp_ns": tick.get("timestamp_ns", None),
        "nc": tick["nc"],
        "nw": tick["nw"],
        "nr": tick["nr"],
        "nm": tick["nm"],
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
    if len(tick_history) > 120:
        tick_history = tick_history[-120:]

    # Log to database periodically
    if metrics["total_ticks"] % 10 == 0:
        db.log_file_operation(tick)

    # Build alert entry if detection occurred
    alert_entry = None
    if is_detected and prob > 0.5:
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
            if len(alert_log) > 50:
                alert_log = alert_log[:50]
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


def _build_alert_message(tick, prob, xgb_pred):
    """Build a human-readable alert message with file details."""
    source = tick.get("source", "simulation")
    family = tick.get("family")
    nc, nw, nr, nm, nu = tick["nc"], tick["nw"], tick["nr"], tick["nm"], tick["nu"]
    file_events = tick.get("file_events", [])

    if source == "real":
        base_msg = (f"Suspicious activity on your system! "
                    f"nc={nc} nw={nw} nr={nr} nm={nm} nu={nu} "
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
    print("[SIM] Simulation started.")

    while simulation_running:
        tick = simulator.generate_next_tick()
        tick["timestamp_ns"] = time.time_ns()
        process_tick(tick)

        speed = simulator.state.simulation_speed
        time.sleep(1.0 / speed)

    print("[SIM] Simulation stopped.")


# ──────────────────────────────────────────────────────────────
# Real Monitoring Loop
# ──────────────────────────────────────────────────────────────

def monitoring_loop():
    """Real monitoring mode loop running in background thread."""
    global monitoring_running

    print("[MONITOR] Real-time monitoring started.")

    while monitoring_running:
        tick = real_monitor.get_tick()
        tick["timestamp"] = metrics["total_ticks"] + 1
        tick["timestamp_ns"] = time.time_ns()
        process_tick(tick)
        time.sleep(1.0)

    print("[MONITOR] Real-time monitoring stopped.")


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
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
def api_history():
    """Get tick history for charts."""
    return jsonify({
        "history": tick_history,
        "alerts": alert_log[:20],
    })


@app.route("/api/models")
def api_models():
    """Get model comparison data."""
    return jsonify({
        "models": model_comparison,
    })


@app.route("/api/folders")
def api_folders():
    """Get available monitoring folders."""
    return jsonify({
        "folders": real_monitor.get_default_folders(),
    })


@app.route("/api/monitor_events")
def api_monitor_events():
    """Get recent file events from real monitoring."""
    return jsonify({
        "events": real_monitor.get_recent_events(),
    })


@app.route("/api/stats")
def api_stats():
    """Get database statistics."""
    return jsonify({
        "db_stats": db.get_stats(),
        "recent_alerts": db.get_recent_alerts(10),
        "attack_sessions": db.get_attack_sessions(10),
    })


# ──────────────────────────────────────────────────────────────
# SocketIO Events
# ──────────────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    """Handle new client connection."""
    print("[WS] Client connected")
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
    print("[WS] Client disconnected")


# ── Simulation Events ──

@socketio.on("start_simulation")
def handle_start_simulation(data=None):
    """Start the simulation mode."""
    global simulation_running, simulation_thread, active_mode

    # Stop monitoring if running
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
def handle_stop_simulation(data=None):
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

    # Stop simulation if running
    if simulation_running:
        handle_stop_simulation()

    if monitoring_running:
        emit("mode_status", {"status": "already_running", "mode": "monitoring"})
        return

    # Get selected folders from frontend
    selected_folders = []
    if data and "folders" in data:
        selected_folders = [f for f in data["folders"] if f.get("selected")]
        selected_folders = [f["path"] for f in selected_folders if f.get("path")]

    # Use defaults if none selected
    if not selected_folders:
        selected_folders = None

    # Reinitialize monitor with selected folders
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
def handle_stop_monitoring(data=None):
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
    family = data.get("family", None)
    success = simulator.force_attack(family)
    emit("attack_forced", {"success": success, "family": family})


@socketio.on("set_speed")
def handle_set_speed(data):
    """Set simulation speed."""
    speed = float(data.get("speed", 1.0))
    simulator.set_speed(speed)
    emit("speed_updated", {"speed": speed})


@socketio.on("reset_metrics")
def handle_reset_metrics(data=None):
    """Reset all metrics."""
    global metrics, tick_history, alert_log
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
    emit("metrics_reset", {"status": "reset"})


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  RANSOMWARE DETECTION SYSTEM")
    print("  Based on Aranyi et al. (2026)")
    print("  Server-Side File Operation Monitoring with ML")
    print("=" * 60)
    print()

    # Initialize ML models
    init_ml_models()
    print()

    # Show available monitoring folders
    folders = real_monitor.get_default_folders()
    print("Available monitoring folders:")
    for f in folders:
        status = "READY" if f["exists"] else "NOT FOUND"
        marker = " [DEFAULT]" if f["selected"] else ""
        print(f"  {f['name']:12s} {f['path']:50s} [{status}]{marker}")
    print()

    # Start the server
    print("Starting web server at http://localhost:5000")
    print("Open your browser and navigate to the URL above.")
    print()

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
